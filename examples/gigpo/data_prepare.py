"""Prepare WebShop assets, search indexes, goal splits, and task rows.

Product JSON files are not supervised prompt and answer data. Task rows contain
only goal indexes. WebShop produces instructions, pages, and rewards during
reset and step calls, so no placeholder parquet file is created.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import importlib
import json
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from huggingface_hub import hf_hub_download

WEBSHOP_REPOSITORY = "https://github.com/princeton-nlp/WebShop.git"
WEBSHOP_REVISION = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"


def asset_manifest(files: dict[str, Path]) -> dict:
    """Return source metadata and SHA-256 values for named local assets."""
    return {
        "webshop_repository": WEBSHOP_REPOSITORY,
        "webshop_revision": WEBSHOP_REVISION,
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        },
    }


@dataclass(frozen=True)
class DataConfig:
    """Configure pinned data, goal splits, shuffling, and evaluation limits.

    ``repo_id`` and ``revision`` identify the dataset. ``local_files_only``
    controls cache-only access. ``seed`` affects training-row shuffling,
    ``train_epochs`` controls traversal count, and ``eval_limit`` selects an
    explicit debug subset when it is not ``None``.
    """

    repo_id: str = "YWZBrandon/webshop-data"
    revision: str = "ce990fff5aee388db2706f07820c578ab68e0453"
    local_files_only: bool = True
    split_protocol: str = "official"
    seed: int = 42
    train_epochs: int = 2
    eval_limit: int | None = None

    def __post_init__(self):
        """Validate split, seed, traversal count, and subset size."""
        if self.split_protocol not in {"verl", "official"}:
            raise ValueError("data.split_protocol must be verl or official.")
        for name, minimum in (("seed", 0), ("train_epochs", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"data.{name} must be an integer >= {minimum}.")
        if self.eval_limit is not None and (
            type(self.eval_limit) is not int or self.eval_limit < 1
        ):
            raise ValueError("data.eval_limit must be a positive integer or null.")
        if type(self.local_files_only) is not bool:
            raise ValueError("data.local_files_only must be boolean.")


def cached_files(config: DataConfig) -> dict[str, Path]:
    """Resolve required files through the default Hugging Face cache.

    The loader opens all three files. Human instructions are not used when
    ``human_goals`` is disabled. Missing assets retain the original cache error.
    """
    return {
        name: Path(
            hf_hub_download(
                repo_id=config.repo_id,
                repo_type="dataset",
                revision=config.revision,
                filename=name,
                local_files_only=config.local_files_only,
            )
        )
        for name in (
            "items_shuffle_1000.json",
            "items_ins_v2_1000.json",
            "items_human_ins.json",
        )
    }


def goal_ranges(count: int, protocol: str) -> dict[str, range]:
    """Return train, evaluation, and test ranges for a goal population.

    Indexes identify the seeded and shuffled server goal list rather than
    products. The ``verl`` protocol evaluates on the test interval, while the
    ``official`` protocol keeps evaluation and test intervals separate. Both
    protocols split goals while sharing the same product catalog.
    """
    if protocol not in {"verl", "official"}:
        raise ValueError("Unknown split protocol.")
    start = 500 if protocol == "verl" else 1500
    if type(count) is not int or count <= start:
        raise ValueError(
            f"Need more than {start} goals for protocol={protocol}; got {count}."
        )
    return {
        "train": range(start, count),
        "eval": range(500) if protocol == "verl" else range(500, 1500),
        "test": range(500),
    }


def task_rows(goals: list[dict], config: DataConfig, split: str) -> list[dict]:
    """Convert ordered goals into replayable task rows for one split.

    Only training rows are shuffled between epochs. AgentRolloutWorkflow expands
    multiple candidates for one goal, so rows must not preduplicate candidate
    groups. ``task_id`` binds the split and goal index without exposing hidden
    environment state or reward answers in model messages.
    """
    indices = list(goal_ranges(len(goals), config.split_protocol)[split])
    if split != "train" and config.eval_limit is not None:
        indices = indices[: config.eval_limit]
    rows = []
    rng = random.Random(config.seed)
    for _ in range(config.train_epochs if split == "train" else 1):
        if split == "train":
            rng.shuffle(indices)
        rows.extend(
            {"task_id": f"webshop-{split}-{i}", "goal_index": i} for i in indices
        )
    return rows


def goal_fingerprint(goals: list[dict]) -> str:
    """Fingerprint the complete ordered goal and price-constraint payload."""
    payload = json.dumps(goals, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def link_assets(source: Path, files: dict[str, Path]):
    """Link cached assets into the source tree without overwriting files."""
    directory = source / "data"
    directory.mkdir(parents=True, exist_ok=True)
    for name, target in files.items():
        destination = directory / name
        if destination.exists() and filecmp.cmp(destination, target, shallow=False):
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing WebShop data: {destination}"
            )
        destination.symlink_to(target.resolve())


def prepare(experiment, *, download: bool = False):
    """Prepare source assets and a matching index for an experiment.

    ``download=True`` permits missing cache entries to be downloaded. Unknown
    indexes and mismatched links fail explicitly. A matching prepared checkout
    can be reused safely.
    """
    source = experiment.source_dir
    if not (source / "web_agent_site").is_dir():
        raise FileNotFoundError(
            f"Clone the pinned Princeton WebShop checkout first: {source}; see README.md."
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != WEBSHOP_REVISION:
        raise ValueError(
            f"Expected WebShop source revision {WEBSHOP_REVISION}; got {revision}."
        )
    files = cached_files(
        replace(experiment.data, local_files_only=False)
        if download
        else experiment.data
    )
    expected = asset_manifest(files)
    manifest = source / ".triotune-webshop.json"
    index = source / "search_engine/indexes_1k"
    if index.exists():
        if not manifest.is_file() or json.loads(manifest.read_text()) != expected:
            raise FileExistsError(
                "Existing indexes_1k has unknown/different data; use a clean source checkout."
            )
        link_assets(source, files)
        return expected
    link_assets(source, files)
    sys.path.insert(0, str(source))
    engine = importlib.import_module("web_agent_site.engine.engine")
    if not Path(engine.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Another WebShop checkout is already imported.")
    # Build only the 1k-product document collection used by this experiment.
    # The loader reads attributes from the source data directory, whose links
    # were validated above.
    random_state = random.getstate()
    try:
        random.seed(experiment.environment_seed)
        products, *_ = engine.load_products(
            filepath=str(files["items_shuffle_1000.json"]),
            num_products=1000,
            human_goals=False,
        )
    finally:
        random.setstate(random_state)
    # Rename the temporary directory only after indexing succeeds so a failed
    # build cannot be mistaken for a complete index.
    with tempfile.TemporaryDirectory(
        prefix="triotune-index-", dir=index.parent
    ) as temporary:
        temporary = Path(temporary)
        documents = temporary / "documents"
        documents.mkdir()
        with (documents / "documents.jsonl").open("w", encoding="utf-8") as stream:
            for product in products:
                options = ", and ".join(
                    f"{name}: {', '.join(values)}"
                    for name, values in product.get("options", {}).items()
                )
                contents = " ".join(
                    [
                        product["Title"],
                        product["Description"],
                        product["BulletPoints"][0],
                        options,
                    ]
                ).lower()
                stream.write(
                    json.dumps(
                        {
                            "id": product["asin"],
                            "contents": contents,
                            "product": product,
                        }
                    )
                    + "\n"
                )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                str(documents),
                "--index",
                str(temporary / "index"),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            check=True,
        )
        (temporary / "index").rename(index)
    manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return expected


def main(argv=None):
    """Prepare data from CLI options and print a JSON asset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "gigpo-webshop-configs.yaml",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing small assets into the default HF cache",
    )
    args = parser.parse_args(argv)
    # Delay the orchestration import so importing data utilities does not create
    # a training service or introduce a circular dependency.
    from .train import load_experiment

    experiment = load_experiment(args.config)
    result = prepare(experiment, download=args.download)
    print(json.dumps({"source_dir": str(experiment.source_dir), **result}, indent=2))


if __name__ == "__main__":
    main()
