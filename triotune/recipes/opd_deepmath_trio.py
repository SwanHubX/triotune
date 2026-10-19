"""Async DeepMath OPD through the cookbook framework.

ModelScope parquet files provide questions only. OPDTrainer owns fresh student
rollouts, fixed teacher scores, distillation targets, updates, and logging.
The independent opd_async.py example is deliberately not imported or modified.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import math
import shutil
import tomllib  # retain legacy TOML loading for explicitly supplied files
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import pytrio as trio

from trio_cookbook.trainer.opd import OPDConfig, OPDTrainer
from trio_cookbook.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "triotune" / "configs" / "opd_deepmath.yaml"
DEEPMATH_SHARDS = 10


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-repo")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--num-shards", type=positive_int, help="Read the first N of 10 train shards.")
    parser.add_argument("--sample-size", type=int, help="Shuffle then select N questions; <=0 uses all loaded rows.")
    parser.add_argument("--question-suffix")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare questions without creating PyTRIO clients.")
    parser.add_argument("--base-model")
    parser.add_argument("--teacher-base-model")
    parser.add_argument("--teacher-model-path")
    parser.add_argument("--lora-rank", type=positive_int)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--group-size", type=positive_int, help="Completions per question; OPD allows 1.")
    parser.add_argument("--max-tokens", type=positive_int)
    parser.add_argument("--max-length", type=positive_int)
    parser.add_argument("--max-concurrent-sampling", type=positive_int)
    parser.add_argument("--max-concurrent-scoring", type=positive_int)
    parser.add_argument("--kl-penalty-coef", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--save-every", type=nonnegative_int)
    parser.add_argument("--seed", type=nonnegative_int)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--project")
    parser.add_argument("--workspace")
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    settings = load_yaml_config(path)
    required = {
        "model": ("name", "lora_rank"),
        "teacher": ("name",),
        "data": ("repo", "revision", "dataset_dir", "num_shards", "sample_size", "seed", "question_suffix"),
        "opd": ("max_steps", "batch_size"),
    }
    for section, fields in required.items():
        values = settings.get(section)
        if not isinstance(values, dict) or any(field not in values for field in fields):
            raise ValueError(f"Config needs [{section}] with {', '.join(fields)}.")
    for section in ("swanlab", "runtime"):
        if not isinstance(settings.get(section, {}), dict):
            raise ValueError(f"[{section}] must be a table.")
    return settings


def modelscope_file_url(repo: str, revision: str, file_path: str) -> str:
    if not isinstance(repo, str) or repo.count("/") != 1:
        raise ValueError("data.repo must be namespace/name.")
    namespace, name = repo.split("/")
    if not namespace or not name:
        raise ValueError("data.repo must be namespace/name.")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("data.revision must be a non-empty string.")
    query = urllib.parse.urlencode({
        "Source": "SDK", "Revision": revision, "FilePath": file_path, "View": "False",
    })
    return (
        "https://www.modelscope.cn/api/v1/datasets/"
        f"{urllib.parse.quote(namespace, safe='')}/{urllib.parse.quote(name, safe='')}/repo?{query}"
    )


def shard_name(index: int) -> str:
    return f"data/train-{index:05d}-of-{DEEPMATH_SHARDS:05d}.parquet"


def validate_data_settings(data: Mapping[str, Any]) -> None:
    # Validate before downloads or model initialization.
    modelscope_file_url(data["repo"], data["revision"], shard_name(0))
    shards = data["num_shards"]
    if isinstance(shards, bool) or not isinstance(shards, int) or not 1 <= shards <= DEEPMATH_SHARDS:
        raise ValueError("data.num_shards must be between 1 and 10.")
    for name in ("sample_size", "seed"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"data.{name} must be an integer.")
    if data["seed"] < 0:
        raise ValueError("data.seed must be nonnegative.")
    if not isinstance(data["question_suffix"], str):
        raise ValueError("data.question_suffix must be a string.")
    if not isinstance(data["dataset_dir"], str) or not data["dataset_dir"].strip():
        raise ValueError("data.dataset_dir must be a non-empty path.")


def download_if_needed(url: str, destination: Path, *, force: bool, timeout: float) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url}\n  -> {destination}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "trio-cookbook-opd"})
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_deepmath(data: Mapping[str, Any], *, force: bool, timeout: float) -> Dataset:
    """Download from ModelScope; datasets only reads the local parquet files."""
    validate_data_settings(data)
    directory = resolve_path(data["dataset_dir"])
    shard_paths = []
    for index in range(data["num_shards"]):
        remote_path = shard_name(index)
        local_path = directory / remote_path
        download_if_needed(
            modelscope_file_url(data["repo"], data["revision"], remote_path),
            local_path, force=force, timeout=timeout,
        )
        shard_paths.append(str(local_path))
    dataset = load_dataset(
        "parquet", data_files=shard_paths, split="train",
        cache_dir=str(directory / ".datasets_cache"),
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("DeepMath parquet reader must return a Dataset.")
    if not len(dataset) or "question" not in dataset.column_names:
        raise ValueError("DeepMath must be non-empty and contain a question column.")
    dataset = dataset.select_columns(["question"]).shuffle(seed=data["seed"])
    if data["sample_size"] > 0:
        dataset = dataset.select(range(min(data["sample_size"], len(dataset))))
    return dataset


def prompt_rows(dataset: Dataset, suffix: str) -> list[dict[str, Any]]:
    """Keep only questions; reference answers and R1 solutions are not targets."""
    rows = []
    for index, row in enumerate(dataset):
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"DeepMath row {index} needs a non-empty string question.")
        content = question.strip()
        if suffix:
            content += "\n\n" + suffix
        rows.append({"messages": [{"role": "user", "content": content}]})
    return rows


async def main(argv: list[str] | None = None) -> None:
    cli = parse_args(argv)
    settings = load_config(resolve_path(cli.config))
    data = dict(settings["data"])
    for option, field in (
        ("dataset_repo", "repo"), ("dataset_revision", "revision"),
        ("num_shards", "num_shards"), ("sample_size", "sample_size"),
        ("question_suffix", "question_suffix"), ("seed", "seed"),
    ):
        if (value := getattr(cli, option)) is not None:
            data[field] = value
    if cli.dataset_dir is not None:
        data["dataset_dir"] = str(cli.dataset_dir)
    validate_data_settings(data)

    opd = dict(settings["opd"])
    for option, field in (
        ("steps", "max_steps"), ("batch_size", "batch_size"), ("group_size", "num_generations"),
        ("max_tokens", "max_completion_length"), ("max_length", "max_length"),
        ("max_concurrent_sampling", "max_concurrent_sampling"),
        ("max_concurrent_scoring", "max_concurrent_scoring"),
        ("kl_penalty_coef", "kl_coef"), ("learning_rate", "learning_rate"),
        ("save_every", "save_every"), ("seed", "seed"),
    ):
        if (value := getattr(cli, option)) is not None:
            opd[field] = value
    if cli.enable_thinking is not None:
        opd["chat_template_kwargs"] = {
            **opd.get("chat_template_kwargs", {}), "enable_thinking": cli.enable_thinking,
        }
    logging = dict(settings.get("swanlab", {}))
    if cli.swanlab is not None:
        logging["report_to"] = "swanlab" if cli.swanlab else None
    for field in ("project", "workspace", "run_name"):
        if (value := getattr(cli, field)) is not None:
            logging[field] = value
    config = OPDConfig(**opd, **logging)
    if config.max_steps is None:
        raise ValueError("opd.max_steps must be bounded because this recipe cycles the prompt pool.")
    if config.prompt_key != "messages":
        raise ValueError("This recipe emits messages; opd.prompt_key must be 'messages'.")

    model = cli.base_model or settings["model"]["name"]
    teacher = cli.teacher_base_model or settings["teacher"]["name"]
    rank = cli.lora_rank if cli.lora_rank is not None else settings["model"]["lora_rank"]
    teacher_path = cli.teacher_model_path or settings["teacher"].get("model_path")
    if any(not isinstance(value, str) or not value.strip() for value in (model, teacher)):
        raise ValueError("Student and teacher names must be non-empty model IDs.")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("model.lora_rank must be positive.")
    if teacher_path is not None and (not isinstance(teacher_path, str) or not teacher_path.strip()):
        raise ValueError("teacher.model_path must be a non-empty checkpoint URI.")
    timeout = settings.get("runtime", {}).get("timeout", 600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("runtime.timeout must be finite and positive.")

    dataset = await asyncio.to_thread(load_deepmath, data, force=cli.force_download, timeout=timeout)
    rows = prompt_rows(dataset, data["question_suffix"])
    print(f"Prepared {len(rows)} DeepMath prompts from {data['repo']}")
    if cli.prepare_only:
        return

    trio.configure(timeout=timeout)
    # OPDTrainer client construction is synchronous; keep it off the event loop.
    trainer = await asyncio.to_thread(
        OPDTrainer, model=model, teacher_model=teacher, teacher_model_path=teacher_path,
        args=config, lora_rank=rank, train_dataset=itertools.cycle(rows),
    )
    state = await trainer.train_async()
    print(f"steps={state.global_step} checkpoint={state.checkpoint_path} metrics={state.metrics}")


if __name__ == "__main__":
    asyncio.run(main())
