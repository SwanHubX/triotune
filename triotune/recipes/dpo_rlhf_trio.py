"""Prepare ModelScope HH-RLHF pairs and train them with PyTRIO DPOTrainer."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import random
import re
import shutil
import tomllib  # retain legacy TOML loading for explicitly supplied files
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trio_cookbook.trainer.dpo import DPOConfig, DPOTrainer
from trio_cookbook.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "triotune" / "configs" / "dpo_rlhf.yaml"
SUBSETS = {"helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"}
TURN = re.compile(r"\n\n(Human|Assistant):")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HH-RLHF with PyTRIO DPOTrainer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--sample-size", type=positive_int)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report-to", choices=("swanlab",))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    settings = load_yaml_config(path)
    for section, fields in {
        "model": ("name", "lora_rank"),
        "data": ("file", "cache_dir", "subsets", "sample_size"),
        "dpo": ("max_steps", "batch_size"),
    }.items():
        values = settings.get(section)
        if not isinstance(values, dict) or any(field not in values for field in fields):
            raise ValueError(f"Config {path} needs [{section}] with {', '.join(fields)}")
    if not isinstance(settings.get("swanlab", {}), dict):
        raise ValueError("[swanlab] must be a table")
    return settings


def modelscope_url(repo: str, revision: str, remote_path: str) -> str:
    namespace, separator, name = repo.partition("/")
    if not namespace or not separator or not name:
        raise ValueError("ModelScope dataset repo must be namespace/name")
    query = urllib.parse.urlencode(
        {"Source": "SDK", "Revision": revision, "FilePath": remote_path, "View": "False"}
    )
    return f"https://www.modelscope.cn/api/v1/datasets/{namespace}/{name}/repo?{query}"


def download_if_needed(url: str, destination: Path, *, force: bool) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "trio-cookbook-dpo"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_transcript(text: str) -> list[dict[str, str]] | None:
    turns = list(TURN.finditer(text))
    if not turns or turns[0].start() != 0 or turns[0].group(1) != "Human":
        return None
    messages = []
    for index, turn in enumerate(turns):
        end = turns[index + 1].start() if index + 1 < len(turns) else len(text)
        content = text[turn.end() : end].strip()
        if not content:
            return None
        messages.append({
            "role": "user" if turn.group(1) == "Human" else "assistant",
            "content": content,
        })
    return messages


def preference_row(source: Mapping[str, Any]) -> dict[str, Any] | None:
    chosen, rejected = source.get("chosen"), source.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    chosen_turns, rejected_turns = parse_transcript(chosen), parse_transcript(rejected)
    if not chosen_turns or not rejected_turns:
        return None
    messages = chosen_turns[:-1]
    if (
        not messages
        or messages != rejected_turns[:-1]
        or messages[-1]["role"] != "user"
        or chosen_turns[-1]["role"] != "assistant"
        or rejected_turns[-1]["role"] != "assistant"
        or chosen_turns[-1]["content"] == rejected_turns[-1]["content"]
    ):
        return None
    return {
        "messages": messages,
        "chosen": [chosen_turns[-1]],
        "rejected": [rejected_turns[-1]],
    }


def prepare_dataset(
    data_file: Path,
    cache_dir: Path,
    settings: Mapping[str, Any],
    *,
    rebuild: bool,
    force: bool,
) -> None:
    if data_file.is_file() and not (rebuild or force):
        return
    subsets = settings["subsets"]
    if (
        not isinstance(subsets, list)
        or not subsets
        or any(not isinstance(subset, str) or subset not in SUBSETS for subset in subsets)
    ):
        raise ValueError(f"data.subsets must contain supported HH-RLHF subsets: {sorted(SUBSETS)}")
    split = settings.get("split", "train")
    if split not in {"train", "test"}:
        raise ValueError("data.split must be train or test")
    sample_size = settings["sample_size"]
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("data.sample_size must be positive")

    rng = random.Random(settings.get("seed", 42))
    sampled: list[dict[str, Any]] = []
    seen = 0
    for subset in subsets:
        remote_path = f"{subset}/{split}.jsonl.gz"
        raw_file = cache_dir / remote_path
        download_if_needed(
            modelscope_url(
                settings.get("repo", "Anthropic/hh-rlhf"),
                settings.get("revision", "master"),
                remote_path,
            ),
            raw_file,
            force=force,
        )
        with gzip.open(raw_file, "rt", encoding="utf-8") as source:
            for line in source:
                raw = json.loads(line)
                pair = preference_row(raw) if isinstance(raw, dict) else None
                if pair is None:
                    continue
                seen += 1
                if len(sampled) < sample_size:
                    sampled.append(pair)
                elif (slot := rng.randrange(seen)) < sample_size:
                    sampled[slot] = pair
    if not sampled:
        raise ValueError("No valid HH-RLHF preference pairs were found")
    rng.shuffle(sampled)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_file.with_suffix(data_file.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for row in sampled:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(data_file)
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(data_file: Path) -> list[dict[str, Any]]:
    with data_file.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or any(
        not isinstance(row, dict) or not all(k in row for k in ("messages", "chosen", "rejected"))
        for row in rows
    ):
        raise ValueError(f"Invalid or empty DPO dataset: {data_file}")
    return rows


def usable_rows(trainer: DPOTrainer, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop pairs that exceed the model's configured token limit."""
    usable = []
    for row in rows:
        try:
            trainer._preferences([row])
        except ValueError as error:
            if "exceeds max_length=" not in str(error):
                raise
        else:
            usable.append(row)
    if not usable:
        raise ValueError("No pairs fit dpo.max_length; increase it or choose another sample")
    return usable


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_config(args.config)
    data = dict(settings["data"])
    if args.sample_size is not None:
        data["sample_size"] = args.sample_size
    data_file = resolve_path(args.data_file or data["file"])
    cache_dir = resolve_path(args.cache_dir or data["cache_dir"])
    prepare_dataset(
        data_file,
        cache_dir,
        data,
        rebuild=args.rebuild_data or args.sample_size is not None,
        force=args.force_download,
    )
    rows = load_rows(data_file)
    print(f"Prepared {len(rows)} HH-RLHF pairs at {data_file}")
    if args.prepare_only:
        return

    dpo = dict(settings["dpo"])
    if args.steps is not None:
        dpo["max_steps"] = args.steps
    if args.batch_size is not None:
        dpo["batch_size"] = args.batch_size
    swanlab = dict(settings.get("swanlab", {}))
    if args.report_to is not None:
        swanlab["report_to"] = args.report_to
    trainer = DPOTrainer(
        model=settings["model"]["name"],
        args=DPOConfig(**dpo, **swanlab),
        lora_rank=settings["model"]["lora_rank"],
    )
    rows = usable_rows(trainer, rows)
    print(f"Training on {len(rows)} pairs after token-length filtering")
    trainer.train_dataset = itertools.cycle(rows)
    trainer.train()


if __name__ == "__main__":
    main()
