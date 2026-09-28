"""Prepare GSM8K data and train GRPO using a TOML recipe configuration.

Launch this recipe with ``bash trio-cookbook/scripts/train_grpo_gms8k.sh``.
Use ``--prepare-only`` to inspect the JSONL data before creating PyTRIO clients.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from numbers import Integral
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from trio_cookbook import GRPOConfig, GRPOTrainer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "trio-cookbook" / "configs" / "grpo_gms8k.toml"

QUESTION_SUFFIX = "Provide a numerical answer without units, written inside \\boxed{}."
FEWSHOT_PREFIX = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]
BOXED_ANSWER = re.compile(r"\\boxed\{([^}]+)\}")
GSM8K_ANSWER = re.compile(r"####\s*(.+)")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def group_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("GRPO group size must be at least 2")
    return parsed


def normalize_answer(value: str) -> str:
    return value.replace(",", "").strip().rstrip(".")


def extract_reference_answer(value: str) -> str:
    match = GSM8K_ANSWER.search(value)
    if match is None:
        raise ValueError(f"GSM8K answer has no #### marker: {value!r}")
    return normalize_answer(match.group(1))


def build_prompt(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K question must be a non-empty string")
    return [
        *(dict(message) for message in FEWSHOT_PREFIX),
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]


def prepare_gsm8k_dataset(data_file: Path, cache_dir: Path, *, rebuild: bool) -> None:
    """Download GSM8K and create the JSONL contract consumed by GRPOTrainer."""

    if data_file.is_file() and not rebuild:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir=str(cache_dir))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(dataset)!r}")

    temporary_file = data_file.with_suffix(data_file.suffix + ".tmp")
    try:
        with temporary_file.open("w", encoding="utf-8") as output:
            for index, source_row in enumerate(dataset):
                question = source_row.get("question")
                source_answer = source_row.get("answer")
                if not isinstance(question, str) or not isinstance(source_answer, str):
                    raise TypeError(f"GSM8K row {index} must contain string question and answer")
                row = {
                    "messages": build_prompt(question),
                    "answer": extract_reference_answer(source_answer),
                    "question": question,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_file.replace(data_file)
    except BaseException:
        temporary_file.unlink(missing_ok=True)
        raise


def load_training_rows(data_file: Path, max_examples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with data_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") if isinstance(row, dict) else None
            valid_messages = (
                isinstance(messages, str)
                or (
                    isinstance(messages, list)
                    and bool(messages)
                    and (
                        all(
                            isinstance(token, Integral) and not isinstance(token, bool)
                            for token in messages
                        )
                        or all(isinstance(message, dict) for message in messages)
                    )
                )
            )
            if not isinstance(row, dict) or not valid_messages or not isinstance(row.get("answer"), str):
                raise ValueError(f"Invalid trainer row at {data_file}:{line_number}")
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"No training rows found in {data_file}")
    return rows


def boxed_answer_reward(
    *,
    completions: list[str | list[int]],
    answer: list[str],
    **_: Any,
) -> list[float]:
    """Return 1 when the last boxed completion equals the GSM8K answer."""

    if len(completions) != len(answer):
        raise ValueError("completions and answer must have the same length")

    rewards: list[float] = []
    for completion, expected in zip(completions, answer, strict=True):
        if not isinstance(completion, str):
            rewards.append(0.0)
            continue
        matches = BOXED_ANSWER.findall(completion)
        predicted = normalize_answer(matches[-1]) if matches else None
        rewards.append(float(predicted == normalize_answer(expected)))
    return rewards


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("rb") as source:
        settings = tomllib.load(source)
    required = {
        "model": ("name", "lora_rank"),
        "data": ("file", "cache_dir"),
        "grpo": ("num_generations", "max_steps"),
    }
    for section, fields in required.items():
        values = settings.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Config {path} must contain a [{section}] section")
        for field in fields:
            if field not in values:
                raise ValueError(f"Config {path} is missing [{section}].{field}")
    if not isinstance(settings.get("swanlab", {}), dict):
        raise ValueError(f"Config {path} must contain a [swanlab] table when provided")
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GSM8K through trio_cookbook.GRPOTrainer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=positive_int, help="override grpo.max_steps")
    parser.add_argument("--batch-size", type=positive_int, help="override grpo.batch_size")
    parser.add_argument("--group-size", type=group_size, help="override grpo.num_generations")
    parser.add_argument("--mode", choices=("disabled", "online", "local", "offline"))
    parser.add_argument("--project")
    parser.add_argument("--workspace", help="SwanLab workspace or organization")
    parser.add_argument(
        "--tag", "--tags", dest="tags", action="append", metavar="TAG",
        help="SwanLab tag; repeat for multiple tags",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--max-examples", type=positive_int)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_config(args.config)
    data = settings["data"]
    data_file = resolve_path(args.data_file or data["file"])
    cache_dir = resolve_path(args.cache_dir or data["cache_dir"])
    max_examples = data.get("max_examples") if args.max_examples is None else args.max_examples
    if max_examples is not None and (
        isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 1
    ):
        raise ValueError("data.max_examples must be a positive integer")

    prepare_gsm8k_dataset(data_file, cache_dir, rebuild=args.rebuild_data)
    rows = load_training_rows(data_file, max_examples)
    print(f"Prepared {len(rows)} GRPO rows at {data_file}")
    if args.prepare_only:
        return

    grpo = dict(settings["grpo"])
    for flag, field in (
        ("steps", "max_steps"),
        ("batch_size", "batch_size"),
        ("group_size", "num_generations"),
    ):
        if (value := getattr(args, flag)) is not None:
            grpo[field] = value
    grpo.setdefault("save_every", grpo.get("max_steps") or 0)

    swanlab_options = dict(settings.get("swanlab", {}))
    for field in ("mode", "project", "workspace", "run_name", "tags"):
        if (value := getattr(args, field)) is not None:
            swanlab_options[field] = value

    config = GRPOConfig(**grpo, **swanlab_options)
    trainer = GRPOTrainer(
        model_name=settings["model"]["name"],
        reward_funcs=boxed_answer_reward,
        args=config,
        train_dataset=rows,
        lora_rank=settings["model"]["lora_rank"],
    )
    trainer.train()


if __name__ == "__main__":
    main()
