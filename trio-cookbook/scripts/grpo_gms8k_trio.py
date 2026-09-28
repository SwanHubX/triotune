"""Train GSM8K with the reusable :class:`GRPOTrainer` API.

The first run downloads ``openai/gsm8k`` into ``.cache/`` and writes the
trainer-ready rows to ``data/gsm8k_train.jsonl``. Use
``--prepare-only`` to inspect that file before creating any PyTRIO clients.

Example:

    uv run python trio-cookbook/scripts/grpo_gms8k_trio.py --prepare-only
    uv run python trio-cookbook/scripts/grpo_gms8k_trio.py \
        --steps 10 --batch-size 4 --group-size 8 --mode disabled
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from trio_cookbook import GRPOConfig, GRPOTrainer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / ".cache" / "huggingface"
DEFAULT_DATA_FILE = REPOSITORY_ROOT / "data" / "gsm8k_train.jsonl"

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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def probability(value: str) -> float:
    parsed = positive_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("value must be <= 1")
    return parsed


def beta_value(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1)")
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
                    "prompt": build_prompt(question),
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
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("prompt"), list)
                or not isinstance(row.get("answer"), str)
            ):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GSM8K through trio_cookbook.GRPOTrainer")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora-rank", type=positive_int, default=32)
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--group-size", type=group_size, default=4)
    parser.add_argument("--max-tokens", type=positive_int, default=1024)
    parser.add_argument("--temperature", type=positive_float, default=1.0)
    parser.add_argument("--top-p", type=probability, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=positive_float, default=4e-5)
    parser.add_argument("--beta1", type=beta_value, default=0.9)
    parser.add_argument("--beta2", type=beta_value, default=0.95)
    parser.add_argument(
        "--save-every",
        type=nonnegative_int,
        help="checkpoint interval; defaults to the final configured step",
    )
    parser.add_argument(
        "--mode",
        choices=("disabled", "online", "local", "offline"),
        default="disabled",
    )
    parser.add_argument("--project", default="trio-cookbook-test")
    parser.add_argument("--run-name", default="grpo-trainer-gsm8k")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--max-examples", type=positive_int)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = args.data_file.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    prepare_gsm8k_dataset(data_file, cache_dir, rebuild=args.rebuild_data)
    rows = load_training_rows(data_file, args.max_examples)
    print(f"Prepared {len(rows)} GRPO rows at {data_file}")
    if args.prepare_only:
        return

    config = GRPOConfig(
        num_generations=args.group_size,
        max_completion_length=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        save_every=args.steps if args.save_every is None else args.save_every,
        seed=args.seed,
        chat_template_kwargs={"enable_thinking": False},
        mode=args.mode,
        project=args.project,
        run_name=args.run_name,
    )
    trainer = GRPOTrainer(
        model_name=args.base_model,
        reward_funcs=boxed_answer_reward,
        args=config,
        train_dataset=rows,
        lora_rank=args.lora_rank,
    )
    trainer.train()


if __name__ == "__main__":
    main()
