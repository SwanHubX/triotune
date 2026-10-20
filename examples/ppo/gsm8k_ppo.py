"""Train GSM8K with PPO and an independent value-model critic.

Run from the repository root:

    uv run python examples/ppo/gsm8k_ppo.py
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from triotune.trainer import PPOConfig, PPOTrainer
from triotune.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("gsm8k_ppo_args.yaml")
BOXED_ANSWER = re.compile(r"\\boxed\{([^{}]+)\}")


def normalize_answer(answer: str) -> str:
    return answer.replace(",", "").replace("$", "").strip().rstrip(".")


def gsm8k_reward(
    *, row: dict[str, Any], completion: str, completion_ids: list[int]
) -> float:
    """Give reward 1 when the final boxed answer matches the reference."""

    del completion_ids
    matches = BOXED_ANSWER.findall(completion)
    prediction = normalize_answer(matches[-1]) if matches else None
    return float(prediction == normalize_answer(row["answer"]))


def resolve_path(path: str | Path) -> Path:
    """Resolve recipe paths from the repository root."""

    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GSM8K with PPO")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    settings = load_yaml_config(args.config)
    config = PPOConfig(**settings)
    train_source = config.train_dataset
    train_dataset = load_dataset("json", data_files={train_source.split: str(resolve_path(train_source.path))}, split=train_source.split)

    valid_dataset = None
    if config.valid_dataset is not None:
        valid_source = config.valid_dataset
        valid_dataset = load_dataset(
            "json",
            data_files={valid_source.split: str(resolve_path(valid_source.path))},
            split=valid_source.split,
        )
        if valid_source.max_examples is not None:
            valid_dataset = valid_dataset.select(
                range(min(valid_source.max_examples, len(valid_dataset)))
            )

    trainer = PPOTrainer(
        args=config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        reward_fn=gsm8k_reward,
    )
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
