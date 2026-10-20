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
    train_dataset = load_dataset(
        "json",
        data_files=str(resolve_path(config.train_dataset.path)),
        split=config.train_dataset.split,
    )
    limit = config.train_dataset.max_examples
    if limit is not None:
        train_dataset = train_dataset.select(range(min(limit, len(train_dataset))))

    trainer = PPOTrainer(
        args=config,
        train_dataset=train_dataset,
        reward_fn=gsm8k_reward,
    )
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
