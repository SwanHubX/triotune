"""Supervise GSM8K solutions with prompt/completion-masked SFT."""

import argparse
import asyncio
from pathlib import Path

from datasets import load_dataset

from triotune.trainer import SFTConfig, SFTTrainer, DatasetConfig
from triotune.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[2]


def load_data(source: DatasetConfig | None, seed: int | None):
    """Load local JSON/JSONL, then shuffle and limit before training."""

    if source is None:
        return None
    if source.path is None:
        raise ValueError("A dataset path is required.")
    path = ROOT / Path(source.path).expanduser()
    dataset = load_dataset(
        "json",
        data_files={source.split: str(path.resolve())},
        split=source.split,
        **source.loader_options,
    )
    if source.shuffle:
        dataset = dataset.shuffle(seed=source.seed if source.seed is not None else seed)
    if source.max_examples is not None:
        dataset = dataset.select(range(min(source.max_examples, len(dataset))))
    if not len(dataset):
        raise ValueError(f"Dataset {path} must contain at least one row.")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML file.")
    config = SFTConfig(**load_yaml_config(parser.parse_args().config))
    if config.train_dataset is None:
        raise ValueError("train_dataset is required.")
    trainer = SFTTrainer(
        args=config,
        train_dataset=load_data(config.train_dataset, config.training.seed),
        eval_dataset=load_data(config.valid_dataset, config.training.seed),
    )
    state = asyncio.run(trainer.train())
    print(f"Completed {state.global_step} updates; checkpoint: {state.checkpoint_path}")


if __name__ == "__main__":
    main()
