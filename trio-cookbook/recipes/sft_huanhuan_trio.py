"""Fine-tune a PyTRIO model on huanhuan-chat's original dialogue pairs."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import tomllib
from pathlib import Path

from trio_cookbook.trainer.sft import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "trio-cookbook" / "configs" / "sft_huanhuan.toml"
SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--limit", type=int, help="Use only the first N dialogue pairs.")
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project")
    parser.add_argument("--run-name")
    parser.add_argument("--swanlab", action="store_true", default=None)
    parser.add_argument("--no-swanlab", dest="swanlab", action="store_false")
    parser.set_defaults(swanlab=None)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        settings = tomllib.load(source)
    for section, fields in {
        "model": ("name", "lora_rank"),
        "data": ("file", "epochs"),
        "sft": ("batch_size", "max_steps"),
    }.items():
        if not isinstance(settings.get(section), dict) or any(
            field not in settings[section] for field in fields
        ):
            raise ValueError(f"Config needs [{section}] with {', '.join(fields)}.")

    data = settings["data"]
    epochs = args.epochs if args.epochs is not None else data["epochs"]
    limit = args.limit if args.limit is not None else data.get("limit")
    if epochs < 1 or (limit is not None and limit < 1):
        parser.error("--epochs and --limit must be positive")
    data_file = args.data_file or Path(data["file"])
    if not data_file.is_absolute():
        data_file = ROOT / data_file

    source = json.loads(data_file.read_text(encoding="utf-8"))
    if limit is not None:
        source = source[:limit]
    rows = [
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["instruction"] + item["input"]},
            ],
            "completion": [{"role": "assistant", "content": item["output"]}],
        }
        for item in source
    ]
    sft = dict(settings["sft"])
    for option, field in (
        ("batch_size", "batch_size"), ("steps", "max_steps"),
        ("max_in_flight", "max_in_flight"),
        ("max_length", "max_length"), ("learning_rate", "learning_rate"),
        ("save_every", "save_every"), ("seed", "seed"),
    ):
        if (value := getattr(args, option)) is not None:
            sft[field] = value
    logging = dict(settings.get("swanlab", {}))
    if args.swanlab is not None:
        logging["report_to"] = "swanlab" if args.swanlab else None
    if args.project is not None:
        logging["project"] = args.project
    if args.run_name is not None:
        logging["run_name"] = args.run_name
    trainer = await asyncio.to_thread(
        SFTTrainer,
        model=args.model or settings["model"]["name"],
        args=SFTConfig(**sft, **logging),
        train_dataset=itertools.chain.from_iterable(rows for _ in range(epochs)),
        lora_rank=args.lora_rank if args.lora_rank is not None else settings["model"]["lora_rank"],
    )
    state = await trainer.train()
    print(f"steps={state.global_step} checkpoint={state.checkpoint_path} metrics={state.metrics}")


if __name__ == "__main__":
    asyncio.run(main())
