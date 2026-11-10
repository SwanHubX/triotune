"""Resume the step-200 WebShop teacher with unchanged native GRPO reward."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

from train.resume_grpo_teacher import load_experiment, run


DEFAULT_CONFIG = (
    Path(__file__).with_name("configs") / "resume-grpo-teacher-native.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--steps",
        type=int,
        help="Override continuation.additional_steps.",
    )
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config and checkpoint-ID syntax without remote calls.",
    )
    args = parser.parse_args()
    experiment = load_experiment(args.config, steps=args.steps)
    if experiment.reward_mode != "native":
        raise ValueError("The native continuation requires reward.mode=native.")
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    asyncio.run(run(experiment, stop_after_step=args.stop_after_step))


if __name__ == "__main__":
    main()
