"""Collect GRPO-teacher experts, then train a fresh student with TCOD-B2F."""

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

from train.collect_tcod_b2f_experts import collect
from train.tcod_b2f import DEFAULT_CONFIG, load_experiment, run


async def prepare_and_train(experiment, *, resume=False, expert_max_tasks=None,
                            stop_after_step=None):
    if resume and expert_max_tasks is not None:
        raise ValueError("Resume uses a frozen expert cache; omit expert_max_tasks.")
    if not resume:
        report = await collect(experiment, max_tasks=expert_max_tasks)
        if expert_max_tasks is not None:
            return {"status": "expert_collection_limited", **report}
    return await run(experiment, stop_after_step=stop_after_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--expert-max-tasks", type=int,
                        help="Collect at most N missing goals and exit before training.")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    experiment = load_experiment(args.config, steps=args.steps, resume_from=args.resume_from)
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    print(json.dumps(asyncio.run(prepare_and_train(
        experiment, resume=experiment.trainer.checkpoint.resume_from is not None,
        expert_max_tasks=args.expert_max_tasks, stop_after_step=args.stop_after_step,
    )), default=str))


if __name__ == "__main__":
    main()
