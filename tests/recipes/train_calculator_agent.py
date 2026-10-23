"""Opt-in live calculator agent probe, PPO/GRPO training, and evaluation.

Run from the repository root with a configured PyTRIO service:
    .venv/bin/python tests/recipes/train_calculator_agent.py --model MODEL --probe-only
    .venv/bin/python tests/recipes/train_calculator_agent.py --model MODEL --steps 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# The repository-only recipe may be invoked directly without installing tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculator_agent import calculator_rows, calculator_workflow

from triotune import PPOConfig, PPOTrainer
from triotune.agent import AgentTrainingLoop, ChatTemplateRenderer, JsonlEpisodeLogger


async def run(args):
    rows = calculator_rows()
    trainer = PPOTrainer(
        args.model,
        args=PPOConfig.for_algorithm(
            "grpo",
            max_steps=args.steps,
            batch_size=1,
            num_generations=2,
            max_tokens=2048,
            seed=1,
            eval_before_train=True,
            eval_after_train=True,
        ),
        train_dataset=rows,
        eval_dataset=rows[:1],
    )
    managed_by_loop = False
    try:
        snapshot = await trainer.create_sampling_snapshot()
        workflow = calculator_workflow(
            logger=JsonlEpisodeLogger(args.trace),
            renderer=ChatTemplateRenderer(
                snapshot.tokenizer, **json.loads(args.template_kwargs)
            ),
        )
        if args.probe_only:
            batch = await workflow.collect(
                rows[:1], snapshot, is_eval=True, strict=False
            )
            print(
                json.dumps(batch.metrics(prefix="probe"), ensure_ascii=False, indent=2)
            )
            episode = batch.episodes[0]
            if not episode.trainable or not episode.score.success:
                raise RuntimeError(
                    f"Calculator probe did not succeed: {episode.status}: {episode.error}"
                )
            await trainer.close()
        else:
            managed_by_loop = True
            state = await AgentTrainingLoop(trainer, workflow).train()
            print(
                json.dumps(
                    {"step": state.global_step, "checkpoint": state.checkpoint_path},
                    indent=2,
                )
            )
    except BaseException as exc:
        if not managed_by_loop:
            await trainer.close(state="crashed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PyTRIO base-model ID")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Sample and validate an episode without optimization",
    )
    parser.add_argument(
        "--trace", type=Path, default=Path("output/calculator_agent_episodes.jsonl")
    )
    parser.add_argument(
        "--template-kwargs",
        default="{}",
        help='JSON chat-template options, e.g. {"enable_thinking": false}',
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
