"""Evaluate base or cloud LoRA policies with the same tasks, scores, and budgets.

GiGPO, GRPO, and PPO evaluation differs only in the actor weights under test.
Evaluation never computes training advantages or updates model parameters. Omit
the checkpoint before training and provide the cloud actor checkpoint URI from
the training report afterward. The PPO critic is not created during evaluation.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import replace
from numbers import Real

from triotune.agentrl import EpisodeResult
from triotune.api.client_api import call_async


def webshop_score_metrics(episodes: Sequence[EpisodeResult]) -> dict[str, float]:
    """Compute the raw WebShop mean score and percentage score in memory.

    Episodes include every independent attempt for each goal, including invalid
    actions, budget exits, and system failures. Only
    ``score.components.task_score`` is consumed; sparse training rewards and
    process rewards are ignored. A normal attempt without a purchase scores zero.
    A system failure without a score also contributes zero and stays in the
    denominator. A recorded score with a missing, nonfinite, or out-of-range
    task score violates the scoring contract and raises an error. Empty
    evaluation input also raises instead of returning a misleading zero score.

    The result includes the mean in the 0 to 1 range, the mean multiplied by
    100, the number of scored attempts, and the number of unscored attempts.
    Every metric uses the ``eval/agent`` prefix and every attempt has equal
    weight in the denominator.
    """
    if not episodes:
        raise ValueError("WebShop Score requires at least one evaluation episode.")
    scores = []
    unscored = 0
    for episode in episodes:
        if episode.score is None:
            unscored += 1
            continue
        value = episode.score.components.get("task_score")
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"WebShop episode {episode.episode_id} requires a finite task_score in [0,1]."
            )
        scores.append(float(value))
    mean = math.fsum(scores) / len(episodes)
    return {
        "eval/agent/task_score_mean": mean,
        "eval/agent/score": mean * 100,
        "eval/agent/task_score_count": float(len(scores)),
        "eval/agent/unscored_episodes": float(unscored),
    }


async def load_actor_config(
    experiment, service, checkpoint: str | None, *, evaluation: bool
):
    """Return an effective PPO config and an optional loaded actor client.

    ``experiment`` is a training experiment and ``service`` is a PyTRIO service
    client. ``checkpoint`` must be the remote actor LoRA URI returned by
    ``save_state``. Evaluation disables the critic and retains actor sampling.
    Loading failures propagate instead of falling back to the base model.
    """
    if checkpoint is not None and not checkpoint.strip():
        raise ValueError("checkpoint must be a nonempty cloud actor LoRA URI.")
    config = experiment.trainer
    if checkpoint and not evaluation:
        raise ValueError("Cloud checkpoint loading is evaluation-only.")
    if evaluation:
        config = replace(config, critic=replace(config.critic, enabled=False))
    client = None
    if checkpoint:
        client = await call_async(
            service, "create_training_client_from_state", path=checkpoint
        )
    return config, client


async def evaluate_policy(loop, *, checkpoint: str | None) -> dict:
    """Evaluate through AgentTrainingLoop and report weights and update counts.

    The loop already owns its dataset, fixed snapshot factory, and interaction
    workflow. The checkpoint value is report metadata because actor loading has
    already completed. Actor and critic update counts are checked before and
    after evaluation to ensure that no algorithm performs training. Framework
    metrics are combined with task scores computed from the complete in-memory
    evaluation batch rather than a potentially incomplete JSONL log.
    """
    state = loop.trainer.state
    before = (state.optimizer_steps, state.critic_optimizer_steps)
    metrics = await loop.evaluate()
    score_metrics = webshop_score_metrics(loop.last_eval_batch.episodes)
    metrics.update(score_metrics)
    await loop.trainer.publish_metrics(score_metrics)
    after = (state.optimizer_steps, state.critic_optimizer_steps)
    if after != before:
        raise RuntimeError("Evaluation changed optimizer state.")
    return {
        "metrics": metrics,
        "optimizer_updates": after[0] - before[0],
        "critic_optimizer_updates": after[1] - before[1],
        "policy_source": "cloud_lora" if checkpoint else "base_model",
        "actor_lora_checkpoint": checkpoint,
    }


def main(argv=None):
    """Parse model, algorithm, config, split, and cloud checkpoint options."""
    from .train import build_parser, run

    args = build_parser(evaluation=True).parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
