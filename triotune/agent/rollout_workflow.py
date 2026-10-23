"""Collect independent episodes into prompt groups suitable for PPO updates.

This layer expands each task into multiple attempts, bounds concurrency, calls
the episode exporter, and optionally persists finalized records. It retains
failed attempts for diagnostics while preventing incomplete prompt groups from
silently entering relative-reward training. Evaluation uses the same execution
path with relaxed batch rejection so failure rates remain observable.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ..trainer.ppo.contracts import Trajectory
from ..trainer.ppo.rollout import gather_cancel_on_error
from .contracts import (
    SYSTEM_STATUSES,
    AgentContractError,
    EpisodeExecutor,
    EpisodeLogger,
    EpisodeResult,
)
from .sampling import SamplingSnapshot


@dataclass(slots=True)
class AgentRolloutBatch:
    """Keep execution evidence separate from trajectories accepted for training.

    ``episodes`` includes failures and every attempted group member. The
    ``trajectories`` list excludes rejected groups, so its length may be smaller
    than the number of individually trainable episode records.
    """

    episodes: list[EpisodeResult]
    trajectories: list[Trajectory]

    def metrics(self, *, prefix: str = "agent") -> dict[str, float]:
        """Aggregate execution metrics with explicit, task-relevant denominators.

        Success and system-failure rates include all attempted episodes; reward
        means use individually trainable, scored episodes. Tool statistics use
        returned observations, excluding calls that failed before a result was
        available. Empty populations yield zero rather than undefined values.
        """
        episodes = self.episodes
        valid = [e for e in episodes if e.trainable]
        count = len(episodes)
        turns = [turn for episode in episodes for turn in episode.turns]
        calls = [result for turn in turns for result in turn.tool_results]
        output = sum(len(turn.generation.tokens) for turn in turns)
        metrics = {
            f"{prefix}/episodes": float(count),
            f"{prefix}/valid_episodes": float(len(valid)),
            # Include system failures in the denominator; expose them separately.
            f"{prefix}/success_rate": sum(e.score.success for e in valid)
            / max(1, count),
            f"{prefix}/system_failure_rate": sum(
                e.status in SYSTEM_STATUSES for e in episodes
            )
            / max(1, count),
            f"{prefix}/tool_calls_mean": len(calls) / max(1, count),
            f"{prefix}/tool_error_rate": sum(r.is_error for r in calls)
            / max(1, len(calls)),
            f"{prefix}/generated_tokens_mean": output / max(1, count),
            f"{prefix}/turns_mean": len(turns) / max(1, count),
            f"{prefix}/reward_mean": sum(e.score.reward for e in valid)
            / max(1, len(valid)),
            f"{prefix}/sample_input_tokens": float(
                sum(t.generation.input_tokens for t in turns)
            ),
            f"{prefix}/sample_output_tokens": float(
                sum(t.generation.output_tokens for t in turns)
            ),
        }
        for status in {e.status for e in episodes}:
            metrics[f"{prefix}/status/{status}"] = float(
                sum(e.status == status for e in episodes)
            )
        return metrics


class EpisodeCollectionError(RuntimeError):
    """Expose rejected collection records without submitting a policy update.

    ``batch`` preserves the original episodes and any otherwise eligible
    trajectories for inspection. The strict collector raises this only after
    completed episode records have been passed to the configured logger.
    """

    def __init__(self, batch: AgentRolloutBatch):
        self.batch = batch
        rejected = [
            f"{e.episode_id}: {e.status} ({e.error})"
            for e in batch.episodes
            if not e.trainable
        ]
        super().__init__("Agent collection rejected episodes: " + "; ".join(rejected))


class AgentRolloutWorkflow:
    """Expand tasks into independent episodes under one frozen policy snapshot.

    Group IDs are scoped to a collect call and shared only by samples of the
    same input row. Any rejected member removes its entire group from training.
    Evaluation records these failures; strict training collection raises.
    """

    def __init__(
        self,
        runner: EpisodeExecutor,
        *,
        max_concurrency: int = 8,
        logger: EpisodeLogger | None = None,
    ):
        """Bind a reusable episode executor, concurrency limit, and audit sink.

        The logger is optional and does not own execution. When configured, its
        failures propagate rather than silently losing experiment evidence.
        """
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer.")
        self.runner, self.max_concurrency, self.logger = runner, max_concurrency, logger

    async def collect(
        self,
        rows: Sequence[Mapping[str, Any]],
        snapshot: SamplingSnapshot,
        *,
        n_samples: int = 1,
        seed: int | None = None,
        is_eval: bool = False,
        strict: bool = True,
    ) -> AgentRolloutBatch:
        """Run n_samples attempts per input row under the supplied policy.

        ``strict=True`` aborts a training collection containing untrainable
        attempts. ``strict=False`` returns their records for evaluation, while
        still excluding their full prompt groups from the trajectory list.
        A malformed trajectory export is classified as a contract error, not
        downgraded to a zero-reward example.
        """
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        semaphore = asyncio.Semaphore(self.max_concurrency)
        # Namespace groups per collection call. Identical row contents in
        # separate calls must not accidentally merge into one GRPO group.
        namespace = uuid4().hex

        async def run(row, row_index, sample_index):
            async with semaphore:
                # Reserve max_turns consecutive seeds for each attempt, leaving
                # room for the runner's per-turn increments without overlap.
                episode = await self.runner.run(
                    copy.deepcopy(row),
                    snapshot,
                    group_id=f"{namespace}:{row_index}",
                    seed=None
                    if seed is None
                    else seed
                    + (row_index * n_samples + sample_index)
                    * self.runner.limits.max_turns,
                    is_eval=is_eval,
                )
                trajectory = None
                if episode.trainable:
                    try:
                        trajectory = self.runner.trajectory_builder.build(episode)
                    except AgentContractError as exc:
                        episode.status, episode.error, episode.score = (
                            "contract_error",
                            str(exc),
                            None,
                        )
                if self.logger is not None:
                    # Persist the post-validation status, including any export
                    # failure, before strict batch rejection can raise.
                    await self.logger.write(episode)
                return episode, trajectory

        results = await gather_cancel_on_error(
            run(row, row_index, sample_index)
            for row_index, row in enumerate(rows)
            for sample_index in range(n_samples)
        )
        # Group-relative baselines require complete groups. A valid sibling
        # still appears in episode diagnostics but cannot train on its own.
        rejected_groups = {e.group_id for e, _ in results if not e.trainable}
        batch = AgentRolloutBatch(
            [e for e, _ in results],
            [
                t
                for e, t in results
                if t is not None and e.group_id not in rejected_groups
            ],
        )
        if strict and rejected_groups:
            raise EpisodeCollectionError(batch)
        return batch
