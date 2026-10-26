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
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ...trainer.ppo.rollout import gather_cancel_on_error
from ...trainer.ppo.trajectory import Trajectory
from ..runtime.sampling import SamplingSnapshot
from ..training.evaluation import evaluation_metrics
from ..types import (
    SYSTEM_STATUSES,
    AgentContractError,
    EpisodeExecutor,
    EpisodeLogger,
    EpisodeResult,
    ExecutionContext,
    RolloutExport,
    _execution_context,
)


@dataclass(slots=True)
class AgentRolloutBatch:
    """Keep execution evidence separate from trajectories accepted for training.

    ``episodes`` includes failures and every attempted group member. The
    ``trajectories`` list excludes rejected groups, so its length may be smaller
    than the number of individually trainable episode records.
    """

    episodes: list[EpisodeResult]
    trajectories: list[Trajectory]
    dropped_groups: int = 0
    refill_batches: int = 0
    # Execution records include all attempts, while exports include only
    # accepted candidates. Group metadata keeps the distinction between a
    # complete group and an explicitly accepted partial group observable.
    exports: list[RolloutExport] = field(default_factory=list)
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            f"{prefix}/accepted_trajectories": float(len(self.trajectories)),
            f"{prefix}/dropped_groups": float(self.dropped_groups),
            f"{prefix}/refill_batches": float(self.refill_batches),
            f"{prefix}/retry_count": float(
                sum(event.retried for e in episodes for event in e.retries)
            ),
            f"{prefix}/transient_failures": float(
                sum(len(e.retries) for e in episodes)
            ),
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
        for stage in ("sampling", "tool", "scoring"):
            metrics[f"{prefix}/retries/{stage}"] = float(
                sum(
                    event.retried and event.stage == stage
                    for e in episodes
                    for event in e.retries
                )
            )
        for status in (
            "completed",
            "invalid_action",
            "max_turns",
            "token_limit",
            "context_limit",
            "sampling_failure",
            "tool_failure",
            "scoring_failure",
            "contract_error",
            "timeout",
            "cancelled",
        ):
            metrics[f"{prefix}/status/{status}"] = float(
                sum(e.status == status for e in episodes)
            )
            metrics[f"{prefix}/status_rate/{status}"] = metrics[
                f"{prefix}/status/{status}"
            ] / max(1, count)
        metrics.update(evaluation_metrics(episodes, prefix=prefix))
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
    same input row. By default every candidate must export successfully; an
    explicit minimum permits partial groups without treating failed candidates
    as zero-reward examples. Evaluation counts all attempts independently of
    how many physical training rows their exporters produce.
    """

    def __init__(
        self,
        runner: EpisodeExecutor,
        *,
        max_concurrency: int = 8,
        logger: EpisodeLogger | None = None,
        min_usable_group_size: int | None = None,
        shared_prefix_policy: str = "deduplicate",
    ):
        """Bind a reusable episode executor, concurrency limit, and audit sink.

        The logger is optional and does not own execution. When configured, its
        failures propagate rather than silently losing experiment evidence.
        ``min_usable_group_size=None`` requires every requested candidate.
        Explicit thresholds count candidates rather than exported fragments.
        ``shared_prefix_policy`` controls whether an original sampled action
        contributes once or once per exported path; causal validation always
        applies, including when repeated contributions are requested.
        """
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer.")
        self.runner, self.max_concurrency, self.logger = runner, max_concurrency, logger
        if min_usable_group_size is not None and (
            type(min_usable_group_size) is not int or min_usable_group_size < 1
        ):
            raise ValueError("min_usable_group_size must be positive or None.")
        if shared_prefix_policy not in {"deduplicate", "repeat"}:
            raise ValueError("shared_prefix_policy must be deduplicate or repeat.")
        self.min_usable_group_size = min_usable_group_size
        self.shared_prefix_policy = shared_prefix_policy

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
        minimum = self.min_usable_group_size or n_samples
        if minimum > n_samples:
            raise ValueError("min_usable_group_size must not exceed n_samples.")
        semaphore = asyncio.Semaphore(self.max_concurrency)
        # Namespace groups per collection call. Identical row contents in
        # separate calls must not accidentally merge into one GRPO group.
        namespace = uuid4().hex

        async def run(row, row_index, sample_index):
            async with semaphore:
                group_id = f"{namespace}:{row_index}"
                rollout_id = f"{namespace}:rollout:{row_index}:{sample_index}"
                context = ExecutionContext(
                    rollout_id=rollout_id,
                    episode_id=uuid4().hex,
                    group_id=group_id,
                    sample_index=sample_index,
                    group_size=n_samples,
                    snapshot_id=snapshot.snapshot_id,
                    policy_version=snapshot.policy_version,
                    task_id=str(row.get("task_id", f"row-{row_index}")),
                )
                # Reserve max_turns consecutive seeds for each attempt, leaving
                # room for the runner's per-turn increments without overlap.
                episode = await self.runner.run(
                    copy.deepcopy(row),
                    snapshot,
                    group_id=group_id,
                    # Assign identity before execution so tools and scorers
                    # can use candidate-specific data while the episode runs.
                    rollout_id=rollout_id,
                    sample_index=sample_index,
                    context=context,
                    seed=None
                    if seed is None
                    else seed
                    + (row_index * n_samples + sample_index)
                    * self.runner.limits.max_turns,
                    is_eval=is_eval,
                )
                try:
                    self._validate_context(episode, context)
                except AgentContractError as exc:
                    # Failed adapters remain attributable to the requested
                    # candidate. Their returned identity cannot split or merge
                    # unrelated PPO groups, even under relaxed collection.
                    episode.status, episode.error, episode.score = (
                        "contract_error",
                        str(exc),
                        None,
                    )
                    for name in (
                        "rollout_id",
                        "episode_id",
                        "group_id",
                        "sample_index",
                        "snapshot_id",
                        "policy_version",
                        "task_id",
                    ):
                        setattr(episode, name, getattr(context, name))
                    episode.context = context
                exported = None
                if episode.trainable:
                    try:
                        exporter = self.runner.trajectory_builder
                        if hasattr(exporter, "export"):
                            exported = exporter.export(episode)
                        else:
                            # Single-row adapters remain usable at the explicit
                            # compatibility boundary. Their output receives the
                            # same provenance and overlap validation as exports.
                            row = exporter.build(episode)
                            row.fragment_id = row.fragment_id or f"{rollout_id}:0"
                            exported = RolloutExport(
                                [row],
                                rollout_id,
                                {row.fragment_id: row.interaction_ids},
                                group_id,
                            )
                        self._validate_export(exported, episode, context)
                    except (AgentContractError, ValueError, TypeError) as exc:
                        episode.status, episode.error, episode.score = (
                            "contract_error",
                            str(exc),
                            None,
                        )
                        exported = None
                if self.logger is not None:
                    # Persist the post-validation status, including any export
                    # failure, before strict batch rejection can raise.
                    token = _execution_context.set(context)
                    try:
                        await self.logger.write(episode)
                    finally:
                        _execution_context.reset(token)
                return episode, exported

        results = await gather_cancel_on_error(
            run(row, row_index, sample_index)
            for row_index, row in enumerate(rows)
            for sample_index in range(n_samples)
        )
        # Group-relative baselines require complete groups. A valid sibling
        # still appears in episode diagnostics but cannot train on its own.
        groups = {}
        for episode, exported in results:
            state = groups.setdefault(
                episode.group_id,
                {
                    "expected": n_samples,
                    "usable_candidates": [],
                    "failed_candidates": [],
                    "min_usable_group_size": minimum,
                },
            )
            state[
                "usable_candidates" if exported is not None else "failed_candidates"
            ].append(episode.sample_index)
        for state in groups.values():
            state["complete"] = len(state["usable_candidates"]) == n_samples
            state["accepted"] = len(state["usable_candidates"]) >= minimum
        rejected_groups = {
            group for group, state in groups.items() if not state["accepted"]
        }
        exports = [
            exported
            for e, exported in results
            if exported is not None and e.group_id not in rejected_groups
        ]
        batch = AgentRolloutBatch(
            [e for e, _ in results],
            [t for exported in exports for t in exported.trajectories],
            dropped_groups=len(rejected_groups),
            exports=exports,
            groups=groups,
        )
        if strict and rejected_groups:
            raise EpisodeCollectionError(batch)
        return batch

    @staticmethod
    def _validate_context(episode, context):
        """Require executor output to retain the collector's trusted identity."""
        episode.__post_init__()
        for name in (
            "rollout_id",
            "episode_id",
            "group_id",
            "sample_index",
            "snapshot_id",
            "policy_version",
            "task_id",
        ):
            if getattr(episode, name) != getattr(context, name):
                raise AgentContractError(f"Executor changed candidate {name}.")
        if episode.context != context:
            raise AgentContractError("Executor must preserve ExecutionContext.")

    def _validate_export(self, exported, episode, context):
        """Validate fragment lineage and count each sampled action explicitly.

        A token's action key is stable across branching paths. Deduplication
        keeps its first occurrence active and masks subsequent copies. Repeat
        mode deliberately trains each copy. Conflicting tokens/probabilities
        under one key always reject the export, irrespective of this policy.
        Different initial prompts cannot share a PPO group; exporters that
        need such layouts must expose separate groups through another workflow.
        """
        if not isinstance(exported, RolloutExport) or not exported.trajectories:
            raise AgentContractError("Export must contain at least one trajectory.")
        if (
            exported.logical_rollout_id != context.rollout_id
            or exported.group_id != context.group_id
        ):
            raise AgentContractError("Export identity differs from its candidate.")
        rows = exported.trajectories
        ids = [row.fragment_id for row in rows]
        if (
            len(set(ids)) != len(ids)
            or None in ids
            or set(ids) != set(exported.fragment_map)
        ):
            raise AgentContractError(
                "Fragment map must identify every row exactly once."
            )
        seen, accepted = {}, []
        source, source_turns = {}, {}
        for index, turn in enumerate(episode.turns):
            identity = (
                episode.interactions[index].interaction_id
                if episode.interactions
                else f"{episode.episode_id}:turn:{index}"
            )
            source_turns[identity] = turn
            for offset, (token, logprob) in enumerate(
                zip(turn.generation.tokens, turn.generation.logprobs, strict=True)
            ):
                source[(identity, offset)] = (token, logprob)
        for row in rows:
            row.__post_init__()
            # Exporters may use dataclasses.replace(), which shares mutable
            # lists by default. Detach masks before deduplication so masking
            # one fragment cannot silently alter an already accepted sibling.
            row.loss_mask = list(row.loss_mask)
            row.turn_ids = list(row.turn_ids) if row.turn_ids is not None else None
            row.token_rewards = (
                list(row.token_rewards) if row.token_rewards is not None else None
            )
            full_tokens = row.prompt_tokens + row.completion_tokens
            checked_contexts = set()
            for name in (
                "rollout_id",
                "episode_id",
                "group_id",
                "sample_index",
                "snapshot_id",
                "policy_version",
                "task_id",
            ):
                if getattr(row, name) != getattr(context, name):
                    raise AgentContractError(f"Export changed candidate {name}.")
            if row.prompt_tokens != list(episode.turns[0].prompt_tokens):
                raise AgentContractError(
                    "Fragments must share the candidate's initial prompt."
                )
            if tuple(exported.fragment_map[row.fragment_id]) != row.interaction_ids:
                raise AgentContractError(
                    "Fragment interactions disagree with row provenance."
                )
            if not set(row.interaction_ids).issubset(source_turns):
                raise AgentContractError("Fragment refers to an unknown interaction.")
            if row.reward != episode.score.reward:
                raise AgentContractError(
                    "Fragments must share the scored candidate outcome."
                )
            if row.action_keys is None:
                raise AgentContractError("Exports require stable action keys.")
            else:
                for i, (key, active) in enumerate(
                    zip(row.action_keys, row.loss_mask, strict=True)
                ):
                    if not active:
                        continue
                    evidence = (row.completion_tokens[i], row.logprobs[i])
                    if source.get(key) != evidence:
                        raise AgentContractError(
                            "Exported action differs from original token evidence."
                        )
                    original = source_turns[key[0]]
                    if (
                        len(row.prompt_tokens) + i
                        != len(original.prompt_tokens) + key[1]
                    ):
                        raise AgentContractError(
                            "Exported action moved to a different causal position."
                        )
                    if key[0] not in checked_contexts:
                        # Behavior probabilities belong to both an action and
                        # its exact conditioning prefix. Matching token IDs
                        # alone is insufficient if an exporter rewrites the
                        # observations or earlier model output in that prefix.
                        end = min(
                            len(full_tokens),
                            len(original.prompt_tokens)
                            + len(original.generation.tokens),
                        )
                        expected = original.prompt_tokens + original.generation.tokens
                        if tuple(full_tokens[:end]) != expected[:end]:
                            raise AgentContractError(
                                "Exported action changed its causal context."
                            )
                        checked_contexts.add(key[0])
                    if key in seen:
                        if seen[key] != evidence:
                            raise AgentContractError(
                                "Shared action has conflicting token evidence."
                            )
                        if self.shared_prefix_policy == "deduplicate":
                            row.loss_mask[i] = False
                            if row.turn_ids is not None:
                                row.turn_ids[i] = -1
                            if row.token_rewards is not None:
                                row.token_rewards[i] = 0.0
                    seen[key] = evidence
            if any(row.loss_mask):
                row.__post_init__()
                accepted.append(row)
        if not accepted:
            raise AgentContractError("Export contains no usable actions.")
        exported.trajectories = accepted
        exported.fragment_map = {
            row.fragment_id: row.interaction_ids for row in accepted
        }
        exported.metadata["shared_prefix_policy"] = self.shared_prefix_policy
        episode.export_metadata = {
            "logical_rollout_id": exported.logical_rollout_id,
            "group_id": exported.group_id,
            "fragment_map": dict(exported.fragment_map),
            **exported.metadata,
        }
