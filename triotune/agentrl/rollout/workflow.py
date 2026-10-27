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
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ...trainer.ppo.rollout import gather_cancel_on_error
from ...trainer.ppo.trajectory import Trajectory
from ..runtime.sampling import SamplingSnapshot
from ..training.evaluation import evaluation_metrics
from ..training.rewards import aggregate_assignments
from ..types import (
    SYSTEM_STATUSES,
    AgentContractError,
    EpisodeExecutor,
    EpisodeLogger,
    EpisodeResult,
    ExecutionContext,
    RolloutExport,
    SampleAcceptance,
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
    # Branch groups are nested counterfactual sets inside accepted logical
    # rollouts. They are separate from candidate prompt groups so fragment count
    # and unfinished sibling paths cannot change candidate-level completeness.
    branch_groups: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            f"{prefix}/branch_groups": float(len(self.branch_groups)),
            f"{prefix}/completed_branches": float(
                sum(
                    len(state["completed_branch_ids"])
                    for state in self.branch_groups.values()
                )
            ),
            f"{prefix}/context_compactions": float(
                sum(len(episode.compactions) for episode in episodes)
            ),
            f"{prefix}/compacted_prompt_tokens_saved": float(
                sum(
                    compaction.original_prompt_tokens
                    - len(compaction.compacted_prompt_tokens)
                    for episode in episodes
                    for compaction in episode.compactions
                )
            ),
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
            "rejected",
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
        accept_sample: SampleAcceptance | None = None,
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
        if shared_prefix_policy not in {"deduplicate", "repeat", "weighted"}:
            raise ValueError(
                "shared_prefix_policy must be deduplicate, repeat, or weighted."
            )
        self.min_usable_group_size = min_usable_group_size
        self.shared_prefix_policy = shared_prefix_policy
        self.accept_sample = accept_sample
        self._resource_started = False
        # One workflow-wide semaphore bounds overlapping collect calls from the
        # asynchronous training pipeline. A per-call semaphore would multiply
        # tool and sampling pressure by the prefetch depth.
        self._capacity = asyncio.Semaphore(max_concurrency)
        self._capacity_limit = max_concurrency
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start session-scoped scorer and tool resources."""
        async with self._lifecycle_lock:
            if self._resource_started:
                return
            start = getattr(self.runner, "start", None)
            if callable(start):
                await start()
            logger_start = getattr(self.logger, "start", None)
            if callable(logger_start):
                result = logger_start()
                if inspect.isawaitable(result):
                    await result
            self._resource_started = True

    async def close(self) -> None:
        """Close resources owned by the underlying episode runner."""
        async with self._lifecycle_lock:
            if not self._resource_started:
                return
            logger_close = getattr(self.logger, "close", None) or getattr(
                self.logger, "aclose", None
            )
            if callable(logger_close):
                result = logger_close()
                if inspect.isawaitable(result):
                    await result
            close = getattr(self.runner, "close", None)
            if callable(close):
                await close()
            self._resource_started = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

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
        if self._capacity_limit != self.max_concurrency:
            # Preserve the historically mutable max_concurrency attribute for
            # callers that tune capacity between collection calls. Changing it
            # while calls are already active is unsupported because replacing a
            # live semaphore would split one global resource budget.
            if type(self.max_concurrency) is not int or self.max_concurrency < 1:
                raise ValueError("max_concurrency must be a positive integer.")
            self._capacity = asyncio.Semaphore(self.max_concurrency)
            self._capacity_limit = self.max_concurrency
        await self.start()
        minimum = self.min_usable_group_size or n_samples
        if minimum > n_samples:
            raise ValueError("min_usable_group_size must not exceed n_samples.")
        # Namespace groups per collection call. Identical row contents in
        # separate calls must not accidentally merge into one GRPO group.
        namespace = uuid4().hex

        async def run(row, row_index, sample_index):
            async with self._capacity:
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
                    resource_owner=getattr(self.runner, "resource_owner", "runner"),
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
                        accepted = (
                            True
                            if self.accept_sample is None
                            else self.accept_sample(episode, exported)
                        )
                        if inspect.isawaitable(accepted):
                            accepted = await accepted
                        if not isinstance(accepted, bool):
                            raise AgentContractError(
                                "Sample acceptance hook must return a boolean."
                            )
                        if not accepted:
                            episode.status = "rejected"
                            episode.error = (
                                "Sample acceptance hook rejected the episode."
                            )
                            exported = None
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
        branch_groups = {}
        for exported in exports:
            for group_id, state in exported.metadata.get("branch_groups", {}).items():
                # Exporters scope branch groups to one logical rollout. Reject
                # collisions instead of merging independently sampled trees.
                if group_id in branch_groups:
                    raise AgentContractError(
                        "Branch group identity is not rollout scoped."
                    )
                branch_groups[group_id] = copy.deepcopy(state)
        batch = AgentRolloutBatch(
            [e for e, _ in results],
            [t for exported in exports for t in exported.trajectories],
            dropped_groups=len(rejected_groups),
            exports=exports,
            groups=groups,
            branch_groups=branch_groups,
        )
        if strict and rejected_groups:
            raise EpisodeCollectionError(batch)
        return batch

    async def collect_iter(
        self,
        rows: Sequence[Mapping[str, Any]],
        snapshot: SamplingSnapshot,
        *,
        n_samples: int = 1,
        seed: int | None = None,
        is_eval: bool = False,
        strict: bool = True,
        max_pending_groups: int | None = None,
    ):
        """Yield completed prompt groups without waiting for the slowest row.

        Each yielded batch is a complete collector transaction for one input
        row, including candidate-group validation and export filtering. It is a
        safe recovery boundary: callers may persist yielded batches and resume
        later from input rows that were never scheduled. In-flight groups are
        cancelled and drained when iteration stops; they are not replayed
        automatically because tools may have non-idempotent side effects.

        ``max_pending_groups`` limits scheduled row groups independently from
        the workflow-wide model/tool semaphore. This keeps task objects and
        retained row payloads bounded even for streaming datasets.
        """
        if max_pending_groups is None:
            max_pending_groups = self.max_concurrency
        if type(max_pending_groups) is not int or max_pending_groups < 1:
            raise ValueError("max_pending_groups must be a positive integer.")
        await self.start()
        indexed = iter(enumerate(rows))
        pending = {}

        def schedule():
            while len(pending) < max_pending_groups:
                try:
                    index, row = next(indexed)
                except StopIteration:
                    return
                group_seed = (
                    None
                    if seed is None
                    else seed + index * n_samples * self.runner.limits.max_turns
                )
                task = asyncio.create_task(
                    self.collect(
                        [row],
                        snapshot,
                        n_samples=n_samples,
                        seed=group_seed,
                        is_eval=is_eval,
                        strict=strict,
                    )
                )
                pending[task] = index

        schedule()
        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    index = pending.pop(task)
                    batch = await task
                    # The source index is diagnostic only and does not replace
                    # collector-owned group, rollout, or sample identities.
                    for state in batch.groups.values():
                        state["source_index"] = index
                    yield batch
                schedule()
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def merge_batches(*batches: AgentRolloutBatch) -> AgentRolloutBatch:
        """Combine completed collector transactions without changing identity.

        This is the safe unit for partial-rollout recovery. A caller persists
        batches already yielded by ``collect_iter`` and later merges them with
        newly collected, previously unscheduled rows. In-flight episodes are not
        reconstructed or replayed, so non-idempotent tool calls remain outside
        the recovery boundary.
        """
        episodes, trajectories, exports = [], [], []
        groups, branch_groups = {}, {}
        dropped = refills = 0
        rollout_ids, episode_ids = set(), set()
        for batch in batches:
            if not isinstance(batch, AgentRolloutBatch):
                raise TypeError("merge_batches accepts AgentRolloutBatch objects.")
            collisions = set(groups).intersection(batch.groups)
            branch_collisions = set(branch_groups).intersection(batch.branch_groups)
            export_ids = {export.logical_rollout_id for export in batch.exports}
            batch_episode_ids = {episode.episode_id for episode in batch.episodes}
            if (
                collisions
                or branch_collisions
                or rollout_ids.intersection(export_ids)
                or episode_ids.intersection(batch_episode_ids)
            ):
                raise AgentContractError(
                    "Recovered collection batches contain duplicate identities."
                )
            episodes.extend(batch.episodes)
            trajectories.extend(batch.trajectories)
            exports.extend(batch.exports)
            groups.update(copy.deepcopy(batch.groups))
            branch_groups.update(copy.deepcopy(batch.branch_groups))
            rollout_ids.update(export_ids)
            episode_ids.update(batch_episode_ids)
            dropped += batch.dropped_groups
            refills += batch.refill_batches
        return AgentRolloutBatch(
            episodes,
            trajectories,
            dropped,
            refills,
            exports,
            groups,
            branch_groups,
        )

    async def resume(
        self,
        partial: AgentRolloutBatch,
        rows: Sequence[Mapping[str, Any]],
        snapshot: SamplingSnapshot,
        **collect_kwargs,
    ) -> AgentRolloutBatch:
        """Collect unstarted rows under the original snapshot and merge results."""
        expected = {
            (episode.snapshot_id, episode.policy_version)
            for episode in partial.episodes
        }
        if expected and expected != {(snapshot.snapshot_id, snapshot.policy_version)}:
            raise AgentContractError(
                "Partial collection can resume only with its original snapshot."
            )
        resumed = await self.collect(rows, snapshot, **collect_kwargs)
        return self.merge_batches(partial, resumed)

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
        if exported.branch_map:
            declared = {branch.branch_id: branch for branch in episode.branches}
            if exported.branch_map != declared or set(exported.branch_map) != set(
                declared
            ):
                raise AgentContractError(
                    "Exported branch map must preserve every declared episode branch."
                )
            group_states = exported.metadata.get("branch_groups")
            if not isinstance(group_states, dict) or not group_states:
                raise AgentContractError(
                    "Branch export requires explicit group metadata."
                )
            expected_group_states = {}
            for branch in exported.branch_map.values():
                group_id = (
                    f"{exported.logical_rollout_id}:branch-group:{branch.group_id}"
                )
                state = expected_group_states.setdefault(
                    group_id,
                    {
                        "branch_ids": [],
                        "completed_branch_ids": [],
                        "parent_interaction_id": branch.parent_interaction_id,
                    },
                )
                state["branch_ids"].append(branch.branch_id)
                if branch.completed:
                    state["completed_branch_ids"].append(branch.branch_id)
            for state in expected_group_states.values():
                state["complete"] = len(state["branch_ids"]) == len(
                    state["completed_branch_ids"]
                )
            if group_states != expected_group_states:
                raise AgentContractError(
                    "Branch group metadata differs from declared branch completion."
                )
            complete_groups = {
                group_id
                for group_id, state in group_states.items()
                if isinstance(state, dict) and state.get("complete") is True
            }
            # A partial sibling set has no unbiased branch-relative baseline.
            # Keep it in branch_map/metadata for diagnostics, but do not train
            # any completed prefix of that group as if the missing branch had a
            # zero reward. Other complete groups in the same tree remain usable.
            rows = [row for row in rows if row.branch_group_id in complete_groups]
            if not rows:
                raise AgentContractError(
                    "Branch export contains no complete branch group."
                )
            exported.trajectories = rows
            exported.fragment_map = {
                row.fragment_id: row.interaction_ids for row in rows
            }
        elif episode.branches or any(row.branch_id is not None for row in rows):
            raise AgentContractError(
                "Branch trajectories require a declared branch map."
            )
        ids = [row.fragment_id for row in rows]
        if (
            len(set(ids)) != len(ids)
            or None in ids
            or set(ids) != set(exported.fragment_map)
        ):
            raise AgentContractError(
                "Fragment map must identify every row exactly once."
            )
        seen, accepted, occurrences = {}, [], {}
        source, source_turns = {}, {}
        interaction_records = {
            interaction.interaction_id: interaction
            for interaction in episode.interactions
        }
        generated_interactions = [
            interaction
            for interaction in episode.interactions
            if interaction.turn is not None
        ]
        if episode.interactions and len(generated_interactions) != len(episode.turns):
            raise AgentContractError(
                "Generated interactions must identify every recorded turn."
            )
        for index, turn in enumerate(episode.turns):
            identity = (
                generated_interactions[index].interaction_id
                if generated_interactions
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
            row.token_advantages = (
                list(row.token_advantages) if row.token_advantages is not None else None
            )
            row.action_weights = (
                list(row.action_weights)
                if row.action_weights is not None
                else [1.0 if active else 0.0 for active in row.loss_mask]
            )
            full_tokens = row.prompt_tokens + row.completion_tokens
            checked_contexts = set()
            for name in (
                "rollout_id",
                "episode_id",
                "sample_index",
                "snapshot_id",
                "policy_version",
                "task_id",
            ):
                if getattr(row, name) != getattr(context, name):
                    raise AgentContractError(f"Export changed candidate {name}.")
            if row.reward_group_id is None:
                if row.group_id != context.group_id:
                    raise AgentContractError("Export changed candidate group_id.")
            elif row.reward_group_id != context.group_id:
                raise AgentContractError(
                    "Compacted fragment changed its reward group identity."
                )
            first_interaction = source_turns[row.interaction_ids[0]]
            if row.prompt_tokens != list(first_interaction.prompt_tokens):
                raise AgentContractError(
                    "Fragment prompt differs from its first interaction request."
                )
            if tuple(exported.fragment_map[row.fragment_id]) != row.interaction_ids:
                raise AgentContractError(
                    "Fragment interactions disagree with row provenance."
                )
            if not set(row.interaction_ids).issubset(source_turns):
                raise AgentContractError("Fragment refers to an unknown interaction.")
            structured_outcome = aggregate_assignments(episode.score.assignments)[0]
            branch = exported.branch_map.get(row.branch_id)
            if row.branch_id is not None and branch is None:
                raise AgentContractError(
                    "Branch fragment refers to an undeclared branch identity."
                )
            expected_reward = (
                float(branch.reward)
                if branch is not None
                else episode.score.reward + structured_outcome
            )
            if row.reward != expected_reward:
                raise AgentContractError(
                    "A fragment must preserve its candidate or branch-local outcome."
                )
            if branch is not None:
                expected_group = (
                    f"{exported.logical_rollout_id}:branch-group:{branch.group_id}"
                )
                expected_path = list(branch.interaction_ids)
                parent = branch.parent_interaction_id
                while parent is not None:
                    expected_path.append(parent)
                    parent = interaction_records[parent].parent_interaction_id
                if (
                    row.branch_completed is not True
                    or row.branch_group_id != expected_group
                    or row.shared_prefix_weight != branch.shared_prefix_weight
                    or row.parent_interaction_id != branch.parent_interaction_id
                    or not set(row.interaction_ids).issubset(expected_path)
                ):
                    raise AgentContractError(
                        "Branch fragment metadata differs from its branch record."
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
                            row.action_weights[i] = 0.0
                            if row.turn_ids is not None:
                                row.turn_ids[i] = -1
                            if row.token_rewards is not None:
                                row.token_rewards[i] = 0.0
                            if row.token_advantages is not None:
                                row.token_advantages[i] = 0.0
                    seen[key] = evidence
                    occurrences.setdefault(key, []).append((row, i))
            if any(row.loss_mask):
                row.__post_init__()
                accepted.append(row)
        if self.shared_prefix_policy == "weighted":
            for key, positions in occurrences.items():
                if len(positions) < 2:
                    continue
                owners = {}
                for row, index in positions:
                    owner = row.logical_sequence_id
                    if owner in owners:
                        # Multiple fragments of one branch are a physical split,
                        # not separate counterfactual evidence. Keep the first
                        # action occurrence and mask later copies exactly once.
                        row.loss_mask[index] = False
                        row.action_weights[index] = 0.0
                        if row.turn_ids is not None:
                            row.turn_ids[index] = -1
                        if row.token_rewards is not None:
                            row.token_rewards[index] = 0.0
                        if row.token_advantages is not None:
                            row.token_advantages[index] = 0.0
                    else:
                        owners[owner] = (row, index)
                if len(owners) < 2:
                    continue
                total = sum(row.shared_prefix_weight for row, _ in owners.values())
                if total <= 0:
                    raise AgentContractError(
                        "Shared branch actions require a positive total prefix weight."
                    )
                for row, index in owners.values():
                    weight = row.shared_prefix_weight / total
                    if weight == 0:
                        row.loss_mask[index] = False
                        row.action_weights[index] = 0.0
                        if row.turn_ids is not None:
                            row.turn_ids[index] = -1
                        if row.token_rewards is not None:
                            row.token_rewards[index] = 0.0
                        if row.token_advantages is not None:
                            row.token_advantages[index] = 0.0
                    else:
                        row.action_weights[index] = weight
            accepted = [row for row in accepted if any(row.loss_mask)]
            for row in accepted:
                row.__post_init__()
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
