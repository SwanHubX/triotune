"""Orchestrate agent sampling and evaluation around the existing PPO trainer.

The workflow supplies execution records and validated trajectories; PPOTrainer
owns advantage preparation, actor/critic optimization, reporting, and saved
model state. This module orders those operations so sampling uses a fixed
policy, updates consume completed batches, and later collection sees refreshed
weights. It adds no new policy-loss or reward mathematics.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, replace
from numbers import Real
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from ...trainer.ppo.rollout import filter_informative_groups
from ...utils.checkpoint import training_limit
from ...utils.dataloader import iter_batches, num_batches
from ..config import AgentConfig
from ..rollout.workflow import AgentRolloutBatch, AgentRolloutWorkflow
from ..runtime.sampling import SamplingSnapshot
from ..types import (
    AgentBatchAdvantageProvider,
    AgentBatchAdvantages,
    AgentContractError,
)
from .evaluation import evaluation_metrics, paired_task_metrics, paired_task_summary


def filter_agent_groups(trajectories):
    """Keep groups with outcome variance or any explicit process signal.

    Equal terminal rewards do not make a process-reward episode uninformative.
    Process rewards are consumed independently of scalar reward normalization.
    """
    informative = {
        t.comparison_group_id for t in filter_informative_groups(trajectories)
    }
    informative.update(
        t.comparison_group_id
        for t in trajectories
        if (t.token_rewards is not None and any(t.token_rewards))
    )
    informative.update(
        t.comparison_group_id
        for t in trajectories
        if t.token_advantages is not None and any(t.token_advantages)
    )
    return [t for t in trajectories if t.comparison_group_id in informative]


if TYPE_CHECKING:
    from ...trainer.ppo.trajectory import PPOState
    from ...trainer.rl_trainer import PPOTrainer


class AgentTrainingLoop:
    """Train from fixed snapshots with optional bounded collection overlap.

    The default path finishes collection before updating. Asynchronous mode
    overlaps completed-snapshot sampling with one serialized PPO update and
    validates the resulting policy age before consumption. The supplied trainer
    still owns all model clients, optimizers, checkpoints, and reporting.
    """

    def __init__(
        self,
        trainer: PPOTrainer,
        workflow: AgentRolloutWorkflow,
        *,
        snapshot_factory: Callable[[], Awaitable[SamplingSnapshot]] | None = None,
        config: AgentConfig | None = None,
        batch_advantage_provider: AgentBatchAdvantageProvider | None = None,
    ):
        """Bind the trainer, execution workflow, and optional snapshot provider.

        The default provider exports the trainer's current actor. A replacement
        can configure stop markers or wrap another sampling backend, but must
        preserve the fixed-policy contract throughout a collected batch.
        Calls that share this trainer's optimizer state must be serialized.
        """
        self.trainer, self.workflow = trainer, workflow
        self.config = config or AgentConfig()
        self.snapshot_factory = snapshot_factory or self._snapshot
        if batch_advantage_provider is not None and not callable(
            getattr(batch_advantage_provider, "advantages", None)
        ):
            raise TypeError("batch_advantage_provider must implement advantages().")
        validate_trainer = getattr(
            batch_advantage_provider, "validate_trainer", None
        )
        if callable(validate_trainer):
            # Algorithm-specific providers can reject trainer settings that
            # would otherwise be silently ignored by a replacement advantage.
            # Validation is synchronous and side-effect free at construction.
            validate_trainer(trainer)
        self.batch_advantage_provider = batch_advantage_provider
        self._last_eval_step: int | None = None
        self._seed_offset = 0
        self.last_eval_batch: AgentRolloutBatch | None = None
        self._baseline_episodes = None
        self.update_history: list[dict[str, Any]] = []
        self.last_train_batch: AgentRolloutBatch | None = None
        self.last_train_snapshot: SamplingSnapshot | None = None
        if trainer._checkpoint.enabled:
            if self.config.async_rollouts or snapshot_factory is not None:
                raise ValueError("Complete agent checkpoints require synchronous rollouts and the trainer snapshot factory.")
            identity = self.config.to_dict()
            identity.pop("log_path", None)
            identity["executor"] = f"{type(workflow.runner).__module__}.{type(workflow.runner).__qualname__}"
            provider_identity = getattr(
                batch_advantage_provider, "checkpoint_identity", None
            )
            if batch_advantage_provider is not None:
                identity["batch_advantage_provider"] = {
                    "type": f"{type(batch_advantage_provider).__module__}."
                    f"{type(batch_advantage_provider).__qualname__}",
                    "settings": (
                        provider_identity() if callable(provider_identity) else None
                    ),
                }
            anchor_observer = getattr(workflow.runner, "anchor_observer", None)
            if anchor_observer is not None:
                identity["anchor_observer"] = (
                    f"{type(anchor_observer).__module__}."
                    f"{type(anchor_observer).__qualname__}"
                )
            identity["limits"] = asdict(workflow.runner.limits)
            trainer._checkpoint.bind_loop("agent", identity=identity, owner=self)

    def checkpoint_state(self):
        """Keep sampler/evaluation progress without persisting completed turns."""
        return {"seed_offset": self._seed_offset, "last_eval_step": self._last_eval_step,
                "baseline": self._baseline_episodes, "updates": self.update_history}

    def restore_checkpoint_state(self, state):
        """Rejoin candidate seed ranges and paired evaluation after optimizer load."""
        self._seed_offset = state["seed_offset"]
        self._last_eval_step = state["last_eval_step"]
        self._baseline_episodes = state["baseline"]
        self.update_history = state["updates"]

    async def _snapshot(self):
        return await self.trainer.create_sampling_snapshot(stop=self.config.stop)

    async def evaluate(
        self, dataset: Iterable[Mapping[str, Any]] | None = None
    ) -> dict[str, float]:
        """Evaluate the current policy through the same agent and scorer.

        The supplied dataset overrides trainer.eval_dataset for this call.
        Collection records model and system failures without rejecting the
        evaluation batch. Metrics are published, but no optimizer operation or
        checkpoint save is performed. Standalone callers close the trainer
        themselves after they finish evaluation.
        """
        trainer, config = self.trainer, self.trainer.args
        dataset = trainer.eval_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("Agent evaluation requires an eval dataset.")
        n_samples = self.config.evaluation.num_generations or config.num_generations
        if max(self.config.evaluation.pass_ks) > n_samples:
            raise ValueError("Evaluation pass_k must not exceed num_generations.")
        snapshot = await self.snapshot_factory()
        # Hold one policy fixed across all evaluation batches so the aggregate
        # describes a single model state, even when the dataset is streamed.
        episodes, trajectories, exports, groups, branch_groups = [], [], [], {}, {}
        row_offset, dropped = 0, 0
        task_ids = set()
        batch_size = config.eval_batch_size or config.batch_size
        batches = iter_batches(dataset, batch_size, max_batches=config.eval_max_batches)
        with tqdm(
            total=num_batches(dataset, batch_size, max_batches=config.eval_max_batches),
            desc="Agent eval",
            unit="batch",
            leave=False,
        ) as progress:
            for batch_rows in batches:
                # Stable identities survive batching and policy refresh. Reject
                # duplicate explicit IDs rather than merging unrelated tasks.
                batch_rows = [dict(row) for row in batch_rows]
                for index, row in enumerate(batch_rows):
                    row.setdefault("task_id", f"eval-row-{row_offset + index}")
                    identity = str(row["task_id"])
                    if identity in task_ids:
                        raise ValueError("Evaluation task_id values must be unique.")
                    task_ids.add(identity)
                batch = await self.workflow.collect(
                    batch_rows,
                    snapshot,
                    n_samples=n_samples,
                    seed=self.config.evaluation.seed
                    + row_offset * n_samples * self.workflow.runner.limits.max_turns,
                    is_eval=True,
                    strict=False,
                )
                episodes.extend(batch.episodes)
                trajectories.extend(batch.trajectories)
                exports.extend(batch.exports)
                groups.update(batch.groups)
                branch_groups.update(batch.branch_groups)
                dropped += batch.dropped_groups
                row_offset += len(batch_rows)
                progress.update(1)
        if not episodes:
            raise ValueError("Agent evaluation dataset is empty.")
        self.last_eval_batch = AgentRolloutBatch(
            episodes,
            trajectories,
            dropped,
            exports=exports,
            groups=groups,
            branch_groups=branch_groups,
        )
        metrics = self.last_eval_batch.metrics(prefix="eval/agent")
        metrics.update(
            evaluation_metrics(episodes, pass_ks=self.config.evaluation.pass_ks)
        )
        if self._baseline_episodes is None:
            self._baseline_episodes = paired_task_summary(episodes)
        else:
            metrics.update(paired_task_metrics(self._baseline_episodes, episodes))
        trainer._eval_sample_input_tokens += int(metrics["eval/agent/sample_input_tokens"])
        trainer._eval_sample_output_tokens += int(metrics["eval/agent/sample_output_tokens"])
        metrics["usage/eval_sample_input_tokens_total"] = float(trainer._eval_sample_input_tokens)
        metrics["usage/eval_sample_output_tokens_total"] = float(trainer._eval_sample_output_tokens)
        await trainer.publish_metrics(metrics)
        self._last_eval_step = trainer.state.global_step
        return metrics

    def _seed(self, n_rows: int) -> int | None:
        """Reserve disjoint per-turn seed ranges across collection calls.

        Advance by the actual number of rows, which may differ between training,
        evaluation, partial batches, and dynamic-sampling refill requests.
        """
        seed = self.trainer.args.seed
        offset = self._seed_offset
        self._seed_offset += (
            n_rows
            * self.trainer.args.num_generations
            * self.workflow.runner.limits.max_turns
        )
        return None if seed is None else seed + offset

    async def _collect(
        self, rows, snapshot: SamplingSnapshot | None = None
    ) -> AgentRolloutBatch:
        """Collect prompt groups and optionally refill uninformative groups.

        The source iterator is consumed once. Dynamic sampling reuses the
        outcome/process group filter and keeps a single policy snapshot
        during all refill requests. Episode diagnostics include rejected groups
        even though their trajectories do not enter the optimizer batch.
        """
        config = self.trainer.args
        snapshot = snapshot or await self.snapshot_factory()
        episodes, accepted, exports, groups, branch_groups = [], [], [], {}, {}
        self.last_train_snapshot = snapshot
        group_count, empty_batches, dropped, refills = 0, 0, 0, 0
        collection_count = 0
        while group_count < config.batch_size:
            batch_rows = list(itertools.islice(rows, config.batch_size - group_count))
            if not batch_rows:
                break
            if collection_count:
                if refills >= self.config.max_refill_batches:
                    raise RuntimeError(
                        "Agent collection exceeded max_refill_batches before filling a batch."
                    )
                refills += 1
            collection_count += 1
            batch = await self.workflow.collect(
                batch_rows,
                snapshot,
                n_samples=config.num_generations,
                seed=self._seed(len(batch_rows)),
                strict=self.config.on_system_failure == "raise",
            )
            episodes.extend(batch.episodes)
            generated = batch.trajectories
            dropped += batch.dropped_groups
            if config.dynamic_sampling:
                generated = filter_agent_groups(generated)
            accepted.extend(generated)
            accepted_ids = {t.rollout_id for t in generated}
            exports.extend(
                e for e in batch.exports if e.logical_rollout_id in accepted_ids
            )
            groups.update(batch.groups)
            branch_groups.update(batch.branch_groups)
            group_count += len({t.reward_group_id or t.group_id for t in generated})
            empty_batches = 0 if generated else empty_batches + 1
            if empty_batches >= config.max_empty_batches:
                raise RuntimeError(
                    "Agent dynamic sampling found too many uninformative groups."
                )
        return AgentRolloutBatch(
            episodes,
            accepted,
            dropped,
            refills,
            exports,
            groups,
            branch_groups,
        )

    async def _update(self, batch, snapshot, progress, *, max_staleness=0):
        """Consume one completed batch and record its exact sampling provenance."""
        trainer = self.trainer
        self.last_train_snapshot = snapshot
        if self.batch_advantage_provider is not None:
            await self._add_batch_advantages(batch)
        previous_optimizer_steps = trainer.state.optimizer_steps
        metrics = await trainer.step(batch.trajectories, max_staleness=max_staleness)
        self.last_train_batch = batch
        self.update_history.append(
            {
                "step": trainer.state.global_step,
                "snapshot_ids": sorted({e.snapshot_id for e in batch.episodes}),
                "policy_versions": sorted(
                    {str(e.policy_version) for e in batch.episodes}
                ),
                "optimizer_steps": trainer.state.optimizer_steps
                - previous_optimizer_steps,
                "action_tokens": sum(
                    sum(t.effective_action_weights) for t in batch.trajectories
                ),
                "context_tokens": sum(
                    len(t.loss_mask) - sum(t.loss_mask) for t in batch.trajectories
                ),
                "metrics": dict(metrics),
            }
        )
        batch_metrics = batch.metrics()
        trainer._sample_input_tokens += int(batch_metrics["agent/sample_input_tokens"])
        trainer._sample_output_tokens += int(batch_metrics["agent/sample_output_tokens"])
        batch_metrics["usage/sample_input_tokens_total"] = float(trainer._sample_input_tokens)
        batch_metrics["usage/sample_output_tokens_total"] = float(trainer._sample_output_tokens)
        await trainer.publish_metrics(batch_metrics)
        progress.update(1)
        if (
            trainer.eval_dataset is not None
            and trainer.args.eval_steps > 0
            and trainer.state.global_step % trainer.args.eval_steps == 0
        ):
            await self.evaluate()
        trainer._checkpoint.publish(trainer)

    async def _add_batch_advantages(self, batch: AgentRolloutBatch) -> None:
        """Validate and attach one provider's batch-level advantages.

        AgentRL changes only ``token_advantages`` in the replacement records;
        providers must treat their inputs as read-only. Running this hook
        immediately before ``trainer.step`` gives a provider the complete
        accepted update batch and executes it exactly once even when the actor
        later performs multiple minibatches or epochs. The PyTRIO boundary is
        therefore identical to ordinary PPO/GRPO.
        """

        provider = self.batch_advantage_provider
        if provider is None:
            return
        rows = provider.advantages(tuple(batch.trajectories))
        if inspect.isawaitable(rows):
            rows = await rows
        mode = "add"
        if isinstance(rows, AgentBatchAdvantages):
            mode, rows = rows.mode, rows.values
        if isinstance(rows, (str, bytes)):
            raise AgentContractError(
                "Batch advantages must contain one numeric row per trajectory."
            )
        try:
            rows = list(rows)
        except TypeError as exc:
            raise AgentContractError(
                "Batch advantages must be an iterable of numeric rows."
            ) from exc
        if len(rows) != len(batch.trajectories):
            raise AgentContractError(
                "Batch advantage rows must align with accepted trajectories."
            )

        prepared = []
        for trajectory, values in zip(batch.trajectories, rows, strict=True):
            if isinstance(values, (str, bytes)):
                raise AgentContractError(
                    "Each batch advantage row must be a numeric sequence."
                )
            try:
                values = list(values)
            except TypeError as exc:
                raise AgentContractError(
                    "Each batch advantage row must be an iterable."
                ) from exc
            if len(values) != len(trajectory.completion_tokens):
                raise AgentContractError(
                    "Batch advantages must align with completion tokens."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            ):
                raise AgentContractError(
                    "Batch advantages must contain only finite numeric values."
                )
            if any(
                not active and float(value) != 0.0
                for value, active in zip(values, trajectory.loss_mask, strict=True)
            ):
                raise AgentContractError(
                    "Batch advantages must be zero at inactive context tokens."
                )
            existing = trajectory.token_advantages or [0.0] * len(values)
            combined = [
                float(base) + float(extra)
                for base, extra in zip(existing, values, strict=True)
            ]
            prepared.append(
                replace(
                    trajectory,
                    token_advantages=combined,
                    token_advantage_mode=(
                        "replace"
                        if mode == "replace"
                        or trajectory.token_advantage_mode == "replace"
                        else "add"
                    ),
                )
            )
        batch.trajectories = prepared

    async def _train_overlapped(self, rows, progress):
        """Pipeline fixed-snapshot collection with one serialized PPO update.

        Prefetched tasks own immutable SamplingSnapshot objects created before
        their corresponding optimizer overlap. Batches are consumed in launch
        order, which bounds numeric policy age by configured queue capacity and
        avoids starving an older behavior policy behind a faster newer request.
        Pending collection is cancelled and drained on every terminal path so
        tools, scorers, and PyTRIO futures do not outlive the training loop.
        """
        trainer = self.trainer
        capacity = self.config.rollout_prefetch_capacity
        pending = []

        async def launch(snapshot):
            if type(snapshot.policy_version) is not int:
                raise AgentContractError(
                    "Asynchronous rollouts require numeric policy_version provenance."
                )
            task = asyncio.create_task(self._collect(rows, snapshot))
            pending.append((snapshot, task))

        snapshot = await self.snapshot_factory()
        if type(snapshot.policy_version) is not int:
            raise AgentContractError(
                "Asynchronous rollouts require numeric policy_version provenance."
            )
        current = await self._collect(rows, snapshot)
        try:
            while current.trajectories and (
                trainer.args.max_steps is None
                or trainer.state.global_step < trainer.args.max_steps
            ):
                while len(pending) < capacity:
                    await launch(snapshot)
                await self._update(
                    current,
                    snapshot,
                    progress,
                    max_staleness=self.config.max_policy_staleness,
                )
                if (
                    trainer.args.max_steps is not None
                    and trainer.state.global_step >= trainer.args.max_steps
                ):
                    break

                next_snapshot, task = pending.pop(0)
                current = await task
                snapshot = next_snapshot
                # Export a replacement only after the optimizer finishes. The
                # new task can overlap the next update, while older queued tasks
                # retain their original version and are checked when consumed.
                replacement = await self.snapshot_factory()
                await launch(replacement)
                if not current.trajectories and current.episodes:
                    await trainer.publish_metrics(current.metrics())
        finally:
            for _, task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*(task for _, task in pending), return_exceptions=True)

    async def train(self, *, stop_after_step: int | None = None) -> PPOState:
        """Run collection, updates, configured evaluations, and final saving.

        Dataset rows are consumed once; max_steps limits policy updates rather
        than generated turns. Periodic checkpointing remains in trainer.step(),
        with a final save here when the current step was not already saved.
        This method owns reporter finalization on success, failure, and
        cancellation, and returns the trainer's existing mutable PPOState.
        """
        trainer, config = self.trainer, self.trainer.args
        try:
            if trainer.train_dataset is None:
                raise ValueError("Agent training requires a train dataset.")
            # Reject one-shot eval iterators before consuming a training batch.
            if (
                trainer.eval_dataset is not None
                and iter(trainer.eval_dataset) is trainer.eval_dataset
            ):
                raise ValueError(
                    "Periodic agent evaluation requires a replayable dataset."
                )
            if stop_after_step is not None and self.config.async_rollouts:
                raise ValueError("Orderly pauses require synchronous agent rollouts.")
            await trainer.initialize()
            initial_step = trainer.state.global_step
            limit, paused = training_limit(config, initial_step, stop_after_step)
            trainer._checkpoint.begin()
            if trainer._checkpoint.evaluation_due(trainer, self._last_eval_step):
                await self.evaluate()
            rows = trainer._checkpoint.rows(trainer)
            with tqdm(
                total=config.max_steps,
                initial=initial_step,
                desc="Agent PPO train",
                unit="step",
            ) as progress:
                while (
                    limit is None
                    or trainer.state.global_step < limit
                ):
                    if self.config.async_rollouts:
                        await self._train_overlapped(rows, progress)
                        break
                    batch = await self._collect(rows)
                    if not batch.trajectories:
                        if batch.episodes:
                            await trainer.publish_metrics(batch.metrics())
                        break
                    await self._update(
                        batch,
                        self.last_train_snapshot,
                        progress,
                    )
            if trainer.state.global_step == initial_step and (
                config.max_steps is None or initial_step < config.max_steps
            ):
                raise ValueError("No trainable agent batch was collected.")
            if not config.save_every or trainer.state.global_step % config.save_every:
                await trainer.save_checkpoint()
            if (
                trainer.eval_dataset is not None
                and not paused
                and config.eval_after_train
                and self._last_eval_step != trainer.state.global_step
            ):
                # A periodic evaluation may already cover the final weights.
                await self.evaluate()
            trainer._checkpoint.publish(trainer)
        except BaseException as exc:
            # Preserve cancellation and the original training exception even
            # if reporter cleanup also fails. Error text must be serializable.
            try:
                await trainer.close(state="crashed", error=str(exc))
            except Exception as cleanup_error:  # noqa: BLE001 -- retain the original training failure
                exc.add_note(f"Reporter cleanup also failed: {cleanup_error}")
            try:
                await self.workflow.close()
            except Exception as cleanup_error:  # noqa: BLE001 -- retain the original training failure
                exc.add_note(f"Agent resource cleanup also failed: {cleanup_error}")
            raise
        else:
            await trainer.close(flush=not paused)
            await self.workflow.close()
        finally:
            trainer._checkpoint.end()
        return trainer.state
