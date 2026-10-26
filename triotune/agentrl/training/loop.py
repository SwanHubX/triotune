"""Orchestrate agent sampling and evaluation around the existing PPO trainer.

The workflow supplies execution records and validated trajectories; PPOTrainer
owns advantage preparation, actor/critic optimization, reporting, and saved
model state. This module orders those operations so sampling uses a fixed
policy, updates consume completed batches, and later collection sees refreshed
weights. It adds no new policy-loss or reward mathematics.
"""

from __future__ import annotations

import itertools
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from ...trainer.ppo.rollout import filter_informative_groups
from ...utils.dataloader import iter_batches, num_batches
from ..config import AgentConfig
from ..rollout.workflow import AgentRolloutBatch, AgentRolloutWorkflow
from ..runtime.sampling import SamplingSnapshot
from .evaluation import evaluation_metrics, paired_task_metrics


def filter_agent_groups(trajectories):
    """Keep groups with outcome variance or any explicit process signal.

    Equal terminal rewards do not make a process-reward episode uninformative.
    Process rewards are consumed independently of scalar reward normalization.
    """
    informative = {t.group_id for t in filter_informative_groups(trajectories)}
    informative.update(
        t.group_id
        for t in trajectories
        if (
            t.token_rewards is not None
            and any(t.token_rewards)
        )
    )
    informative.update(
        t.group_id
        for t in trajectories
        if t.token_advantages is not None and any(t.token_advantages)
    )
    return [t for t in trajectories if t.group_id in informative]


if TYPE_CHECKING:
    from ...trainer.ppo.trajectory import PPOState
    from ...trainer.rl_trainer import PPOTrainer


class AgentTrainingLoop:
    """Collect before updating; one snapshot covers all turns and prompt groups.

    This loop does not overlap optimizer updates with collection. The supplied
    PPOTrainer owns all model clients, optimizers, checkpoints and reporting.
    """

    def __init__(
        self,
        trainer: PPOTrainer,
        workflow: AgentRolloutWorkflow,
        *,
        snapshot_factory: Callable[[], Awaitable[SamplingSnapshot]] | None = None,
        config: AgentConfig | None = None,
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
        self._last_eval_step: int | None = None
        self._seed_offset = 0
        self.last_eval_batch: AgentRolloutBatch | None = None
        self._baseline_episodes = None
        self.update_history: list[dict[str, Any]] = []
        self.last_train_batch: AgentRolloutBatch | None = None
        self.last_train_snapshot: SamplingSnapshot | None = None

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
        episodes, trajectories, exports, groups = [], [], [], {}
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
                dropped += batch.dropped_groups
                row_offset += len(batch_rows)
                progress.update(1)
        if not episodes:
            raise ValueError("Agent evaluation dataset is empty.")
        self.last_eval_batch = AgentRolloutBatch(episodes, trajectories, dropped, exports=exports, groups=groups)
        metrics = self.last_eval_batch.metrics(prefix="eval/agent")
        metrics.update(
            evaluation_metrics(episodes, pass_ks=self.config.evaluation.pass_ks)
        )
        if self._baseline_episodes is None:
            self._baseline_episodes = list(episodes)
        else:
            metrics.update(paired_task_metrics(self._baseline_episodes, episodes))
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

    async def _collect(self, rows) -> AgentRolloutBatch:
        """Collect prompt groups and optionally refill uninformative groups.

        The source iterator is consumed once. Dynamic sampling reuses the
        outcome/process group filter and keeps a single policy snapshot
        during all refill requests. Episode diagnostics include rejected groups
        even though their trajectories do not enter the optimizer batch.
        """
        config = self.trainer.args
        snapshot = await self.snapshot_factory()
        episodes, accepted, exports, groups = [], [], [], {}
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
            exports.extend(e for e in batch.exports if e.logical_rollout_id in accepted_ids)
            groups.update(batch.groups)
            group_count += len({t.group_id for t in generated})
            empty_batches = 0 if generated else empty_batches + 1
            if empty_batches >= config.max_empty_batches:
                raise RuntimeError(
                    "Agent dynamic sampling found too many uninformative groups."
                )
        return AgentRolloutBatch(episodes, accepted, dropped, refills, exports, groups)

    async def train(self) -> PPOState:
        """Run collection, updates, configured evaluations, and final saving.

        Dataset rows are consumed once; max_steps limits policy updates rather
        than generated turns. Periodic checkpointing remains in trainer.step(),
        with a final save here when the current step was not already saved.
        This method owns reporter finalization on success, failure, and
        cancellation, and returns the trainer's existing mutable PPOState.
        """
        trainer, config = self.trainer, self.trainer.args
        initial_step = trainer.state.global_step
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
            await trainer.initialize()
            if trainer.eval_dataset is not None and config.eval_before_train:
                await self.evaluate()
            rows = iter(trainer.train_dataset)
            with tqdm(
                total=config.max_steps,
                initial=initial_step,
                desc="Agent PPO train",
                unit="step",
            ) as progress:
                while (
                    config.max_steps is None
                    or trainer.state.global_step < config.max_steps
                ):
                    batch = await self._collect(rows)
                    if not batch.trajectories:
                        if batch.episodes:
                            await trainer.publish_metrics(batch.metrics())
                        break
                    # All model/tool interactions finish before parameters can
                    # change. The next _collect call obtains a new snapshot.
                    previous_optimizer_steps = trainer.state.optimizer_steps
                    metrics = await trainer.step(batch.trajectories)
                    self.last_train_batch = batch
                    self.update_history.append(
                        {
                            "step": trainer.state.global_step,
                            "snapshot_ids": sorted(
                                {e.snapshot_id for e in batch.episodes}
                            ),
                            "optimizer_steps": trainer.state.optimizer_steps
                            - previous_optimizer_steps,
                            "action_tokens": sum(
                                sum(t.loss_mask) for t in batch.trajectories
                            ),
                            "context_tokens": sum(
                                len(t.loss_mask) - sum(t.loss_mask)
                                for t in batch.trajectories
                            ),
                            "metrics": dict(metrics),
                        }
                    )
                    await trainer.publish_metrics(batch.metrics())
                    progress.update(1)
                    if (
                        trainer.eval_dataset is not None
                        and config.eval_steps > 0
                        and trainer.state.global_step % config.eval_steps == 0
                    ):
                        await self.evaluate()
            if trainer.state.global_step == initial_step:
                raise ValueError("No trainable agent batch was collected.")
            if not config.save_every or trainer.state.global_step % config.save_every:
                await trainer.save_checkpoint()
            if (
                trainer.eval_dataset is not None
                and config.eval_after_train
                and self._last_eval_step != trainer.state.global_step
            ):
                # A periodic evaluation may already cover the final weights.
                await self.evaluate()
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
            await trainer.close()
            await self.workflow.close()
        return trainer.state
