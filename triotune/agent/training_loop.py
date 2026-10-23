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

from ..trainer.ppo.rollout import filter_informative_groups
from ..utils.dataloader import iter_batches, num_batches
from .rollout_workflow import AgentRolloutBatch, AgentRolloutWorkflow
from .sampling import SamplingSnapshot

if TYPE_CHECKING:
    from ..trainer.ppo.contracts import PPOState
    from ..trainer.rl_trainer import PPOTrainer


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
    ):
        """Bind the trainer, execution workflow, and optional snapshot provider.

        The default provider exports the trainer's current actor. A replacement
        can configure stop markers or wrap another sampling backend, but must
        preserve the fixed-policy contract throughout a collected batch.
        Calls that share this trainer's optimizer state must be serialized.
        """
        self.trainer, self.workflow = trainer, workflow
        self.snapshot_factory = snapshot_factory or trainer.create_sampling_snapshot
        self._last_eval_step: int | None = None
        self._seed_offset = 0

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
        snapshot = await self.snapshot_factory()
        # Hold one policy fixed across all evaluation batches so the aggregate
        # describes a single model state, even when the dataset is streamed.
        episodes, trajectories = [], []
        batch_size = config.eval_batch_size or config.batch_size
        batches = iter_batches(dataset, batch_size, max_batches=config.eval_max_batches)
        with tqdm(
            total=num_batches(dataset, batch_size, max_batches=config.eval_max_batches),
            desc="Agent eval",
            unit="batch",
            leave=False,
        ) as progress:
            for batch_rows in batches:
                batch = await self.workflow.collect(
                    batch_rows,
                    snapshot,
                    n_samples=config.num_generations,
                    seed=self._seed(len(batch_rows)),
                    is_eval=True,
                    strict=False,
                )
                episodes.extend(batch.episodes)
                trajectories.extend(batch.trajectories)
                progress.update(1)
        if not episodes:
            raise ValueError("Agent evaluation dataset is empty.")
        metrics = AgentRolloutBatch(episodes, trajectories).metrics(prefix="eval/agent")
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
        existing raw-reward group filter and keeps a single policy snapshot
        during all refill requests. Episode diagnostics include rejected groups
        even though their trajectories do not enter the optimizer batch.
        """
        config = self.trainer.args
        snapshot = await self.snapshot_factory()
        episodes, accepted = [], []
        group_count, empty_batches = 0, 0
        while group_count < config.batch_size:
            batch_rows = list(itertools.islice(rows, config.batch_size - group_count))
            if not batch_rows:
                break
            batch = await self.workflow.collect(
                batch_rows,
                snapshot,
                n_samples=config.num_generations,
                seed=self._seed(len(batch_rows)),
            )
            episodes.extend(batch.episodes)
            generated = batch.trajectories
            if config.dynamic_sampling:
                generated = filter_informative_groups(generated)
            accepted.extend(generated)
            group_count += len({t.group_id for t in generated})
            empty_batches = 0 if generated else empty_batches + 1
            if empty_batches >= config.max_empty_batches:
                raise RuntimeError(
                    "Agent dynamic sampling found too many uninformative groups."
                )
        return AgentRolloutBatch(episodes, accepted)

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
                        break
                    # All model/tool interactions finish before parameters can
                    # change. The next _collect call obtains a new snapshot.
                    await trainer.step(batch.trajectories)
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
            raise
        else:
            await trainer.close()
        return trainer.state
