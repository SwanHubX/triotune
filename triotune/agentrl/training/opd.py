"""Connect exact-token multi-turn agent collection to vanilla OPD.

The ordinary OPD trainer owns optimization but assumes one static prompt and
one completion.  An agent environment instead alternates model actions and
external observations.  This module preserves that separation of concerns:

* ``AgentRolloutWorkflow`` owns environment execution and records the exact
  prompt, sampled token IDs, and behavior log-probabilities for every turn.
* ``episode_turn_rollouts`` converts each generated turn into an independent
  OPD action while keeping the preceding conversation as masked context.
* ``OPDTrainer`` asks the fixed teacher to score those exact student actions
  and remains the only object allowed to update student parameters.

Environment rewards are retained for diagnostics and task evaluation.  They do
not enter the vanilla OPD loss; the only optimization signal is the teacher
minus student log-probability signal already implemented by the OPD trainer.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ...trainer.opd.contracts import OPDResult, OPDRollout
from ...utils.checkpoint import training_limit
from ..config import AgentConfig
from ..diagnostics.logging import JsonlEvaluationSummaryLogger
from ..rollout.workflow import AgentRolloutBatch, AgentRolloutWorkflow
from ..runtime.sampling import (
    SamplingSnapshot,
    create_sampling_snapshot,
    validate_training_snapshot,
)
from ..training.evaluation import evaluation_metrics
from ..types import AgentContractError, EpisodeResult


def episode_turn_rollouts(
    episodes: Sequence[EpisodeResult],
    *,
    behavior_policy_version: int,
    teacher_group: str = "default",
) -> list[OPDRollout]:
    """Convert exact generated turns into completion-only OPD actions.

    Each returned record has one model request as its prompt and one generated
    response as its completion.  Tool observations and earlier model actions
    therefore condition the teacher and student in exactly the same way, while
    only the current response receives an OPD target.  Flattening a whole agent
    episode would incorrectly train on environment text and is deliberately
    rejected by this interface.

    All episodes must come from the current immutable sampling snapshot.  A
    stale or mixed batch fails before teacher scoring, which prevents behavior
    log-probabilities from being attached to another policy version.
    """

    if not episodes:
        raise ValueError("Agent OPD conversion requires at least one episode.")
    if (
        isinstance(behavior_policy_version, bool)
        or not isinstance(behavior_policy_version, int)
        or behavior_policy_version < 0
    ):
        raise ValueError("behavior_policy_version must be a nonnegative integer.")
    if not isinstance(teacher_group, str) or not teacher_group.strip():
        raise ValueError("teacher_group must be a nonempty string.")
    teacher_group = teacher_group.strip()

    if any(not isinstance(episode, EpisodeResult) for episode in episodes):
        raise TypeError("Agent OPD inputs must be EpisodeResult records.")
    snapshot_ids = {episode.snapshot_id for episode in episodes}
    if len(snapshot_ids) != 1:
        raise AgentContractError(
            "One Agent OPD update must use exactly one sampling snapshot."
        )

    rollouts = []
    for episode_index, episode in enumerate(episodes):
        episode.__post_init__()
        if not episode.trainable:
            raise AgentContractError(
                "Agent OPD cannot train an unscored or system-failed episode."
            )
        if episode.policy_version != behavior_policy_version:
            raise AgentContractError(
                "Episode policy version differs from the current OPD student."
            )
        if not episode.turns:
            raise AgentContractError("A trainable Agent OPD episode has no turns.")

        for turn_index, turn in enumerate(episode.turns):
            generated = turn.generation
            # Length termination applies to the current action.  Episode-level
            # limits apply only to the final generated action because earlier
            # turns completed normally and produced valid environment states.
            truncated = generated.stop_reason in {
                "length",
                "max_tokens",
                "max_length",
            } or (
                turn_index == len(episode.turns) - 1
                and episode.status in {"max_turns", "token_limit", "context_limit"}
            )
            rollouts.append(
                OPDRollout(
                    prompt_tokens=list(turn.prompt_tokens),
                    completion_tokens=list(generated.tokens),
                    behavior_logprobs=list(generated.logprobs),
                    prompt_index=episode_index,
                    sample_index=episode.sample_index,
                    behavior_policy_version=behavior_policy_version,
                    teacher_group=teacher_group,
                    truncated=truncated,
                )
            )
    return rollouts


@dataclass(slots=True)
class AgentOPDUpdate:
    """Keep environment evidence beside one completed OPD optimizer result."""

    result: OPDResult
    batch: AgentRolloutBatch
    rollouts: list[OPDRollout]


class AgentOPDTrainingLoop:
    """Run synchronous multi-turn on-policy distillation through AgentRL.

    The loop intentionally mirrors the established PPO agent lifecycle without
    importing PPO reward, advantage, critic, or reference-policy components.
    One iteration exports a fixed student snapshot, completes all environment
    episodes, converts every turn, teacher-scores the original actions, performs
    one OPD update, then discards the snapshot before the next iteration.
    """

    def __init__(
        self,
        trainer: Any,
        workflow: AgentRolloutWorkflow,
        *,
        config: AgentConfig | None = None,
        teacher_group: str = "default",
    ) -> None:
        if not callable(
            getattr(trainer, "create_sampling_snapshot", None)
        ) or not callable(getattr(trainer, "step_rollouts_async", None)):
            raise TypeError(
                "AgentOPDTrainingLoop requires an OPD trainer with agent rollout APIs."
            )
        if not isinstance(workflow, AgentRolloutWorkflow):
            raise TypeError("workflow must be an AgentRolloutWorkflow.")
        self.trainer = trainer
        self.workflow = workflow
        self.config = config or AgentConfig()
        if self.config.async_rollouts:
            raise ValueError(
                "Vanilla Agent OPD currently requires synchronous fresh rollouts."
            )
        if not isinstance(teacher_group, str) or not teacher_group.strip():
            raise ValueError("teacher_group must be a nonempty string.")
        self.teacher_group = teacher_group.strip()
        if self.teacher_group not in trainer.teacher_groups:
            raise ValueError(f"Unknown teacher group {self.teacher_group!r}.")

        self._seed_offset = 0
        self._last_eval_step: int | None = None
        self.last_train_batch: AgentRolloutBatch | None = None
        self.last_eval_batch: AgentRolloutBatch | None = None
        self.last_snapshot: SamplingSnapshot | None = None
        self.update_history: list[dict[str, Any]] = []
        self.evaluation_logger = (
            JsonlEvaluationSummaryLogger(
                Path(self.config.log_path).with_name("eval-episodes.jsonl")
            )
            if self.config.log_path is not None
            else None
        )

        # Complete checkpoints must also capture the agent-side seed cursor and
        # update provenance.  The trainer continues to own remote model state.
        if trainer._checkpoint.enabled:
            identity = self.config.to_dict()
            identity.pop("log_path", None)
            identity.update(
                {
                    "teacher_group": self.teacher_group,
                    "executor": (
                        f"{type(workflow.runner).__module__}."
                        f"{type(workflow.runner).__qualname__}"
                    ),
                }
            )
            trainer._checkpoint.bind_loop(
                "agent_opd",
                identity=identity,
                owner=self,
            )

    def checkpoint_state(self) -> dict[str, Any]:
        """Persist deterministic collection progress, not completed episodes."""

        return {
            "seed_offset": self._seed_offset,
            "last_eval_step": self._last_eval_step,
            "updates": self.update_history,
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore the exact next seed and prior update audit history."""

        expected = {"seed_offset", "last_eval_step", "updates"}
        if set(state) != expected:
            raise ValueError("Agent OPD checkpoint loop state is incomplete.")
        seed_offset = state["seed_offset"]
        if type(seed_offset) is not int or seed_offset < 0:
            raise ValueError("Agent OPD checkpoint seed_offset is invalid.")
        self._seed_offset = seed_offset
        self._last_eval_step = state["last_eval_step"]
        self.update_history = list(state["updates"])

    def _seed(self, row_count: int, attempts: int) -> int | None:
        """Reserve disjoint per-turn seed ranges for future collection calls."""

        if type(row_count) is not int or row_count < 1:
            raise ValueError("row_count must be a positive integer.")
        if type(attempts) is not int or attempts < 1:
            raise ValueError("attempts must be a positive integer.")
        seed = self.trainer.args.seed
        offset = self._seed_offset
        self._seed_offset += (
            row_count * attempts * self.workflow.runner.limits.max_turns
        )
        return None if seed is None else seed + offset

    @staticmethod
    def _accepted_episodes(batch: AgentRolloutBatch) -> list[EpisodeResult]:
        """Select episodes whose validated export entered the accepted batch."""

        accepted_ids = {item.logical_rollout_id for item in batch.exports}
        episodes = [
            episode
            for episode in batch.episodes
            if (episode.rollout_id or episode.episode_id) in accepted_ids
        ]
        if not episodes:
            raise AgentContractError(
                "Agent OPD collection produced no accepted episodes."
            )
        return episodes

    async def collect_and_update(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> AgentOPDUpdate:
        """Execute one complete collect, teacher-score, and update transaction."""

        if not rows:
            raise ValueError("Agent OPD update requires at least one task row.")
        snapshot = await create_sampling_snapshot(self.trainer, stop=self.config.stop)
        validate_training_snapshot(snapshot)
        self.last_snapshot = snapshot
        attempts = self.trainer.args.num_generations
        batch = await self.workflow.collect(
            rows,
            snapshot,
            n_samples=attempts,
            seed=self._seed(len(rows), attempts),
            strict=self.config.on_system_failure == "raise",
        )
        episodes = self._accepted_episodes(batch)
        accepted_tasks = len({episode.group_id for episode in episodes})
        rollouts = episode_turn_rollouts(
            episodes,
            behavior_policy_version=self.trainer.state.behavior_policy_version,
            teacher_group=self.teacher_group,
        )
        result = await self.trainer.step_rollouts_async(
            rollouts,
            prompt_count=accepted_tasks,
        )
        self.last_train_batch = batch
        record = {
            "step": result.step,
            "snapshot_id": snapshot.snapshot_id,
            "behavior_policy_version": result.behavior_policy_version,
            "requested_tasks": len(rows),
            "tasks": accepted_tasks,
            "episodes": len(episodes),
            "turns": len(rollouts),
            "action_tokens": sum(len(item.completion_tokens) for item in rollouts),
            "metrics": dict(result.metrics),
        }
        self.update_history.append(record)

        execution_metrics = batch.metrics(prefix="agent")
        execution_metrics.update(
            {
                "agent/opd_requested_tasks": float(len(rows)),
                "agent/opd_tasks": float(accepted_tasks),
                "agent/opd_episodes": float(len(episodes)),
                "agent/opd_turns": float(len(rollouts)),
                "agent/opd_action_tokens": float(record["action_tokens"]),
            }
        )
        await self.trainer.publish_metrics(execution_metrics)
        # If the optimizer step saved remote state, commit the agent-side seed
        # and history only after both records describe the same update boundary.
        self.trainer._checkpoint.publish(self.trainer)
        if self.trainer._sampler_checkpoint.due(self.trainer.state.global_step):
            await self.trainer.save_sampler_checkpoint_async()
        return AgentOPDUpdate(result, batch, rollouts)

    async def evaluate(
        self,
        dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Evaluate task behavior without teacher scoring or optimization."""

        trainer = self.trainer
        dataset = trainer.eval_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("Agent OPD evaluation requires an eval dataset.")
        attempts = (
            self.config.evaluation.num_generations or trainer.args.num_generations
        )
        if max(self.config.evaluation.pass_ks) > attempts:
            raise ValueError("Evaluation pass_k must not exceed num_generations.")
        snapshot = await create_sampling_snapshot(
            trainer,
            stop=self.config.stop,
            temperature=self.config.evaluation.temperature,
            top_p=self.config.evaluation.top_p,
        )
        batch_size = trainer.args.eval_batch_size or trainer.args.batch_size
        max_rows = (
            None
            if trainer.args.eval_max_batches is None
            else batch_size * trainer.args.eval_max_batches
        )
        rows = list(
            itertools.islice(iter(dataset), max_rows)
            if max_rows is not None
            else dataset
        )
        if not rows:
            raise ValueError("Agent OPD evaluation dataset is empty.")
        task_ids = set()
        for index, row in enumerate(rows):
            row = dict(row)
            row.setdefault("task_id", f"eval-row-{index}")
            task_id = str(row["task_id"])
            if task_id in task_ids:
                raise ValueError("Evaluation task_id values must be unique.")
            task_ids.add(task_id)
            rows[index] = row

        episodes = []
        with tqdm(
            total=len(rows),
            desc="Agent OPD eval",
            unit="task",
            leave=False,
        ) as progress:
            # A single queue spans the entire validation set. Completed tasks
            # immediately free a model slot instead of waiting for every task
            # in a fixed-size batch, which removes repeated straggler tails.
            async for batch in self.workflow.collect_iter(
                rows,
                snapshot,
                n_samples=attempts,
                seed=self.config.evaluation.seed,
                is_eval=True,
                strict=False,
                max_pending_groups=self.config.max_concurrency,
                log_episodes=False,
            ):
                episodes.extend(batch.episodes)
                if self.evaluation_logger is not None:
                    await self.evaluation_logger.write_batch(
                        batch.episodes,
                        step=trainer.state.global_step,
                    )
                progress.update(1)
        self.last_eval_batch = AgentRolloutBatch(episodes, [])
        metrics = self.last_eval_batch.metrics(prefix="eval/agent")
        metrics.update(
            evaluation_metrics(
                episodes,
                pass_ks=self.config.evaluation.pass_ks,
                prefix="eval/agent",
            )
        )
        await trainer.publish_metrics(metrics)
        self._last_eval_step = trainer.state.global_step
        return metrics

    async def train(self, *, stop_after_step: int | None = None):
        """Consume task rows once and return the trainer's final OPD state."""

        trainer = self.trainer
        if trainer.train_dataset is None:
            raise ValueError("Agent OPD training requires a train dataset.")
        if (
            trainer.eval_dataset is not None
            and iter(trainer.eval_dataset) is trainer.eval_dataset
        ):
            raise ValueError("Periodic Agent OPD evaluation requires replayable data.")

        limit, paused = training_limit(
            trainer.args,
            trainer.state.global_step,
            stop_after_step,
        )
        trainer._checkpoint.begin()
        initial_step = trainer.state.global_step
        try:
            if trainer._checkpoint.evaluation_due(trainer, self._last_eval_step):
                await self.evaluate()
            rows = trainer._checkpoint.rows(trainer)
            with tqdm(
                total=trainer.args.max_steps,
                initial=initial_step,
                desc="Agent OPD train",
                unit="step",
            ) as progress:
                while limit is None or trainer.state.global_step < limit:
                    batch_rows = list(itertools.islice(rows, trainer.args.batch_size))
                    if not batch_rows:
                        break
                    update = await self.collect_and_update(batch_rows)
                    progress.set_postfix(
                        {
                            label: f"{update.result.metrics[name]:.4f}"
                            for label, name in (
                                ("loss", "train/loss"),
                                ("reverse_kl", "distill/reverse_kl_sampled"),
                                ("turns", "data/completions"),
                            )
                            if name in update.result.metrics
                        },
                        refresh=False,
                    )
                    progress.update(1)
                    if (
                        trainer.eval_dataset is not None
                        and trainer.args.eval_steps > 0
                        and trainer.state.global_step % trainer.args.eval_steps == 0
                        and self._last_eval_step != trainer.state.global_step
                    ):
                        await self.evaluate()
                        trainer._checkpoint.publish(trainer)

            if trainer.state.global_step == initial_step and (
                trainer.args.max_steps is None or initial_step < trainer.args.max_steps
            ):
                raise ValueError("No trainable Agent OPD batch was collected.")
            if (
                not trainer.args.save_every
                or trainer.state.global_step % trainer.args.save_every
            ):
                await trainer.save_checkpoint_async()
                trainer._checkpoint.publish(trainer)
            if (
                trainer.eval_dataset is not None
                and not paused
                and trainer.args.eval_after_train
                and self._last_eval_step != trainer.state.global_step
            ):
                await self.evaluate()
            if not paused and trainer._sampler_checkpoint.due(
                trainer.state.global_step, final=True
            ):
                trainer._checkpoint.publish(trainer)
                await trainer.save_sampler_checkpoint_async()
            trainer._metrics_api.log_text(
                key="checkpoint/path",
                data=trainer.state.checkpoint_path,
                step=trainer.state.global_step,
            )
            # Final evaluation progress belongs to the same saved weights even
            # when sampler exports are disabled or already saved this step.
            trainer._checkpoint.publish(trainer)
            await trainer.close(flush=not paused)
        except BaseException as error:
            try:
                await trainer.close(state="crashed", error=str(error))
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(f"Reporter cleanup also failed: {cleanup_error}")
            try:
                await self.workflow.close()
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(f"Agent resource cleanup also failed: {cleanup_error}")
            raise
        else:
            await self.workflow.close()
        finally:
            trainer._checkpoint.end()
        return trainer.state


__all__ = [
    "AgentOPDTrainingLoop",
    "AgentOPDUpdate",
    "episode_turn_rollouts",
]
