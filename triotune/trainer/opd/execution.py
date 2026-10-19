"""Composable execution plan for one OPD rollout-and-target transaction."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ...api.loss_api import Objective
from .contracts import OPDPrompt, OPDRollout, OPDTeacherTarget, OPDTrainingSample
from .loss import SampledReverseKLLoss
from .metrics import OPDMetricReducer
from .rollout import OPDRolloutPhase
from .targets import OPDTargetBuilder
from .teacher_phase import OPDTeacherPhase

# ---------------------------------------------------------------------------
# Replaceable phase protocols
# ---------------------------------------------------------------------------


class RolloutPhase(Protocol):
    """Extension contract for student rollout generation."""

    async def sample_prompt(
        self,
        *,
        sampler: Any,
        prompt: OPDPrompt,
        prompt_index: int,
        behavior_policy_version: int,
        config: Any,
        seed_offset: int,
        sampling_params_factory: Any = None,
        model_input_factory: Any = None,
    ) -> list[OPDRollout]: ...


class TeacherPhase(Protocol):
    """Extension contract for scoring an already sampled completion."""

    async def score(
        self,
        *,
        teacher_clients: Mapping[str, Any],
        teacher_weights: Mapping[str, float],
        rollout: OPDRollout,
        model_input_factory: Any = None,
    ) -> OPDTeacherTarget: ...


class TargetBuilder(Protocol):
    """Extension contract for teacher-target aggregation and datum creation."""

    def build(self, target: OPDTeacherTarget) -> OPDTrainingSample: ...


# ---------------------------------------------------------------------------
# Immutable phase composition and batch transaction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OPDExecutionPlan:
    """Bind OPD phases and loss once before the first remote operation.

    The default plan is pure on-policy distillation:

    ``student rollout -> routed teacher scoring -> reverse-KL target -> IS loss``.

    It intentionally has no reward, critic, GAE, reference-policy, or mixed-RL
    branch.  Researchers can replace a phase or objective by constructing a new
    plan, while ``OPDTrainer`` keeps dataset and optimizer lifecycle unchanged.
    """

    rollout_phase: RolloutPhase
    teacher_phase: TeacherPhase
    target_builder: TargetBuilder
    objective: Objective
    metrics: OPDMetricReducer

    def __post_init__(self) -> None:
        # Protocol annotations aid static typing, while this small runtime check
        # gives configuration-driven experiments an early, actionable failure.
        required = (
            (self.rollout_phase, "sample_prompt"),
            (self.teacher_phase, "score"),
            (self.target_builder, "build"),
            (self.objective, "forward_backward_async"),
            (self.metrics, "training"),
            (self.metrics, "evaluation"),
            (self.metrics, "evaluation_totals"),
        )
        for component, method in required:
            if not callable(getattr(component, method, None)):
                raise TypeError(f"OPD component must implement {method}().")

    @classmethod
    def from_config(cls, config: Any) -> "OPDExecutionPlan":
        """Create the default immutable plan from validated OPD config."""

        # Capture objective-defining values now. Later config mutation may tune
        # scheduling, but it cannot silently change the bound distillation loss.
        loss = SampledReverseKLLoss(
            coefficient=config.kl_coef,
            discount_factor=config.discount_factor,
            importance_ratio_cap=config.importance_ratio_cap,
        )
        return cls(
            rollout_phase=OPDRolloutPhase(),
            teacher_phase=OPDTeacherPhase(),
            target_builder=OPDTargetBuilder(loss),
            objective=loss.objective,
            metrics=OPDMetricReducer(),
        )

    async def materialize(
        self,
        prompts: Sequence[OPDPrompt],
        *,
        sampler: Any,
        teacher_clients: Mapping[str, Any],
        teacher_groups: Mapping[str, Mapping[str, float]],
        config: Any,
        behavior_policy_version: int,
        seed_offset: int,
        sampling_params_factory: Any = None,
        model_input_factory: Any = None,
    ) -> list[OPDTrainingSample]:
        """Run bounded rollout/scoring and return samples in stable input order.

        Student and teacher calls may overlap across prompts, matching the
        current PyTRIO implementation.  The phase interfaces do not assume that
        overlap, so a future separated inference/training scheduler can invoke
        the same phases serially without changing loss or target code.
        """

        if not prompts:
            raise ValueError("OPD execution requires at least one prompt.")
        # Sampling and scoring have independent service limits. Separate
        # semaphores allow one prompt's teacher scoring to overlap another
        # prompt's rollout without exceeding either remote concurrency budget.
        sampling_slots = asyncio.Semaphore(config.max_concurrent_sampling)
        scoring_slots = asyncio.Semaphore(config.max_concurrent_scoring)
        pending = iter(enumerate(prompts))
        results: list[list[OPDTrainingSample] | None] = [None] * len(prompts)

        async def process_prompt(
            prompt_index: int, prompt: OPDPrompt
        ) -> list[OPDTrainingSample]:
            async with sampling_slots:
                rollouts = await self.rollout_phase.sample_prompt(
                    sampler=sampler,
                    prompt=prompt,
                    prompt_index=prompt_index,
                    behavior_policy_version=behavior_policy_version,
                    config=config,
                    seed_offset=seed_offset,
                    sampling_params_factory=sampling_params_factory,
                    model_input_factory=model_input_factory,
                )

            async def score(rollout: OPDRollout) -> OPDTrainingSample:
                # Only the remote teacher call occupies a scoring slot. Target
                # construction is local and should not reduce scoring capacity.
                teacher_weights = teacher_groups.get(rollout.teacher_group)
                if teacher_weights is None:
                    raise ValueError(
                        f"Unknown teacher group {rollout.teacher_group!r}."
                    )
                async with scoring_slots:
                    target = await self.teacher_phase.score(
                        teacher_clients=teacher_clients,
                        teacher_weights=teacher_weights,
                        rollout=rollout,
                        model_input_factory=model_input_factory,
                    )
                return self.target_builder.build(target)

            # A TaskGroup drains or cancels sibling teacher requests if any one
            # completion fails, so no partial prompt group reaches the update.
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(score(rollout)) for rollout in rollouts]
            return [task.result() for task in tasks]

        async def worker() -> None:
            # ``next`` contains no await, so cooperative workers cannot consume
            # the same prompt.  Each result is written to its original index.
            for prompt_index, prompt in pending:
                results[prompt_index] = await process_prompt(prompt_index, prompt)

        workers = min(
            len(prompts),
            config.max_concurrent_sampling + config.max_concurrent_scoring,
        )
        # Bound local task creation as well as remote operations. This matters
        # when a caller provides a batch much larger than both concurrency caps.
        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
        if any(result is None for result in results):
            raise RuntimeError("OPD execution did not finish every prompt.")
        # Flatten only after every worker finishes; indexing above preserves
        # prompt order even when remote requests complete out of order.
        return [
            sample
            for prompt_samples in results
            if prompt_samples is not None
            for sample in prompt_samples
        ]


__all__ = [
    "OPDExecutionPlan",
    "RolloutPhase",
    "TargetBuilder",
    "TeacherPhase",
]
