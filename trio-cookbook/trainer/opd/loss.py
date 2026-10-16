"""Sampled reverse-KL signal and its remote policy-loss surrogate.

OPD does not transmit a teacher vocabulary distribution.  The student samples
an action under its behavior policy and the teacher scores that same action.
The detached token signal ``log p_teacher - log p_behavior`` is then optimized
through PyTRIO's importance-sampling policy objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ...api.loss_api import BuiltinPyTrioObjective, LossSpec, Objective
from .contracts import validate_logprobs


# ---------------------------------------------------------------------------
# Token-level sampled reverse-KL signal
# ---------------------------------------------------------------------------


def distillation_advantages(
    student: Sequence[float],
    teacher: Sequence[float],
    *,
    coef: float,
    discount: float,
) -> list[float]:
    """Compute detached negative sampled reverse-KL token advantages.

    With ``discount=0`` the signal at token ``t`` is
    ``coef * (log p_teacher(a_t) - log p_behavior(a_t))``.  A positive discount
    optionally propagates later distillation signal to earlier sampled tokens.
    No group centering is applied, so a single completion retains its signal.
    """

    student_values = validate_logprobs(student, source="Student completion")
    teacher_values = validate_logprobs(teacher, source="Teacher completion")
    if len(student_values) != len(teacher_values):
        raise ValueError("Student and teacher completion logprobs must align.")
    if (
        isinstance(coef, bool)
        or not isinstance(coef, (int, float))
        or not math.isfinite(float(coef))
        or coef <= 0
        or isinstance(discount, bool)
        or not isinstance(discount, (int, float))
        or not math.isfinite(float(discount))
        or not 0 <= discount <= 1
    ):
        raise ValueError("coef must be positive and discount must be in [0, 1].")

    # These values are computed before the remote forward/backward request and
    # are therefore detached score-function signals, not differentiable logits.
    advantages = [
        float(coef) * (teacher_logp - student_logp)
        for student_logp, teacher_logp in zip(
            student_values, teacher_values, strict=True
        )
    ]
    # Reverse accumulation implements G_t = signal_t + gamma * G_{t+1} without
    # allocating a second token buffer. A zero discount keeps each token local.
    future = 0.0
    for index in range(len(advantages) - 1, -1, -1):
        future = advantages[index] + float(discount) * future
        if not math.isfinite(future):
            raise ValueError("Distillation advantages overflowed.")
        advantages[index] = future
    return advantages


# ---------------------------------------------------------------------------
# Loss component bound into an execution plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SampledReverseKLLoss:
    """Configurable OPD signal paired with a PyTRIO policy objective.

    Replacing this object inside :class:`OPDExecutionPlan` is the intended
    research extension point for another token-credit rule or compatible
    policy surrogate.  The default remains un-clipped importance sampling.
    """

    coefficient: float = 1.0
    discount_factor: float = 0.0
    objective: Objective = field(
        # PyTRIO computes the current/behavior likelihood ratio remotely. The
        # local module supplies behavior logprobs and detached token advantages.
        default_factory=lambda: BuiltinPyTrioObjective(
            LossSpec("importance_sampling")
        )
    )

    def __post_init__(self) -> None:
        # Validate the scalars once when the execution plan is bound.  Use a
        # legal one-token distribution to share the exact checks with runtime.
        distillation_advantages(
            [-1.0],
            [-1.0],
            coef=self.coefficient,
            discount=self.discount_factor,
        )
        if not callable(getattr(self.objective, "forward_backward_async", None)):
            raise TypeError("objective must implement forward_backward_async().")

    def advantages(
        self, student: Sequence[float], teacher: Sequence[float]
    ) -> list[float]:
        """Build token advantages using this immutable loss configuration."""

        return distillation_advantages(
            student,
            teacher,
            coef=self.coefficient,
            discount=self.discount_factor,
        )


__all__ = ["SampledReverseKLLoss", "distillation_advantages"]
