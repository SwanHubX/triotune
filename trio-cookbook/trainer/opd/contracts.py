"""Data contracts exchanged by the OPD rollout, teacher, and target phases.

These objects keep algorithm state independent from PyTRIO response objects.
Each phase receives a small, validated value object, which makes it possible to
replace one phase in an experiment without changing ``OPDTrainer``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any


# ---------------------------------------------------------------------------
# Primitive validation shared by phase contracts
# ---------------------------------------------------------------------------


def validate_tokens(tokens: Sequence[int], *, source: str) -> list[int]:
    """Return a defensive copy of a non-empty token-ID sequence."""

    # ``bool`` is an ``Integral`` in Python, so reject it explicitly instead of
    # silently turning True/False into model token IDs 1/0.
    if (
        isinstance(tokens, (str, bytes))
        or not isinstance(tokens, Sequence)
        or not tokens
        or any(
            not isinstance(token, Integral)
            or isinstance(token, bool)
            or token < 0
            for token in tokens
        )
    ):
        raise ValueError(f"{source} must contain nonnegative integer token IDs.")
    return [int(token) for token in tokens]


def validate_logprobs(
    values: Sequence[float | None], *, source: str
) -> list[float]:
    """Return finite, nonpositive token log-probabilities."""

    # Log-probabilities cannot exceed zero. Rejecting NaN and infinities here
    # prevents invalid teacher or sampler output from reaching remote training.
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
        or any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) > 0
            for value in values
        )
    ):
        raise ValueError(f"{source} must contain finite, nonpositive logprobs.")
    return [float(value) for value in values]


# ---------------------------------------------------------------------------
# Contracts exchanged by rollout, teacher scoring, and target construction
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OPDRollout:
    """One student completion sampled by a named behavior-policy version."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    behavior_logprobs: list[float]
    prompt_index: int
    sample_index: int
    behavior_policy_version: int

    def __post_init__(self) -> None:
        # Copy caller-owned sequences at the phase boundary. Downstream phases
        # can then rely on stable token/logprob alignment for this rollout.
        self.prompt_tokens = validate_tokens(self.prompt_tokens, source="Prompt")
        self.completion_tokens = validate_tokens(
            self.completion_tokens, source="Student completion"
        )
        self.behavior_logprobs = validate_logprobs(
            self.behavior_logprobs, source="Student completion"
        )
        if len(self.completion_tokens) != len(self.behavior_logprobs):
            raise ValueError("Student completion tokens and logprobs must align.")
        # These identifiers restore deterministic output order and make policy
        # staleness visible without leaking SDK-specific request metadata.
        for name in ("prompt_index", "sample_index", "behavior_policy_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")


@dataclass(slots=True)
class OPDTeacherTarget:
    """Teacher token scores aligned to one existing student rollout."""

    rollout: OPDRollout
    teacher_logprobs: list[float]

    def __post_init__(self) -> None:
        # Teacher scores refer to the exact actions sampled by the student, so
        # one score is required for every completion token.
        self.teacher_logprobs = validate_logprobs(
            self.teacher_logprobs, source="Teacher completion"
        )
        if len(self.teacher_logprobs) != len(self.rollout.completion_tokens):
            raise ValueError("Teacher logprobs must align with completion tokens.")


@dataclass(slots=True)
class OPDTrainingSample:
    """Materialized PyTRIO datum plus statistics retained outside the SDK."""

    datum: Any
    rollout: OPDRollout
    teacher_logprobs: list[float]
    advantages: list[float]

    def __post_init__(self) -> None:
        # Keep algorithm statistics beside the materialized datum. PyTRIO owns
        # remote execution, while local evaluation still needs these raw sums.
        self.teacher_logprobs = validate_logprobs(
            self.teacher_logprobs, source="Teacher completion"
        )
        self.advantages = [float(value) for value in self.advantages]
        expected = len(self.rollout.completion_tokens)
        if len(self.teacher_logprobs) != expected or len(self.advantages) != expected:
            raise ValueError("OPD target fields must align with completion tokens.")
        if any(not math.isfinite(value) for value in self.advantages):
            raise ValueError("Distillation advantages must be finite.")

    @property
    def tokens(self) -> int:
        return len(self.rollout.completion_tokens)

    @property
    def student_sum(self) -> float:
        return sum(self.rollout.behavior_logprobs)

    @property
    def teacher_sum(self) -> float:
        return sum(self.teacher_logprobs)

    @property
    def advantage_sum(self) -> float:
        return sum(self.advantages)


# ---------------------------------------------------------------------------
# Trainer lifecycle contracts
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OPDState:
    """Mutable trainer lifecycle state retained across optimizer steps."""

    global_step: int = 0
    behavior_policy_version: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(slots=True)
class OPDResult:
    """Metrics and policy version produced by one completed OPD update."""

    step: int
    metrics: dict[str, float]
    num_datums: int
    behavior_policy_version: int
    checkpoint_path: str | None = None


__all__ = [
    "OPDResult",
    "OPDRollout",
    "OPDState",
    "OPDTeacherTarget",
    "OPDTrainingSample",
    "validate_logprobs",
    "validate_tokens",
]
