"""Data contracts exchanged by the OPD rollout, teacher, and target phases.

These objects keep algorithm state independent from PyTRIO response objects.
Each phase receives a small, validated value object, which makes it possible to
replace one phase in an experiment without changing ``OPDTrainer``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class OPDTeacherSpec:
    """One fixed teacher model and its optional LoRA checkpoint."""

    teacher_id: str
    model: str
    model_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("teacher_id", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if self.model_path is not None:
            if not isinstance(self.model_path, str) or not self.model_path.strip():
                raise ValueError("model_path must be a non-empty checkpoint URI.")
            object.__setattr__(self, "model_path", self.model_path.strip())


@dataclass(frozen=True, slots=True)
class OPDTeacherGroup:
    """A named route containing positive, unnormalized teacher weights."""

    name: str
    teachers: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Teacher group name must be a non-empty string.")
        if not isinstance(self.teachers, Mapping) or not self.teachers:
            raise ValueError("Teacher group must contain at least one teacher.")
        weights: dict[str, float] = {}
        for teacher_id, weight in self.teachers.items():
            if not isinstance(teacher_id, str) or not teacher_id.strip():
                raise ValueError("Teacher IDs must be non-empty strings.")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, Real)
                or not math.isfinite(float(weight))
                or weight <= 0
            ):
                raise ValueError("Teacher weights must be finite and positive.")
            weights[teacher_id.strip()] = float(weight)
        object.__setattr__(self, "name", self.name.strip())
        # Keep a defensive copy; OPDTrainer copies it again when binding the
        # execution topology, so later caller mutations cannot affect a run.
        object.__setattr__(self, "teachers", weights)


@dataclass(slots=True)
class OPDPrompt:
    """A rendered prompt plus framework-owned teacher-group routing metadata."""

    text: str
    tokens: list[int]
    teacher_group: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Prompt text must be a string.")
        self.tokens = validate_tokens(self.tokens, source="Prompt")
        if not isinstance(self.teacher_group, str) or not self.teacher_group.strip():
            raise ValueError("teacher_group must be a non-empty string.")
        self.teacher_group = self.teacher_group.strip()


@dataclass(slots=True)
class OPDRollout:
    """One student completion sampled by a named behavior-policy version."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    behavior_logprobs: list[float]
    prompt_index: int
    sample_index: int
    behavior_policy_version: int
    teacher_group: str = "default"

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
        if not isinstance(self.teacher_group, str) or not self.teacher_group.strip():
            raise ValueError("teacher_group must be a non-empty string.")
        self.teacher_group = self.teacher_group.strip()


@dataclass(slots=True)
class OPDTeacherTarget:
    """Normalized ensemble scores aligned to one existing student rollout.

    ``teacher_logprobs`` is the weighted mean logprob used for interpretable
    metrics. ``teacher_weight_sum`` retains the original group scale used by
    the reverse-KL target builder.
    """

    rollout: OPDRollout
    teacher_logprobs: list[float]
    teacher_weight_sum: float = 1.0

    def __post_init__(self) -> None:
        # Teacher scores refer to the exact actions sampled by the student, so
        # one score is required for every completion token.
        self.teacher_logprobs = validate_logprobs(
            self.teacher_logprobs, source="Teacher completion"
        )
        if len(self.teacher_logprobs) != len(self.rollout.completion_tokens):
            raise ValueError("Teacher logprobs must align with completion tokens.")
        if (
            isinstance(self.teacher_weight_sum, bool)
            or not isinstance(self.teacher_weight_sum, Real)
            or not math.isfinite(float(self.teacher_weight_sum))
            or self.teacher_weight_sum <= 0
        ):
            raise ValueError("teacher_weight_sum must be finite and positive.")
        self.teacher_weight_sum = float(self.teacher_weight_sum)


@dataclass(slots=True)
class OPDTrainingSample:
    """Materialized datum plus normalized ensemble statistics and weight scale."""

    datum: Any
    rollout: OPDRollout
    teacher_logprobs: list[float]
    advantages: list[float]
    teacher_weight_sum: float = 1.0

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
        if (
            isinstance(self.teacher_weight_sum, bool)
            or not isinstance(self.teacher_weight_sum, Real)
            or not math.isfinite(float(self.teacher_weight_sum))
            or self.teacher_weight_sum <= 0
        ):
            raise ValueError("teacher_weight_sum must be finite and positive.")
        self.teacher_weight_sum = float(self.teacher_weight_sum)

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
    "OPDPrompt",
    "OPDResult",
    "OPDRollout",
    "OPDState",
    "OPDTeacherGroup",
    "OPDTeacherSpec",
    "OPDTeacherTarget",
    "OPDTrainingSample",
    "validate_logprobs",
    "validate_tokens",
]
