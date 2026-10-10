"""Local OPD state, step results, and scored completion records."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OPDState:
    global_step: int = 0
    behavior_policy_version: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(slots=True)
class OPDResult:
    step: int
    metrics: dict[str, float]
    num_datums: int
    behavior_policy_version: int
    checkpoint_path: str | None = None


@dataclass(slots=True)
class _ScoredCompletion:
    datum: Any
    tokens: int
    student_sum: float
    teacher_sum: float
    advantage_sum: float
