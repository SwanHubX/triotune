"""OPD's sampled reverse-KL signal and next-token alignment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from ...utils.datums import materialize_datum


@dataclass(slots=True)
class _ScoredCompletion:
    """One student completion aligned with its teacher distillation signal."""

    datum: Any
    tokens: int
    student_sum: float
    teacher_sum: float
    advantage_sum: float


def validate_tokens(tokens: Sequence[int], *, source: str) -> list[int]:
    if not tokens or any(not isinstance(token, Integral) or isinstance(token, bool)
                         or token < 0 for token in tokens):
        raise ValueError(f"{source} must contain nonnegative integer token IDs.")
    return [int(token) for token in tokens]


def validate_logprobs(values: Sequence[float | None], *, source: str) -> list[float]:
    if not values or any(not isinstance(value, Real) or isinstance(value, bool)
                         or not math.isfinite(value) or value > 0 for value in values):
        raise ValueError(f"{source} must contain finite, nonpositive logprobs.")
    return [float(value) for value in values]


def teacher_completion_logprobs(
    values: Sequence[float | None], prompt_length: int, completion_length: int,
) -> list[float]:
    """PyTRIO scores token i at position i, including a possible initial None."""
    if len(values) != prompt_length + completion_length:
        raise ValueError("Teacher logprobs must align with the full prompt plus completion.")
    return validate_logprobs(values[prompt_length:], source="Teacher completion")


def distillation_advantages(
    student: Sequence[float], teacher: Sequence[float], *, coef: float, discount: float,
) -> list[float]:
    """Negative sampled reverse KL, optionally with discounted future credit.

    No group centering: even one sample must retain its distillation signal.
    Advantages are constants for the remote importance-sampling surrogate.
    """
    student = validate_logprobs(student, source="Student completion")
    teacher = validate_logprobs(teacher, source="Teacher completion")
    if len(student) != len(teacher):
        raise ValueError("Student and teacher completion logprobs must align.")
    if (
        not math.isfinite(coef)
        or coef <= 0
        or not math.isfinite(discount)
        or not 0 <= discount <= 1
    ):
        raise ValueError("coef must be positive and discount must be in [0, 1].")
    advantages = [coef * (q - p) for p, q in zip(student, teacher, strict=True)]
    future = 0.0
    for index in range(len(advantages) - 1, -1, -1):
        future = advantages[index] + discount * future
        if not math.isfinite(future):
            raise ValueError("Distillation advantages overflowed.")
        advantages[index] = future
    return advantages


def build_opd_datum(
    prompt: Sequence[int], completion: Sequence[int], student: Sequence[float],
    advantages: Sequence[float],
) -> Any:
    """Mask prompt positions and shift once; use with auto_shift=False."""
    prompt = validate_tokens(prompt, source="Prompt")
    completion = validate_tokens(completion, source="Completion")
    student = validate_logprobs(student, source="Student completion")
    if len(completion) != len(student) or len(completion) != len(advantages):
        raise ValueError("Completion tokens, logprobs and advantages must align.")
    if any(not math.isfinite(value) for value in advantages):
        raise ValueError("Distillation advantages must be finite.")
    padding = len(prompt) - 1
    return materialize_datum(
        prompt + completion[:-1],
        {
            # -100 excludes prompt tokens from PyTRIO's active-token loss
            # denominator as well as their gradient; zero advantages alone
            # would still dilute the completion signal for long prompts.
            "target_tokens": [-100] * padding + completion,
            "logprobs": [0.0] * padding + student,
            "advantages": [0.0] * padding + list(advantages),
        },
    )
