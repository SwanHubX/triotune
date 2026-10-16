"""OPD-specific causal alignment and PyTRIO Datum construction."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from ...api.loss_api import PolicyLossInput
from .contracts import validate_logprobs, validate_tokens


# ---------------------------------------------------------------------------
# Next-token alignment
# ---------------------------------------------------------------------------


def build_opd_datum(
    prompt: Sequence[int],
    completion: Sequence[int],
    student: Sequence[float],
    advantages: Sequence[float],
    *,
    model_input_factory: Callable[[Sequence[int]], Any] | None = None,
    datum_factory: Callable[..., Any] | None = None,
) -> Any:
    """Mask prompt positions and align one sampled completion for policy loss.

    The returned datum is already next-token shifted and therefore must be used
    with ``auto_shift=False``.  Prompt targets use ``-100`` so they contribute
    neither gradients nor active-token normalization.
    """

    prompt_tokens = validate_tokens(prompt, source="Prompt")
    completion_tokens = validate_tokens(completion, source="Completion")
    student_logprobs = validate_logprobs(student, source="Student completion")
    advantage_values = [float(value) for value in advantages]
    # Completion tokens, behavior logprobs, and detached advantages describe
    # the same sampled actions and must therefore have identical lengths.
    if (
        len(completion_tokens) != len(student_logprobs)
        or len(completion_tokens) != len(advantage_values)
    ):
        raise ValueError("Completion tokens, logprobs and advantages must align.")
    if any(not math.isfinite(value) for value in advantage_values):
        raise ValueError("Distillation advantages must be finite.")

    # For prompt=[p0, p1] and completion=[a0, a1], the causal model consumes
    # [p0, p1, a0] and predicts [-100, a0, a1]. The ignored prompt target keeps
    # prompt length from diluting completion-only loss normalization.
    padding = len(prompt_tokens) - 1
    return PolicyLossInput(
        model_tokens=prompt_tokens + completion_tokens[:-1],
        target_tokens=[-100] * padding + completion_tokens,
        # Prompt positions are masked by target_tokens. Their neutral auxiliary
        # values merely preserve the one-to-one wire-field alignment.
        behavior_logprobs=[0.0] * padding + student_logprobs,
        advantages=[0.0] * padding + advantage_values,
    ).to_datum(
        model_input_factory=model_input_factory,
        datum_factory=datum_factory,
    )


__all__ = ["build_opd_datum"]
