"""Routed fixed-teacher scoring for existing student completions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ...utils.scoring import completion_logprobs, score_completion_async
from .contracts import OPDRollout, OPDTeacherTarget

# ---------------------------------------------------------------------------
# Full-sequence response alignment
# ---------------------------------------------------------------------------


def teacher_completion_logprobs(
    values: Sequence[float | None],
    prompt_length: int,
    completion_length: int,
) -> list[float]:
    """Extract the teacher-scored completion suffix from a full sequence."""

    # The generic utility validates full prompt+completion length and discards
    # prompt positions, which are conditioning context rather than OPD targets.
    return completion_logprobs(
        values,
        prompt_length=prompt_length,
        completion_length=completion_length,
        source="Teacher",
    )


# ---------------------------------------------------------------------------
# Fixed-teacher scoring phase
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OPDTeacherPhase:
    """Score sampled tokens and aggregate a routed geometric teacher ensemble."""

    async def score(
        self,
        *,
        teacher_clients: Mapping[str, Any],
        teacher_weights: Mapping[str, float],
        rollout: OPDRollout,
        model_input_factory: Any = None,
    ) -> OPDTeacherTarget:
        if not isinstance(teacher_clients, Mapping) or not teacher_clients:
            raise ValueError("teacher_clients must contain at least one teacher.")
        if not isinstance(teacher_weights, Mapping) or not teacher_weights:
            raise ValueError("teacher_weights must contain at least one teacher.")

        weighted_logprobs = [0.0] * len(rollout.completion_tokens)
        weight_sum = 0.0
        for teacher_id, raw_weight in teacher_weights.items():
            if teacher_id not in teacher_clients:
                raise ValueError(f"Unknown teacher ID {teacher_id!r}.")
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, Real)
                or not math.isfinite(float(raw_weight))
                or raw_weight <= 0
            ):
                raise ValueError("Teacher weights must be finite and positive.")
            weight = float(raw_weight)

            # Score the exact student-sampled token sequence. Teachers never
            # generate replacement tokens and never receive optimizer updates.
            # Calls are intentionally sequential inside one scoring slot, so
            # max_concurrent_scoring remains an upper bound on remote requests.
            logprobs = await score_completion_async(
                teacher_clients[teacher_id],
                rollout.prompt_tokens,
                rollout.completion_tokens,
                model_input_factory=model_input_factory,
                source=f"Teacher {teacher_id!r}",
            )
            for index, logprob in enumerate(logprobs):
                weighted_logprobs[index] += weight * logprob
            weight_sum += weight

        # The normalized log score represents the weighted geometric ensemble
        # for metrics. OPDTargetBuilder restores the raw weight scale in the
        # reverse-KL signal, so user weights are not silently normalized away.
        ensemble_logprobs = [value / weight_sum for value in weighted_logprobs]
        return OPDTeacherTarget(
            rollout,
            ensemble_logprobs,
            teacher_weight_sum=weight_sum,
        )


__all__ = ["OPDTeacherPhase", "teacher_completion_logprobs"]
