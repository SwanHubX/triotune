"""Fixed-teacher scoring phase for existing student completions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    """Score sampled tokens without sampling or changing teacher weights."""

    async def score(
        self,
        *,
        teacher_client: Any,
        rollout: OPDRollout,
        model_input_factory: Any = None,
    ) -> OPDTeacherTarget:
        # Score the exact student-sampled token sequence. The teacher never
        # generates replacement tokens and never receives an optimizer update.
        teacher_logprobs = await score_completion_async(
            teacher_client,
            rollout.prompt_tokens,
            rollout.completion_tokens,
            model_input_factory=model_input_factory,
            source="Teacher",
        )
        # Contract construction rechecks one-to-one token alignment before the
        # target can enter loss construction.
        return OPDTeacherTarget(rollout, teacher_logprobs)


__all__ = ["OPDTeacherPhase", "teacher_completion_logprobs"]
