"""Convert teacher scores into immutable OPD training targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import OPDTeacherTarget, OPDTrainingSample
from .datum import build_opd_datum
from .loss import SampledReverseKLLoss


# ---------------------------------------------------------------------------
# Teacher target to remote training datum
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OPDTargetBuilder:
    """Apply one token-credit rule and materialize its aligned policy datum."""

    loss: SampledReverseKLLoss
    model_input_factory: Any = None
    datum_factory: Any = None

    def build(self, target: OPDTeacherTarget) -> OPDTrainingSample:
        rollout = target.rollout
        # Convert aligned policy scores into detached token credit before the
        # model-facing datum is created.
        advantages = self.loss.advantages(
            rollout.behavior_logprobs,
            target.teacher_logprobs,
        )
        # Datum construction owns causal shifting and prompt masking; this phase
        # only connects the algorithm signal to that representation.
        datum = build_opd_datum(
            rollout.prompt_tokens,
            rollout.completion_tokens,
            rollout.behavior_logprobs,
            advantages,
            model_input_factory=self.model_input_factory,
            datum_factory=self.datum_factory,
        )
        # Preserve the raw scores next to the remote datum for metrics and
        # evaluation without depending on SDK-internal tensor containers.
        return OPDTrainingSample(
            datum=datum,
            rollout=rollout,
            teacher_logprobs=target.teacher_logprobs,
            advantages=advantages,
        )


__all__ = ["OPDTargetBuilder"]
