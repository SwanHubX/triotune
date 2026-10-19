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
        # Restore the unnormalized group weight in both policy terms:
        #   sum_j w_j log p_teacher_j - (sum_j w_j) log p_behavior.
        # This is the sampled reverse-KL signal for a weighted geometric
        # ensemble and preserves the user's intended group scale.
        weighted_student = [
            target.teacher_weight_sum * value
            for value in rollout.behavior_logprobs
        ]
        weighted_teacher = [
            target.teacher_weight_sum * value
            for value in target.teacher_logprobs
        ]
        advantages = self.loss.advantages(
            weighted_student,
            weighted_teacher,
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
            teacher_weight_sum=target.teacher_weight_sum,
        )


__all__ = ["OPDTargetBuilder"]
