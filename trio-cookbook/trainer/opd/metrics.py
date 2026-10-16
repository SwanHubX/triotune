"""Algorithm-specific OPD metric reduction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import OPDTrainingSample


# ---------------------------------------------------------------------------
# Remote backend metric normalization
# ---------------------------------------------------------------------------


def remote_metrics(*responses: Any) -> dict[str, float]:
    """Normalize finite backend loss and gradient metrics."""

    metrics: dict[str, float] = {}
    # Different PyTRIO versions may attach metrics to forward and optimizer
    # results under different names. Preserve all finite numeric values first.
    for response in responses:
        raw = getattr(response, "metrics", None)
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                metrics[str(key)] = float(value)
    # Publish stable trainer-level aliases without discarding backend-specific
    # keys that may still be useful when comparing experiments.
    for key in ("loss_mean", "loss", "loss:mean", "scalar/loss_mean"):
        if key in metrics:
            metrics["train/loss"] = metrics[key]
            break
    for key in ("grad_norm", "gradient_norm"):
        if key in metrics:
            metrics["optim/grad_norm"] = metrics[key]
            break
    return metrics


# ---------------------------------------------------------------------------
# Streaming token-weighted sufficient statistics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OPDMetricTotals:
    """Streaming sufficient statistics for token-weighted OPD metrics."""

    prompt_count: int = 0
    completion_count: int = 0
    token_count: int = 0
    student_logprob_sum: float = 0.0
    teacher_logprob_sum: float = 0.0
    advantage_sum: float = 0.0

    def add(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prompt_count: int,
    ) -> None:
        """Accumulate one materialized batch without retaining its datums."""

        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count < 0
        ):
            raise ValueError("prompt_count must be a nonnegative integer.")
        # Store sums rather than per-sample values so long evaluations do not
        # retain PyTRIO datums or grow memory with dataset size.
        self.prompt_count += prompt_count
        self.completion_count += len(samples)
        self.token_count += sum(sample.tokens for sample in samples)
        self.student_logprob_sum += sum(sample.student_sum for sample in samples)
        self.teacher_logprob_sum += sum(sample.teacher_sum for sample in samples)
        self.advantage_sum += sum(sample.advantage_sum for sample in samples)

    def distillation(self, *, prefix: str) -> dict[str, float]:
        """Return token means under the requested metric namespace."""

        if self.token_count <= 0:
            raise ValueError("OPD metrics require at least one completion token.")
        # All signal metrics use completion tokens as the denominator. Prompt
        # length therefore cannot change the reported distillation scale.
        return {
            f"{prefix}/reverse_kl_sampled": (
                self.student_logprob_sum - self.teacher_logprob_sum
            )
            / self.token_count,
            f"{prefix}/student_logprob": self.student_logprob_sum
            / self.token_count,
            f"{prefix}/teacher_logprob": self.teacher_logprob_sum
            / self.token_count,
            f"{prefix}/advantage_mean": self.advantage_sum / self.token_count,
        }


# ---------------------------------------------------------------------------
# Training and evaluation metric views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OPDMetricReducer:
    """Reduce token-weighted OPD samples for training or evaluation."""

    def distillation(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prefix: str,
    ) -> dict[str, float]:
        # Reuse the streaming representation even for one training batch so the
        # batch and full-evaluation paths cannot drift mathematically.
        totals = OPDMetricTotals()
        totals.add(samples, prompt_count=0)
        return totals.distillation(prefix=prefix)

    def training(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prompt_count: int,
        behavior_policy_version: int,
    ) -> dict[str, float]:
        totals = OPDMetricTotals()
        totals.add(samples, prompt_count=prompt_count)
        metrics = totals.distillation(prefix="distill")
        metrics.update(
            {
                "data/prompts": float(totals.prompt_count),
                "data/completions": float(totals.completion_count),
                "data/completion_tokens": float(totals.token_count),
                "policy/behavior_version": float(behavior_policy_version),
            }
        )
        return metrics

    def evaluation(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prompt_count: int,
    ) -> dict[str, float]:
        totals = OPDMetricTotals()
        totals.add(samples, prompt_count=prompt_count)
        return self.evaluation_totals(totals)

    def evaluation_totals(self, totals: OPDMetricTotals) -> dict[str, float]:
        """Reduce streaming evaluation totals without retaining all samples."""

        metrics = totals.distillation(prefix="eval")
        metrics.update(
            {
                "eval/prompts": float(totals.prompt_count),
                "eval/completions": float(totals.completion_count),
                "eval/tokens": float(totals.token_count),
            }
        )
        return metrics


__all__ = ["OPDMetricReducer", "OPDMetricTotals", "remote_metrics"]
