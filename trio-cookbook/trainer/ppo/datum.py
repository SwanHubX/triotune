"""Causal alignment shared by scoring, native PPO and custom objectives."""

from collections.abc import Sequence

import numpy as np

from ...utils.datums import materialize_datum
from .contracts import PreparedBatch, Trajectory


def build_datum(
    trajectory: Trajectory,
    mask: Sequence[bool] | None = None,
    *,
    advantages: Sequence[float] | None = None,
    old_logprobs: Sequence[float] | None = None,
):
    t = trajectory
    mask = t.loss_mask if mask is None else mask
    n, prefix = len(t.completion_tokens), len(t.prompt_tokens) - 1
    if len(mask) != n:
        raise ValueError("Datum mask must align with completion tokens.")
    targets = [-100] * prefix + [
        token if active else -100
        for token, active in zip(t.completion_tokens, mask, strict=True)
    ]
    fields = {"target_tokens": np.asarray(targets, dtype=np.int32)}
    if (advantages is None) != (old_logprobs is None):
        raise ValueError("Native PPO needs both advantages and old logprobs.")
    if advantages is not None:
        if len(advantages) != n or len(old_logprobs) != n:
            raise ValueError("Native PPO inputs must align with completion tokens.")
        fields.update(
            advantages=np.asarray([0.0] * prefix + list(advantages), dtype=np.float32),
            logprobs=np.asarray([0.0] * prefix + list(old_logprobs), dtype=np.float32),
        )
    return materialize_datum(t.prompt_tokens + t.completion_tokens[:-1], fields)


def batch_datums(batch: PreparedBatch, *, native: bool = False):
    return [
        build_datum(
            t,
            mask,
            advantages=adv if native else None,
            old_logprobs=prox if native else None,
        )
        for t, mask, adv, prox in zip(
            batch.trajectories,
            batch.masks,
            batch.advantages,
            batch.proximal_logprobs,
            strict=True,
        )
    ]


def completion_logprobs(
    result, trajectories: Sequence[Trajectory]
) -> list[list[float]]:
    if len(result.loss_fn_outputs) != len(trajectories):
        raise ValueError("Forward output count must match trajectories.")
    rows = []
    for output, t in zip(result.loss_fn_outputs, trajectories, strict=True):
        raw = output["logprobs"]
        values = np.asarray(raw.tolist(), dtype=np.float64).reshape(-1)
        if (
            values.size != len(t.prompt_tokens) + len(t.completion_tokens) - 1
            or not np.isfinite(values).all()
        ):
            raise ValueError("Forward logprobs must be finite and causally aligned.")
        rows.append(values[len(t.prompt_tokens) - 1 :].tolist())
    return rows
