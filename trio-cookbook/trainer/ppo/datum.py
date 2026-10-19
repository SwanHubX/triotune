"""Build the causal PyTRIO datums shared by every PPO update path.

PPO keeps trajectory contracts local while this module owns the final SDK
boundary.  It masks prompt and tool positions with ``-100``, aligns optional
native-PPO fields to the same next-token positions, and slices remote forward
outputs back to completion-only logprobs.  No reward or loss mathematics is
implemented here.
"""

from collections.abc import Sequence

import numpy as np

from ...utils.datums import materialize_datum
from .contracts import PreparedBatch, Trajectory

IGNORE_INDEX = -100


def build_datum(
    trajectory: Trajectory,
    mask: Sequence[bool] | None = None,
    *,
    advantages: Sequence[float] | None = None,
    old_logprobs: Sequence[float] | None = None,
):
    """Convert one trajectory into a causal, next-token-aligned PyTRIO Datum.

    ``prompt_tokens + completion_tokens[:-1]`` is the model input.  Its labels
    contain ignored prompt positions followed by completion targets, so the
    first completion token is predicted from the final prompt token.  Native
    PPO receives advantages and proximal logprobs in the same aligned shape.
    """

    t = trajectory
    mask = t.loss_mask if mask is None else mask
    n, prefix = len(t.completion_tokens), len(t.prompt_tokens) - 1
    if len(mask) != n or not all(isinstance(active, bool) for active in mask):
        raise ValueError("Datum mask must align with completion tokens.")
    targets = [IGNORE_INDEX] * prefix + [
        token if active else IGNORE_INDEX
        for token, active in zip(t.completion_tokens, mask, strict=True)
    ]
    fields = {"target_tokens": np.asarray(targets, dtype=np.int32)}
    if (advantages is None) != (old_logprobs is None):
        raise ValueError("Native PPO needs both advantages and old logprobs.")
    if advantages is not None:
        if len(advantages) != n or len(old_logprobs) != n:
            raise ValueError("Native PPO inputs must align with completion tokens.")
        if any(
            not np.isfinite(value)
            for value in (*advantages, *old_logprobs)
        ):
            raise ValueError("Native PPO fields must contain finite values.")
        fields.update(
            advantages=np.asarray([0.0] * prefix + list(advantages), dtype=np.float32),
            logprobs=np.asarray([0.0] * prefix + list(old_logprobs), dtype=np.float32),
        )
    return materialize_datum(t.prompt_tokens + t.completion_tokens[:-1], fields)


def batch_datums(batch: PreparedBatch, *, native: bool = False):
    """Materialize every prepared trajectory for one PyTRIO request."""

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
    """Extract finite completion logprobs from a causal PyTRIO forward result."""

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
