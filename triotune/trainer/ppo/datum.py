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
from .trajectory import PreparedBatch, Trajectory

IGNORE_INDEX = -100


class TrainingDatums(list):
    """SDK rows with an ordered, local-only provenance sidecar.

    The list itself is the exact payload accepted by PyTRIO. ``provenance[i]``
    describes row i without injecting unsupported fields into SDK loss inputs.
    The actor retains this sidecar for request/response auditing; serializing
    the list for a remote call does not require server-side metadata support.
    """

    def __init__(self, datums, trajectories):
        super().__init__(datums)
        self.provenance = tuple(t.provenance for t in trajectories)
        self.prefix_groups = shared_prefix_groups(trajectories)
        if len(self) != len(self.provenance):
            raise ValueError("Datum provenance must identify every training row.")


def shared_prefix_groups(trajectories: Sequence[Trajectory]):
    """Describe maximal shared model-input prefixes without altering SDK rows.

    PyTRIO currently accepts ordinary Datum lists and exposes no portable tree
    attention flag. This local sidecar records maximal trie nodes shared by two
    or more physical rows, allowing compatible clients or diagnostics to reuse
    prefix work while the standard client safely ignores it. Token IDs and row
    order remain unchanged, so fallback execution has identical PPO semantics.
    """
    inputs = [
        tuple(trajectory.prompt_tokens + trajectory.completion_tokens[:-1])
        for trajectory in trajectories
    ]
    groups = []
    pending = [(list(range(len(trajectories))), 0)] if trajectories else []
    while pending:
        indices, offset = pending.pop()
        buckets = {}
        for index in indices:
            if offset < len(inputs[index]):
                buckets.setdefault(inputs[index][offset], []).append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            children = {}
            for index in members:
                if offset + 1 < len(inputs[index]):
                    children.setdefault(inputs[index][offset + 1], []).append(index)
            shared_children = [child for child in children.values() if len(child) > 1]
            ended = len(members) - sum(len(child) for child in children.values())
            if len(children) != 1 or ended:
                groups.append(
                    {
                        "prefix_length": offset + 1,
                        "row_indices": tuple(members),
                    }
                )
            pending.extend((child, offset + 1) for child in shared_children)
    return tuple(groups)


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
        if any(not np.isfinite(value) for value in (*advantages, *old_logprobs)):
            raise ValueError("Native PPO fields must contain finite values.")
        fields.update(
            advantages=np.asarray([0.0] * prefix + list(advantages), dtype=np.float32),
            logprobs=np.asarray([0.0] * prefix + list(old_logprobs), dtype=np.float32),
        )
    return materialize_datum(t.prompt_tokens + t.completion_tokens[:-1], fields)


def batch_datums(batch: PreparedBatch, *, native: bool = False):
    """Materialize every prepared trajectory for one PyTRIO request."""

    return TrainingDatums(
        [
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
        ],
        batch.trajectories,
    )


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
