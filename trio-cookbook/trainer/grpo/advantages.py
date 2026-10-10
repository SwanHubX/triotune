"""Group-relative reward and advantage computation."""

from __future__ import annotations

import math
from collections.abc import Iterable


def group_advantages(
    rewards: Iterable[float],
    *,
    scale: str = "none",
    epsilon: float = 1e-4,
) -> tuple[list[float], dict[str, float]]:
    """Compute GRPO's per-group centered advantages.

    The subtraction is always group-local.  TRL may gather examples across
    distributed processes before reshaping; a remote PyTRIO client does not
    know those local boundaries, so this function requires one complete group.
    """

    values = [float(value) for value in rewards]
    if len(values) < 2:
        raise ValueError("A GRPO group must contain at least two rewards.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Rewards must be finite before computing advantages.")
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    variance = sum(value * value for value in centered) / len(centered)
    std = math.sqrt(variance)
    if scale == "none":
        denominator = 1.0
    elif scale == "group":
        denominator = std + epsilon
    else:
        raise ValueError("scale must be 'none' or 'group' for a single prompt group.")
    advantages = [value / denominator for value in centered]
    return advantages, {
        "reward/mean": mean,
        "reward/std": std,
        "advantage/mean": sum(advantages) / len(advantages),
        "advantage/std": math.sqrt(sum(value * value for value in advantages) / len(advantages)),
        "advantage/degenerate": float(std == 0.0),
    }
