"""Token GAE skips masked prompt/tool positions and bootstraps truncations."""

import math
from collections.abc import Sequence


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    truncated: bool = False,
) -> tuple[list[float], list[float]]:
    if len(rewards) != len(mask) or len(values) != len(mask) + 1:
        raise ValueError("GAE needs T rewards/mask and T+1 values including bootstrap.")
    if any(not math.isfinite(x) for x in (*rewards, *values, discount, gae_lambda)):
        raise ValueError("GAE inputs must be finite.")
    if not 0 <= discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE discount and lambda must be in [0, 1].")
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_value, running = values[-1] if truncated else 0.0, 0.0
    for t in reversed(range(len(rewards))):
        if mask[t]:
            delta = rewards[t] + discount * next_value - values[t]
            running = delta + discount * gae_lambda * running
            advantages[t], returns[t] = running, running + values[t]
            next_value = values[t]
    return advantages, returns
