"""Generalized advantage estimation over token or generated-turn timesteps.

The functions in this module are deliberately independent of PyTRIO.  They
transform validated rollout metadata into token-aligned advantages and returns;
the actor later sends those frozen targets through PyTRIO's native or custom
loss boundary.
"""

import math
from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _validate_gae_inputs(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    discount: float,
    gae_lambda: float,
) -> None:
    """Validate the completion-aligned arrays shared by both recurrences."""

    if len(rewards) != len(mask) or len(values) != len(mask) + 1:
        raise ValueError("GAE needs T rewards/mask and T+1 values including bootstrap.")
    if not all(isinstance(active, bool) for active in mask):
        raise ValueError("GAE mask must contain booleans.")
    if any(not math.isfinite(x) for x in (*rewards, *values, discount, gae_lambda)):
        raise ValueError("GAE inputs must be finite.")
    if not 0 <= discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE discount and lambda must be in [0, 1].")


def _validate_turn_ids(
    turn_ids: Sequence[int] | None,
    mask: Sequence[bool],
) -> list[int]:
    """Return validated turn IDs for active completion tokens.

    Active IDs may contain gaps, but they must be temporally nondecreasing.
    Inactive prompt/tool positions use ``-1`` and do not consume a GAE step.
    """

    if turn_ids is None:
        raise ValueError("Turn-level GAE requires completion-aligned turn_ids.")
    if len(turn_ids) != len(mask):
        raise ValueError("turn_ids must align with completion tokens.")
    if any(
        isinstance(turn_id, bool) or not isinstance(turn_id, int)
        for turn_id in turn_ids
    ):
        raise ValueError("turn_ids must contain integers.")

    previous = -1
    for turn_id, active in zip(turn_ids, mask, strict=True):
        if not active:
            if turn_id != -1:
                raise ValueError("Inactive completion tokens must use turn_id -1.")
            continue
        if not 0 <= turn_id < len(mask):
            raise ValueError("Active turn_ids must be in [0, T).")
        if turn_id < previous:
            raise ValueError("Active turn_ids must be temporally nondecreasing.")
        previous = turn_id
    return list(turn_ids)


# ---------------------------------------------------------------------------
# Token and turn recurrences
# ---------------------------------------------------------------------------

def compute_token_level_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    truncated: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute GAE with every active generated token as one timestep.

    ``values`` contains one extra bootstrap value.  A truncated rollout uses
    that value; a normally terminated rollout bootstraps from zero.  Masked
    tool or padding positions are skipped without consuming discount or lambda.
    """

    _validate_gae_inputs(rewards, values, mask, discount, gae_lambda)
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_value, running = values[-1] if truncated else 0.0, 0.0

    for index in reversed(range(len(rewards))):
        if not mask[index]:
            continue
        delta = rewards[index] + discount * next_value - values[index]
        running = delta + discount * gae_lambda * running
        advantages[index], returns[index] = running, running + values[index]
        next_value = values[index]
    return advantages, returns


def compute_turn_level_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    turn_ids: Sequence[int] | None,
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    truncated: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute GAE with every nonempty generated turn as one timestep.

    Rewards are summed within a turn, while the turn value is read from its
    first active action-token state.  The resulting advantage and return are
    broadcast to all active tokens in that turn so the PyTRIO policy and critic
    interfaces remain token aligned.  Missing numeric turn IDs are allowed and
    do not introduce artificial discount steps.
    """

    _validate_gae_inputs(rewards, values, mask, discount, gae_lambda)
    aligned_turn_ids = _validate_turn_ids(turn_ids, mask)

    # Preserve temporal order without assuming that turn IDs are contiguous.
    turn_order: list[int] = []
    turn_positions: dict[int, list[int]] = {}
    for index, (turn_id, active) in enumerate(
        zip(aligned_turn_ids, mask, strict=True)
    ):
        if not active:
            continue
        if turn_id not in turn_positions:
            turn_order.append(turn_id)
            turn_positions[turn_id] = []
        turn_positions[turn_id].append(index)

    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_value, running = values[-1] if truncated else 0.0, 0.0

    for turn_id in reversed(turn_order):
        positions = turn_positions[turn_id]
        first_position = positions[0]
        turn_value = values[first_position]
        turn_reward = sum(rewards[index] for index in positions)
        delta = turn_reward + discount * next_value - turn_value
        running = delta + discount * gae_lambda * running
        turn_return = running + turn_value

        for index in positions:
            advantages[index] = running
            returns[index] = turn_return
        next_value = turn_value
    return advantages, returns


def compute_trajectory_level_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    truncated: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute one GAE step for the complete generated trajectory.

    All active token rewards form one outcome.  The first active value is the
    trajectory baseline and the final bootstrap is used only for truncation;
    the resulting targets are broadcast to active tokens for token alignment.
    """

    _validate_gae_inputs(rewards, values, mask, discount, gae_lambda)
    active = [index for index, is_active in enumerate(mask) if is_active]
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    if not active:
        return advantages, returns
    value = values[active[0]]
    bootstrap = values[-1] if truncated else 0.0
    delta = sum(rewards[index] for index in active) + discount * bootstrap - value
    advantage = delta
    target = advantage + value
    for index in active:
        advantages[index] = advantage
        returns[index] = target
    return advantages, returns


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    truncated: bool = False,
    timestep_unit: str = "token",
    turn_ids: Sequence[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Dispatch GAE without changing the historical token-level default."""

    if timestep_unit == "token":
        return compute_token_level_gae(
            rewards,
            values,
            mask,
            discount=discount,
            gae_lambda=gae_lambda,
            truncated=truncated,
        )
    if timestep_unit == "turn":
        return compute_turn_level_gae(
            rewards,
            values,
            mask,
            turn_ids,
            discount=discount,
            gae_lambda=gae_lambda,
            truncated=truncated,
        )
    if timestep_unit == "trajectory":
        return compute_trajectory_level_gae(
            rewards,
            values,
            mask,
            discount=discount,
            gae_lambda=gae_lambda,
            truncated=truncated,
        )
    raise ValueError("timestep_unit must be 'token', 'turn' or 'trajectory'.")
