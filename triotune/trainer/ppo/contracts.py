"""PPO data contracts shared by rollout, actor, critic, and PyTRIO datums."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class Trajectory:
    """One unpadded prompt/completion sample and its aligned training fields.

    Auxiliary arrays describe completion tokens; ``values`` additionally holds
    one final bootstrap prediction for critic-based PPO.
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    reward: float
    group_id: str
    loss_mask: list[bool] | None = None
    truncated: bool = False
    reference_logprobs: list[float] | None = None
    values: list[float] | None = None
    token_rewards: list[float] | None = None
    turn_ids: list[int] | None = None

    def __post_init__(self) -> None:
        if not self.prompt_tokens or not self.completion_tokens:
            raise ValueError(
                "A trajectory needs nonempty prompt and completion tokens."
            )
        if any(
            isinstance(t, bool) or not isinstance(t, int) or t < 0
            for t in self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("Token IDs must be nonnegative integers.")
        n = len(self.completion_tokens)
        if self.logprobs is None:
            raise ValueError(
                "Behavior logprobs are required for every completion token."
            )
        if self.loss_mask is None:
            self.loss_mask = [True] * n
        if len(self.loss_mask) != n or not all(
            isinstance(x, bool) for x in self.loss_mask
        ):
            raise ValueError("loss_mask must contain one boolean per completion token.")
        if not any(self.loss_mask):
            raise ValueError(
                "A trajectory must contain at least one active action token."
            )
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError(
                "group_id must be a nonempty string identifying the prompt group."
            )
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite.")
        for name in ("logprobs", "reference_logprobs", "values", "token_rewards"):
            values = getattr(self, name)
            if values is not None and (
                len(values) != n + int(name == "values")
                or any(not math.isfinite(x) for x in values)
            ):
                raise ValueError(
                    f"{name} must be finite and aligned with completion tokens"
                    + (" plus one bootstrap value." if name == "values" else ".")
                )
        if self.turn_ids is not None:
            if len(self.turn_ids) != n or any(
                isinstance(turn_id, bool) or not isinstance(turn_id, int)
                for turn_id in self.turn_ids
            ):
                raise ValueError(
                    "turn_ids must contain one integer per completion token."
                )
            previous = -1
            for turn_id, active in zip(self.turn_ids, self.loss_mask, strict=True):
                if not active:
                    if turn_id != -1:
                        raise ValueError(
                            "Inactive completion tokens must use turn_id -1."
                        )
                    continue
                if not 0 <= turn_id < n:
                    raise ValueError("Active turn_ids must be in [0, T).")
                if turn_id < previous:
                    raise ValueError(
                        "Active turn_ids must be temporally nondecreasing."
                    )
                previous = turn_id


@dataclass(slots=True)
class PreparedBatch:
    """Frozen actor targets prepared once before minibatch optimization."""

    trajectories: list[Trajectory]
    advantages: list[list[float]]
    returns: list[list[float]]
    masks: list[list[bool]]
    proximal_logprobs: list[list[float]]
    prompt_mean_scale: float = 1.0


@dataclass(slots=True)
class PPOState:
    """Mutable lifecycle state returned by the root PPO trainer."""

    global_step: int = 0
    optimizer_steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    critic_optimizer_steps: int = 0
    critic_checkpoint_path: str | None = None
