"""Composable PPO-family settings; presets only select settings, not trainers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ...api.cli_args import BaseTrainerConfig


@dataclass(frozen=True, slots=True)
class NormConfig:
    mean_level: Literal["batch", "group"] | None = "group"
    std_level: Literal["batch", "group"] | None = "group"
    mean_leave1out: bool = False
    std_unbiased: bool = True
    eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("mean_leave1out", "std_unbiased"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        for level in (self.mean_level, self.std_level):
            if level not in (None, "batch", "group"):
                raise ValueError("Normalization levels must be batch, group or None.")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Normalization eps must be finite and positive.")


@dataclass(slots=True)
class PPOConfig(BaseTrainerConfig):
    """One policy loop with interchangeable baselines, reductions and losses.

    Defaults describe PPO with independent PyTRIO actor and critic clients.
    ``for_algorithm('grpo')`` disables the critic without changing the loop.
    max_steps counts collected rollout batches; optimizer_steps also counts
    their minibatches and update_epochs. Configuration stays in this package
    so every PPO-family switch has one implementation location.
    """

    use_critic: bool = True
    critic_learning_rate: float | None = None
    critic_lora_rank: int | None = None
    critic_token_ids: tuple[int, int] | None = None
    critic_value_scale: float = 1.0
    critic_value_bias: float = 0.0
    critic_eps_clip: float = 0.2
    critic_n_minibatches: int = 1
    critic_update_epochs: int = 1
    num_generations: int = 4
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    prompt_key: str = "messages"
    chat_template_kwargs: dict = field(default_factory=dict)
    reward_norm: NormConfig | None = None
    adv_norm: NormConfig | None = field(
        default_factory=lambda: NormConfig("batch", "batch")
    )
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    discount: float = 1.0
    gae_lambda: float = 1.0
    kl_ctl: float = 0.0
    kl_estimator: str = "k1"
    mask_no_eos_with_zero: bool = False
    eps_clip: float = 0.2
    eps_clip_higher: float | None = None
    c_clip: float | None = None
    loss_type: str = "ppo"
    importance_sampling_level: str = "token"
    loss_aggregation: str = "token_mean"
    loss_aggregation_divisor: float | None = None
    use_decoupled_loss: bool = False
    recompute_logprob: bool = False
    rejection_lower: float | None = None
    rejection_upper: float | None = None
    sapo_tau_pos: float = 1.0
    sapo_tau_neg: float = 1.05
    ppo_n_minibatches: int = 1
    update_epochs: int = 1
    loss_backend: str = "custom"
    dynamic_sampling: bool = False
    max_empty_batches: int = 32
    overlong_tokens: int = 0
    overlong_penalty_factor: float = 1.0
    eval_after_train: bool = False

    def __post_init__(self) -> None:
        super(PPOConfig, self).__post_init__()
        for name in (
            "use_critic",
            "use_decoupled_loss",
            "recompute_logprob",
            "dynamic_sampling",
            "mask_no_eos_with_zero",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        for name in ("reward_norm", "adv_norm"):
            if getattr(self, name) is not None and not isinstance(
                getattr(self, name), NormConfig
            ):
                raise ValueError(f"{name} must be a NormConfig or None.")
        self._non_empty_string("prompt_key")
        for name in (
            "num_generations",
            "max_tokens",
            "ppo_n_minibatches",
            "update_epochs",
            "max_empty_batches",
            "critic_n_minibatches",
            "critic_update_epochs",
        ):
            self._integer(name, minimum=1)
        self._integer("overlong_tokens", minimum=0)
        if self.critic_learning_rate is not None:
            self._positive("critic_learning_rate")
        if self.critic_lora_rank is not None:
            self._integer("critic_lora_rank", minimum=1)
        self._positive("critic_value_scale")
        self._nonnegative("critic_eps_clip")
        if not math.isfinite(self.critic_value_bias):
            raise ValueError("critic_value_bias must be finite.")
        if self.critic_token_ids is not None and (
            not isinstance(self.critic_token_ids, (tuple, list))
            or len(self.critic_token_ids) != 2
            or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0
                for t in self.critic_token_ids
            )
            or self.critic_token_ids[0] == self.critic_token_ids[1]
        ):
            raise ValueError(
                "critic_token_ids must contain two distinct nonnegative token IDs (positive, negative)."
            )
        for name in (
            "temperature",
            "reward_scaling",
            "reward_clip",
            "sapo_tau_pos",
            "sapo_tau_neg",
        ):
            self._positive(name)
        for name in ("eps_clip", "kl_ctl", "overlong_penalty_factor"):
            self._nonnegative(name)
        for name in ("top_p", "discount", "gae_lambda"):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or not 0 <= value <= 1
                or (name == "top_p" and value == 0)
            ):
                raise ValueError(f"{name} is outside its probability range.")
        if not math.isfinite(self.reward_bias) or self.eps_clip > 1:
            raise ValueError("reward_bias must be finite and eps_clip <= 1.")
        for name in ("eps_clip_higher", "rejection_lower", "rejection_upper"):
            if getattr(self, name) is not None:
                self._positive(name)
        if self.c_clip is not None and (
            not math.isfinite(self.c_clip) or self.c_clip <= 1
        ):
            raise ValueError("c_clip must be finite and > 1.")
        choices = {
            "kl_estimator": ("k1", "k2", "k3"),
            "loss_type": ("ppo", "sapo", "cispo"),
            "importance_sampling_level": ("token", "sequence"),
            "loss_aggregation": ("token_mean", "seq_mean", "prompt_mean", "constant"),
            "loss_backend": ("custom", "native"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")
        if self.loss_aggregation == "constant":
            self._positive("loss_aggregation_divisor")
        if self.loss_type == "sapo" and self.use_decoupled_loss:
            raise ValueError("SAPO requires use_decoupled_loss=False.")
        if self.loss_type == "cispo" and (
            self.importance_sampling_level != "token" or self.eps_clip_higher is None
        ):
            raise ValueError("CISPO requires token ratios and eps_clip_higher.")
        if (
            self.rejection_lower is not None
            and self.rejection_upper is not None
            and self.rejection_lower > self.rejection_upper
        ):
            raise ValueError("rejection_lower must not exceed rejection_upper.")
        if (
            self.rejection_lower is not None or self.rejection_upper is not None
        ) and not self.use_decoupled_loss:
            raise ValueError("Behavior rejection requires use_decoupled_loss=True.")
        if self.overlong_tokens > self.max_tokens:
            raise ValueError("overlong_tokens must not exceed max_tokens.")
        if self.dynamic_sampling and self.num_generations < 2:
            raise ValueError(
                "Dynamic sampling requires at least two responses per prompt."
            )
        if (
            self.loss_aggregation != "constant"
            and self.loss_aggregation_divisor is not None
        ):
            raise ValueError(
                "loss_aggregation_divisor only applies to constant reduction."
            )
        if self.loss_backend == "native" and not self.native_compatible:
            raise ValueError(
                "Native PPO only supports token_mean, token ratios and ordinary clipping; use custom."
            )
        if self.eval_steps or self.eval_before_train or self.eval_after_train:
            raise NotImplementedError(
                "This PPO package does not implement scheduled evaluation."
            )
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise NotImplementedError(
                "Use temperature=1 and top_p=1: target-only training logprobs cannot "
                "reconstruct temperature/nucleus normalization."
            )

    @property
    def native_compatible(self) -> bool:
        return (
            self.loss_type == "ppo"
            and self.importance_sampling_level == "token"
            and self.loss_aggregation == "token_mean"
            and self.c_clip is None
            and self.rejection_lower is None
            and self.rejection_upper is None
            and not self.use_decoupled_loss
        )

    @classmethod
    def for_algorithm(cls, name: str, **overrides) -> PPOConfig:
        """AReaL-style presets. Explicit keyword overrides always win."""
        options = {"use_critic": False, "reward_norm": NormConfig()}
        if name == "ppo":
            options = {"use_critic": True, "reward_norm": None}
        elif name == "grpo":
            pass
        elif name == "drgrpo":
            options["reward_norm"] = NormConfig("group", None)
        elif name == "liteppo":
            options["reward_norm"] = NormConfig("group", "batch")
        elif name == "rloo":
            options["reward_norm"] = NormConfig("group", None, mean_leave1out=True)
        elif name == "reinforce":
            options["reward_norm"] = None
        elif name == "reinforce_baseline":
            options["reward_norm"] = NormConfig("group", None)
        elif name == "gspo":
            options["importance_sampling_level"] = "sequence"
        elif name == "sapo":
            options["loss_type"] = "sapo"
        elif name == "cispo":
            options.update(loss_type="cispo", eps_clip=1.0, eps_clip_higher=4.0)
        elif name == "dapo":
            options.update(
                dynamic_sampling=True,
                eps_clip=0.2,
                eps_clip_higher=0.28,
                overlong_tokens=128,
            )
        else:
            raise ValueError(f"Unknown PPO-family preset: {name!r}.")
        return cls(**{**options, **overrides})
