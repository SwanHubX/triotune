"""GRPO configuration.

This is deliberately smaller than ``trl.GRPOConfig``.  TRL's configuration
also contains local Transformers/Accelerate/vLLM switches; those would be
misleading for PyTRIO, where the service owns model loading and execution.
Although configuration objects are often shared utilities, this one is kept
under ``trainer`` because its fields (group size, reward scaling and the GRPO
loss contract) are algorithm-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(slots=True)
class GRPOConfig:
    """Configuration for one client-side, text-generation GRPO loop."""

    output_dir: str | None = None
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    # PyTRIO uses -1 to mean that top-k filtering is disabled.
    top_k: int = -1
    batch_size: int = 1
    max_steps: int | None = None
    learning_rate: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)
    reward_weights: list[float] | None = None
    scale_rewards: str = "none"
    reward_epsilon: float = 1e-4
    skip_zero_advantage: bool = True
    save_every: int = 0
    seed: int | None = None
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    mode: str = "disabled"
    project: str = "trio-cookbook-grpo"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise ValueError("GRPO requires num_generations >= 2.")
        if self.max_completion_length < 1:
            raise ValueError("max_completion_length must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided.")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (unlimited) or a positive integer.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and > 0.")
        for name, value in (("learning_rate", self.learning_rate), ("eps", self.eps)):
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and > 0.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if self.scale_rewards not in {"none", "group", "batch"}:
            raise ValueError("scale_rewards must be 'none', 'group', or 'batch'.")
        if self.save_every < 0:
            raise ValueError("save_every must be >= 0.")
        if self.mode not in {"disabled", "online", "local", "offline"}:
            raise ValueError("mode must be 'disabled', 'online', 'local', or 'offline'.")
        if not self.project.strip():
            raise ValueError("project must not be empty.")
        if self.workspace is not None and (
            not isinstance(self.workspace, str) or not self.workspace.strip()
        ):
            raise ValueError("workspace must be a non-empty string when provided.")
        if self.tags is not None and (
            not isinstance(self.tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags)
        ):
            raise ValueError("tags must be a list of non-empty strings when provided.")
        if self.reward_epsilon <= 0 or not math.isfinite(self.reward_epsilon):
            raise ValueError("reward_epsilon must be finite and positive.")
        if not self.prompt_key.strip():
            raise ValueError("prompt_key must not be empty.")
        reserved = {"tokenize", "add_generation_prompt"}.intersection(self.chat_template_kwargs)
        if reserved:
            raise ValueError(
                "chat_template_kwargs must not override " + ", ".join(sorted(reserved)) + "."
            )
        if self.loss_fn != "importance_sampling":
            raise ValueError(
                "The first PyTRIO implementation supports GRPO with "
                "loss_fn='importance_sampling' only."
            )

    @property
    def sampling_params(self) -> dict[str, Any]:
        """Parameters passed to ``SamplingClient.sample``.

        A caller can translate this plain mapping to its SDK's
        ``SamplingParams`` class.  This avoids importing a particular PyTRIO
        release from the cookbook.
        """

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_completion_length,
            "seed": self.seed,
        }

    def optimizer_params(self) -> dict[str, float]:
        """Return SDK-neutral Adam parameters for the remote client adapter."""

        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
