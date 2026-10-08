"""Configuration for the PyTRIO-backed DPO trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class DPOConfig:
    """The supported, text-only subset of TRL's DPO configuration."""

    beta: float = 0.1
    batch_size: int = 1
    max_steps: int | None = None
    max_length: int | None = 1024
    learning_rate: float = 5e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    seed: int | None = None
    add_eos_token: bool = True
    save_every: int = 0
    mode: str = "local"
    project: str = "trio-cookbook-dpo"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided.")
        if self.max_length is not None and self.max_length < 2:
            raise ValueError("max_length must be at least 2 when provided.")
        for name in ("learning_rate", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if self.save_every < 0:
            raise ValueError("save_every must be nonnegative.")
        if self.mode not in {"disabled", "online", "local", "offline"}:
            raise ValueError("mode must be disabled, online, local, or offline.")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("project must be a non-empty string.")

    def optimizer_params(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
