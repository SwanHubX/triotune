"""Configuration for text-only PyTRIO supervised fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class SFTConfig:
    batch_size: int = 1
    max_in_flight: int = 32
    max_steps: int | None = None
    max_length: int | None = 1024
    dataset_text_field: str = "text"
    completion_only_loss: bool | None = None
    assistant_only_loss: bool = False
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 2e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    seed: int | None = None
    save_every: int = 0
    report_to: Literal["swanlab"] | None = None
    project: str = "trio-cookbook-sft"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError("batch_size and max_steps must be positive.")
        if self.max_in_flight < 1:
            raise ValueError("max_in_flight must be positive.")
        if self.max_length is not None and self.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self.save_every < 0:
            raise ValueError("save_every must be nonnegative.")
        if not self.dataset_text_field.strip():
            raise ValueError("dataset_text_field must be non-empty.")
        if self.report_to not in (None, "swanlab"):
            raise ValueError("report_to must be 'swanlab' or None.")
        if self.report_to and not self.project.strip():
            raise ValueError("project must be non-empty.")
        if {"tokenize", "add_generation_prompt", "return_dict", "return_assistant_tokens_mask"} & self.chat_template_kwargs.keys():
            raise ValueError("chat_template_kwargs contains a reserved tokenizer argument.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("learning_rate and eps must be finite and positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1).")

    def optimizer_params(self) -> dict[str, float]:
        return {"learning_rate": self.learning_rate, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps}
