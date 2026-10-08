"""Configuration for the PyTRIO-backed DPO trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class DPOConfig:
    """Configure text-only DPO training with PyTRIO.

    Parameters:
        beta (`float`, defaults to `0.1`):
            Scale for the policy-versus-reference log-probability difference in the DPO loss.
        batch_size (`int`, defaults to `1`):
            Number of chosen/rejected preference pairs processed per training step.
        max_steps (`int` or `None`, defaults to `None`):
            Maximum number of training steps. `None` trains until the dataset is exhausted.
        max_length (`int` or `None`, defaults to `1024`):
            Maximum prompt-plus-completion token length. Overlong pairs raise an error; `None` disables the limit.
        learning_rate (`float`, defaults to `5e-6`):
            Learning rate passed to the remote Adam optimizer.
        beta1 (`float`, defaults to `0.9`):
            Adam coefficient for the first-moment estimate.
        beta2 (`float`, defaults to `0.95`):
            Adam coefficient for the second-moment estimate.
        eps (`float`, defaults to `1e-8`):
            Adam denominator smoothing constant.
        seed (`int` or `None`, defaults to `None`):
            Seed for the training client and final inference samples; `None` leaves it unspecified.
        prompt_key (`str`, defaults to `"messages"`):
            Dataset column containing the prompt shared by the chosen and rejected completions.
        add_eos_token (`bool`, defaults to `True`):
            Append the tokenizer's EOS token to each completion when it is missing.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Extra keyword arguments passed to the tokenizer's chat template.
        save_every (`int`, defaults to `0`):
            Interval between periodic state checkpoints. `0` disables periodic saves; training still saves a final state.
        logging_step (`int`, defaults to `1`):
            Interval for logging averaged training scalars to SwanLab; the final partial interval is also logged.
        report_to (`"swanlab"` or `None`, defaults to `None`):
            Enable SwanLab reporting, or use terminal progress only when `None`.
        project (`str`, defaults to `"trio-cookbook-dpo"`):
            SwanLab project name.
        workspace (`str` or `None`, defaults to `None`):
            SwanLab workspace; `None` uses the SDK default.
        tags (`list[str]` or `None`, defaults to `None`):
            Tags attached to the SwanLab run.
        run_name (`str` or `None`, defaults to `None`):
            SwanLab run name; `None` derives a name from the model.
        log_dir (`str` or `None`, defaults to `None`):
            Local directory for SwanLab logs; `None` uses the SDK default.
    """

    beta: float = 0.1
    batch_size: int = 1
    max_steps: int | None = None
    max_length: int | None = 1024
    learning_rate: float = 5e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    seed: int | None = None
    prompt_key: str = "messages"
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    save_every: int = 0
    logging_step: int = 1
    report_to: Literal["swanlab"] | None = None
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
        if isinstance(self.logging_step, bool) or not isinstance(self.logging_step, int) or self.logging_step < 1:
            raise ValueError("logging_step must be a positive integer.")
        if not isinstance(self.prompt_key, str) or not self.prompt_key.strip():
            raise ValueError("prompt_key must be a non-empty string.")
        if {"tokenize", "add_generation_prompt"}.intersection(self.chat_template_kwargs):
            raise ValueError("chat_template_kwargs cannot override tokenize or add_generation_prompt.")
        if self.report_to is not None and self.report_to != "swanlab":
            raise ValueError("report_to must be 'swanlab' or None.")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("project must be a non-empty string.")

    def optimizer_params(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
