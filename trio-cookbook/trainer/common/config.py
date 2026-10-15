"""Public configuration contracts shared by all PyTRIO trainers.

This module is the trainer's contract layer: common run controls live on
``BaseTrainerConfig`` while algorithm subclasses add only their own behavior.
The classes intentionally retain the existing flat constructor fields so
recipes and TOML mappings remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal


@dataclass(slots=True)
class BaseTrainerConfig:
    """Common optimizer, lifecycle, and reporting options for every trainer."""

    batch_size: int = 1
    max_steps: int | None = None
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    seed: int | None = None
    save_every: int = 0
    report_to: Literal["swanlab"] | None = None
    project: str = "trio-cookbook"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        self._integer("batch_size", minimum=1)
        if self.max_steps is not None:
            self._integer("max_steps", minimum=1)
        self._integer("save_every", minimum=0)
        for name in ("learning_rate", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if self.report_to not in (None, "swanlab"):
            raise ValueError("report_to must be 'swanlab' or None.")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("project must be a non-empty string.")
        if self.workspace is not None and (
            not isinstance(self.workspace, str) or not self.workspace.strip()
        ):
            raise ValueError("workspace must be a non-empty string when provided.")
        if self.tags is not None and (
            not isinstance(self.tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags)
        ):
            raise ValueError("tags must be a list of non-empty strings when provided.")

    def _integer(self, name: str, *, minimum: int) -> None:
        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    def optimizer_params(self) -> dict[str, float]:
        """Return SDK-neutral Adam parameters for the PyTRIO adapter."""

        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }


@dataclass(slots=True)
class _StepLoggingConfig(BaseTrainerConfig):
    """Common contract for trainers that aggregate metric windows."""

    logging_step: int = 1

    def __post_init__(self) -> None:
        BaseTrainerConfig.__post_init__(self)
        self._integer("logging_step", minimum=1)


@dataclass(slots=True)
class GRPOConfig(_StepLoggingConfig):
    """Configure client-side, text-generation GRPO training."""

    learning_rate: float = 4e-5
    project: str = "trio-cookbook-grpo"
    output_dir: bool = False
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)
    reward_weights: list[float] | None = None
    scale_rewards: str = "none"
    reward_epsilon: float = 1e-4
    skip_zero_advantage: bool = True
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    stop: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _StepLoggingConfig.__post_init__(self)
        self._integer("num_generations", minimum=2)
        self._integer("max_completion_length", minimum=1)
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (unlimited) or a positive integer.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and > 0.")
        if self.scale_rewards not in {"none", "group", "batch"}:
            raise ValueError("scale_rewards must be 'none', 'group', or 'batch'.")
        if self.reward_epsilon <= 0 or not math.isfinite(self.reward_epsilon):
            raise ValueError("reward_epsilon must be finite and positive.")
        if not isinstance(self.prompt_key, str) or not self.prompt_key.strip():
            raise ValueError("prompt_key must be a non-empty string.")
        if not isinstance(self.stop, bool):
            raise TypeError("stop must be a boolean.")
        if not isinstance(self.output_dir, bool):
            raise TypeError("output_dir must be a boolean.")
        reserved = {"tokenize", "add_generation_prompt"}.intersection(
            self.chat_template_kwargs
        )
        if reserved:
            raise ValueError(
                "chat_template_kwargs must not override "
                + ", ".join(sorted(reserved))
                + "."
            )
        if self.loss_fn != "importance_sampling":
            raise ValueError("GRPO supports loss_fn='importance_sampling' only.")

    @property
    def sampling_params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_completion_length,
            "seed": self.seed,
        }


@dataclass(slots=True)
class DPOConfig(_StepLoggingConfig):
    """Configure text-only DPO training with PyTRIO."""

    learning_rate: float = 5e-6
    project: str = "trio-cookbook-dpo"
    beta: float = 0.1
    max_length: int | None = 1024
    prompt_key: str = "messages"
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _StepLoggingConfig.__post_init__(self)
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        if not isinstance(self.prompt_key, str) or not self.prompt_key.strip():
            raise ValueError("prompt_key must be a non-empty string.")
        if {"tokenize", "add_generation_prompt"}.intersection(
            self.chat_template_kwargs
        ):
            raise ValueError(
                "chat_template_kwargs cannot override tokenize or add_generation_prompt."
            )


@dataclass(slots=True)
class SFTConfig(BaseTrainerConfig):
    """Configure text-only supervised fine-tuning with PyTRIO."""

    learning_rate: float = 2e-5
    project: str = "trio-cookbook-sft"
    max_in_flight: int = 32
    max_length: int | None = 1024
    dataset_text_field: str = "text"
    completion_only_loss: bool | None = None
    assistant_only_loss: bool = False
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        BaseTrainerConfig.__post_init__(self)
        self._integer("max_in_flight", minimum=1)
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        if not isinstance(self.dataset_text_field, str) or not self.dataset_text_field.strip():
            raise ValueError("dataset_text_field must be non-empty.")
        reserved = {
            "tokenize",
            "add_generation_prompt",
            "return_dict",
            "return_assistant_tokens_mask",
        }.intersection(self.chat_template_kwargs)
        if reserved:
            raise ValueError("chat_template_kwargs contains a reserved tokenizer argument.")


@dataclass(slots=True)
class OPDConfig(_StepLoggingConfig):
    """Configure sampled, token-level reverse-KL distillation.

    Parameters:
        batch_size (`int`, defaults to `4`): Prompts per optimizer update.
        max_steps (`int` or `None`, defaults to `None`): Optimizer step limit.
        learning_rate (`float`, defaults to `1e-5`): Remote Adam learning rate.
        beta1 (`float`, defaults to `0.9`): Adam first-moment coefficient.
        beta2 (`float`, defaults to `0.95`): Adam second-moment coefficient.
        eps (`float`, defaults to `1e-8`): Adam denominator smoothing value.
        seed (`int` or `None`, defaults to `None`): Sampling seed base.
        save_every (`int`, defaults to `0`): Periodic checkpoint interval.
        report_to (`"swanlab"` or `None`, defaults to `None`): Reporting backend.
        project (`str`, defaults to `"trio-cookbook-opd"`): SwanLab project.
        workspace (`str` or `None`, defaults to `None`): SwanLab workspace.
        tags (`list[str]` or `None`, defaults to `None`): SwanLab tags.
        run_name (`str` or `None`, defaults to `None`): SwanLab run name.
        log_dir (`str` or `None`, defaults to `None`): SwanLab log directory.
        logging_step (`int`, defaults to `1`): Metric aggregation interval.
        num_generations (`int`, defaults to `1`): Completions per prompt.
        max_completion_length (`int`, defaults to `512`): Completion token limit.
        max_length (`int` or `None`, defaults to `2048`): Total token budget.
        max_concurrent_sampling (`int`, defaults to `16`): Sampling concurrency.
        max_concurrent_scoring (`int`, defaults to `32`): Teacher concurrency.
        kl_coef (`float`, defaults to `1.0`): Reverse-KL signal coefficient.
        discount_factor (`float`, defaults to `0.0`): Future-signal discount.
        prompt_key (`str`, defaults to `"messages"`): Dataset prompt column.
        add_generation_prompt (`bool`, defaults to `True`): Add assistant prefix.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`): Template options.
        stop (`list[str]` or `None`, defaults to `None`): Sampling stop strings.
    """

    batch_size: int = 4
    project: str = "trio-cookbook-opd"
    num_generations: int = 1
    max_completion_length: int = 512
    max_length: int | None = 2048
    max_concurrent_sampling: int = 16
    max_concurrent_scoring: int = 32
    kl_coef: float = 1.0
    discount_factor: float = 0.0
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    stop: list[str] | None = None

    def __post_init__(self) -> None:
        _StepLoggingConfig.__post_init__(self)
        for name in (
            "num_generations",
            "max_completion_length",
            "max_concurrent_sampling",
            "max_concurrent_scoring",
        ):
            self._integer(name, minimum=1)
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        if self.seed is not None:
            self._integer("seed", minimum=0)
        if not math.isfinite(self.kl_coef) or self.kl_coef <= 0:
            raise ValueError("kl_coef must be finite and positive.")
        if not math.isfinite(self.discount_factor) or not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1].")
        if not isinstance(self.prompt_key, str) or not self.prompt_key.strip():
            raise ValueError("prompt_key must be a non-empty string.")
        if not isinstance(self.add_generation_prompt, bool):
            raise ValueError("add_generation_prompt must be boolean.")
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a dictionary.")
        if {"tokenize", "add_generation_prompt"}.intersection(
            self.chat_template_kwargs
        ):
            raise ValueError("chat_template_kwargs contains a reserved tokenizer argument.")
        if self.stop is not None and (
            not isinstance(self.stop, list)
            or not self.stop
            or any(not isinstance(value, str) or not value for value in self.stop)
        ):
            raise ValueError("stop must be a non-empty list of non-empty strings or None.")


__all__ = [
    "BaseTrainerConfig",
    "DPOConfig",
    "GRPOConfig",
    "OPDConfig",
    "SFTConfig",
]
