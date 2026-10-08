"""Configuration for PyTRIO-backed GRPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal


@dataclass(slots=True)
class GRPOConfig:
    """Configure client-side, text-generation GRPO training.

    Parameters:
        output_dir (`bool`, defaults to `False`):
            Whether to log the final state checkpoint URI to SwanLab when reporting is enabled.
        num_generations (`int`, defaults to `8`):
            Number of completions sampled for each prompt in a GRPO group.
        max_completion_length (`int`, defaults to `512`):
            Maximum number of tokens generated per completion.
        temperature (`float`, defaults to `1.0`):
            Sampling temperature; higher values make token selection more random.
        top_p (`float`, defaults to `1.0`):
            Nucleus-sampling probability threshold; `1.0` considers the full distribution.
        top_k (`int`, defaults to `-1`):
            Number of highest-probability tokens kept for sampling; `-1` disables top-k filtering.
        batch_size (`int`, defaults to `1`):
            Number of prompts processed per training step.
        max_steps (`int` or `None`, defaults to `None`):
            Maximum number of training steps. `None` trains until the dataset is exhausted.
        learning_rate (`float`, defaults to `4e-5`):
            Learning rate passed to the remote Adam optimizer.
        beta1 (`float`, defaults to `0.9`):
            Adam coefficient for the first-moment estimate.
        beta2 (`float`, defaults to `0.95`):
            Adam coefficient for the second-moment estimate.
        eps (`float`, defaults to `1e-8`):
            Adam denominator smoothing constant.
        loss_fn (`str`, defaults to `"importance_sampling"`):
            Remote training loss function. Only `"importance_sampling"` is currently supported.
        loss_fn_config (`dict[str, Any]`, defaults to `{}`):
            Extra configuration passed to the remote loss function.
        reward_weights (`list[float]` or `None`, defaults to `None`):
            Weight for each reward function; `None` gives every function weight `1.0`.
        scale_rewards (`str`, defaults to `"none"`):
            Advantage scaling: `"none"` centers rewards only, `"group"` divides by each group's standard deviation,
            and `"batch"` divides by the standard deviation across the entire step.
        reward_epsilon (`float`, defaults to `1e-4`):
            Smoothing constant added to the reward standard deviation when scaling advantages.
        skip_zero_advantage (`bool`, defaults to `True`):
            Skip groups whose completion advantages are all zero.
        save_every (`int`, defaults to `0`):
            Interval between state checkpoints. `0` disables state checkpoint saves.
        logging_step (`int`, defaults to `1`):
            Interval for logging averaged training scalars to SwanLab; the final partial interval is also logged.
        seed (`int` or `None`, defaults to `None`):
            Seed for the training client and sampling; `None` leaves it unspecified.
        prompt_key (`str`, defaults to `"messages"`):
            Dataset column containing each prompt.
        add_generation_prompt (`bool`, defaults to `True`):
            Add the assistant-generation prefix when rendering chat messages.
        stop (`bool`, defaults to `True`):
            Stop generation at the tokenizer EOS or chat end marker when available.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Extra keyword arguments passed to the tokenizer's chat template.
        report_to (`"swanlab"` or `None`, defaults to `None`):
            Enable SwanLab reporting, or use terminal progress only when `None`.
        project (`str`, defaults to `"trio-cookbook-grpo"`):
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

    output_dir: bool = False
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
    logging_step: int = 1
    seed: int | None = None
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    stop: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    report_to: Literal["swanlab"] | None = None
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
        if isinstance(self.logging_step, bool) or not isinstance(self.logging_step, int) or self.logging_step < 1:
            raise ValueError("logging_step must be a positive integer.")
        if self.report_to is not None and self.report_to != "swanlab":
            raise ValueError("report_to must be 'swanlab' or None.")
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
        if not isinstance(self.stop, bool):
            raise TypeError("stop must be a boolean.")
        if not isinstance(self.output_dir, bool):
            raise TypeError("output_dir must be a boolean.")
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
