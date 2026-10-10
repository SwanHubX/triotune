"""Configuration for PyTRIO on-policy distillation (OPD)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class OPDConfig:
    """Configure sampled, token-level reverse-KL distillation.

    Sampling uses temperature=1, top_p=1 and top_k=-1 so rollout logprobs
    describe the student distribution used by training. No additional student
    scoring pass or local PyTorch custom backward is needed.

    Parameters:
        batch_size (`int`, defaults to `4`):
            Prompts per optimizer update; the last batch may be smaller.
        num_generations (`int`, defaults to `1`):
            Student completions per prompt. OPD needs no reward group or centering;
            one completion minimizes rollout cost, more reduce sampling variance.
        max_completion_length (`int`, defaults to `512`):
            Maximum generated tokens, including any sampled EOS. No EOS is appended.
        max_length (`int` or `None`, defaults to `2048`):
            Prompt plus completion budget. Generation is capped to the remaining
            space; prompts leaving no completion space fail. None disables this cap.
        max_steps (`int` or `None`, defaults to `None`):
            Total optimizer step limit; None consumes the dataset once.
        max_concurrent_sampling (`int`, defaults to `16`):
            Maximum simultaneous student sample requests within a step.
        max_concurrent_scoring (`int`, defaults to `32`):
            Maximum simultaneous teacher compute_logprobs requests. Separate
            limits let teacher scoring overlap other prompts' student generation.
        kl_coef (`float`, defaults to `1.0`):
            Multiply teacher_logprob - student_logprob by this positive coefficient.
            One uses the unscaled sampled reverse-KL descent direction.
        discount_factor (`float`, defaults to `0.0`):
            Discounted future sum of token advantages, in [0, 1]. Zero uses only
            the current token's KL signal (the standard local OPD surrogate);
            one adds the entire remaining sequence's signal to each token.
        prompt_key (`str`, defaults to `"messages"`):
            Dataset prompt column; supports text, token IDs or text chat messages.
        add_generation_prompt (`bool`, defaults to `True`):
            Ask the student chat template to append an assistant generation prefix.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Extra student chat-template arguments, e.g. enable_thinking=False.
            tokenize and add_generation_prompt are controlled by the trainer.
        stop (`list[str]` or `None`, defaults to `None`):
            Optional stop strings passed to sampling; None uses normal model EOS
            stopping. A custom stop changes which completion prefixes are trained.
        learning_rate (`float`, defaults to `1e-5`):
            Remote Adam learning rate; a conservative starting point for distillation.
        beta1 (`float`, defaults to `0.9`):
            Adam first-moment decay coefficient.
        beta2 (`float`, defaults to `0.95`):
            Adam second-moment decay coefficient.
        eps (`float`, defaults to `1e-8`):
            Positive smoothing constant in Adam's denominator.
        seed (`int` or `None`, defaults to `None`):
            Training-client seed and sampling seed base. Sampling offsets it by
            completed prompt count and prompt index to avoid repeating the same
            random stream, including when manual steps have different sizes.
        save_every (`int`, defaults to `0`):
            Periodic state checkpoint interval; zero disables periodic saves.
            train/train_async still save the final state after successful updates.
        logging_step (`int`, defaults to `1`):
            SwanLab scalar averaging interval; the final partial window is flushed.
        report_to (`"swanlab"` or `None`, defaults to `None`):
            Enable SwanLab reporting; None shows terminal progress only.
        project (`str`, defaults to `"trio-cookbook-opd"`):
            SwanLab project name.
        workspace (`str` or `None`, defaults to `None`):
            SwanLab workspace; None uses the SDK default.
        tags (`list[str]` or `None`, defaults to `None`):
            Tags attached to the SwanLab run.
        run_name (`str` or `None`, defaults to `None`):
            SwanLab run name; None uses <student model name>-opd.
        log_dir (`str` or `None`, defaults to `None`):
            Local SwanLab log directory; None uses the SDK default.
    """

    batch_size: int = 4
    num_generations: int = 1
    max_completion_length: int = 512
    max_length: int | None = 2048
    max_steps: int | None = None
    max_concurrent_sampling: int = 16
    max_concurrent_scoring: int = 32
    kl_coef: float = 1.0
    discount_factor: float = 0.0
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    stop: list[str] | None = None
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    seed: int | None = None
    save_every: int = 0
    logging_step: int = 1
    report_to: Literal["swanlab"] | None = None
    project: str = "trio-cookbook-opd"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None

    def __post_init__(self) -> None:
        for name in ("batch_size", "num_generations", "max_completion_length",
                     "max_concurrent_sampling", "max_concurrent_scoring", "logging_step"):
            self._integer(name, minimum=1)
        if self.max_steps is not None:
            self._integer("max_steps", minimum=1)
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        self._integer("save_every", minimum=0)
        if self.seed is not None:
            self._integer("seed", minimum=0)
        for name in ("kl_coef", "learning_rate", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("beta1", "beta2", "discount_factor"):
            value = getattr(self, name)
            upper = value <= 1 if name == "discount_factor" else value < 1
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or not upper:
                raise ValueError(f"{name} is outside its supported range.")
        for name in ("prompt_key", "project"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.add_generation_prompt, bool):
            raise ValueError("add_generation_prompt must be boolean.")
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a dictionary.")
        if {"tokenize", "add_generation_prompt"} & self.chat_template_kwargs.keys():
            raise ValueError("chat_template_kwargs contains a reserved tokenizer argument.")
        if self.stop is not None and (
            not isinstance(self.stop, list) or not self.stop
            or any(not isinstance(value, str) or not value for value in self.stop)
        ):
            raise ValueError("stop must be a non-empty list of non-empty strings or None.")
        if self.report_to not in (None, "swanlab"):
            raise ValueError("report_to must be 'swanlab' or None.")

    def _integer(self, name: str, *, minimum: int) -> None:
        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    def optimizer_params(self) -> dict[str, float]:
        return {"learning_rate": self.learning_rate, "beta1": self.beta1,
                "beta2": self.beta2, "eps": self.eps}
