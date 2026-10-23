"""Validated, serializable settings for synchronous agent experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import EpisodeLimits


def positive_int(value: Any, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bound transient operation retries; max_attempts includes the first call.

    Tools are retried only when both retry_tools and the executor's
    is_retry_safe(call) opt-in are true. Deadlines still bound the whole episode.
    Contract violations and model actions are never retried.
    """

    max_attempts: int = 1
    initial_delay: float = 0.1
    max_delay: float = 2.0
    multiplier: float = 2.0
    retry_tools: bool = False

    def __post_init__(self) -> None:
        positive_int(self.max_attempts, "max_attempts")
        for name in ("initial_delay", "max_delay", "multiplier"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.multiplier < 1 or self.max_delay < self.initial_delay:
            raise ValueError("multiplier must be >= 1 and max_delay >= initial_delay.")
        if type(self.retry_tools) is not bool:
            raise ValueError("retry_tools must be a boolean.")

    def delay(self, attempt: int) -> float:
        delay = self.initial_delay
        for _ in range(attempt - 1):
            delay = min(self.max_delay, delay * self.multiplier)
        return delay


@dataclass(frozen=True, slots=True)
class AgentEvaluationConfig:
    """Use a stable seed across policy versions for paired task comparisons."""

    num_generations: int | None = None
    seed: int = 1729
    pass_ks: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.num_generations is not None:
            positive_int(self.num_generations, "evaluation.num_generations")
        positive_int(self.seed, "evaluation.seed", minimum=0)
        object.__setattr__(self, "pass_ks", tuple(self.pass_ks))
        if not self.pass_ks or len(set(self.pass_ks)) != len(self.pass_ks):
            raise ValueError("pass_ks must be nonempty and unique.")
        for k in self.pass_ks:
            positive_int(k, "pass_k")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configure the supplied JSON agent; adapters remain explicitly composed.

    token_append preserves sampled IDs while deriving only the next-message
    suffix from a tokenizer template. chat_template requires exact round trips.
    Temperature and top-p remain 1 in PyTRIOAgentSampler for PPO correctness.
    """

    limits: EpisodeLimits = field(default_factory=EpisodeLimits)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    evaluation: AgentEvaluationConfig = field(default_factory=AgentEvaluationConfig)
    max_concurrency: int = 8
    on_system_failure: str = "raise"
    max_refill_batches: int = 8
    parser: str = "strict_json"
    prompt_key: str = "messages"
    renderer: str = "token_append"
    advertise_tools: bool = False
    template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    stop: str | list[str] | list[int] | None = None
    log_path: str | None = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("limits", EpisodeLimits),
            ("retry", RetryPolicy),
            ("evaluation", AgentEvaluationConfig),
        ):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, cls(**value))
            elif not isinstance(value, cls):
                raise TypeError(f"{name} must be a {cls.__name__} or mapping.")
        positive_int(self.max_concurrency, "max_concurrency")
        positive_int(self.max_refill_batches, "max_refill_batches")
        if not isinstance(self.prompt_key, str) or not self.prompt_key.strip():
            raise ValueError("prompt_key must be a nonempty string.")
        if self.on_system_failure not in {"raise", "drop_group"}:
            raise ValueError("on_system_failure must be raise or drop_group.")
        if self.parser not in {"json", "strict_json"} or self.renderer not in {
            "chat_template",
            "token_append",
        }:
            raise ValueError("Unsupported parser or renderer.")
        if type(self.advertise_tools) is not bool or not isinstance(
            self.template_kwargs, dict
        ):
            raise ValueError(
                "advertise_tools must be boolean and template_kwargs a mapping."
            )
        if {"tokenize", "add_generation_prompt", "tools"}.intersection(
            self.template_kwargs
        ):
            raise ValueError("Reserved chat-template control in template_kwargs.")
        if self.stop is not None and not (
            isinstance(self.stop, str)
            and self.stop
            or isinstance(self.stop, list)
            and self.stop
            and (
                all(isinstance(s, str) and s for s in self.stop)
                or all(type(s) is int and s >= 0 for s in self.stop)
            )
        ):
            raise ValueError(
                "stop must be a nonempty string, string list, token list, or null."
            )
        if self.log_path is not None and (
            not isinstance(self.log_path, str) or not self.log_path.strip()
        ):
            raise ValueError("log_path must be a nonempty string or null.")

    @classmethod
    def from_dict(cls, settings: Mapping[str, Any]) -> AgentConfig:
        return cls(**dict(settings))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def build_workflow(self, *, tokenizer, tools, scorer, logger=None):
        """Compose the built-in JSON adapters with application tools/scoring."""
        from .diagnostics.logging import JsonlEpisodeLogger
        from .rollout.workflow import AgentRolloutWorkflow
        from .runtime.conversation import (
            ChatTemplateRenderer,
            JsonActionParser,
            JsonObservationFormatter,
            TokenAppendRenderer,
        )
        from .runtime.runner import EpisodeRunner

        renderer = (
            TokenAppendRenderer
            if self.renderer == "token_append"
            else ChatTemplateRenderer
        )
        if logger is None and self.log_path is not None:
            logger = JsonlEpisodeLogger(self.log_path)
        return AgentRolloutWorkflow(
            EpisodeRunner(
                parser=JsonActionParser(allow_plain_text=self.parser == "json"),
                renderer=renderer(
                    tokenizer,
                    advertise_tools=self.advertise_tools,
                    **self.template_kwargs,
                ),
                tools=tools,
                scorer=scorer,
                observations=JsonObservationFormatter(),
                limits=self.limits,
                retry=self.retry,
                prompt_key=self.prompt_key,
            ),
            max_concurrency=self.max_concurrency,
            logger=logger,
        )
