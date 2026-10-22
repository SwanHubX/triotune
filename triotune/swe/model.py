"""A tool-aware adapter around the same public PyTRIO API used by trainers."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

import pytrio as trio

from ..api.client_api import call_async
from ..dataset.prompt import _token_list
from .contracts import SampledTurn


@dataclass(slots=True)
class SWESamplingConfig:
    """Limits for one model call. An episode must keep one policy snapshot."""

    max_tokens: int = 2048
    max_context_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("max_tokens", "max_context_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_tokens >= self.max_context_tokens:
            raise ValueError("max_tokens must leave space for a prompt.")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite.")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature must be >= 0 and top_p in (0, 1].")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None.")
        reserved = {
            "tools",
            "tokenize",
            "add_generation_prompt",
            "return_dict",
            "return_assistant_tokens_mask",
        }
        if not isinstance(self.chat_template_kwargs, dict) or reserved.intersection(
            self.chat_template_kwargs
        ):
            raise ValueError(
                "chat_template_kwargs must not override rendering controls."
            )


class PyTRIOSWEModel:
    """Use an injected SamplingClient, including a trained LoRA snapshot.

    create() is for inference from a base model or a saved model_path. For RL,
    construct this class with save_weights_and_get_sampling_client()'s result.
    """

    def __init__(
        self,
        sampling_client,
        tokenizer,
        *,
        policy_version: str,
        config: SWESamplingConfig | None = None,
    ):
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version must identify the sampling snapshot.")
        self.client, self.tokenizer = sampling_client, tokenizer
        self.policy_version = policy_version
        self.config = config or SWESamplingConfig()

    @classmethod
    async def create(
        cls,
        model: str,
        *,
        model_path: str | None = None,
        config: SWESamplingConfig | None = None,
    ):
        service = await asyncio.to_thread(trio.ServiceClient)
        client = await call_async(
            service, "create_sampling_client", base_model=model, model_path=model_path
        )
        tokenizer = await asyncio.to_thread(client.get_tokenizer)
        return cls(
            client,
            tokenizer,
            policy_version=model_path or f"base:{model}",
            config=config,
        )

    async def sample(self, messages: list[dict], tools: list[dict]) -> SampledTurn:
        config = self.config
        rendered = await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            **config.chat_template_kwargs,
        )
        prompt = _token_list(rendered, source="SWE chat template")
        if any(t < 0 for t in prompt):
            raise ValueError("The SWE template returned negative token IDs.")
        if len(prompt) + config.max_tokens > config.max_context_tokens:
            raise ValueError(
                "SWE context budget exceeded; history is never silently truncated."
            )
        response = await call_async(
            self.client,
            "sample",
            prompt=trio.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=trio.SamplingParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
            ),
        )
        if len(response.sequences) != 1:
            raise ValueError("SWE sampling requires exactly one response.")
        sequence = response.sequences[0]
        tokens = tuple(sequence.tokens)
        text = await asyncio.to_thread(
            self.tokenizer.decode, tokens, skip_special_tokens=False
        )
        reason = sequence.stop_reason
        truncated = reason in ("length", "max_tokens", "max_length")
        if reason not in (
            "stop",
            "eos",
            "stop_token",
            "length",
            "max_tokens",
            "max_length",
        ):
            truncated = len(tokens) >= config.max_tokens
        return SampledTurn(
            tuple(prompt),
            tokens,
            tuple(sequence.logprobs),
            text,
            self.policy_version,
            truncated,
        )

    @property
    def supports_ppo(self) -> bool:
        """Match PPO's existing unmodified-distribution logprob requirement."""
        return self.config.temperature == 1.0 and self.config.top_p == 1.0
