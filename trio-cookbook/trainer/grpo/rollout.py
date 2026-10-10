"""Text-only GRPO rollout adapter for ``SamplingClient``.

It returns grouped completions and behavior logprobs, which are GRPO contracts;
the lower-level future handling it uses lives in ``utils.client``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ...utils.client import call, call_async, sampling_params
from ..common.prompts import as_prompt_text
from .types import GRPOBatch, SequenceSample


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence_values(result: Any) -> list[Any]:
    sequences = _field(result, "sequences")
    if sequences is None:
        sequences = _field(result, "samples")
    if sequences is None and isinstance(result, (list, tuple)):
        sequences = result
    if sequences is None:
        raise TypeError("SamplingClient.sample() returned no sequences/samples.")
    return list(sequences)


def parse_sample(raw: Any, *, prompt_tokens: list[int], decode: Callable[[list[int]], str] | None = None) -> SequenceSample:
    tokens = _field(raw, "tokens")
    if tokens is None:
        tokens = _field(raw, "completion_tokens")
    if tokens is None:
        raise ValueError("Each sampled sequence must contain completion tokens.")
    tokens = [int(token) for token in tokens]
    logprobs = _field(raw, "logprobs")
    if logprobs is None:
        raise ValueError("PyTRIO sampling must return behavior logprobs for GRPO.")
    logprobs = [float(value) for value in logprobs]
    # Some APIs return log P(next token) for the complete prompt+completion.
    # Keep only completion positions rather than silently changing alignment.
    if len(logprobs) == len(prompt_tokens) + len(tokens):
        logprobs = logprobs[-len(tokens) :]
    if len(tokens) != len(logprobs):
        raise ValueError(f"Sample has {len(tokens)} tokens but {len(logprobs)} logprobs.")
    text = _field(raw, "text")
    if text is None:
        text = _field(raw, "completion")
    if text is None and decode is not None:
        text = decode(tokens)
    return SequenceSample(tokens=tokens, logprobs=logprobs, text=text, metadata={"raw": raw})


class SamplingAdapter:
    def __init__(self, client: Any, *, params_factory: Any = None, decode: Callable[[list[int]], str] | None = None, stop: list[str] | None = None):
        self.client = client
        self.params_factory = params_factory
        self.decode = decode
        self.stop = stop

    def sample(self, group: GRPOBatch, config: Any) -> GRPOBatch:
        prompt = group.sampling_prompt if group.sampling_prompt is not None else group.prompt
        # Trainer-created groups carry a normalized text prompt. Keep this
        # assertion for callers that construct GRPOBatch directly.
        as_prompt_text(group.prompt)
        raw = call(
            self.client,
            "sample",
            prompt=prompt,
            num_samples=config.num_generations,
            sampling_params=sampling_params(config, self.params_factory, stop=self.stop),
        )
        samples = [parse_sample(value, prompt_tokens=group.prompt_tokens, decode=self.decode) for value in _sequence_values(raw)]
        if len(samples) != config.num_generations:
            raise ValueError(f"SamplingClient returned {len(samples)} samples; expected {config.num_generations}.")
        return GRPOBatch(group.prompt_id, group.prompt, group.prompt_tokens, samples, group.row, group.behavior_policy_version, group.sampling_prompt)

    async def sample_async(self, group: GRPOBatch, config: Any) -> GRPOBatch:
        # Trainer-created groups carry a normalized text prompt. Keep this
        # assertion for callers that construct GRPOBatch directly.
        as_prompt_text(group.prompt)
        raw = await call_async(
            self.client,
            "sample",
            prompt=group.sampling_prompt if group.sampling_prompt is not None else group.prompt,
            num_samples=config.num_generations,
            sampling_params=sampling_params(config, self.params_factory, stop=self.stop),
        )
        samples = [parse_sample(value, prompt_tokens=group.prompt_tokens, decode=self.decode) for value in _sequence_values(raw)]
        if len(samples) != config.num_generations:
            raise ValueError(f"SamplingClient returned {len(samples)} samples; expected {config.num_generations}.")
        return GRPOBatch(group.prompt_id, group.prompt, group.prompt_tokens, samples, group.row, group.behavior_policy_version, group.sampling_prompt)
