"""Data contracts shared by the GRPO modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SequenceSample:
    """One text completion returned by a PyTRIO sampler."""

    tokens: list[int]
    logprobs: list[float]
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GRPOBatch:
    """A prompt and its group of completions under one behavior policy."""

    prompt_id: str
    prompt: Any
    prompt_tokens: list[int]
    samples: list[SequenceSample]
    row: dict[str, Any] = field(default_factory=dict)
    behavior_policy_version: int = 0
    sampling_prompt: Any | None = None


@dataclass(slots=True)
class GRPODatum:
    """SDK-neutral representation of the aligned PyTRIO policy datum."""

    model_input: Any
    target_tokens: list[int]
    logprobs: list[float]
    advantages: list[float]
    prompt_id: str
    sample_id: int

    @property
    def loss_fn_inputs(self) -> dict[str, list[float] | list[int]]:
        return {
            "target_tokens": self.target_tokens,
            "logprobs": self.logprobs,
            "advantages": self.advantages,
        }


@dataclass(slots=True)
class GRPOResult:
    """Result and metrics for one optimizer step."""

    step: int
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    num_datums: int = 0
    skipped_groups: int = 0
    checkpoint_path: str | None = None


def as_prompt_text(prompt: Any) -> str:
    """Convert a text prompt to a reward-facing string.

    Multimodal message structures are intentionally rejected instead of being
    silently stringified, which would make reward and token alignment wrong.
    """

    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (bytes, bytearray)):
        return bytes(prompt).decode("utf-8")
    if isinstance(prompt, dict):
        if isinstance(prompt.get("text"), str):
            return prompt["text"]
        if isinstance(prompt.get("content"), str):
            return prompt["content"]
        # Text-only mode deliberately drops image/audio/video content instead
        # of routing it into a tokenizer that cannot represent it.
        if "content" in prompt and isinstance(prompt["content"], list):
            return "".join(
                part.get("text", "")
                for part in prompt["content"]
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if "messages" in prompt:
            return as_prompt_text(prompt["messages"])
    if isinstance(prompt, (list, tuple)) and prompt and all(isinstance(item, dict) for item in prompt):
        return "\n".join(as_prompt_text(item) for item in prompt)
    if isinstance(prompt, (list, tuple)) and all(isinstance(token, int) for token in prompt):
        return ""
    raise TypeError("A prompt must be text or a token sequence; multimodal messages are unsupported.")
