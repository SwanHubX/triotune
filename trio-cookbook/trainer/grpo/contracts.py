"""Data contracts exchanged across the GRPO rollout and update pipeline."""

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
    # This is the exact rendered text sent to reward functions. Conversational
    # inputs are rendered with the tokenizer's chat template before a batch is
    # created, so rollout and training use the same token sequence.
    prompt: str
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
        """Return fields consumed by PyTRIO's importance-sampling loss."""

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


@dataclass(slots=True)
class TrainerState:
    """Mutable local state tracked across GRPO training steps."""

    # Number of completed trainer steps.
    global_step: int = 0
    # Version of the behavior policy used to generate the next rollout batch.
    behavior_policy_version: int = 0
    # Metrics collected during the most recently completed trainer step.
    metrics: dict[str, float] = field(default_factory=dict)
    # URI of the most recently saved weights and optimizer state.
    checkpoint_path: str | None = None
