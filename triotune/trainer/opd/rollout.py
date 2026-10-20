"""Student on-policy rollout phase for OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...api.client_api import call_async, sdk_object
from .contracts import OPDPrompt, OPDRollout

# ---------------------------------------------------------------------------
# SDK response compatibility
# ---------------------------------------------------------------------------


def _field(value: Any, name: str, default: Any = None) -> Any:
    # PyTRIO-compatible test clients and SDK versions may return mappings or
    # attribute objects. Normalize access without leaking that variation later.
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


# ---------------------------------------------------------------------------
# Student rollout phase
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OPDRolloutPhase:
    """Sample and validate every completion for one rendered prompt."""

    async def sample_prompt(
        self,
        *,
        sampler: Any,
        prompt: OPDPrompt,
        prompt_index: int,
        behavior_policy_version: int,
        config: Any,
        seed_offset: int,
        sampling_params_factory: Any = None,
        model_input_factory: Any = None,
    ) -> list[OPDRollout]:
        """Return a complete prompt group before teacher scoring begins."""

        # Reserve the prompt portion of the total sequence budget before asking
        # the remote sampler for completion tokens.
        max_tokens = config.max_completion_length
        if config.max_length is not None:
            max_tokens = min(max_tokens, config.max_length - len(prompt.tokens))
        seed = config.seed
        if seed is not None:
            # A monotonically increasing offset avoids replaying the same random
            # stream for a prompt position in later optimizer steps.
            seed += seed_offset + prompt_index
        # OPD samples from the full student distribution. Top-p/top-k filtering
        # would change the behavior policy represented by returned logprobs.
        params = sdk_object(
            "SamplingParams",
            {
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "seed": seed,
                "stop": config.stop,
            },
            sampling_params_factory,
        )
        prompt_input = (
            model_input_factory(prompt.tokens)
            if model_input_factory is not None
            else prompt.tokens
        )
        response = await call_async(
            sampler,
            "sample",
            prompt=prompt_input,
            num_samples=config.num_generations,
            sampling_params=params,
        )
        # Accept both names used by supported sampling-client response shapes.
        sequences = _field(response, "sequences")
        if sequences is None:
            sequences = _field(response, "samples")
        if sequences is None:
            raise TypeError("Student sampler returned no sequences/samples.")
        sequences = list(sequences)
        if len(sequences) != config.num_generations:
            raise ValueError(
                "Student sampler returned an unexpected number of completions."
            )

        # Constructing all contracts here validates the complete response before
        # any teacher request for this prompt is submitted.
        rollouts = []
        for sample_index, sequence in enumerate(sequences):
            tokens = _field(sequence, "tokens")
            if tokens is None:
                tokens = _field(sequence, "completion_tokens")
            logprobs = _field(sequence, "logprobs")
            if tokens is None:
                raise ValueError("Student completion must contain token IDs.")
            if logprobs is None:
                raise ValueError("Student sampling must return completion logprobs.")
            reason = _field(sequence, "stop_reason")
            reason = getattr(reason, "value", getattr(reason, "name", reason))
            reason = str(reason).lower() if reason is not None else ""
            # The contract copies and validates the values before any teacher
            # request can observe this prompt group.
            rollout = OPDRollout(
                prompt_tokens=prompt.tokens,
                completion_tokens=tokens,
                behavior_logprobs=logprobs,
                prompt_index=prompt_index,
                sample_index=sample_index,
                behavior_policy_version=behavior_policy_version,
                # Routing is framework metadata: the student rollout cannot
                # alter which teacher group will score its sampled actions.
                teacher_group=prompt.teacher_group,
                truncated=(
                    reason in ("length", "max_tokens", "max_length")
                    or reason.endswith("_length")
                    or (not reason and len(tokens) >= max_tokens)
                ),
            )
            if len(rollout.completion_tokens) > max_tokens:
                raise ValueError(
                    "Student completion exceeds the requested token budget."
                )
            rollouts.append(rollout)
        return rollouts


__all__ = ["OPDRolloutPhase"]
