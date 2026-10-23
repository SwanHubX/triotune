"""Adapt fixed PyTRIO policies to the agent's token-level sampling boundary.

This module translates SDK requests and responses only. Snapshot creation and
refresh belong to the trainer; episode budgets and deadlines belong to the
runner. Keeping those lifecycles outside the sampler allows one policy snapshot
to serve every turn and task group collected before the next optimizer update.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytrio as trio

from ...api.client_api import call_async
from ...utils.token_usage import compute_sample_tokens
from ..contracts import AgentContractError, AgentSampler, Generation, validate_tokens


class PyTRIOAgentSampler:
    """Adapt an existing immutable sampling client; never refresh it mid-episode.

    Sampling stays at temperature=top_p=1 so returned behavior logprobs match
    the target-logprob contract used by Triotune's PPO implementation.
    """

    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        *,
        stop: str | Sequence[str] | Sequence[int] | None = None,
    ):
        """Wrap a sampling client and its tokenizer without changing its weights.

        ``stop`` is forwarded to PyTRIO as a string, string sequence, or token-ID
        sequence. The caller must use stop conventions compatible with its
        parser and chat template, especially when stop markers are not returned.
        """
        self.client, self.tokenizer, self.stop = client, tokenizer, stop

    async def generate(
        self, prompt_tokens: Sequence[int], *, max_tokens: int, seed: int | None = None
    ) -> Generation:
        """Sample one assistant response and validate its training provenance.

        Args:
            prompt_tokens: Exact serialized context, including observations.
            max_tokens: Maximum generated tokens for this request.
            seed: Optional backend seed selected by the collector and runner.

        Returns:
            Original sampled IDs, aligned behavior scores, and request usage.

        Raises:
            AgentContractError: The backend returns multiple sequences, omits
                behavior scores, exceeds the budget, or violates Generation's
                token-alignment requirements.

        Transport errors are deliberately left for the runner to classify.
        """
        validate_tokens(prompt_tokens, name="prompt_tokens")
        result = await call_async(
            self.client,
            "sample",
            prompt=trio.ModelInput.from_ints(list(prompt_tokens)),
            # Grouped exploration is expanded into independent episodes by the
            # collector; a single turn must have exactly one continuation.
            num_samples=1,
            sampling_params=trio.SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=1.0,
                seed=seed,
                stop=self.stop,
            ),
        )
        if len(result.sequences) != 1:
            raise AgentContractError(
                "An agent request must return exactly one sequence."
            )
        sequence = result.sequences[0]
        if sequence.logprobs is None:
            raise AgentContractError(
                "Agent sampling requires original behavior logprobs."
            )
        tokens = tuple(sequence.tokens)
        if len(tokens) > max_tokens:
            raise AgentContractError("Sampler exceeded the requested output budget.")
        # Decode the original tokens: never trust response text as token identity.
        text = await asyncio.to_thread(
            self.tokenizer.decode, list(tokens), skip_special_tokens=True
        )
        usage = compute_sample_tokens(result, prompt_tokens=len(prompt_tokens))
        reason = sequence.stop_reason
        # Compatible clients may use backend-specific termination labels. Fall
        # back to the observed length when no recognized label is available.
        if reason not in {
            "stop",
            "eos",
            "stop_token",
            "length",
            "max_tokens",
            "max_length",
        }:
            reason = "length" if len(tokens) >= max_tokens else "stop"
        return Generation(
            tokens,
            tuple(sequence.logprobs),
            text,
            reason,
            usage.input_tokens,
            usage.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class SamplingSnapshot:
    """One fixed policy shared by all episodes in a collected batch.

    snapshot_id is provenance, not a server-side staleness guarantee. The
    training loop must finish collection before mutating actor weights.

    ``sampler`` encapsulates the actual fixed policy and ``tokenizer`` supplies
    the default conversation renderer. Freezing this wrapper does not freeze
    arbitrary adapter internals; providers remain responsible for policy
    immutability throughout collection.
    """

    snapshot_id: str
    sampler: AgentSampler
    tokenizer: Any

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ValueError("snapshot_id must be a nonempty string.")
