"""Score a fixed completion with a PyTRIO sampling client.

Sampling and scoring are different operations. Sampling chooses new tokens,
whereas ``compute_logprobs`` evaluates tokens that have already been chosen.
Several algorithms need the latter operation:

* DPO sums reference-policy token logprobs into one sequence score.
* OPD compares student and teacher logprobs token by token.
* PPO can use token logprobs from a reference or proximal policy for KL
  rewards, clipping, and policy-drift diagnostics.

PyTRIO scores the complete ``prompt + completion`` sequence, but these
algorithms consume only completion positions. This module centralizes that
boundary::

    prompt tokens + completion tokens
                    |
                    v
             PyTRIO ModelInput
                    |
                    v
            compute_logprobs()
                    |
                    v
       full-sequence scores -> completion-only scores

The module deliberately does not sum scores, calculate KL penalties, or build
advantages. Those transformations have algorithm-specific meanings and stay
in their respective trainer modules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

from ..api.client_api import call, call_async
from .datums import materialize_model_input


def _tokens(values: Sequence[int], *, source: str) -> list[int]:
    """Copy and validate one model-token sequence.

    Args:
        values: Token IDs supplied by an algorithm-specific caller.
        source: Human-readable input name used in validation errors.

    Returns:
        A defensive list of ordinary Python integers.

    Raises:
        TypeError: If ``values`` is not a token sequence.
        ValueError: If the sequence is empty or contains an invalid token ID.
    """

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{source} tokens must be a sequence.")
    tokens = list(values)
    if not tokens or any(
        not isinstance(token, Integral) or isinstance(token, bool) or token < 0
        for token in tokens
    ):
        raise ValueError(
            f"{source} tokens must contain nonnegative integer token IDs."
        )
    return [int(token) for token in tokens]


def completion_logprobs(
    values: Sequence[float | None],
    *,
    prompt_length: int,
    completion_length: int,
    source: str = "Scored",
) -> list[float]:
    """Extract finite completion scores from a full-sequence SDK response.

    PyTRIO returns one position for every input token and may use ``None`` for
    the first, unscored token.  Prompt positions are intentionally ignored;
    every completion position must contain a finite, nonpositive logprob.

    Args:
        values: Full-sequence scores returned by ``compute_logprobs``.
        prompt_length: Number of prompt tokens at the start of the sequence.
        completion_length: Number of completion tokens following the prompt.
        source: Policy/model label used in validation errors.

    Returns:
        Completion-only token logprobs in their original order.

    Raises:
        TypeError: If ``values`` is not a sequence.
        ValueError: If lengths do not align or a completion score is missing,
            non-finite, or positive.
    """

    # Lengths define the only valid boundary between prompt and completion.
    # Validate them before slicing so malformed callers cannot silently shift
    # scores onto the wrong target tokens.
    for name, length, minimum in (
        ("prompt_length", prompt_length, 1),
        ("completion_length", completion_length, 1),
    ):
        if (
            not isinstance(length, Integral)
            or isinstance(length, bool)
            or length < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{source} logprobs must be a sequence.")

    # Copy the SDK result before checking it. Callers may pass a mutable list,
    # but scoring must be deterministic for the duration of this operation.
    scores = list(values)
    expected = int(prompt_length) + int(completion_length)
    if len(scores) != expected:
        raise ValueError(
            f"{source} logprobs must align with the full prompt plus completion."
        )

    # Prompt scores are conditioning context, not training targets. Only the
    # suffix associated with generated completion tokens is returned.
    completion = scores[int(prompt_length) :]
    if any(
        value is None
        or not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) > 0
        for value in completion
    ):
        raise ValueError(
            f"{source} completion must contain finite, nonpositive logprobs."
        )
    return [float(value) for value in completion if value is not None]


def score_completion(
    client: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    *,
    model_input_factory: Callable[[Sequence[int]], Any] | None = None,
    source: str = "Scored",
) -> list[float]:
    """Synchronously score one fixed completion without sampling new tokens.

    Args:
        client: Sampling client exposing ``compute_logprobs``.
        prompt_tokens: Conditioning token IDs.
        completion_tokens: Existing completion token IDs to evaluate.
        model_input_factory: Optional ``ModelInput`` constructor. The default
            lazily imports PyTRIO; tests can inject a lightweight factory.
        source: Policy/model label used in response-validation errors.

    Returns:
        One logprob for each completion token.
    """

    # Validate both segments independently so an empty or malformed completion
    # is rejected before any remote request is submitted.
    prompt = _tokens(prompt_tokens, source="Prompt")
    completion = _tokens(completion_tokens, source="Completion")

    # The scoring API requires the exact full causal context. Keeping prompt
    # and completion separate until here makes the later slicing boundary
    # explicit and prevents off-by-one alignment errors.
    model_input = materialize_model_input(
        prompt + completion,
        model_input_factory=model_input_factory,
    )
    values = call(client, "compute_logprobs", model_input)

    # Remove prompt positions while preserving one-to-one alignment between
    # the returned scores and ``completion_tokens``.
    return completion_logprobs(
        values,
        prompt_length=len(prompt),
        completion_length=len(completion),
        source=source,
    )


async def score_completion_async(
    client: Any,
    prompt_tokens: Sequence[int],
    completion_tokens: Sequence[int],
    *,
    model_input_factory: Callable[[Sequence[int]], Any] | None = None,
    source: str = "Scored",
) -> list[float]:
    """Asynchronously score one fixed completion without blocking the loop.

    This is the asynchronous counterpart of :func:`score_completion`; input
    validation and output alignment are intentionally identical.

    Args:
        client: Sampling client exposing ``compute_logprobs_async`` or a
            compatible synchronous fallback.
        prompt_tokens: Conditioning token IDs.
        completion_tokens: Existing completion token IDs to evaluate.
        model_input_factory: Optional ``ModelInput`` constructor.
        source: Policy/model label used in response-validation errors.

    Returns:
        One logprob for each completion token.
    """

    # Perform local validation before yielding control to a remote request.
    prompt = _tokens(prompt_tokens, source="Prompt")
    completion = _tokens(completion_tokens, source="Completion")

    # Score exactly the same causal sequence as the synchronous path.
    model_input = materialize_model_input(
        prompt + completion,
        model_input_factory=model_input_factory,
    )
    values = await call_async(client, "compute_logprobs", model_input)

    # Apply the shared alignment checks after resolving the remote response.
    return completion_logprobs(
        values,
        prompt_length=len(prompt),
        completion_length=len(completion),
        source=source,
    )


__all__ = [
    "completion_logprobs",
    "score_completion",
    "score_completion_async",
]
