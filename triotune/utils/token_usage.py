"""Token accounting at the PyTRIO request boundary.

The helpers in this module deliberately count tokens rather than estimating
money.  PyTRIO exposes sampling input/output usage, but its public model
catalog and response types do not expose prices or cached-input usage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any


def _nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _model_input_length(datum: Any) -> int:
    model_input = getattr(datum, "model_input", None)
    length = getattr(model_input, "length", None)
    if length is None:
        try:
            length = len(model_input)
        except TypeError as error:
            raise TypeError("Training datums must expose model_input.length.") from error
    return _nonnegative_int(length, name="model_input.length")


def compute_train_tokens(datums: Iterable[Any]) -> int:
    """Count model-input tokens submitted in one training compute request."""

    return sum(_model_input_length(datum) for datum in datums)


@dataclass(frozen=True, slots=True)
class SampleTokens:
    """Input and generated token usage for one sampling request."""

    input_tokens: int
    output_tokens: int


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def compute_sample_tokens(
    response: Any,
    *,
    prompt_tokens: int | None = None,
) -> SampleTokens:
    """Read sampling usage, with tokenized-text fallbacks for compatible clients.

    PyTRIO 0.2.9 returns ``input_tokens`` and ``output_tokens`` directly.  The
    fallback keeps injected clients and older compatible response shapes
    measurable without inspecting a model name or maintaining a price table.
    """

    fallback_input = (
        None
        if prompt_tokens is None
        else _nonnegative_int(prompt_tokens, name="prompt_tokens")
    )
    raw_input = _field(response, "input_tokens")
    if raw_input is None or (raw_input == 0 and fallback_input):
        if fallback_input is None:
            raise ValueError(
                "Sampling response has no input_tokens; prompt_tokens is required."
            )
        input_tokens = fallback_input
    else:
        input_tokens = _nonnegative_int(raw_input, name="response.input_tokens")

    raw_output = _field(response, "output_tokens")
    if raw_output is None:
        sequences = _field(response, "sequences")
        if sequences is None:
            sequences = _field(response, "samples")
        if sequences is None:
            raise ValueError("Sampling response has no output token information.")
        output_tokens = 0
        for sequence in sequences:
            tokens = _field(sequence, "tokens")
            if tokens is None:
                tokens = _field(sequence, "completion_tokens")
            if tokens is None:
                raise ValueError("Sampling sequence has no token IDs.")
            output_tokens += len(tokens)
    else:
        output_tokens = _nonnegative_int(raw_output, name="response.output_tokens")

    return SampleTokens(input_tokens, output_tokens)


SampleUsageCallback = Callable[[SampleTokens], None]


__all__ = [
    "SampleTokens",
    "SampleUsageCallback",
    "compute_sample_tokens",
    "compute_train_tokens",
]
