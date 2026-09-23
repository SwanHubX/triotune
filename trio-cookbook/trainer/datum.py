"""GRPO token alignment and remote Datum construction.

The SDK materialization helper is small, but the required fields and one-token
shift are specific to GRPO policy learning, so this module intentionally stays
under ``trainer`` rather than being presented as a generic data utility.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .types import GRPODatum, SequenceSample


def build_aligned_datum(
    prompt_tokens: list[int],
    sample: SequenceSample,
    advantage: float,
    *,
    prompt_id: str,
    sample_id: int,
    model_input_builder: Callable[[list[int]], Any] | None = None,
) -> GRPODatum:
    """Construct the exact one-token-shifted representation expected by GRPO.

    TRL computes a mask from local logits.  PyTRIO's policy loss instead reads
    aligned ``target_tokens``, old ``logprobs`` and ``advantages`` from the
    Datum, so prompt positions receive zero values and only completion tokens
    carry the sequence advantage.
    """

    if not prompt_tokens:
        raise ValueError("prompt_tokens must contain at least one token.")
    if not sample.tokens:
        raise ValueError("Empty completions cannot contribute to GRPO.")
    if len(sample.tokens) != len(sample.logprobs):
        raise ValueError("Completion tokens and behavior logprobs must have equal length.")
    if not math.isfinite(float(advantage)):
        raise ValueError("Advantages must be finite.")
    model_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * (len(prompt_tokens) - 1) + sample.tokens
    old_logprobs = [0.0] * (len(prompt_tokens) - 1) + sample.logprobs
    advantages = [0.0] * (len(prompt_tokens) - 1) + [float(advantage)] * len(sample.tokens)
    n = len(model_tokens)
    if not len(target_tokens) == len(old_logprobs) == len(advantages) == n:
        raise AssertionError("PyTRIO GRPO fields lost token alignment.")
    if any(not math.isfinite(value) for value in old_logprobs + advantages):
        raise ValueError("logprobs and advantages must be finite.")
    return GRPODatum(
        model_input=model_input_builder(model_tokens) if model_input_builder else model_tokens,
        target_tokens=target_tokens,
        logprobs=old_logprobs,
        advantages=advantages,
        prompt_id=prompt_id,
        sample_id=sample_id,
    )


def validate_datum(datum: GRPODatum) -> None:
    n = len(datum.model_input) if hasattr(datum.model_input, "__len__") else len(datum.target_tokens)
    fields = datum.loss_fn_inputs
    if any(len(values) != n for values in fields.values()):
        raise ValueError("model_input and all PyTRIO loss_fn_inputs must have equal lengths.")
    if any(not math.isfinite(float(value)) for value in datum.logprobs + datum.advantages):
        raise ValueError("Non-finite values are not valid PyTRIO loss inputs.")


def to_client_datum(datum: GRPODatum, factory: Callable[..., Any] | None = None) -> Any:
    """Materialize a SDK Datum without importing tinker/trl at module import."""

    validate_datum(datum)
    inputs = datum.loss_fn_inputs
    if factory is not None:
        return factory(model_input=datum.model_input, loss_fn_inputs=inputs)
    # Installed PyTRIO packages can provide their own public Datum class.  The
    # fallback dataclass remains useful for fake clients and unit tests.
    for module_name in ("pytrio", "tinker"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        cls = getattr(module, "Datum", None)
        if cls is not None:
            return cls(model_input=datum.model_input, loss_fn_inputs=inputs)
    return datum
