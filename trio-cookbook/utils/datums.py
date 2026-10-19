"""Materialize algorithm-prepared data as PyTRIO SDK objects.

This module is the final conversion boundary between local trainer data and
PyTRIO requests. It creates two SDK objects:

* ``ModelInput`` contains the token sequence consumed by the model.
* ``Datum`` combines that model input with aligned ``loss_fn_inputs``.

Algorithms remain responsible for deciding what each loss field means. For
example, SFT supplies labels, DPO supplies masked labels, and GRPO/PPO supply
behavior logprobs and advantages. This module intentionally does not create
masks, advantages, rewards, or algorithm-specific loss fields.

The optional factories serve two purposes: tests can inject lightweight fake
objects, and compatibility callers can provide constructors from a different
SDK version without changing trainer code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _default_model_input(tokens: Sequence[int]) -> Any:
    """Create the installed PyTRIO ``ModelInput`` from integer tokens.

    PyTRIO is imported inside the function so modules that only inspect local
    contracts do not need to initialize the SDK at import time.
    """

    import pytrio as trio

    return trio.ModelInput.from_ints(list(tokens))


def materialize_model_input(
    tokens: Sequence[int] | Any,
    *,
    model_input_factory: Callable[[Sequence[int]], Any] | None = None,
) -> Any:
    """Materialize token IDs as a PyTRIO ``ModelInput`` when needed.

    With no explicit factory, lists and tuples are converted through the
    installed PyTRIO SDK, while an existing model-input object passes through
    unchanged. An explicit factory is always applied and therefore owns all
    compatibility conversion for that call.

    Keeping this operation in one place gives training datums and
    inference-only scoring the same conversion semantics.

    Args:
        tokens: Raw token IDs or an already materialized model-input object.
        model_input_factory: Optional constructor used instead of the installed
            PyTRIO ``ModelInput.from_ints`` implementation.

    Returns:
        A materialized model-input object, or the original object when it does
        not require conversion.
    """

    # An injected factory takes precedence so tests and compatibility adapters
    # observe exactly the same input that the real SDK constructor would see.
    if model_input_factory is not None:
        return model_input_factory(tokens)

    # Plain token containers use the default PyTRIO representation. Other
    # objects are assumed to already satisfy the SDK model-input contract.
    if isinstance(tokens, (list, tuple)):
        return _default_model_input(tokens)
    return tokens


def _default_datum(model_input: Any, loss_fn_inputs: Mapping[str, Any]) -> Any:
    """Create the installed PyTRIO ``Datum`` with copied loss inputs."""

    import pytrio as trio

    return trio.Datum(model_input=model_input, loss_fn_inputs=dict(loss_fn_inputs))


def materialize_datum(
    model_input: Sequence[int] | Any,
    loss_fn_inputs: Mapping[str, Any],
    *,
    model_input_factory: Callable[[Sequence[int]], Any] | None = None,
    datum_factory: Callable[..., Any] | None = None,
) -> Any:
    """Materialize one aligned PyTRIO Datum through optional factories.

    ``model_input`` may already be an SDK ``ModelInput``. When a token
    sequence is supplied, the default PyTRIO conversion is used. Factories
    keep this helper usable with fake clients and avoid importing PyTRIO in
    modules that only inspect the contract.

    Args:
        model_input: Raw model token IDs or an existing SDK model-input object.
        loss_fn_inputs: Already aligned fields consumed by the selected loss,
            such as ``target_tokens``, ``logprobs``, or ``advantages``.
        model_input_factory: Optional constructor for the model-input object.
        datum_factory: Optional constructor for the final datum object.

    Returns:
        A PyTRIO-compatible datum containing the materialized model input and a
        defensive copy of ``loss_fn_inputs``.
    """

    # Normalize the model-facing half first. This keeps SDK construction out of
    # algorithm modules and gives scoring and training identical input objects.
    model_input = materialize_model_input(
        model_input,
        model_input_factory=model_input_factory,
    )

    # Never retain the caller's mapping directly: later caller mutations must
    # not change a datum that has already been prepared for a remote request.
    values = dict(loss_fn_inputs)
    if datum_factory is not None:
        return datum_factory(model_input=model_input, loss_fn_inputs=values)
    return _default_datum(model_input, values)


def causal_datum(
    input_ids: Sequence[int],
    target_tokens: Sequence[int],
    *,
    model_input_factory: Callable[[Sequence[int]], Any] | None = None,
    datum_factory: Callable[..., Any] | None = None,
) -> Any:
    """Build the standard one-token-shifted causal-language-model Datum.

    The caller supplies a complete token sequence and targets that are already
    aligned to next-token prediction. For example::

        input_ids     = [token_0, token_1, token_2]
        model_input   = [token_0, token_1]
        target_tokens = [label_1, label_2]

    This helper removes only the final input token. It does not shift targets,
    create prompt masks, or add algorithm-specific auxiliary fields.

    Args:
        input_ids: Complete causal token sequence, including the final target.
        target_tokens: Pre-aligned next-token labels. Its length must equal
            ``len(input_ids) - 1``; ``-100`` may be used by callers as an
            ignored label where the selected loss supports it.
        model_input_factory: Optional constructor for the model-input object.
        datum_factory: Optional constructor for the final datum object.

    Returns:
        A datum whose model input and targets have identical logical lengths.

    Raises:
        ValueError: If the sequence cannot form a causal pair or target lengths
            do not match the one-token shift.
    """

    # Copy both sequences so validation and materialization cannot be affected
    # by mutations to caller-owned containers.
    tokens = list(input_ids)
    targets = list(target_tokens)
    if len(tokens) < 2:
        raise ValueError("input_ids must contain at least two tokens.")
    if len(targets) != len(tokens) - 1:
        raise ValueError("target_tokens must align with input_ids[:-1].")

    # Causal language modeling predicts every target from the token immediately
    # before it, so the final token is a label but never a model-input position.
    return materialize_datum(
        tokens[:-1],
        {"target_tokens": targets},
        model_input_factory=model_input_factory,
        datum_factory=datum_factory,
    )
