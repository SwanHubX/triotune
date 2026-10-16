"""Algorithm-neutral PyTRIO Datum materialization.

Algorithms still calculate their own token alignment and loss fields. This
module only turns aligned token IDs and a loss-input mapping into the SDK
objects shared by SFT, DPO, GRPO, and OPD.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _default_model_input(tokens: Sequence[int]) -> Any:
    """Create a PyTRIO ModelInput lazily so unit tests can use fake factories."""

    import pytrio as trio

    return trio.ModelInput.from_ints(list(tokens))


def _default_datum(model_input: Any, loss_fn_inputs: Mapping[str, Any]) -> Any:
    """Create the installed PyTRIO Datum class."""

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
    """

    if model_input_factory is not None:
        model_input = model_input_factory(model_input)
    elif isinstance(model_input, (list, tuple)):
        model_input = _default_model_input(model_input)
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

    The caller supplies already aligned ``input_ids`` and targets. This helper
    only removes the final input position; prompt masks and algorithm-specific
    auxiliary fields remain the caller's responsibility.
    """

    tokens = list(input_ids)
    targets = list(target_tokens)
    if len(tokens) < 2:
        raise ValueError("input_ids must contain at least two tokens.")
    if len(targets) != len(tokens) - 1:
        raise ValueError("target_tokens must align with input_ids[:-1].")
    return materialize_datum(
        tokens[:-1],
        {"target_tokens": targets},
        model_input_factory=model_input_factory,
        datum_factory=datum_factory,
    )
