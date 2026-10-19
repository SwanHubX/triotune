"""Common contracts for PyTRIO built-in training objectives.

This module is the SDK-neutral boundary between an algorithm and a PyTRIO
``forward``/``forward_backward`` request.  It deliberately does not calculate
rewards, advantages, masks, or any loss value: rollout and algorithm modules
prepare an already next-token-aligned sequence, while PyTRIO owns the actual
forward and backward implementation.

The contract has three layers:

* :class:`LossSpec` names one concrete PyTRIO loss and validates its options;
* ``*LossInput`` classes validate the aligned fields required by that loss;
* :class:`BuiltinPyTrioObjective` turns the spec into the common remote-call
  keyword arguments used by SFT, GRPO, and OPD.

``custom_ce`` is retained as a PyTRIO compatibility name.  It is a CE-family
loss, but it is not one of the four policy objectives listed in the public
cookbook configuration.  DPO remains a custom objective and is intentionally
outside this built-in loss registry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal, Protocol, TypeAlias

# ---------------------------------------------------------------------------
# Loss names and field families
# ---------------------------------------------------------------------------


LossName: TypeAlias = Literal[
    "cross_entropy",
    "custom_ce",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
]
LossFamily: TypeAlias = Literal["supervised", "policy"]

# Keep these sets in sync with playground/Trio-Train/api/loss_types.py.  Sets
# are intentionally immutable: callers should select a loss through LossSpec,
# rather than mutate a process-wide registry.
CE_LOSS_FNS: frozenset[str] = frozenset({"cross_entropy", "custom_ce"})
POLICY_LOSS_FNS: frozenset[str] = frozenset(
    {"importance_sampling", "ppo", "cispo", "dro"}
)
BUILTIN_LOSS_FNS: frozenset[str] = CE_LOSS_FNS | POLICY_LOSS_FNS

_POLICY_FIELDS = frozenset({"target_tokens", "logprobs", "advantages"})
_SUPERVISED_FIELDS = frozenset({"target_tokens"})
_OPTIONAL_FIELDS = {
    "supervised": frozenset({"weights"}),
    "policy": frozenset(),
}
_CONFIG_FIELDS = {
    "cross_entropy": frozenset(),
    "custom_ce": frozenset(),
    "importance_sampling": frozenset(),
    "ppo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
    "cispo": frozenset({"clip_low_threshold", "clip_high_threshold"}),
    "dro": frozenset({"beta"}),
}
_DEFAULT_CONFIG = {
    "ppo": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    "cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0},
    "dro": {"beta": 0.05},
}


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _materialize_sequence(value: Any, *, field_name: str) -> list[Any]:
    """Copy a one-dimensional sequence and unwrap SDK tensor-data values."""

    if hasattr(value, "data") and not isinstance(value, (list, tuple)):
        value = value.data
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a one-dimensional sequence.")
    values = list(value)
    if any(isinstance(item, (list, tuple)) for item in values):
        raise TypeError(f"{field_name} must be one-dimensional.")
    return values


def _token_sequence(value: Any, *, field_name: str, allow_ignore: bool) -> list[int]:
    """Validate token IDs while permitting PyTRIO's ``-100`` ignore label."""

    values = _materialize_sequence(value, field_name=field_name)
    if not values:
        raise ValueError(f"{field_name} must contain at least one token.")
    tokens: list[int] = []
    for index, item in enumerate(values):
        if not isinstance(item, Integral) or isinstance(item, bool):
            raise TypeError(f"{field_name}[{index}] must be an integer token ID.")
        token = int(item)
        if token < 0 and not (allow_ignore and token == -100):
            raise ValueError(
                f"{field_name}[{index}] must be nonnegative"
                + (" or -100." if allow_ignore else ".")
            )
        tokens.append(token)
    return tokens


def _finite_sequence(
    value: Any,
    *,
    field_name: str,
    nonnegative: bool = False,
) -> list[float]:
    """Copy numeric values and reject booleans, NaN, infinity, and bad weights."""

    values = _materialize_sequence(value, field_name=field_name)
    if not values:
        raise ValueError(f"{field_name} must contain at least one value.")
    result: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{field_name}[{index}] must be a real number.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{index}] must be finite.")
        if nonnegative and number < 0:
            raise ValueError(f"{field_name}[{index}] must be nonnegative.")
        result.append(number)
    return result


def _length_of_model_input(model_input: Any) -> int:
    """Derive an SDK model-input length without importing PyTRIO eagerly."""

    try:
        return len(model_input)
    except TypeError:
        to_ints = getattr(model_input, "to_ints", None)
        if callable(to_ints):
            return len(to_ints())
    raise TypeError("datum.model_input must expose a length or to_ints().")


def _validate_mapping_fields(
    fields: Mapping[str, Any],
    *,
    spec: "LossSpec",
    expected_length: int | None = None,
) -> None:
    """Validate a raw ``Datum.loss_fn_inputs`` mapping for one LossSpec."""

    missing = spec.required_fields - set(fields)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{spec.name} loss inputs are missing: {names}.")
    unknown = set(fields) - spec.allowed_fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{spec.name} loss inputs contain unsupported fields: {names}.")

    for field_name, raw in fields.items():
        if field_name in {"target_tokens"}:
            values = _token_sequence(
                raw, field_name=field_name, allow_ignore=True
            )
        elif field_name == "weights":
            values = _finite_sequence(raw, field_name=field_name, nonnegative=True)
        else:
            values = _finite_sequence(raw, field_name=field_name)
        if expected_length is not None and len(values) != expected_length:
            raise ValueError(
                f"{field_name} length {len(values)} does not match model_input "
                f"length {expected_length}."
            )


# ---------------------------------------------------------------------------
# Objective specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LossSpec:
    """A validated concrete PyTRIO loss name and its loss-local options.

    ``config`` contains only parameters consumed by the selected PyTRIO loss;
    rollout, reward, advantage, and optimizer scheduling options do not belong
    here.  The values are copied at construction so a caller cannot change the
    request by mutating the original dictionary later.
    """

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate the loss name and configuration immediately."""

        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("loss name must be a non-empty string.")
        name = self.name.strip()
        if name not in BUILTIN_LOSS_FNS:
            supported = ", ".join(sorted(BUILTIN_LOSS_FNS))
            raise ValueError(f"Unsupported PyTRIO loss {name!r}; choose {supported}.")
        if not isinstance(self.config, Mapping):
            raise TypeError("loss config must be a mapping.")
        copied = dict(self.config)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "config", copied)
        self.validate()

    @property
    def family(self) -> LossFamily:
        """Return ``supervised`` for CE or ``policy`` for policy gradients."""

        return "supervised" if self.name in CE_LOSS_FNS else "policy"

    @property
    def loss_fn(self) -> str:
        """Expose the PyTRIO keyword spelling alongside ``name``."""

        return self.name

    @property
    def loss_fn_config(self) -> dict[str, Any]:
        """Return a defensive copy under PyTRIO's request-field spelling."""

        return dict(self.config)

    @property
    def input_family(self) -> LossFamily:
        """Readable alias used by algorithm builders and configuration code."""

        return self.family

    @property
    def is_policy(self) -> bool:
        """Whether the loss consumes behavior log-probabilities and advantages."""

        return self.family == "policy"

    @property
    def required_fields(self) -> frozenset[str]:
        """The required ``Datum.loss_fn_inputs`` keys for this loss."""

        return _POLICY_FIELDS if self.is_policy else _SUPERVISED_FIELDS

    @property
    def optional_fields(self) -> frozenset[str]:
        """Optional keys accepted by this loss's Datum contract."""

        return _OPTIONAL_FIELDS[self.family]

    @property
    def allowed_fields(self) -> frozenset[str]:
        """Required plus optional keys accepted by ``loss_fn_inputs``."""

        return self.required_fields | self.optional_fields

    @property
    def effective_config(self) -> dict[str, Any]:
        """Return explicit defaults merged with caller-provided options."""

        values = dict(_DEFAULT_CONFIG.get(self.name, {}))
        values.update(self.config)
        return values

    def validate(self) -> None:
        """Validate options supported by the selected PyTRIO loss."""

        unknown = set(self.config) - _CONFIG_FIELDS[self.name]
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"{self.name} does not support config fields: {names}.")
        for key, value in self.config.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{self.name} config {key!r} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{self.name} config {key!r} must be finite.")

        values = self.effective_config
        if self.name in {"ppo", "cispo"}:
            low = float(values["clip_low_threshold"])
            high = float(values["clip_high_threshold"])
            if low < 0 or high < 0 or low > high:
                raise ValueError(
                    f"{self.name} clip thresholds require 0 <= low <= high."
                )
        if self.name == "dro" and float(values["beta"]) < 0:
            raise ValueError("dro beta must be nonnegative.")

    def validate_loss_inputs(self, loss_inputs: Mapping[str, Any] | Any) -> None:
        """Validate a loss-input object or a raw ``loss_fn_inputs`` mapping."""

        if isinstance(loss_inputs, _AlignedLossInput):
            if loss_inputs.family != self.family:
                raise TypeError(
                    f"{self.name} requires {self.family} inputs, got "
                    f"{loss_inputs.family}."
                )
            _validate_mapping_fields(
                loss_inputs.loss_fn_inputs,
                spec=self,
                expected_length=len(loss_inputs.model_tokens),
            )
            return
        if not isinstance(loss_inputs, Mapping):
            raise TypeError("loss inputs must be a loss-input object or mapping.")
        _validate_mapping_fields(loss_inputs, spec=self)

    def validate_datum(self, datum: Any) -> None:
        """Validate one SDK-neutral or materialized PyTRIO Datum."""

        fields = getattr(datum, "loss_fn_inputs", None)
        if not isinstance(fields, Mapping):
            raise TypeError("datum.loss_fn_inputs must be a mapping.")
        model_input = getattr(datum, "model_input", None)
        expected_length = _length_of_model_input(model_input)
        _validate_mapping_fields(fields, spec=self, expected_length=expected_length)

    def validate_datums(self, datums: Sequence[Any]) -> None:
        """Validate a non-empty batch of materialized Datums."""

        if not isinstance(datums, Sequence) or isinstance(datums, (str, bytes)):
            raise TypeError("datums must be a sequence of PyTRIO Datum objects.")
        if not datums:
            raise ValueError("datums must contain at least one item.")
        for datum in datums:
            self.validate_datum(datum)

    def request_kwargs(self) -> dict[str, Any]:
        """Build the common PyTRIO request arguments.

        Inputs are already aligned by the owning algorithm, so auto-shifting is
        disabled for every built-in objective. Empty loss configuration is
        omitted for compatibility with SDK versions that only accept it when a
        concrete loss option is present.
        """

        values = {
            "loss_fn": self.name,
            "auto_shift": False,
        }
        if self.config:
            values["loss_fn_config"] = dict(self.config)
        return values


# ---------------------------------------------------------------------------
# Aligned loss-input objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _AlignedLossInput:
    """Shared implementation for aligned CE and policy input contracts."""

    model_tokens: list[int]
    target_tokens: list[int]
    family: LossFamily = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize token sequences and require one-to-one causal alignment."""

        self.model_tokens = _token_sequence(
            self.model_tokens, field_name="model_tokens", allow_ignore=False
        )
        self.target_tokens = _token_sequence(
            self.target_tokens, field_name="target_tokens", allow_ignore=True
        )
        if len(self.model_tokens) != len(self.target_tokens):
            raise ValueError("model_tokens and target_tokens must have equal lengths.")

    @property
    def loss_fn_inputs(self) -> dict[str, list[int] | list[float]]:
        """Return a fresh mapping suitable for ``trio.Datum.loss_fn_inputs``."""

        return {"target_tokens": list(self.target_tokens)}

    @property
    def input_ids(self) -> list[int]:
        """PyTRIO/Trio-Train spelling for the aligned model token sequence."""

        return list(self.model_tokens)

    @property
    def target_ids(self) -> list[int]:
        """PyTRIO/Trio-Train spelling for the aligned target sequence."""

        return list(self.target_tokens)

    def to_datum(
        self,
        *,
        model_input_factory: Any = None,
        datum_factory: Any = None,
    ) -> Any:
        """Materialize this aligned contract as a PyTRIO ``Datum`` lazily."""

        from ..utils.datums import materialize_datum

        return materialize_datum(
            self.model_tokens,
            self.loss_fn_inputs,
            model_input_factory=model_input_factory,
            datum_factory=datum_factory,
        )


@dataclass(init=False, slots=True)
class SupervisedLossInput(_AlignedLossInput):
    """Aligned CE input: targets plus optional per-token nonnegative weights."""

    weights: list[float] | None = None
    family: LossFamily = field(init=False, default="supervised", repr=False)

    def __init__(
        self,
        model_tokens: Any | None = None,
        target_tokens: Any | None = None,
        weights: Any | None = None,
        *,
        input_ids: Any | None = None,
        target_ids: Any | None = None,
    ) -> None:
        """Accept cookbook names and the flat Trio-Train compatibility names."""

        if model_tokens is None:
            model_tokens = input_ids
        elif input_ids is not None:
            raise TypeError("pass either model_tokens or input_ids, not both.")
        if target_tokens is None:
            target_tokens = target_ids
        elif target_ids is not None:
            raise TypeError("pass either target_tokens or target_ids, not both.")
        if model_tokens is None or target_tokens is None:
            raise TypeError("model_tokens/input_ids and target_tokens/target_ids are required.")
        self.model_tokens = model_tokens
        self.target_tokens = target_tokens
        self.weights = weights
        self.family = "supervised"
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate CE targets and optional weights after base token checks."""

        _AlignedLossInput.__post_init__(self)
        if self.weights is not None:
            self.weights = _finite_sequence(
                self.weights, field_name="weights", nonnegative=True
            )
            if len(self.weights) != len(self.model_tokens):
                raise ValueError("weights must align with model_tokens.")

    @property
    def loss_fn_inputs(self) -> dict[str, list[int] | list[float]]:
        """Return CE fields in PyTRIO's expected names."""

        values = _AlignedLossInput.loss_fn_inputs.fget(self)  # type: ignore[union-attr]
        if self.weights is not None:
            values["weights"] = list(self.weights)
        return values


@dataclass(init=False, slots=True)
class PolicyLossInput(_AlignedLossInput):
    """Aligned policy input shared by IS, PPO, CISPO, and DRO."""

    logprobs: list[float]
    advantages: list[float]
    family: LossFamily = field(init=False, default="policy", repr=False)

    def __init__(
        self,
        model_tokens: Any | None = None,
        target_tokens: Any | None = None,
        logprobs: Any | None = None,
        advantages: Any | None = None,
        *,
        input_ids: Any | None = None,
        target_ids: Any | None = None,
        behavior_logprobs: Any | None = None,
    ) -> None:
        """Accept ``logprobs`` and the descriptive ``behavior_logprobs`` name."""

        if model_tokens is None:
            model_tokens = input_ids
        elif input_ids is not None:
            raise TypeError("pass either model_tokens or input_ids, not both.")
        if target_tokens is None:
            target_tokens = target_ids
        elif target_ids is not None:
            raise TypeError("pass either target_tokens or target_ids, not both.")
        if logprobs is None:
            logprobs = behavior_logprobs
        elif behavior_logprobs is not None:
            raise TypeError("pass either logprobs or behavior_logprobs, not both.")
        if model_tokens is None or target_tokens is None:
            raise TypeError("model_tokens/input_ids and target_tokens/target_ids are required.")
        if logprobs is None or advantages is None:
            raise TypeError("logprobs/behavior_logprobs and advantages are required.")
        self.model_tokens = model_tokens
        self.target_tokens = target_tokens
        self.logprobs = logprobs
        self.advantages = advantages
        self.family = "policy"
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate behavior log-probabilities and token advantages."""

        _AlignedLossInput.__post_init__(self)
        self.logprobs = _finite_sequence(self.logprobs, field_name="logprobs")
        self.advantages = _finite_sequence(self.advantages, field_name="advantages")
        expected = len(self.model_tokens)
        if len(self.logprobs) != expected or len(self.advantages) != expected:
            raise ValueError(
                "logprobs and advantages must align with model_tokens."
            )

    @property
    def loss_fn_inputs(self) -> dict[str, list[int] | list[float]]:
        """Return the common policy fields consumed by all four policy losses."""

        values = _AlignedLossInput.loss_fn_inputs.fget(self)  # type: ignore[union-attr]
        values.update(
            logprobs=list(self.logprobs),
            advantages=list(self.advantages),
        )
        return values

    @property
    def behavior_logprobs(self) -> list[float]:
        """Readable alias for the PyTRIO wire field named ``logprobs``."""

        return list(self.logprobs)


# Names used by Trio-Train are exported as concrete specializations.  They
# inherit the same validation and wire representation, so algorithm code does
# not need a separate schema for each policy loss.
@dataclass(init=False, slots=True)
class LossInputCrossEntropy(SupervisedLossInput):
    """PyTRIO ``cross_entropy`` input specialization."""


@dataclass(init=False, slots=True)
class LossInputPolicy(PolicyLossInput):
    """Base specialization for PyTRIO policy-loss inputs."""


@dataclass(init=False, slots=True)
class LossInputImportanceSampling(PolicyLossInput):
    """PyTRIO ``importance_sampling`` input specialization."""


@dataclass(init=False, slots=True)
class LossInputPPO(PolicyLossInput):
    """PyTRIO ``ppo`` input specialization."""


@dataclass(init=False, slots=True)
class LossInputCISPO(PolicyLossInput):
    """PyTRIO ``cispo`` input specialization."""


@dataclass(init=False, slots=True)
class LossInputDRO(PolicyLossInput):
    """PyTRIO ``dro`` input specialization."""


# ---------------------------------------------------------------------------
# Remote objective adapter
# ---------------------------------------------------------------------------


class Objective(Protocol):
    """Remote loss execution shared by built-in and custom objectives."""

    def forward_backward(self, client: Any, datums: Sequence[Any]) -> Any: ...

    async def forward_backward_async(
        self, client: Any, datums: Sequence[Any]
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class BuiltinPyTrioObjective:
    """Adapter that submits one validated built-in objective to PyTRIO.

    The adapter owns only loss-specific request construction and Datum
    validation.  It intentionally does not call ``optim_step``; update
    scheduling remains an algorithm concern, which also keeps it compatible
    with asynchronous rollout trainers.
    """

    spec: LossSpec | str

    def __post_init__(self) -> None:
        """Coerce a convenient string input into a validated LossSpec."""

        if not isinstance(self.spec, LossSpec):
            object.__setattr__(self, "spec", LossSpec(str(self.spec)))

    @property
    def loss_fn(self) -> str:
        """Expose the exact name sent to PyTRIO."""

        return self.spec.name

    @property
    def loss_fn_config(self) -> dict[str, Any]:
        """Return a defensive copy of loss-local options."""

        return dict(self.spec.config)

    def request_kwargs(self) -> dict[str, Any]:
        """Return stable PyTRIO keyword arguments for a forward request."""

        return self.spec.request_kwargs()

    def validate_datums(self, datums: Sequence[Any]) -> None:
        """Validate all fields before submitting a remote operation."""

        self.spec.validate_datums(datums)

    def submit(self, client: Any, datums: Sequence[Any]) -> Any:
        """Submit ``forward_backward`` and return the SDK future/result."""

        self.validate_datums(datums)
        function = getattr(client, "forward_backward", None)
        if function is None:
            raise AttributeError("PyTRIO client does not expose forward_backward().")
        return function(list(datums), **self.request_kwargs())

    def forward_backward(self, client: Any, datums: Sequence[Any]) -> Any:
        """Run and resolve a synchronous forward/backward request."""

        self.validate_datums(datums)
        from .client_api import call

        return call(
            client,
            "forward_backward",
            list(datums),
            **self.request_kwargs(),
        )

    def forward(self, client: Any, datums: Sequence[Any]) -> Any:
        """Run and resolve a synchronous evaluation forward request."""

        self.validate_datums(datums)
        from .client_api import call

        return call(client, "forward", list(datums), **self.request_kwargs())

    async def forward_async(self, client: Any, datums: Sequence[Any]) -> Any:
        """Run and resolve an asynchronous evaluation forward request."""

        self.validate_datums(datums)
        from .client_api import call_async

        return await call_async(
            client,
            "forward",
            list(datums),
            **self.request_kwargs(),
        )

    async def submit_async(self, client: Any, datums: Sequence[Any]) -> Any:
        """Submit through the shared async client adapter without blocking."""

        self.validate_datums(datums)
        from .client_api import submit_async

        return await submit_async(
            client, "forward_backward", list(datums), **self.request_kwargs()
        )

    async def forward_backward_async(self, client: Any, datums: Sequence[Any]) -> Any:
        """Resolve an async forward/backward result through the client adapter."""

        self.validate_datums(datums)
        from .client_api import call_async

        return await call_async(
            client, "forward_backward", list(datums), **self.request_kwargs()
        )


BuiltinObjective = BuiltinPyTrioObjective


# ---------------------------------------------------------------------------
# Small functional façade
# ---------------------------------------------------------------------------


def make_loss_spec(name: str, config: Mapping[str, Any] | None = None) -> LossSpec:
    """Construct a validated LossSpec without exposing dataclass details."""

    return LossSpec(name=name, config={} if config is None else config)


def forward_backward(
    client: Any,
    datums: Sequence[Any],
    objective: BuiltinPyTrioObjective | LossSpec | str,
) -> Any:
    """Submit one built-in objective using one common call shape."""

    if not isinstance(objective, BuiltinPyTrioObjective):
        objective = BuiltinPyTrioObjective(objective)
    return objective.forward_backward(client, datums)


__all__ = [
    "BUILTIN_LOSS_FNS",
    "CE_LOSS_FNS",
    "POLICY_LOSS_FNS",
    "BuiltinObjective",
    "BuiltinPyTrioObjective",
    "LossFamily",
    "LossInputCISPO",
    "LossInputCrossEntropy",
    "LossInputDRO",
    "LossInputImportanceSampling",
    "LossInputPPO",
    "LossInputPolicy",
    "LossName",
    "LossSpec",
    "Objective",
    "PolicyLossInput",
    "SupervisedLossInput",
    "forward_backward",
    "make_loss_spec",
]
