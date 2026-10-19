"""Public configuration and client contracts for trainer algorithms."""

from .cli_args import (
    BaseTrainerConfig,
    DPOConfig,
    NormConfig,
    OPDConfig,
    PPOConfig,
    SFTConfig,
)
from .client_api import (
    call,
    call_async,
    optimizer_params,
    sampling_params,
    sdk_object,
    submit_async,
    wait,
    wait_async,
)
from .loss_api import (
    BUILTIN_LOSS_FNS,
    CE_LOSS_FNS,
    POLICY_LOSS_FNS,
    BuiltinObjective,
    BuiltinPyTrioObjective,
    LossSpec,
    PolicyLossInput,
    SupervisedLossInput,
)

__all__ = [
    "BUILTIN_LOSS_FNS",
    "CE_LOSS_FNS",
    "POLICY_LOSS_FNS",
    "BaseTrainerConfig",
    "BuiltinObjective",
    "BuiltinPyTrioObjective",
    "DPOConfig",
    "LossSpec",
    "NormConfig",
    "OPDConfig",
    "PPOConfig",
    "PolicyLossInput",
    "SFTConfig",
    "SupervisedLossInput",
    "call",
    "call_async",
    "optimizer_params",
    "sampling_params",
    "sdk_object",
    "submit_async",
    "wait",
    "wait_async",
]
