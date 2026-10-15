"""Public contracts and shared support for trainer algorithms."""

from .config import BaseTrainerConfig, DPOConfig, GRPOConfig, OPDConfig, SFTConfig

__all__ = [
    "BaseTrainerConfig",
    "DPOConfig",
    "GRPOConfig",
    "OPDConfig",
    "SFTConfig",
]
