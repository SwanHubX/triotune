"""Client-side training utilities for PyTRIO.

The package intentionally contains no model implementation.  Model weights,
tokenization backends, sampling and optimization stay behind the PyTRIO
clients supplied by the caller.
"""

import sys
from types import ModuleType

from .trainer import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, RenderedPrompt, render_prompt
from .trainer import OPDConfig, OPDTrainer
from .trainer.common import client as _client_module
from .trainer.common.client import call, call_async, wait, wait_async

# Preserve the former public helper path without retaining duplicate source.
# A lightweight package facade supports both ``utils.call`` and
# ``utils.client.call`` while the implementation stays in trainer.common.
_utils_compat = ModuleType(f"{__name__}.utils")
_utils_compat.__path__ = []
_utils_compat.client = _client_module
for _name in _client_module.__all__:
    setattr(_utils_compat, _name, getattr(_client_module, _name))
_utils_compat.__all__ = [*_client_module.__all__, "client"]
sys.modules[f"{__name__}.utils"] = _utils_compat
sys.modules[f"{__name__}.utils.client"] = _client_module
utils = _utils_compat
del _client_module, _name, _utils_compat

__all__ = [
    "DPOConfig",
    "DPOTrainer",
    "GRPOConfig",
    "GRPOTrainer",
    "OPDConfig",
    "OPDTrainer",
    "RenderedPrompt",
    "call",
    "call_async",
    "render_prompt",
    "wait",
    "wait_async",
]
