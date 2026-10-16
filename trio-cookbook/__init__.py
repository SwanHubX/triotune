"""Client-side training utilities for PyTRIO.

The package intentionally contains no model implementation.  Model weights,
tokenization backends, sampling and optimization stay behind the PyTRIO
clients supplied by the caller.
"""

from . import utils
from .api.client_api import call, call_async, wait, wait_async
from .trainer import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, RenderedPrompt, render_prompt
from .trainer import OPDConfig, OPDTrainer

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
