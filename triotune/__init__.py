"""Client-side training utilities for PyTRIO.

The package intentionally contains no model implementation.  Model weights,
tokenization backends, sampling and optimization stay behind the PyTRIO
clients supplied by the caller.
"""

from . import utils
from .api.client_api import call, call_async, wait, wait_async
from .trainer import DPOConfig, DPOTrainer, OPDConfig, OPDTrainer
from .trainer import PPOConfig, PPOTrainer, RenderedPrompt, render_prompt

__all__ = [
    "DPOConfig",
    "DPOTrainer",
    "OPDConfig",
    "OPDTrainer",
    "PPOConfig",
    "PPOTrainer",
    "RenderedPrompt",
    "call",
    "call_async",
    "render_prompt",
    "wait",
    "wait_async",
]
