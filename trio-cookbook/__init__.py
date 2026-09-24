"""Client-side training utilities for PyTRIO.

The package intentionally contains no model implementation.  Model weights,
tokenization backends, sampling and optimization stay behind the PyTRIO
clients supplied by the caller.
"""

from .trainer import GRPOConfig, GRPOTrainer, RenderedPrompt, render_prompt
from .utils import call, call_async, wait, wait_async

__all__ = [
    "GRPOConfig",
    "GRPOTrainer",
    "RenderedPrompt",
    "call",
    "call_async",
    "render_prompt",
    "wait",
    "wait_async",
]
