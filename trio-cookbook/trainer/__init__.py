"""Modular, text-only GRPO trainer for a remote PyTRIO client."""

from .config import GRPOConfig
from .grpo_trainer import GRPOTrainer, TrainerState
from .prompts import RenderedPrompt, render_prompt
from .types import GRPOBatch, GRPODatum, GRPOResult, SequenceSample

__all__ = [
    "GRPOBatch",
    "GRPOConfig",
    "GRPODatum",
    "GRPOResult",
    "GRPOTrainer",
    "RenderedPrompt",
    "SequenceSample",
    "TrainerState",
    "render_prompt",
]
