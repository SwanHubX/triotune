"""Text-only GRPO and DPO trainers backed by PyTRIO."""

from .config import GRPOConfig
from .dpo_config import DPOConfig
from .dpo_trainer import DPOResult, DPOState, DPOTrainer
from .grpo_trainer import GRPOTrainer, TrainerState
from .prompts import RenderedPrompt, render_prompt
from .types import GRPOBatch, GRPODatum, GRPOResult, SequenceSample

__all__ = [
    "DPOConfig",
    "DPOResult",
    "DPOState",
    "DPOTrainer",
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
