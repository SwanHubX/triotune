"""Text-only GRPO, DPO, and SFT trainers backed by PyTRIO."""

from .grpo_config import GRPOConfig
from .dpo_config import DPOConfig
from .dpo_trainer import DPOResult, DPOState, DPOTrainer
from .grpo_trainer import GRPOTrainer, TrainerState
from .sft_config import SFTConfig
from .sft_trainer import SFTState, SFTTrainer
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
    "SFTConfig",
    "SFTState",
    "SFTTrainer",
    "TrainerState",
    "render_prompt",
]
