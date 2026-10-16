"""Composable PPO-family training, with explicit PyTRIO capability boundaries."""

from .actor import PPOActor
from .advantages import prepare_advantages
from .config import NormConfig, PPOConfig
from .contracts import PPOState, PreparedBatch, Trajectory
from .critic import Critic, PyTRIOCritic
from .critic_loss import value_loss
from .gae import compute_gae
from .trainer import PPOTrainer

__all__ = [
    "Critic",
    "NormConfig",
    "PPOActor",
    "PPOConfig",
    "PPOState",
    "PPOTrainer",
    "PyTRIOCritic",
    "PreparedBatch",
    "Trajectory",
    "compute_gae",
    "prepare_advantages",
    "value_loss",
]
