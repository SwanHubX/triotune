"""Synchronous training loops and algorithm-specific agent adapters."""

from .opd import AgentOPDTrainingLoop, AgentOPDUpdate, episode_turn_rollouts

__all__ = ["AgentOPDTrainingLoop", "AgentOPDUpdate", "episode_turn_rollouts"]
