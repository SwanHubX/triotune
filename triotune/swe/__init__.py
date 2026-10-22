"""PyTRIO sampling and training contracts for external SWE runtimes.

Tool execution lives in Triotune-SWEAgent and Triotune-Environment. Importing
this module creates no model clients, processes, or environments.
"""

from .contracts import SampledTurn, trajectory_from_turns
from .model import ContextBudgetError, PyTRIOSWEModel, SWESamplingConfig
from .retry import RetryConfig

__all__ = [
    "ContextBudgetError",
    "PyTRIOSWEModel",
    "RetryConfig",
    "SWESamplingConfig",
    "SampledTurn",
    "trajectory_from_turns",
]
