"""PyTRIO sampling and training contracts for external SWE runtimes.

Tool execution lives in Triotune-SWEAgent and Triotune-Environment. Importing
this module creates no model clients, processes, or environments.
"""

from .contracts import SampledTurn, trajectory_from_turns
from .model import PyTRIOSWEModel, SWESamplingConfig

__all__ = [
    "PyTRIOSWEModel",
    "SWESamplingConfig",
    "SampledTurn",
    "trajectory_from_turns",
]
