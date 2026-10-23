"""Expose the public building blocks for tool-using agent RL with PyTRIO.

Typical composition starts with an EpisodeRunner containing conversation,
action, tool, and scoring adapters. AgentRolloutWorkflow expands tasks into
grouped attempts and exports PPO trajectories. AgentTrainingLoop connects the
workflow to PPOTrainer for model updates, snapshot refresh, and evaluation.

Import protocols and data records here when implementing an adapter; concrete
implementations are also exported for local Python tools, JSON conversations,
PyTRIO sampling, exact-token trajectory construction, and JSONL episode logs.
The package provides orchestration and client adapters, while model execution
and optimization remain behind the existing PyTRIO/trainer boundaries.
"""

from .contracts import (
    ActionParseError,
    ActionParser,
    AgentAction,
    AgentContractError,
    AgentSampler,
    AgentTurn,
    ConversationRenderer,
    EpisodeExecutor,
    EpisodeLimits,
    EpisodeLogger,
    EpisodeResult,
    EpisodeScore,
    EpisodeScorer,
    EpisodeStatus,
    Generation,
    ObservationFormatter,
    ToolActionError,
    ToolCall,
    ToolExecutor,
    ToolResult,
    ToolSystemError,
    TrajectoryExporter,
)
from .conversation import (
    ChatTemplateRenderer,
    JsonActionParser,
    JsonObservationFormatter,
)
from .episode_logging import JsonlEpisodeLogger
from .episode_runner import EpisodeRunner
from .rollout_workflow import (
    AgentRolloutBatch,
    AgentRolloutWorkflow,
    EpisodeCollectionError,
)
from .sampling import PyTRIOAgentSampler, SamplingSnapshot
from .tool_executor import FunctionTool, FunctionToolExecutor
from .training_loop import AgentTrainingLoop
from .trajectory_builder import EpisodeTrajectoryBuilder

# Keep adapter contracts and their default implementations available through
# one stable import surface. Runtime state remains owned by the objects callers
# construct; importing the package does not start clients or execute tools.
__all__ = [
    "ActionParseError",
    "ActionParser",
    "AgentAction",
    "AgentContractError",
    "AgentRolloutBatch",
    "AgentRolloutWorkflow",
    "AgentSampler",
    "AgentTrainingLoop",
    "AgentTurn",
    "ChatTemplateRenderer",
    "ConversationRenderer",
    "EpisodeCollectionError",
    "EpisodeExecutor",
    "EpisodeLimits",
    "EpisodeLogger",
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeScore",
    "EpisodeScorer",
    "EpisodeStatus",
    "EpisodeTrajectoryBuilder",
    "FunctionTool",
    "FunctionToolExecutor",
    "Generation",
    "JsonActionParser",
    "JsonObservationFormatter",
    "JsonlEpisodeLogger",
    "ObservationFormatter",
    "PyTRIOAgentSampler",
    "SamplingSnapshot",
    "ToolActionError",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "ToolSystemError",
    "TrajectoryExporter",
]
