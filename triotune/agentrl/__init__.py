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

from .config import AgentConfig, AgentEvaluationConfig, RetryPolicy
from .diagnostics.logging import JsonlEpisodeLogger
from .diagnostics.validation import probe_tokenizer
from .rollout.trajectory import EpisodeTrajectoryBuilder
from .rollout.workflow import (
    AgentRolloutBatch,
    AgentRolloutWorkflow,
    EpisodeCollectionError,
)
from .runtime.conversation import (
    ChatTemplateRenderer,
    JsonActionParser,
    JsonObservationFormatter,
    TokenAppendRenderer,
)
from .runtime.runner import EpisodeRunner
from .runtime.sampling import PyTRIOAgentSampler, SamplingSnapshot
from .runtime.tools import FunctionTool, FunctionToolExecutor
from .training.evaluation import evaluation_metrics, paired_task_metrics
from .training.loop import AgentTrainingLoop
from .training.rewards import (
    CompositeScorer,
    CostConfig,
    RewardTerm,
    ToolCostScorer,
    TurnRewardScorer,
)
from .types import (
    ActionParseError,
    ActionParser,
    AgentAction,
    AgentContractError,
    AgentEvent,
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
    ExecutionContext,
    Generation,
    GenerationEvent,
    Interaction,
    ObservationFormatter,
    RetryEvent,
    RewardEvent,
    RolloutExport,
    ToolActionError,
    ToolCall,
    ToolCallEvent,
    ToolExecutor,
    ToolResult,
    ToolResultEvent,
    ToolSystemError,
    TrajectoryExporter,
    current_execution_context,
)

# Keep adapter types and their default implementations available through one
# stable import surface. Runtime state remains owned by the objects callers
# construct; importing the package does not start clients or execute tools.
__all__ = [
    "ActionParseError",
    "ActionParser",
    "AgentAction",
    "AgentConfig",
    "AgentContractError",
    "AgentEvaluationConfig",
    "AgentEvent",
    "AgentRolloutBatch",
    "AgentRolloutWorkflow",
    "AgentSampler",
    "AgentTrainingLoop",
    "AgentTurn",
    "ChatTemplateRenderer",
    "CompositeScorer",
    "ConversationRenderer",
    "CostConfig",
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
    "ExecutionContext",
    "FunctionTool",
    "FunctionToolExecutor",
    "Generation",
    "GenerationEvent",
    "Interaction",
    "JsonActionParser",
    "JsonObservationFormatter",
    "JsonlEpisodeLogger",
    "ObservationFormatter",
    "PyTRIOAgentSampler",
    "RetryEvent",
    "RetryPolicy",
    "RewardEvent",
    "RewardTerm",
    "RolloutExport",
    "SamplingSnapshot",
    "TokenAppendRenderer",
    "ToolActionError",
    "ToolCall",
    "ToolCallEvent",
    "ToolCostScorer",
    "ToolExecutor",
    "ToolResult",
    "ToolResultEvent",
    "ToolSystemError",
    "TrajectoryExporter",
    "TurnRewardScorer",
    "current_execution_context",
    "evaluation_metrics",
    "paired_task_metrics",
    "probe_tokenizer",
]
