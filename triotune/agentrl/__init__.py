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
    CompactionPlan,
    ContextCompactor,
    JsonActionParser,
    JsonObservationFormatter,
    NativeToolContinuationRenderer,
    PrefixCache,
    TokenAppendRenderer,
)
from .runtime.runner import EpisodeRunner, RuntimeBoundary
from .runtime.sampling import PyTRIOAgentSampler, SamplingSnapshot
from .runtime.tools import FunctionTool, FunctionToolExecutor
from .sdk.openai import (
    Agent,
    ExternalOpenAIBackend,
    OpenAIClient,
    OpenAIEpisodeExecutor,
    OpenAISession,
    OpenAITraceBranch,
    PyTRIOBackend,
    Qwen35ToolParser,
    function_tool,
)
from .training.evaluation import evaluation_metrics, paired_task_metrics
from .training.loop import AgentTrainingLoop
from .training.rewards import (
    CompositeScorer,
    CostConfig,
    RewardAggregator,
    RewardAssignment,
    RewardLedger,
    RewardSubmitter,
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
    ContextCompaction,
    ConversationCompactor,
    ConversationRenderer,
    EpisodeBranch,
    EpisodeExecutor,
    EpisodeLimits,
    EpisodeLogger,
    EpisodeResult,
    EpisodeScore,
    EpisodeScorer,
    EpisodeStatus,
    ExecutionContext,
    FailureClassifier,
    Generation,
    GenerationEvent,
    Interaction,
    InteractionScorer,
    ObservationFormatter,
    RetryEvent,
    RewardEvent,
    RolloutExport,
    SampleAcceptance,
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
    "Agent",
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
    "OpenAIEpisodeExecutor",
    "ChatTemplateRenderer",
    "CompactionPlan",
    "CompositeScorer",
    "ContextCompaction",
    "ContextCompactor",
    "ConversationCompactor",
    "ConversationRenderer",
    "CostConfig",
    "EpisodeBranch",
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
    "ExternalOpenAIBackend",
    "FailureClassifier",
    "FunctionTool",
    "FunctionToolExecutor",
    "Generation",
    "GenerationEvent",
    "Interaction",
    "InteractionScorer",
    "JsonActionParser",
    "JsonObservationFormatter",
    "JsonlEpisodeLogger",
    "NativeToolContinuationRenderer",
    "ObservationFormatter",
    "OpenAIClient",
    "OpenAISession",
    "OpenAITraceBranch",
    "PrefixCache",
    "PyTRIOAgentSampler",
    "PyTRIOBackend",
    "Qwen35ToolParser",
    "RetryEvent",
    "RetryPolicy",
    "RewardAggregator",
    "RewardAssignment",
    "RewardEvent",
    "RewardLedger",
    "RewardSubmitter",
    "RewardTerm",
    "RolloutExport",
    "RuntimeBoundary",
    "SampleAcceptance",
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
    "function_tool",
    "paired_task_metrics",
    "probe_tokenizer",
]
