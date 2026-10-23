"""Define the data and adapter contracts shared by agent RL components.

Execution adapters exchange original generations, parsed actions, tool results,
and scored episodes through these types. The PPO-specific conversion happens
in the trajectory exporter, keeping tool implementations and agent frameworks
independent of optimizer details.

Protocols describe replaceable behavior through structural typing: adapters
implement the documented methods without inheriting from a framework base
class. Dataclasses carry execution records; their validation protects token
alignment, reward finiteness, and the distinction between model and system
failures before a record can reach the training boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from ..trainer.ppo.contracts import Trajectory
    from .runtime.sampling import SamplingSnapshot


class AgentContractError(ValueError):
    """Report invalid adapter data, including rewritten tokens or missing scores.

    This represents an integration failure. It must reject the episode rather
    than teach the policy that otherwise valid model behavior earned no reward.
    """


class ActionParseError(ValueError):
    """Identify model output that violates the configured action protocol.

    The original generation remains available for scoring and training as a
    task failure. Adapter implementation errors use AgentContractError instead.
    """


class ToolActionError(ValueError):
    """Describe recoverable errors in a model-selected tool or its arguments.

    Tool implementations raise this for domain checks such as division by zero.
    The executor returns the error as an observation so the model can correct
    its next action within the remaining episode budget.
    """


class ToolSystemError(RuntimeError):
    """Reject an episode because a tool implementation or service failed.

    This failure category keeps infrastructure availability separate from the
    quality of the model's decision and prevents accidental zero-reward labels.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def validate_tokens(tokens: Sequence[int], *, name: str) -> None:
    """Require a nonempty sequence of Python integer token IDs.

    ``name`` identifies the adapter field in error messages. Boolean values are
    explicitly rejected because Python otherwise treats them as integers.
    """
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise AgentContractError(f"{name} must contain nonnegative integer tokens.")


@dataclass(frozen=True, slots=True)
class Generation:
    """Store one model response before action parsing changes its representation.

    Attributes:
        tokens: Original sampled output IDs, including returned special tokens.
        logprobs: Behavior-policy log probabilities in exactly the same order.
        text: Decoded view used for parsing; never a replacement for token IDs.
        stop_reason: Backend termination label used to identify length limits.
        input_tokens: Reported input usage for this model request.
        output_tokens: Reported output usage for this model request.

    Usage counters describe request accounting and need not equal the length of
    the final concatenated trajectory. Sampling adapters populate them from the
    backend response, with compatible fallbacks where available.
    """

    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str
    stop_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        validate_tokens(self.tokens, name="generation.tokens")
        if len(self.tokens) != len(self.logprobs) or any(
            isinstance(value, bool) or not math.isfinite(value)
            for value in self.logprobs
        ):
            raise AgentContractError(
                "Behavior logprobs must be finite and token aligned."
            )
        if not isinstance(self.text, str) or not isinstance(self.stop_reason, str):
            raise AgentContractError("Generation text and stop_reason must be strings.")
        for value in (self.input_tokens, self.output_tokens):
            if type(value) is not int or value < 0:
                raise AgentContractError("Sampling usage must be nonnegative integers.")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Describe one requested tool invocation with arguments still unexecuted.

    ``call_id`` connects the request to its observation and is unique within a
    generated turn. Domain validation belongs to the registered tool; this
    record only requires a name and an object with string argument keys.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.call_id, str)
            or not self.call_id
            or not isinstance(self.name, str)
            or not self.name
        ):
            raise AgentContractError("Tool calls need nonempty string IDs and names.")
        if not isinstance(self.arguments, dict) or any(
            not isinstance(key, str) for key in self.arguments
        ):
            raise AgentContractError(
                "Tool arguments must be an object with string keys."
            )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Carry a tool observation back to the agent and the episode log.

    ``is_error`` marks recoverable model-action errors, not infrastructure
    failures. The runner checks that the name and ID match the original call
    before allowing the observation into the conversation.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or type(self.is_error) is not bool:
            raise AgentContractError(
                "Tool results need string content and a boolean error flag."
            )


@dataclass(frozen=True, slots=True)
class AgentAction:
    """Separate conversation content from the decision that controls execution.

    ``assistant_message`` is appended to the next model request. Exactly one of
    a nonempty ``tool_calls`` tuple or a non-None ``final_text`` is required.
    Keeping both views allows a parser to preserve the original assistant text
    while exposing structured tool arguments to an independent executor.
    """

    assistant_message: dict[str, Any]
    tool_calls: tuple[ToolCall, ...] = ()
    final_text: str | None = None

    def __post_init__(self) -> None:
        if bool(self.tool_calls) == (self.final_text is not None):
            raise AgentContractError("An action needs either tools or a final answer.")
        if self.assistant_message.get("role") != "assistant":
            raise AgentContractError("An action must carry an assistant message.")
        if self.final_text is not None and not isinstance(self.final_text, str):
            raise AgentContractError("A final answer must be a string.")
        if len({call.call_id for call in self.tool_calls}) != len(self.tool_calls):
            raise AgentContractError("Tool call IDs must be unique within a turn.")
        for call in self.tool_calls:
            call.__post_init__()


@dataclass(slots=True)
class AgentTurn:
    """Record one generation and any tool observations produced from it.

    ``prompt_tokens`` are the actual IDs sent to the sampler. ``action`` stays
    None when output is truncated or cannot be parsed. ``tool_results`` is
    filled as calls complete, so a failed episode can retain partial progress.
    """

    prompt_tokens: tuple[int, ...]
    generation: Generation
    action: AgentAction | None = None
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RetryEvent:
    """One transient failed operation, including the exhausted final attempt."""

    stage: str
    turn: int
    attempt: int
    error_type: str
    error: str
    retried: bool


# Completion does not imply success: a valid final answer can still be wrong.
# Budget exits and invalid actions are model-side outcomes, while the system
# statuses below prevent infrastructure failures from entering policy updates.
EpisodeStatus = Literal[
    "completed",
    "invalid_action",
    "max_turns",
    "token_limit",
    "context_limit",
    "sampling_failure",
    "tool_failure",
    "scoring_failure",
    "contract_error",
    "timeout",
]
SYSTEM_STATUSES = frozenset(
    {"sampling_failure", "tool_failure", "scoring_failure", "contract_error", "timeout"}
)


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    """Outcome reward and optional original-output-token process rewards.

    Each process_rewards key is a zero-based generated-turn index. Its vector
    contains one reward per original output token, including any EOS token.
    These rewards are consumed by existing PPO GAE, not treated as advantages.

    ``reward`` is the terminal task score. ``success`` is an explicit metric,
    independent of the numerical reward scale. ``components`` exposes finite
    diagnostic scores; components are not implicitly summed into the reward.
    Process reward lengths are checked by the exporter against recorded turns.
    """

    reward: float
    success: bool
    components: dict[str, float] = field(default_factory=dict)
    process_rewards: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise AgentContractError("success must be a boolean.")
        values = [self.reward, *self.components.values()]
        if any(not isinstance(k, str) or not k.strip() for k in self.components):
            raise AgentContractError("Reward component names must be nonempty strings.")
        for turn, rewards in self.process_rewards.items():
            if type(turn) is not int or turn < 0:
                raise AgentContractError(
                    "Process reward turn IDs must be nonnegative integers."
                )
            values.extend(rewards)
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            raise AgentContractError("Rewards must be finite numeric scalars.")


@dataclass(slots=True)
class EpisodeResult:
    """Accumulate the auditable result of one task execution.

    ``episode_id`` identifies one attempt, ``group_id`` connects attempts of
    the same task for relative-reward training, and ``snapshot_id`` records
    which fixed sampling policy was supplied. ``is_eval`` labels provenance
    without selecting a different execution path. ``error`` retains the reason
    for abnormal termination; system failures normally have no score.
    """

    episode_id: str
    group_id: str
    snapshot_id: str
    is_eval: bool = False
    turns: list[AgentTurn] = field(default_factory=list)
    status: EpisodeStatus = "completed"
    final_text: str | None = None
    score: EpisodeScore | None = None
    error: str | None = None
    task_id: str | None = None
    sample_index: int = 0
    retries: list[RetryEvent] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def trainable(self) -> bool:
        """Check episode-level eligibility before trajectory export.

        At least one generation and a score are required. This does not yet
        prove token-prefix validity or group eligibility: the exporter and
        collector enforce those checks separately.
        """
        return (
            bool(self.turns)
            and self.status not in SYSTEM_STATUSES
            and self.score is not None
        )


@dataclass(frozen=True, slots=True)
class EpisodeLimits:
    """Bound the work allowed during one episode, independently of PPO updates.

    Token limits distinguish one response, all generated responses combined,
    and the full model context including observations. Timeout values are
    seconds. ``episode_timeout`` covers rendering, generation, tools, and
    scoring; sampling and tool deadlines additionally bound individual calls.
    """

    max_turns: int = 8
    max_tokens_per_turn: int = 256
    max_generated_tokens: int = 2048
    max_context_tokens: int = 8192
    sampling_timeout: float = 60.0
    tool_timeout: float = 10.0
    episode_timeout: float = 180.0

    def __post_init__(self) -> None:
        for name in (
            "max_turns",
            "max_tokens_per_turn",
            "max_generated_tokens",
            "max_context_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("sampling_timeout", "tool_timeout", "episode_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


class AgentSampler(Protocol):
    """Generate original tokens and behavior scores from a fixed policy.

    Implementations must honor the requested output budget and must not refresh
    weights between turns. Backend errors propagate to the runner for failure
    classification; missing or misaligned scores violate the data contract.
    """

    async def generate(
        self, prompt_tokens: Sequence[int], *, max_tokens: int, seed: int | None = None
    ) -> Generation: ...


class ConversationRenderer(Protocol):
    """Serialize messages and advertised tool schemas into actual input IDs.

    Rendering is synchronous and may run in a worker thread. Successive inputs
    must preserve previously sampled tokens for the default concat exporter.
    A renderer may additionally define render_continuation(episode, messages,
    tools); the runner uses it after the first generation to append context
    without re-encoding existing output IDs.
    """

    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> list[int]: ...


class ActionParser(Protocol):
    """Interpret a generation without replacing its original training tokens.

    Raise ActionParseError for malformed model actions. Use AgentContractError
    for invalid adapter results or inconsistencies unrelated to model behavior.
    """

    def parse(self, generation: Generation) -> AgentAction: ...


class ToolExecutor(Protocol):
    """Expose available tool schemas and execute one already-parsed request.

    Implementations return recoverable errors as ToolResult observations and
    raise ToolSystemError for execution infrastructure failures. They do not
    construct model prompts or assign training rewards.
    """

    @property
    def schemas(self) -> Sequence[Mapping[str, Any]]: ...

    async def execute(self, call: ToolCall) -> ToolResult: ...


class ObservationFormatter(Protocol):
    """Translate tool results into messages understood by the model protocol.

    Pair the formatter with its ActionParser: native tool messages and a JSON
    conversation protocol can require different role and call-ID conventions.
    """

    def format(self, results: Sequence[ToolResult]) -> list[dict[str, Any]]: ...


class EpisodeScorer(Protocol):
    """Score a completed or model-limited execution using task-specific context.

    ``row`` may contain ground truth that should never be added to the model's
    messages. Scorers can return terminal and process rewards together, and
    must support concurrent calls from separate episodes without mixing state.
    """

    async def score(
        self, row: Mapping[str, Any], episode: EpisodeResult
    ) -> EpisodeScore: ...


class EpisodeLogger(Protocol):
    """Persist a finalized episode, including failures and partial tool results.

    The collector calls this after export validation. Implementations choose
    serialization, storage, and redaction policies; logging errors propagate
    to the caller rather than silently claiming the record was persisted.
    """

    async def write(self, episode: EpisodeResult) -> None: ...


class TrajectoryExporter(Protocol):
    """Validate request continuity and convert a scored episode for PPO.

    ``validate_prefix`` runs before each request so an incompatible conversation
    can fail early. ``build`` performs final validation and returns the existing
    PPO trajectory type. Alternative layouts must preserve action identity,
    behavior scores, masks, and reward alignment across this boundary.
    """

    def validate_prefix(
        self, episode: EpisodeResult, prompt_tokens: list[int]
    ) -> None: ...

    def build(self, episode: EpisodeResult) -> Trajectory: ...


class EpisodeExecutor(Protocol):
    """Replace the complete agent loop while retaining collection and training.

    An SDK adapter implements ``run`` and returns a complete EpisodeResult with
    original token data. The collector also needs ``limits.max_turns`` to
    allocate seed ranges and ``trajectory_builder`` to validate/export records.
    Implementations must preserve the supplied snapshot and propagate asyncio
    cancellation rather than converting cancellation into a task reward.
    """

    limits: EpisodeLimits
    trajectory_builder: TrajectoryExporter

    async def run(
        self,
        row: Mapping[str, Any],
        snapshot: SamplingSnapshot,
        *,
        group_id: str,
        seed: int | None = None,
        is_eval: bool = False,
    ) -> EpisodeResult: ...
