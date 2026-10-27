"""Typed execution records and replaceable interfaces for Agent RL.

Execution adapters exchange original generations, parsed actions, tool results,
and scored episodes through these types. The PPO-specific conversion happens
in the trajectory exporter, keeping tool implementations and agent frameworks
independent of optimizer details.

Protocols describe replaceable behavior through structural typing: adapters
implement the documented methods without inheriting from a framework base
class. Dataclasses carry execution records; their validation protects token
alignment, reward finiteness, provenance, and the distinction between model
and system failures before a record can reach the training boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from ..trainer.ppo.trajectory import Trajectory
    from .runtime.sampling import SamplingSnapshot


class AgentContractError(ValueError):
    """Report invalid adapter data, including rewritten tokens or missing scores.

    This represents an integration failure. It must reject the episode rather
    than teach the policy that otherwise valid model behavior earned no reward.
    """


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Immutable identity shared by execution, scoring, logging, and export.

    A rollout is one candidate execution, an episode is its recorded attempt,
    and a group contains candidates for one input task. Interaction identities
    belong to model requests within that attempt. Child contexts preserve all
    candidate fields and change only the interaction and its parent. Resources
    belong to ``resource_owner``; recording this value does not transfer their
    ownership or authorize an adapter to close a shared client.
    """

    rollout_id: str
    episode_id: str
    group_id: str
    sample_index: int
    snapshot_id: str
    policy_version: int | str | None = None
    interaction_id: str | None = None
    parent_interaction_id: str | None = None
    group_size: int = 1
    task_id: str | None = None
    resource_owner: str = "runner"

    def __post_init__(self) -> None:
        for name in (
            "rollout_id",
            "episode_id",
            "group_id",
            "snapshot_id",
            "resource_owner",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise AgentContractError(f"{name} must be a nonempty string.")
        for name in ("interaction_id", "parent_interaction_id", "task_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise AgentContractError(f"{name} must be a nonempty string or None.")
        if (
            type(self.sample_index) is not int
            or type(self.group_size) is not int
            or not 0 <= self.sample_index < self.group_size
        ):
            raise AgentContractError("Candidate index must be within group_size.")
        if self.interaction_id is None and self.parent_interaction_id is not None:
            raise AgentContractError("Only an interaction can have a parent.")
        if (
            self.interaction_id is not None
            and self.interaction_id == self.parent_interaction_id
        ):
            raise AgentContractError("An interaction cannot be its own parent.")
        if self.policy_version is not None and (
            type(self.policy_version) not in (int, str)
            or isinstance(self.policy_version, int)
            and self.policy_version < 0
            or isinstance(self.policy_version, str)
            and not self.policy_version
        ):
            raise AgentContractError(
                "policy_version must be nonnegative or a nonempty label."
            )


# Context variables are local to an asynchronous task and are copied into
# asyncio.to_thread workers. Tool functions can inspect candidate identity
# without accepting model-controlled identity arguments or mutating shared
# executor state. The runner restores the enclosing value even on cancellation.
_execution_context: ContextVar[ExecutionContext | None] = ContextVar(
    "triotune_execution_context", default=None
)


def current_execution_context() -> ExecutionContext | None:
    """Return the current adapter's trusted identity, or None outside execution.

    Processes do not share context variables. An out-of-process adapter must
    serialize ExecutionContext explicitly and establish it in its own worker.
    Background work that outlives an episode must manage its own resource scope.
    """
    return _execution_context.get()


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
class ContextCompaction:
    """Audit one exact-token context reset before a generated interaction.

    ``turn_index`` identifies the first turn sampled from the compacted prompt.
    The new prompt IDs are stored verbatim because a summary is model context,
    not behavior-policy output, and must never be reconstructed by decoding old
    actions. Retained and summarized interaction IDs explain which prior tool
    exchanges remain literal and which were replaced by application-provided
    summaries. Export uses these records to split one logical rollout into
    independently causal physical rows.
    """

    turn_index: int
    original_prompt_tokens: int
    compacted_prompt_tokens: tuple[int, ...]
    retained_interaction_ids: tuple[str, ...] = ()
    summarized_interaction_ids: tuple[str, ...] = ()
    removed_messages: int = 0

    def __post_init__(self) -> None:
        if type(self.turn_index) is not int or self.turn_index < 1:
            raise AgentContractError("Compaction turn_index must be positive.")
        if (
            type(self.original_prompt_tokens) is not int
            or self.original_prompt_tokens < 1
            or type(self.removed_messages) is not int
            or self.removed_messages < 1
        ):
            raise AgentContractError(
                "Compaction token and removed-message counts must be positive."
            )
        validate_tokens(self.compacted_prompt_tokens, name="compacted_prompt_tokens")
        for name in ("retained_interaction_ids", "summarized_interaction_ids"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise AgentContractError(
                    f"{name} must contain unique nonempty interaction IDs."
                )
        if set(self.retained_interaction_ids).intersection(
            self.summarized_interaction_ids
        ):
            raise AgentContractError(
                "An interaction cannot be both retained and summarized."
            )


@dataclass(frozen=True, slots=True)
class RetryEvent:
    """One transient failed operation, including the exhausted final attempt."""

    stage: str
    turn: int
    attempt: int
    error_type: str
    error: str
    retried: bool


@dataclass(slots=True)
class Interaction:
    """One model request and the environment work selected by its output.

    Token spans are half-open offsets in this request's prompt-plus-output
    sequence. A failed request keeps its input span and has no output span.
    ``turn`` references the original generation, parsed action, and partial
    tool results; it is absent when sampling never produced a valid generation.
    Latency covers sampling and tools, including bounded retry backoff, but not
    terminal episode scoring. ``reward`` is the raw local total, equal to
    ``outcome_reward + sum(process_rewards)``. Outcome is recorded only on the
    last generated interaction; process rewards and direct advantages retain
    their original output-token offsets. These are scorer signals before PPO
    shaping, normalization, and GAE, not critic returns or extra reward sources.
    """

    context: ExecutionContext
    input_token_span: tuple[int, int]
    output_token_span: tuple[int, int] | None = None
    turn: AgentTurn | None = None
    reward: float = 0.0
    process_rewards: tuple[float, ...] = ()
    latency: float = 0.0
    retries: list[RetryEvent] = field(default_factory=list)
    status: str = "running"
    resource_owner: str = "runner"
    # Append optional diagnostic fields to preserve positional construction of
    # existing Interaction records. Direct advantages never enter reward.
    outcome_reward: float = 0.0
    token_advantages: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.context.interaction_id is None:
            raise AgentContractError("Interaction context requires an interaction_id.")
        for span in (self.input_token_span, self.output_token_span):
            if span is not None and (
                len(span) != 2
                or any(type(value) is not int for value in span)
                or not 0 <= span[0] <= span[1]
            ):
                raise AgentContractError(
                    "Interaction token spans must be half-open nonnegative ranges."
                )
        if not math.isfinite(self.latency) or self.latency < 0:
            raise AgentContractError(
                "Interaction latency must be finite and nonnegative."
            )
        if self.turn is not None:
            n = len(self.turn.prompt_tokens)
            if self.input_token_span != (0, n) or self.output_token_span != (
                n,
                n + len(self.turn.generation.tokens),
            ):
                raise AgentContractError(
                    "Interaction spans must match its original request and generation."
                )
        if not all(
            math.isfinite(value)
            for value in (
                self.reward,
                self.outcome_reward,
                *self.process_rewards,
                *self.token_advantages,
            )
        ):
            raise AgentContractError("Interaction rewards must be finite.")
        if self.turn is not None and any(
            vector and len(vector) != len(self.turn.generation.tokens)
            for vector in (self.process_rewards, self.token_advantages)
        ):
            raise AgentContractError(
                "Interaction reward vectors must align with original output tokens."
            )

    @property
    def interaction_id(self) -> str:
        return self.context.interaction_id

    @property
    def parent_interaction_id(self) -> str | None:
        return self.context.parent_interaction_id

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.turn.action.tool_calls if self.turn and self.turn.action else ()

    @property
    def tool_results(self) -> list[ToolResult]:
        return self.turn.tool_results if self.turn else []


@dataclass(frozen=True, slots=True)
class EpisodeBranch:
    """Describe one trainable suffix in an episode interaction tree.

    ``interaction_ids`` contains only the interactions created after the fork.
    ``parent_interaction_id`` identifies the state from which that suffix starts;
    the exporter follows parent links to recover the immutable shared prefix.
    Sibling alternatives use the same rollout-local ``group_id`` so reward
    normalization compares only branches produced from the same decision state.

    ``reward`` is the complete branch-local outcome rather than an increment to
    the episode outcome. A completed branch therefore requires a finite reward,
    while an unfinished branch can retain ``None`` and remains visible in export
    metadata without entering PPO. ``shared_prefix_weight`` controls how this
    branch contributes to sampled actions that also occur in sibling paths when
    the collector uses weighted shared-prefix training. Branch-local actions
    always retain unit weight.
    """

    branch_id: str
    interaction_ids: tuple[str, ...]
    group_id: str
    parent_interaction_id: str | None = None
    reward: float | None = None
    completed: bool = True
    shared_prefix_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("branch_id", "group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise AgentContractError(f"{name} must be a nonempty string.")
        if self.parent_interaction_id is not None and (
            not isinstance(self.parent_interaction_id, str)
            or not self.parent_interaction_id
        ):
            raise AgentContractError(
                "parent_interaction_id must be a nonempty string or None."
            )
        if (
            not isinstance(self.interaction_ids, tuple)
            or not self.interaction_ids
            or len(set(self.interaction_ids)) != len(self.interaction_ids)
            or any(
                not isinstance(value, str) or not value
                for value in self.interaction_ids
            )
        ):
            raise AgentContractError(
                "A branch needs an ordered tuple of unique interaction IDs."
            )
        if type(self.completed) is not bool:
            raise AgentContractError("Branch completed must be a boolean.")
        if self.completed and self.reward is None:
            raise AgentContractError("A completed branch requires a local reward.")
        if self.reward is not None and (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(float(self.reward))
        ):
            raise AgentContractError("Branch reward must be finite or None.")
        if (
            isinstance(self.shared_prefix_weight, bool)
            or not isinstance(self.shared_prefix_weight, (int, float))
            or not math.isfinite(float(self.shared_prefix_weight))
            or self.shared_prefix_weight < 0
        ):
            raise AgentContractError(
                "shared_prefix_weight must be finite and nonnegative."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentEvent:
    """Append-only semantic evidence scoped to an immutable execution context.

    Event timestamps use monotonic seconds within the process. Event IDs are
    globally unique, while parent interaction IDs describe conversation lineage
    rather than event order. JSON serialization includes ``kind`` so readers
    do not need Python classes to distinguish event payloads.
    """

    context: ExecutionContext
    kind: str
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: float = field(default_factory=time.monotonic)
    latency: float = 0.0
    status: str = "completed"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationEvent(AgentEvent):
    """A successful model response together with its original token evidence."""

    kind: str = field(default="generation", init=False)
    generation: Generation


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallEvent(AgentEvent):
    """A requested tool call recorded before execution, including failed calls."""

    kind: str = field(default="tool_call", init=False)
    call: ToolCall


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultEvent(AgentEvent):
    """An observation linked to its tool request by the original call ID."""

    kind: str = field(default="tool_result", init=False)
    result: ToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardEvent(AgentEvent):
    """A raw local reward total with separate outcome/process/advantage views.

    These records are observations of scorer decisions, not an additional
    reward source. Export consumes EpisodeScore exactly once and never sums
    event payloads into that score a second time. ``reward`` already includes
    ``outcome_reward`` and ``process_rewards``; token_advantages are policy-only.
    """

    kind: str = field(default="reward", init=False)
    reward: float
    process_rewards: tuple[float, ...] = ()
    outcome_reward: float = 0.0
    token_advantages: tuple[float, ...] = ()


@dataclass(slots=True)
class RolloutExport:
    """Physical training rows belonging to one logical candidate execution.

    Each fragment ID maps to the interactions represented by that row. The
    collector validates row provenance, common initial prompts, and overlapping
    action positions before accepting an export. Metadata describes export
    semantics; it must contain JSON-compatible values when persisted.
    """

    trajectories: list[Trajectory]
    logical_rollout_id: str
    fragment_map: dict[str, tuple[str, ...]]
    group_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # The branch map includes unfinished siblings even though only completed
    # branches may produce trajectories. This makes partial tree collection
    # auditable without turning missing counterfactual outcomes into rewards.
    branch_map: dict[str, EpisodeBranch] = field(default_factory=dict)


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
    "cancelled",
    "rejected",
]
SYSTEM_STATUSES = frozenset(
    {
        "sampling_failure",
        "tool_failure",
        "scoring_failure",
        "contract_error",
        "timeout",
        "cancelled",
    }
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
    # Structured submissions are kept beside legacy fields so a scorer can
    # describe outcome, process and direct-advantage semantics without making
    # the execution layer understand PPO tensor shapes.
    assignments: tuple[Any, ...] = ()

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
        if not isinstance(self.assignments, (tuple, list)):
            raise AgentContractError("Reward assignments must be a sequence.")
        for assignment in self.assignments:
            if not hasattr(assignment, "target") or not hasattr(assignment, "value"):
                raise AgentContractError(
                    "Reward assignments must expose target and value."
                )
            validate = getattr(assignment, "__post_init__", None)
            if callable(validate):
                validate()
            if (
                isinstance(assignment.value, bool)
                or not isinstance(assignment.value, (int, float))
                or not math.isfinite(float(assignment.value))
            ):
                raise AgentContractError(
                    "Reward assignment values must be finite scalars."
                )
        object.__setattr__(self, "assignments", tuple(self.assignments))


@dataclass(slots=True)
class EpisodeResult:
    """Accumulate the auditable result of one task execution.

    ``episode_id`` identifies one attempt, ``rollout_id`` identifies the
    logical task execution, ``group_id`` connects candidates for relative
    reward training, and ``snapshot_id`` records the concrete sampler. The
    optional ``policy_version`` makes the model generation explicit without
    requiring consumers to parse snapshot IDs. ``is_eval`` labels provenance
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
    # ``rollout_id`` identifies the logical task execution that may later be
    # exported into one or more physical training rows. It is intentionally
    # separate from ``episode_id`` (one sampled attempt) and ``group_id``
    # (the PPO-relative comparison group). Appending the field preserves the
    # positional constructor shape used by earlier callers.
    rollout_id: str | None = None
    # The snapshot's policy version is carried with the episode so later
    # exporters and trainers can reject stale or mixed-policy data explicitly.
    policy_version: int | str | None = None
    context: ExecutionContext | None = None
    interactions: list[Interaction] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    export_metadata: dict[str, Any] = field(default_factory=dict)
    branches: list[EpisodeBranch] = field(default_factory=list)
    compactions: list[ContextCompaction] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate identity fields before tools or scorers observe a record.

        Candidate identity is part of the execution contract. A scorer that
        compares candidates must never receive a missing or negative index, and
        a later exporter must not silently attach a row to another snapshot.
        Optional values remain accepted for hand-built legacy records.
        """
        for name in ("episode_id", "group_id", "snapshot_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise AgentContractError(f"{name} must be a nonempty string.")
        if self.rollout_id is not None and (
            not isinstance(self.rollout_id, str) or not self.rollout_id
        ):
            raise AgentContractError("rollout_id must be a nonempty string or None.")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise AgentContractError("sample_index must be a nonnegative integer.")
        if self.policy_version is not None and (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, (int, str))
            or (isinstance(self.policy_version, str) and not self.policy_version)
        ):
            raise AgentContractError(
                "policy_version must be a nonempty integer/string or None."
            )
        if self.context is not None:
            for name in (
                "rollout_id",
                "episode_id",
                "group_id",
                "sample_index",
                "snapshot_id",
                "policy_version",
                "task_id",
            ):
                if getattr(self, name) != getattr(self.context, name):
                    raise AgentContractError(
                        f"Episode {name} differs from ExecutionContext."
                    )
        known = set()
        interaction_by_id = {}
        for interaction in self.interactions:
            interaction.__post_init__()
            if interaction.interaction_id in known or (
                interaction.parent_interaction_id is not None
                and interaction.parent_interaction_id not in known
            ):
                raise AgentContractError(
                    "Interaction lineage must be unique and parent-before-child."
                )
            for name in (
                "rollout_id",
                "episode_id",
                "group_id",
                "sample_index",
                "snapshot_id",
                "policy_version",
            ):
                if getattr(interaction.context, name) != getattr(self, name):
                    raise AgentContractError(f"Interaction changed candidate {name}.")
            known.add(interaction.interaction_id)
            interaction_by_id[interaction.interaction_id] = interaction
        branch_ids, branch_suffixes, branch_parents = set(), set(), {}
        for branch in self.branches:
            if not isinstance(branch, EpisodeBranch):
                raise AgentContractError(
                    "Episode branches must be EpisodeBranch records."
                )
            branch.__post_init__()
            if branch.branch_id in branch_ids:
                raise AgentContractError("Branch IDs must be unique within an episode.")
            branch_ids.add(branch.branch_id)
            if branch.parent_interaction_id is not None and (
                branch.parent_interaction_id not in interaction_by_id
            ):
                raise AgentContractError("Branch parent interaction is unknown.")
            previous = branch.parent_interaction_id
            for interaction_id in branch.interaction_ids:
                interaction = interaction_by_id.get(interaction_id)
                if interaction is None:
                    raise AgentContractError("Branch refers to an unknown interaction.")
                if interaction.parent_interaction_id != previous:
                    raise AgentContractError(
                        "Branch interactions must form one parent-linked suffix."
                    )
                if interaction_id in branch_suffixes:
                    raise AgentContractError(
                        "A branch-local interaction cannot belong to two suffixes."
                    )
                branch_suffixes.add(interaction_id)
                previous = interaction_id
            expected_parent = branch_parents.setdefault(
                branch.group_id, branch.parent_interaction_id
            )
            if expected_parent != branch.parent_interaction_id:
                raise AgentContractError(
                    "Branches in one group must start from the same parent interaction."
                )
        previous_compaction = 0
        generated_ids = [
            interaction.interaction_id
            for interaction in self.interactions
            if interaction.turn is not None
        ]
        for compaction in self.compactions:
            if not isinstance(compaction, ContextCompaction):
                raise AgentContractError(
                    "Episode compactions must be ContextCompaction records."
                )
            compaction.__post_init__()
            if (
                compaction.turn_index <= previous_compaction
                or compaction.turn_index > len(self.turns)
            ):
                raise AgentContractError(
                    "Compaction turn indices must be ordered generated-turn boundaries."
                )
            if (
                compaction.turn_index < len(self.turns)
                and tuple(self.turns[compaction.turn_index].prompt_tokens)
                != compaction.compacted_prompt_tokens
            ):
                raise AgentContractError(
                    "Compaction tokens differ from the sampled turn prompt."
                )
            prior_ids = set(generated_ids[: compaction.turn_index])
            if not set(
                compaction.retained_interaction_ids
                + compaction.summarized_interaction_ids
            ).issubset(prior_ids):
                raise AgentContractError(
                    "Compaction provenance refers to a non-prior interaction."
                )
            previous_compaction = compaction.turn_index

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
            and self.status != "rejected"
            and not self.export_metadata.get("system_failure", False)
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


class ConversationCompactor(Protocol):
    """Replace old message blocks without changing recorded sampled actions."""

    async def compact(
        self, messages: Sequence[Mapping[str, Any]], episode: EpisodeResult
    ) -> Any: ...


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


class InteractionScorer(Protocol):
    """Score one completed interaction without replacing episode outcome scoring."""

    async def score(
        self, row: Mapping[str, Any], episode: EpisodeResult, interaction: Interaction
    ) -> Any: ...


class FailureClassifier(Protocol):
    """Map an adapter exception to a stable episode status and optional message."""

    def classify(self, stage: str, error: BaseException, episode: EpisodeResult): ...


class SampleAcceptance(Protocol):
    """Decide whether a scored execution may enter a training batch."""

    def __call__(
        self, episode: EpisodeResult, export: RolloutExport | None
    ) -> bool | Awaitable[bool]: ...


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
    can fail early. ``export`` returns physical rows and their logical source
    mapping. Alternative layouts must preserve action identity, exact causal
    contexts, behavior scores, masks, and reward alignment across this boundary.
    A single-row ``build`` method may be exposed by concrete adapters for direct
    callers, but it is not required by this protocol.
    """

    def validate_prefix(
        self, episode: EpisodeResult, prompt_tokens: list[int]
    ) -> None: ...

    def export(self, episode: EpisodeResult) -> RolloutExport:
        """Return one or more rows with explicit logical and fragment identity."""
        ...


class EpisodeExecutor(Protocol):
    """Replace the complete agent loop while retaining collection and training.

    An SDK adapter implements ``run`` and returns a complete EpisodeResult with
    original token data. Identity arguments are supplied before execution so
    tools and scorers can make candidate-aware decisions. Context is supplied
    explicitly across process boundaries; in-process adapters can also inspect
    current_execution_context(). The collector needs ``limits.max_turns`` to
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
        rollout_id: str | None = None,
        sample_index: int = 0,
        context: ExecutionContext | None = None,
        seed: int | None = None,
        is_eval: bool = False,
    ) -> EpisodeResult: ...
