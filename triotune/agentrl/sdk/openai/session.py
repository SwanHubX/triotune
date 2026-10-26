"""Session and request provenance for OpenAI agent calls."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any
from uuid import uuid4

from ...training.rewards import RewardAssignment
from ...types import AgentContractError, AgentEvent, EpisodeScore, ExecutionContext


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(slots=True)
class OpenAIInteraction:
    """One OpenAI request and its completion/tool evidence."""

    request_id: str
    interaction_id: str
    parent_interaction_id: str | None
    endpoint: str
    started_at: float
    context: ExecutionContext
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    server_request_id: str | None = None
    error: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    logprobs: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reward: Any = None
    status: str = "running"
    latency: float | None = None
    reward_submissions: list[Any] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OpenAITrajectoryExport:
    """API trace export, not an exact-token PPO ``RolloutExport``.

    API usage and text logprobs do not establish the original prompt/action
    token IDs. This trace must not be re-tokenized into behavior-policy data.
    """

    session_id: str
    logical_rollout_id: str
    interactions: tuple[dict[str, Any], ...]
    rewards: tuple[Any, ...]
    metadata: dict[str, Any]


class OpenAISession:
    """Own one session ID and preserve request/interaction lineage."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        context: ExecutionContext | None = None,
        training: bool = False,
        rollout_id: str | None = None,
        group_id: str | None = None,
        sample_index: int | None = None,
        snapshot_id: str | None = None,
        policy_version: int | str | None = None,
        resource_owner: str | None = None,
    ) -> None:
        """Bind collector identity or create independently identified inference records.

        Training provenance requires a complete root ExecutionContext with a
        known policy version. Legacy scalar identity arguments remain useful
        for inference; with a context they may only repeat its existing values.
        Binding training provenance does not make external API traces trainable.
        """
        if type(training) is not bool:
            raise AgentContractError("training must be boolean.")
        if training and context is None:
            raise AgentContractError(
                "Training sessions require a collector ExecutionContext."
            )
        identity = {
            "rollout_id": rollout_id,
            "group_id": group_id,
            "sample_index": sample_index,
            "snapshot_id": snapshot_id,
            "policy_version": policy_version,
            "resource_owner": resource_owner,
        }
        if context is not None:
            if not isinstance(context, ExecutionContext):
                raise AgentContractError("context must be an ExecutionContext.")
            context.__post_init__()
            if (
                context.interaction_id is not None
                or context.parent_interaction_id is not None
            ):
                raise AgentContractError(
                    "Session context must identify a root candidate."
                )
            if any(
                value is not None
                and (
                    type(value) is not type(getattr(context, key))
                    or value != getattr(context, key)
                )
                for key, value in identity.items()
            ):
                raise AgentContractError(
                    "Session identity overrides conflict with ExecutionContext."
                )
            if training and context.policy_version is None:
                raise AgentContractError(
                    "Training sessions require a known policy_version."
                )
        else:
            index = sample_index if sample_index is not None else 0
            context = ExecutionContext(
                rollout_id=rollout_id
                if rollout_id is not None
                else f"rollout_{uuid4().hex}",
                episode_id=f"episode_{uuid4().hex}",
                group_id=group_id if group_id is not None else f"group_{uuid4().hex}",
                sample_index=index,
                group_size=max(1, index + 1),
                snapshot_id=snapshot_id
                if snapshot_id is not None
                else f"external_{uuid4().hex}",
                policy_version=policy_version,
                resource_owner=resource_owner
                if resource_owner is not None
                else "openai-session",
            )
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise AgentContractError("session_id must be a nonempty string.")
        self._session_id = (
            session_id if session_id is not None else f"sess_{uuid4().hex}"
        )
        self._context = context
        self._identity = replace(context)
        self._training = training
        self.interactions: list[OpenAIInteraction] = []
        self.rewards: list[Any] = []
        self.events: list[AgentEvent] = []
        self._reward_ids: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, OpenAIInteraction] = {}
        self._interaction_contexts: dict[str, ExecutionContext] = {}
        self.request_to_interaction: dict[str, str] = {}
        self.response_to_interaction: dict[str, str] = {}
        self.status = "open"
        self.error: str | None = None
        self.released = False
        self.running = False

    @property
    def session_id(self) -> str:
        """Return the SDK session identity, independent of candidate identity."""
        return self._session_id

    @property
    def context(self) -> ExecutionContext:
        """Return the immutable candidate identity supplied by the collector."""
        return self._context

    @property
    def training(self) -> bool:
        return self._training

    @property
    def rollout_id(self) -> str:
        return self.context.rollout_id

    @property
    def group_id(self) -> str:
        return self.context.group_id

    def validate_identity(self, context: ExecutionContext | None = None) -> None:
        """Reject changed provenance before API calls, tool evidence or export.

        Interaction records remain mutable for results/status. Their context,
        parent and lookup identities must still match the original allocation;
        adapters cannot replace them with another candidate's otherwise valid
        context. A snapshot of the root identity also detects replacements of
        private attributes without relying on Python property setters alone.
        """
        if self.context != self._identity or (
            context is not None and context != self._identity
        ):
            raise AgentContractError(
                "Session identity differs from its ExecutionContext."
            )
        expected_requests: dict[str, str] = {}
        expected_responses: dict[str, str] = {}
        for item in self.interactions:
            expected = self._interaction_contexts.get(item.interaction_id)
            if (
                expected is None
                or item.context != expected
                or item.parent_interaction_id != expected.parent_interaction_id
                or self._by_id.get(item.interaction_id) is not item
                or self.request_to_interaction.get(item.request_id)
                != item.interaction_id
                or (
                    item.response_id is not None
                    and self.response_to_interaction.get(item.response_id)
                    != item.interaction_id
                )
            ):
                raise AgentContractError(
                    "Interaction identity differs from its allocated ExecutionContext."
                )
            expected_requests[item.request_id] = item.interaction_id
            if item.response_id is not None:
                expected_responses[item.response_id] = item.interaction_id
        # Check the reverse lookup tables as well as each record.  Without this
        # comparison, an adapter could add a second request/response alias for
        # an existing interaction and later use that unallocated alias to move
        # rewards or parent links across an otherwise valid session boundary.
        if (
            self.request_to_interaction != expected_requests
            or self.response_to_interaction != expected_responses
        ):
            raise AgentContractError(
                "Session request or response identity mappings were modified."
            )
        trusted_contexts = {self.context, *self._interaction_contexts.values()}
        if any(
            not isinstance(event, AgentEvent) or event.context not in trusted_contexts
            for event in self.events
        ):
            raise AgentContractError("Session event provenance was modified.")

    def get_interaction(
        self,
        interaction_id: str | None = None,
        *,
        request_id: str | None = None,
        response_id: str | None = None,
    ) -> OpenAIInteraction:
        """Resolve exactly one session-local reference without guessing its type.

        Request IDs are the locally allocated IDs sent in X-Client-Request-Id;
        response IDs are the provider's completion/response IDs. References
        from another session are never accepted as reward or parent targets.
        """
        self.validate_identity()
        if (
            sum(
                value is not None for value in (interaction_id, request_id, response_id)
            )
            != 1
        ):
            raise AgentContractError(
                "Supply exactly one interaction, request or response ID."
            )
        if request_id is not None:
            interaction_id = self.request_to_interaction.get(request_id)
        elif response_id is not None:
            interaction_id = self.response_to_interaction.get(response_id)
        if interaction_id not in self._by_id:
            raise AgentContractError("Unknown interaction reference in this session.")
        return self._by_id[interaction_id]

    def _require_idle(self) -> None:
        self.validate_identity()
        if self.running or any(item.status == "running" for item in self.interactions):
            raise AgentContractError(
                "Session still has active agent work or a request in flight."
            )

    def finish(self, status: str = "completed", *, error: str | None = None) -> None:
        """Seal a quiescent session; successful, failed and cancelled traces persist.

        Repeating the same terminal status is harmless. A terminal session
        cannot be reopened or changed to another outcome. Failed/cancelled
        sessions remain exportable for diagnostics, but do not accept rewards.
        """
        self._require_idle()
        if status not in {"completed", "failed", "cancelled"}:
            raise AgentContractError("Invalid terminal session status.")
        if self.status != "open":
            if self.status != status:
                raise AgentContractError(
                    "A terminal session cannot change its outcome."
                )
            return
        if status == "completed" and (
            not self.interactions
            or any(item.status != "completed" for item in self.interactions)
        ):
            raise AgentContractError("Completion requires completed request evidence.")
        self.status, self.error = status, error
        self.events.append(
            AgentEvent(
                context=self.context,
                kind="session_end",
                status=status,
                details={"error": error},
            )
        )

    def release(self) -> None:
        """Freeze writes after the owner has finished or aborted the session.

        Read-only export remains available after release, so a proxy can hand
        its completed trace to the caller without keeping it in its registry.
        """
        self._require_idle()
        if self.status == "open":
            raise AgentContractError("Finish or abort the session before release.")
        self.released = True

    @classmethod
    def _normalized(cls, value: Any) -> Any:
        """Ignore omitted optional fields when matching SDK messages to requests."""
        value = cls._jsonable(value)
        if isinstance(value, dict):
            return {
                key: cls._normalized(item)
                for key, item in value.items()
                if item is not None
                and not (
                    key == "content" and item == "" and value.get("role") == "assistant"
                )
            }
        if isinstance(value, list):
            return [cls._normalized(item) for item in value]
        return value

    def _parent(self, endpoint, payload, parent_interaction_id):
        """Prefer explicit lineage; otherwise find a complete message-prefix match.

        Repeated prompts are independent roots. Only a request containing a
        prior request AND its actual output can infer that request as parent.
        Responses previous_response_id is an explicit provider reference and
        must identify a completed response in this same session.
        """
        parent = (
            self.get_interaction(parent_interaction_id)
            if parent_interaction_id is not None
            else None
        )
        previous_id = (
            payload.get("previous_response_id") if endpoint == "/v1/responses" else None
        )
        if previous_id is not None:
            previous = self.get_interaction(response_id=previous_id)
            if previous.endpoint != "/v1/responses" or (
                parent is not None and parent is not previous
            ):
                raise AgentContractError("Conflicting parent response reference.")
            parent = previous
        if parent is not None:
            if parent.status != "completed":
                raise AgentContractError("Parent interaction must be completed.")
            if parent.endpoint != endpoint:
                raise AgentContractError(
                    "Parent interaction must use the same API endpoint."
                )
            return parent.interaction_id
        key = "messages" if endpoint == "/v1/chat/completions" else "input"
        child = self._normalized(payload.get(key, []))
        if not isinstance(child, list):
            return None
        matches = []
        for candidate in self.interactions:
            if candidate.endpoint != endpoint or candidate.status != "completed":
                continue
            # A previous_response_id request contains only a delta. Matching
            # that delta plus output would omit its inherited server history
            # and could attach an unrelated root as a child. Require an
            # explicit reference to this candidate instead of guessing history.
            if (
                key == "input"
                and candidate.request.get("previous_response_id") is not None
            ):
                continue
            prefix = self._normalized(candidate.request.get(key, []))
            if key == "input" and isinstance(prefix, str):
                prefix = [{"role": "user", "content": prefix}]
            if not isinstance(prefix, list):
                continue
            if key == "messages":
                choices = candidate.response.get("choices", [])
                output = (
                    [choices[0]["message"]]
                    if len(choices) == 1 and choices[0].get("message")
                    else []
                )
            else:
                output = candidate.response.get("output", [])
            if not output:
                continue
            expected = prefix + self._normalized(output)
            if child[: len(expected)] == expected:
                matches.append((len(expected), candidate.interaction_id))
        if not matches:
            return None
        longest = max(length for length, _ in matches)
        parents = [identity for length, identity in matches if length == longest]
        if len(parents) != 1:
            raise AgentContractError(
                "Ambiguous message prefix requires an explicit parent."
            )
        return parents[0]

    def start_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        parent_interaction_id: str | None = None,
    ) -> OpenAIInteraction:
        """Allocate request and child interaction identity before an API call."""
        self.validate_identity()
        if self.status != "open" or self.released:
            raise AgentContractError(
                "A terminal or released session cannot start requests."
            )
        if any(item.status == "running" for item in self.interactions):
            raise AgentContractError(
                "Concurrent requests require separate OpenAI sessions."
            )
        parent = self._parent(endpoint, payload, parent_interaction_id)
        interaction_id = f"int_{uuid4().hex}"
        interaction = OpenAIInteraction(
            request_id=f"req_{uuid4().hex}",
            interaction_id=interaction_id,
            parent_interaction_id=parent,
            endpoint=endpoint,
            started_at=time.monotonic(),
            context=replace(
                self.context,
                interaction_id=interaction_id,
                parent_interaction_id=parent,
            ),
            request=copy.deepcopy(payload),
        )
        self.interactions.append(interaction)
        self._by_id[interaction_id] = interaction
        self._interaction_contexts[interaction_id] = replace(interaction.context)
        self.request_to_interaction[interaction.request_id] = interaction_id
        self.events.append(
            AgentEvent(
                context=interaction.context,
                kind="openai_request",
                details={"request_id": interaction.request_id, "endpoint": endpoint},
            )
        )
        return interaction

    def complete_request(self, interaction: OpenAIInteraction, response: Any) -> None:
        """Capture IDs, usage and available token logprobs from a response."""
        self.validate_identity()
        if (
            self._by_id.get(interaction.interaction_id) is not interaction
            or interaction.status != "running"
        ):
            raise AgentContractError("Only this session's active request can complete.")
        response_id = _value(response, "id", None)
        if response_id is not None and (
            not isinstance(response_id, str)
            or not response_id
            or response_id in self.response_to_interaction
        ):
            raise AgentContractError(
                "Response IDs must be nonempty and unique within the session."
            )
        interaction.response = self._jsonable(response)
        interaction.response_id = response_id
        if response_id is not None:
            self.response_to_interaction[response_id] = interaction.interaction_id
        interaction.server_request_id = _value(response, "_request_id", None)
        usage = _value(response, "usage", None)
        if usage is not None:
            interaction.input_tokens = _value(
                usage, "prompt_tokens", _value(usage, "input_tokens", None)
            )
            interaction.output_tokens = _value(
                usage, "completion_tokens", _value(usage, "output_tokens", None)
            )
        choices = _value(response, "choices", ()) or ()
        if choices:
            interaction.logprobs = self._jsonable(_value(choices[0], "logprobs", None))
            raw_calls = _value(_value(choices[0], "message", {}), "tool_calls", ())
            interaction.tool_calls.extend(self._jsonable(raw_calls or ()))
        output = _value(response, "output", ()) or ()
        for item in output:
            if _value(item, "type") == "function_call":
                interaction.tool_calls.append(self._jsonable(item))
        response_logprobs = [
            self._jsonable(_value(part, "logprobs"))
            for item in output
            for part in (_value(item, "content", ()) or ())
            if _value(part, "logprobs") is not None
        ]
        if response_logprobs:
            interaction.logprobs = response_logprobs
        interaction.status = _value(response, "status", "completed") or "completed"
        if choices and _value(choices[0], "finish_reason") in {
            "length",
            "content_filter",
        }:
            interaction.status = "incomplete"
        interaction.latency = time.monotonic() - interaction.started_at
        self.events.append(
            AgentEvent(
                context=interaction.context,
                kind="openai_completion",
                status=interaction.status,
                latency=interaction.latency,
                details={
                    "response_id": interaction.response_id,
                    "request_id": interaction.request_id,
                },
            )
        )
        if interaction.status != "completed" and not self.running:
            self.finish("cancelled" if interaction.status == "cancelled" else "failed")

    def record_tool_results(self, interaction: OpenAIInteraction, results) -> None:
        """Append results only to a completed API request in the open session."""
        self.validate_identity()
        if (
            self.status != "open"
            or self.released
            or self._by_id.get(interaction.interaction_id) is not interaction
            or interaction.status != "completed"
        ):
            raise AgentContractError(
                "Tool results require this open session's completed request."
            )
        interaction.tool_results.extend(self._jsonable(results))

    def _reward_target(self, assignment: RewardAssignment, explicit_target=None):
        """Resolve top-level and nested assignments with identical position rules.

        Positions always refer to session request indexes. An explicit ID can
        supply a missing interaction/turn position, but cannot override one.
        Trajectory contributions remain session scoped even inside a score.
        """
        if not isinstance(assignment, RewardAssignment):
            raise AgentContractError(
                "Session assignments require RewardAssignment objects."
            )
        assignment.__post_init__()
        if assignment.target == "trajectory":
            if explicit_target is not None:
                raise AgentContractError(
                    "Trajectory rewards cannot target one interaction."
                )
            return None
        if assignment.position is None:
            if explicit_target is None or assignment.target == "token":
                raise AgentContractError(
                    "Reward requires a position or explicit interaction reference; token rewards require an offset."
                )
            return explicit_target
        index = (
            assignment.position[0]
            if assignment.target == "token"
            else assignment.position
        )
        if index >= len(self.interactions):
            raise AgentContractError(
                "Reward position refers to an unknown interaction."
            )
        positioned = self.interactions[index]
        if explicit_target is not None and explicit_target is not positioned:
            raise AgentContractError(
                "Reward position conflicts with its explicit target."
            )
        return positioned

    def submit_reward(
        self,
        reward: Any,
        *,
        submission_id: str | None = None,
        interaction_id: str | None = None,
        request_id: str | None = None,
        response_id: str | None = None,
    ) -> Any:
        """Bind a delayed reward only after successful session completion.

        Interaction/turn positions index recorded requests; token positions
        retain (request index, token offset) without inventing token arrays.
        Trajectory assignments and EpisodeScore outcomes belong to the session.
        EpisodeScore process vectors and nested assignments bind independently;
        only legacy scalar submissions default to the final interaction.
        Idempotency compares both the complete reward payload and its resolved
        target, so reuse of a submission ID cannot silently move a reward.
        """
        self._require_idle()
        if self.status != "completed" or self.released:
            raise AgentContractError("Rewards require a completed, unreleased session.")
        if submission_id is not None and (
            not isinstance(submission_id, str) or not submission_id
        ):
            raise AgentContractError("submission_id must be a nonempty string.")
        if isinstance(reward, (RewardAssignment, EpisodeScore)):
            reward.__post_init__()
        elif type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError("Session reward must be finite or a structured reward.")
        target = None
        if any(
            value is not None for value in (interaction_id, request_id, response_id)
        ):
            target = self.get_interaction(
                interaction_id, request_id=request_id, response_id=response_id
            )
        # Resolve and serialize every contribution before touching stored
        # rewards, events or idempotency keys. A bad nested target rejects the
        # entire score, including its otherwise valid outcome and process data.
        bindings = []
        if isinstance(reward, EpisodeScore):
            if target is not None:
                raise AgentContractError(
                    "EpisodeScore is session scoped; target individual assignments instead."
                )
            bindings.append(
                (
                    None,
                    {
                        "reward": reward.reward,
                        "success": reward.success,
                        "components": copy.deepcopy(reward.components),
                    },
                )
            )
            for index, values in reward.process_rewards.items():
                if index >= len(self.interactions):
                    raise AgentContractError(
                        "Process reward refers to an unknown interaction."
                    )
                for offset, value in enumerate(values):
                    assignment = RewardAssignment(
                        "token",
                        value,
                        source="episode_score/process",
                        position=(index, offset),
                    )
                    bindings.append(
                        (self._reward_target(assignment), self._jsonable(assignment))
                    )
            for assignment in reward.assignments:
                bindings.append(
                    (self._reward_target(assignment), self._jsonable(assignment))
                )
        elif isinstance(reward, RewardAssignment):
            target = self._reward_target(reward, target)
            bindings.append((target, self._jsonable(reward)))
        else:
            target = target or self.interactions[-1]
            bindings.append((target, reward))
        payload = {
            "reward": self._jsonable(reward),
            "bindings": [
                {
                    "interaction_id": item.interaction_id if item else None,
                    "reward": value,
                }
                for item, value in bindings
            ],
        }
        json.dumps(payload, allow_nan=False)
        if submission_id is not None and submission_id in self._reward_ids:
            if self._reward_ids[submission_id] != payload:
                raise AgentContractError(
                    "submission_id was already used with different reward content or target."
                )
            return reward
        self.rewards.append(copy.deepcopy(reward))
        if submission_id is not None:
            self._reward_ids[submission_id] = copy.deepcopy(payload)
        for item, value in bindings:
            if item is not None:
                item.reward = copy.deepcopy(value)
                item.reward_submissions.append(copy.deepcopy(value))
        self.events.append(
            AgentEvent(
                context=target.context if target is not None else self.context,
                kind="reward_submission",
                details={
                    "reward": self._jsonable(reward),
                    "submission_id": submission_id,
                    "interaction_id": target.interaction_id if target else None,
                    "bindings": copy.deepcopy(payload["bindings"]),
                },
            )
        )
        return reward

    def export_trajectory(
        self, *, metadata: dict[str, Any] | None = None
    ) -> OpenAITrajectoryExport:
        """Export API evidence without fabricating token IDs or PPO logprobs."""
        self._require_idle()
        if self.status == "open":
            raise AgentContractError("Finish or abort the session before export.")
        return OpenAITrajectoryExport(
            session_id=self.session_id,
            logical_rollout_id=self.rollout_id,
            interactions=tuple(self._jsonable(item) for item in self.interactions),
            rewards=tuple(self._jsonable(item) for item in self.rewards),
            metadata={
                **(metadata or {}),
                "group_id": self.group_id,
                "episode_id": self.context.episode_id,
                "group_size": self.context.group_size,
                "task_id": self.context.task_id,
                "sample_index": self.context.sample_index,
                "snapshot_id": self.context.snapshot_id,
                "policy_version": self.context.policy_version,
                "resource_owner": self.context.resource_owner,
                "context": asdict(self.context),
                "training_context": self.training,
                "trainable": False,
                "status": self.status,
                "error": self.error,
                "released": self.released,
                "events": self._jsonable(self.events),
            },
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [OpenAISession._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): OpenAISession._jsonable(item) for key, item in value.items()
            }
        if is_dataclass(value):
            return OpenAISession._jsonable(asdict(value))
        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                return OpenAISession._jsonable(method())
        if hasattr(value, "__dict__"):
            return OpenAISession._jsonable(vars(value))
        raise TypeError(f"Unsupported session evidence: {type(value).__name__}.")
