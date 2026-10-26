"""Track OpenAI request lineage, evidence, rewards, and terminal lifecycle.

An ``OpenAISession`` is the SDK-side owner of one logical conversation.  It
keeps the public session ID separate from collector identity, allocates one
child ``ExecutionContext`` per request, maps local/provider IDs, and validates
all delayed reward targets before mutating state.  It is intentionally a trace
and contract boundary: exporting an API trace does not fabricate a PPO
trajectory when a provider did not return exact token evidence.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any
from uuid import uuid4

from ...training.rewards import RewardAssignment
from ...types import (
    AgentContractError,
    AgentEvent,
    EpisodeScore,
    ExecutionContext,
    ToolResult,
    ToolResultEvent,
)


def _value(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an SDK response object or a JSON mapping."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(slots=True)
class OpenAIInteraction:
    """One request and all completion, sampling, tool, and reward evidence.

    ``status`` describes request progress independently from the containing
    session.  The request and response IDs are immutable lookup keys in normal
    operation; session validation detects an adapter that rewrites them after
    allocation.  PyTRIO fields remain optional so ordinary external provider
    traces can be recorded without pretending that text was re-tokenized.
    """

    request_id: str
    interaction_id: str
    parent_interaction_id: str | None
    endpoint: str
    started_at: float
    context: ExecutionContext
    request: dict[str, Any] = field(default_factory=dict)
    identity_payload: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    server_request_id: str | None = None
    error: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    logprobs: Any = None
    # PyTRIO-backed requests retain original IDs instead of reconstructing them
    # from decoded response text, which would invalidate behavior-policy data.
    input_ids: tuple[int, ...] | None = None
    output_ids: tuple[int, ...] | None = None
    stop_reason: str | None = None
    # The source distinguishes fixed-snapshot sampling from provider traces.
    sampling_source: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reward: Any = None
    status: str = "running"
    latency: float | None = None
    reward_submissions: list[Any] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OpenAITrajectoryExport:
    """Serializable API trace export, not an exact-token PPO ``RolloutExport``.

    External API usage and text logprobs do not establish original token IDs.
    A PyTRIO backend can retain exact arrays in interaction records, but those
    records still require an explicit episode/trajectory exporter before PPO.
    This trace must not be re-tokenized into fabricated behavior-policy data.
    """

    session_id: str
    logical_rollout_id: str
    interactions: tuple[dict[str, Any], ...]
    rewards: tuple[Any, ...]
    metadata: dict[str, Any]


class OpenAISession:
    """Own one session ID and preserve request/interaction lineage.

    Sessions begin ``open`` and become terminal only after all running work is
    idle.  Completed sessions accept delayed rewards; failed or cancelled
    sessions retain evidence for export but reject training labels.  Release
    freezes writes while keeping read-only inspection available to the caller.
    """

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
        max_generated_tokens: int | None = None,
        episode_timeout: float | None = None,
    ) -> None:
        """Bind collector identity or create independently identified inference records.

        Training provenance requires a complete root ExecutionContext with a
        known policy version. Legacy scalar identity arguments remain useful
        for inference; with a context they may only repeat its existing values.
        Binding training provenance does not make external API traces trainable.
        """
        if type(training) is not bool:
            raise AgentContractError("training must be boolean.")
        if max_generated_tokens is not None and (
            type(max_generated_tokens) is not int or max_generated_tokens < 1
        ):
            raise AgentContractError("max_generated_tokens must be positive or None.")
        if episode_timeout is not None and (
            isinstance(episode_timeout, bool)
            or not isinstance(episode_timeout, (int, float))
            or not math.isfinite(float(episode_timeout))
            or episode_timeout <= 0
        ):
            raise AgentContractError(
                "episode_timeout must be finite and positive or None."
            )
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
        # This in-process binding stores live source objects, never IDs or
        # credentials in exported metadata. Labels alone cannot distinguish
        # two clients serving different weights under the same version name.
        self._sampling_binding: tuple[Any, ...] | None = None
        self.interactions: list[OpenAIInteraction] = []
        self.rewards: list[Any] = []
        self.events: list[AgentEvent] = []
        self.retries: list[Any] = []
        self.failure_classification: tuple[str, str] | None = None
        self.accepted: bool | None = None
        self._reward_ids: dict[str, dict[str, Any]] = {}
        self._by_id: dict[str, OpenAIInteraction] = {}
        self._interaction_contexts: dict[str, ExecutionContext] = {}
        self.request_to_interaction: dict[str, str] = {}
        self.response_to_interaction: dict[str, str] = {}
        self.status = "open"
        self.error: str | None = None
        self.released = False
        self.running = False
        self._max_generated_tokens = max_generated_tokens
        self._episode_timeout = episode_timeout
        self._started_at = None

    def configure_limits(
        self,
        *,
        max_generated_tokens: int | None = None,
        episode_timeout: float | None = None,
    ) -> None:
        """Configure one attempt's total budgets before its first request.

        The session owns these limits so direct clients, the built-in Agent,
        proxy calls, and third-party callbacks cannot silently use different
        timeout or generated-token accounting. Existing interactions cannot be
        reconfigured; this prevents a later continuation from extending an
        already running episode.
        """
        if self.interactions or self.running or self.status != "open" or self.released:
            raise AgentContractError(
                "Session limits must be configured before execution."
            )
        if max_generated_tokens is not None and (
            type(max_generated_tokens) is not int or max_generated_tokens < 1
        ):
            raise AgentContractError("max_generated_tokens must be positive or None.")
        if episode_timeout is not None and (
            isinstance(episode_timeout, bool)
            or not isinstance(episode_timeout, (int, float))
            or not math.isfinite(float(episode_timeout))
            or episode_timeout <= 0
        ):
            raise AgentContractError(
                "episode_timeout must be finite and positive or None."
            )
        self._max_generated_tokens = max_generated_tokens
        self._episode_timeout = episode_timeout

    def begin_execution(self) -> None:
        """Start the monotonic clock used by all requests in this session."""
        if self._started_at is None:
            self._started_at = time.monotonic()

    def remaining_timeout(self) -> float | None:
        """Return seconds remaining in the total episode budget."""
        if self._episode_timeout is None or self._started_at is None:
            return None
        return self._episode_timeout - (time.monotonic() - self._started_at)

    def remaining_generated_tokens(self) -> int | None:
        """Return the unsampled output budget, using original IDs when present."""
        if self._max_generated_tokens is None:
            return None
        used = 0
        for item in self.interactions:
            if item.output_ids is not None:
                used += len(item.output_ids)
            elif item.output_tokens is not None:
                used += item.output_tokens
        return self._max_generated_tokens - used

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
        """Return whether this session was explicitly bound to training provenance."""
        return self._training

    @property
    def rollout_id(self) -> str:
        """Expose the collector rollout identity, not the SDK session ID."""
        return self.context.rollout_id

    @property
    def group_id(self) -> str:
        """Expose the collector group identity associated with this candidate."""
        return self.context.group_id

    def bind_sampling_source(self, *, snapshot, model: str) -> None:
        """Pin the actual sampler resources for this session's entire rollout.

        Training identity is checked by the backend before binding. Repeated
        requests may use different adapter wrappers only if the snapshot,
        sampler, tokenizer, and underlying sampling client are the same live
        objects. This check does not refresh weights or close trainer-owned
        resources; immutable policy internals remain the provider's contract.
        """
        self.validate_identity()
        sampler = snapshot.sampler
        if not self.training or self.status != "open" or self.released:
            raise AgentContractError(
                "Sampling source binding requires an open training session."
            )
        if any(
            item.input_ids is None
            for item in self.interactions
            if item.status != "running"
        ):
            raise AgentContractError(
                "An external-provider trace cannot become a fixed-policy rollout."
            )
        source = (
            snapshot,
            sampler,
            snapshot.tokenizer,
            getattr(sampler, "client", None),
            getattr(sampler, "tokenizer", None),
        )
        settings = (model, copy.deepcopy(getattr(sampler, "stop", None)))
        if self._sampling_binding is None:
            self._sampling_binding = (*source, settings)
        elif (
            any(
                left is not right
                for left, right in zip(self._sampling_binding[:-1], source, strict=True)
            )
            or self._sampling_binding[-1] != settings
        ):
            raise AgentContractError(
                "Session sampling source changed within a rollout."
            )

    @property
    def sampling_source_bound(self) -> bool:
        """Indicate whether requests are pinned to live fixed-policy resources."""
        return self._sampling_binding is not None

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
        """Reject lifecycle operations while agent or request work is active."""
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

    def _tool_result_bindings(
        self, endpoint, payload, parent_id, *, explicit_parent=False
    ):
        """Resolve third-party tool observations without trusting a bare call ID.

        Chat and full-history Responses requests normally echo the ancestor
        request/output prefix. Only calls at verified output positions can own
        subsequent observations. An explicit parent or Responses previous ID
        may instead scope a delta's result IDs to that response; native PyTRIO
        still enforces full causal history at its own sampling boundary.
        Unrelated sessions, fabricated assistant calls, repeated
        results within one payload, and conflicting replays are all rejected.

        This method plans changes only. Every observation is validated before
        request allocation or tool-result mutation, retaining the existing
        atomic failure and empty-session capacity rules.
        """
        key = "messages" if endpoint == "/v1/chat/completions" else "input"
        items = self._normalized(payload.get(key, []))
        if not isinstance(items, list):
            return []

        def is_result(item):
            return isinstance(item, dict) and (
                item.get("role") == "tool"
                if key == "messages"
                else item.get("type") == "function_call_output"
            )

        if not any(is_result(item) for item in items):
            return []
        if parent_id is None:
            raise AgentContractError("Tool results require a verified session parent.")
        ancestors = []
        owner = self.get_interaction(parent_id)
        while owner is not None:
            ancestors.append(owner)
            owner = (
                self.get_interaction(owner.parent_interaction_id)
                if owner.parent_interaction_id is not None
                else None
            )
        positions = {}
        for owner in ancestors:
            prefix = self._normalized(owner.request.get(key, []))
            if key == "input" and isinstance(prefix, str):
                prefix = [{"role": "user", "content": prefix}]
            if not isinstance(prefix, list):
                continue
            if key == "messages":
                choices = owner.response.get("choices", [])
                output = (
                    [choices[0]["message"]]
                    if len(choices) == 1 and choices[0].get("message")
                    else []
                )
            else:
                # A delta input is not a complete history. Its response can be
                # referenced explicitly below, but never inferred as a prefix.
                if owner.request.get("previous_response_id") is not None:
                    continue
                output = owner.response.get("output", [])
            expected = prefix + self._normalized(output)
            if output and items[: len(expected)] == expected:
                for offset in range(len(output)):
                    positions[len(prefix) + offset] = owner
        pending = {}

        def register_calls(owner, calls):
            if not isinstance(calls, list):
                raise AgentContractError("Recorded tool calls must be a list.")
            for call in calls:
                if not isinstance(call, dict):
                    raise AgentContractError("Recorded tool calls must be objects.")
                identity = call.get("id") if key == "messages" else call.get("call_id")
                function = call.get("function", {}) if key == "messages" else call
                if not isinstance(function, dict):
                    raise AgentContractError(
                        "Recorded function descriptors must be objects."
                    )
                name = function.get("name")
                if (
                    not isinstance(identity, str)
                    or not identity
                    or not isinstance(name, str)
                    or not name
                ):
                    raise AgentContractError(
                        "Recorded tools require a call ID and name."
                    )
                if identity in pending:
                    raise AgentContractError(
                        "Ambiguous tool-call ID in this conversation."
                    )
                pending[identity] = (owner, name)

        if (
            explicit_parent
            or key == "input"
            and payload.get("previous_response_id") is not None
        ) and ancestors[0] not in positions.values():
            # Explicit references retain the existing external-provider delta
            # contract. Native PyTRIO still checks complete causal messages in
            # its backend before sampling; no token prefix is inferred here.
            register_calls(ancestors[0], ancestors[0].tool_calls)
        bindings, seen = [], set()
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            calls = (
                item.get("tool_calls", [])
                if key == "messages"
                else [item]
                if item.get("type") == "function_call"
                else []
            )
            if calls:
                owner = positions.get(index)
                if owner is None:
                    raise AgentContractError(
                        "Tool observations cannot follow an unverified assistant call."
                    )
                register_calls(owner, calls)
            if not is_result(item):
                continue
            identity = (
                item.get("tool_call_id") if key == "messages" else item.get("call_id")
            )
            if not isinstance(identity, str) or identity not in pending:
                raise AgentContractError(
                    "Tool result call ID does not belong to this session history."
                )
            owner, name = pending.pop(identity)
            position = (owner.interaction_id, identity)
            if position in seen:
                raise AgentContractError(
                    "A request cannot contain duplicate tool results."
                )
            seen.add(position)
            content = item.get("content") if key == "messages" else item.get("output")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise AgentContractError(
                        "Recorded tool observations must be text or finite JSON."
                    ) from exc
            existing = [
                value
                for value in owner.tool_results
                if value.get("call_id") == identity
            ]
            if existing:
                if any(
                    value.get("content") != content or value.get("name") != name
                    for value in existing
                ):
                    raise AgentContractError(
                        "Tool result conflicts with recorded observation."
                    )
                continue
            bindings.append((owner, ToolResult(identity, name, content)))
        return bindings

    def start_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        parent_interaction_id: str | None = None,
        request_id: str | None = None,
        identity_payload: dict[str, Any] | None = None,
    ) -> OpenAIInteraction:
        """Allocate request and child identity before backend execution.

        Allocation precedes the network/backend call so failures still have a
        stable request ID, context, and latency record.  A session permits only
        one in-flight request; independent concurrent candidates must use
        separate sessions.
        """
        self.validate_identity()
        self.begin_execution()
        remaining = self.remaining_timeout()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Session episode timeout exceeded.")
        if self.status != "open" or self.released:
            raise AgentContractError(
                "A terminal or released session cannot start requests."
            )
        if any(item.status == "running" for item in self.interactions):
            raise AgentContractError(
                "Concurrent requests require separate OpenAI sessions."
            )
        parent = self._parent(endpoint, payload, parent_interaction_id)
        if request_id is not None and (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 128
            or request_id in self.request_to_interaction
        ):
            raise AgentContractError(
                "Request IDs must be new strings of 1 to 128 characters."
            )
        results = self._tool_result_bindings(
            endpoint, payload, parent, explicit_parent=parent_interaction_id is not None
        )
        interaction_id = f"int_{uuid4().hex}"
        interaction = OpenAIInteraction(
            request_id=request_id or f"req_{uuid4().hex}",
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
            identity_payload=copy.deepcopy(
                payload if identity_payload is None else identity_payload
            ),
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
        for owner, result in results:
            # The third-party tool already ran before this request arrived.
            # Persist its observation before contacting the model, so a later
            # provider timeout cannot erase completed external side effects.
            self.record_tool_results(owner, [result])
            self.events.append(ToolResultEvent(context=owner.context, result=result))
        return interaction

    def replay_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        request_id: str | None,
        parent_interaction_id: str | None = None,
    ) -> Any | None:
        """Resolve an explicit duplicate request without prompt-based guessing.

        Only an exact endpoint, payload, and resolved parent match can replay a
        completed response. The provider is not called again, no interaction or
        tool event is appended, and the stored response is copied before it is
        returned. Reusing an ID for different content or while the original is
        active/failed is a contract error rather than a second completion.
        """
        if request_id is None:
            return None
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise AgentContractError("Request IDs must contain 1 to 128 characters.")
        interaction_id = self.request_to_interaction.get(request_id)
        if interaction_id is None:
            return None
        interaction = self._by_id[interaction_id]
        parent = self._parent(endpoint, payload, parent_interaction_id)
        if (
            interaction.endpoint != endpoint
            or interaction.parent_interaction_id != parent
            or self._normalized(interaction.identity_payload)
            != self._normalized(payload)
        ):
            raise AgentContractError(
                "Request ID was reused with different request content."
            )
        if interaction.status != "completed" or not interaction.response:
            raise AgentContractError("Only a completed request can be replayed.")
        return copy.deepcopy(interaction.response)

    def complete_request(self, interaction: OpenAIInteraction, response: Any) -> None:
        """Capture provider IDs, usage, logprobs, and optional raw token evidence.

        The response may be an official SDK model or a JSON-compatible backend
        result.  PyTRIO's private evidence envelope is validated against the
        interaction context before it is stored, preventing stale snapshots or
        misaligned logprobs from entering the trace.
        """
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
        # The private evidence envelope is produced only by the fixed-snapshot
        # backend; validating it here keeps transport adapters from claiming a
        # different policy version or silently dropping behavior logprobs.
        sampling = _value(response, "_triotune_sampling", None)
        if sampling is not None:
            if not isinstance(sampling, dict):
                raise AgentContractError("Sampling evidence must be an object.")
            input_ids = sampling.get("input_ids")
            output_ids = sampling.get("output_ids")
            logprobs = sampling.get("logprobs")
            source = sampling.get("sampling_source")
            stop_reason = sampling.get("stop_reason")
            if (
                not isinstance(input_ids, list)
                or any(type(token) is not int or token < 0 for token in input_ids)
                or not input_ids
                or not isinstance(output_ids, list)
                or any(type(token) is not int or token < 0 for token in output_ids)
                or not output_ids
                or not isinstance(logprobs, list)
                or len(logprobs) != len(output_ids)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in logprobs
                )
                or not isinstance(stop_reason, str)
                or not stop_reason
                or not isinstance(source, str)
                or not source
            ):
                raise AgentContractError(
                    "Sampling evidence must contain aligned token IDs and logprobs."
                )
            if (
                sampling.get("snapshot_id") != interaction.context.snapshot_id
                or sampling.get("policy_version") != interaction.context.policy_version
            ):
                raise AgentContractError(
                    "Sampling evidence does not match the interaction snapshot."
                )
            interaction.input_ids = tuple(input_ids)
            interaction.output_ids = tuple(output_ids)
            interaction.logprobs = tuple(float(value) for value in logprobs)
            interaction.stop_reason = stop_reason
            interaction.sampling_source = source
        if (
            self._max_generated_tokens is not None
            and interaction.output_ids is None
            and interaction.output_tokens is None
        ):
            raise AgentContractError(
                "Generated-token budgets require exact IDs or provider output usage."
            )
        remaining_tokens = self.remaining_generated_tokens()
        if remaining_tokens is not None and remaining_tokens < 0:
            raise AgentContractError("Session total generated-token budget exceeded.")
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
        """Append tool observations only to this session's completed request."""
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
        if (
            assignment.target == "token"
            and positioned.output_ids is not None
            and assignment.position[1] >= len(positioned.output_ids)
        ):
            raise AgentContractError(
                "Token reward offset exceeds the original sampled output."
            )
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

        When exact output IDs are available, token positions and dense process
        vectors must match those arrays rather than decoded answer text or
        provider usage counters. External traces without exact IDs preserve
        their deferred token-validation semantics.
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
                original = self.interactions[index].output_ids
                if original is not None and len(values) != len(original):
                    raise AgentContractError(
                        "Process rewards must align with original sampled output tokens."
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
        """Export terminal API evidence without fabricating PPO token data.

        Caller metadata is merged for diagnostics, but trusted identity fields,
        status, release state, and ``trainable=False`` are written last so an
        exporter cannot relabel another candidate or opt an API trace into PPO.
        """
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
                "accepted": self.accepted,
                "failure_classification": self.failure_classification,
                "retries": self._jsonable(self.retries),
                "events": self._jsonable(self.events),
            },
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Convert SDK models and dataclasses into finite JSON-safe evidence."""
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
