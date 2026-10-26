"""Session and request provenance for OpenAI agent calls."""

from __future__ import annotations

import copy
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
        rollout_id: str | None = None,
        group_id: str | None = None,
        sample_index: int = 0,
        snapshot_id: str | None = None,
        policy_version: int | str | None = None,
        resource_owner: str = "openai-session",
    ) -> None:
        self.session_id = session_id or f"sess_{uuid4().hex}"
        self.rollout_id = rollout_id or self.session_id
        self.group_id = group_id or self.session_id
        self.context = ExecutionContext(
            rollout_id=self.rollout_id,
            episode_id=self.session_id,
            group_id=self.group_id,
            sample_index=sample_index,
            group_size=max(1, sample_index + 1),
            snapshot_id=snapshot_id or self.session_id,
            policy_version=policy_version,
            resource_owner=resource_owner,
        )
        self.interactions: list[OpenAIInteraction] = []
        self.rewards: list[Any] = []
        self.events: list[AgentEvent] = []
        self._reward_ids: set[str] = set()
        self.running = False

    def start_request(
        self, endpoint: str, payload: dict[str, Any]
    ) -> OpenAIInteraction:
        """Allocate request and child interaction identity before an API call."""
        if any(item.status == "running" for item in self.interactions):
            raise AgentContractError(
                "Concurrent requests require separate OpenAI sessions."
            )
        parent = self.interactions[-1].interaction_id if self.interactions else None
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
        interaction.response = self._jsonable(response)
        interaction.response_id = _value(response, "id", None)
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

    def record_tool_results(self, interaction: OpenAIInteraction, results) -> None:
        interaction.tool_results.extend(self._jsonable(results))

    def submit_reward(self, reward: Any, *, submission_id: str | None = None) -> Any:
        """Validate a delayed reward and attach it to the terminal interaction.

        An optional caller-owned submission ID makes retries idempotent. The
        reward does not cause a failed or incomplete API request to become a
        trainable PPO sample.
        """
        if isinstance(reward, (RewardAssignment, EpisodeScore)):
            reward.__post_init__()
        elif type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError("Session reward must be finite or a structured reward.")
        if submission_id is not None and submission_id in self._reward_ids:
            return reward
        self.rewards.append(reward)
        if submission_id is not None:
            self._reward_ids.add(submission_id)
        if self.interactions:
            self.interactions[-1].reward = self._jsonable(reward)
        self.events.append(
            AgentEvent(
                context=self.context,
                kind="reward_submission",
                details={
                    "reward": self._jsonable(reward),
                    "submission_id": submission_id,
                },
            )
        )
        return reward

    def export_trajectory(
        self, *, metadata: dict[str, Any] | None = None
    ) -> OpenAITrajectoryExport:
        """Export API evidence without fabricating token IDs or PPO logprobs."""
        return OpenAITrajectoryExport(
            session_id=self.session_id,
            logical_rollout_id=self.rollout_id,
            interactions=tuple(self._jsonable(item) for item in self.interactions),
            rewards=tuple(self._jsonable(item) for item in self.rewards),
            metadata={
                **(metadata or {}),
                "group_id": self.group_id,
                "sample_index": self.context.sample_index,
                "snapshot_id": self.context.snapshot_id,
                "policy_version": self.context.policy_version,
                "resource_owner": self.context.resource_owner,
                "trainable": False,
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
