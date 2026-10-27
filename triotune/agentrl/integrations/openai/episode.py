"""Bridge fixed-policy SDK executions into the existing EpisodeRunner/PPO path."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Callable
from dataclasses import replace
from weakref import WeakSet

from ...runtime.runner import EpisodeRunner
from ...runtime.sampling import SamplingSnapshot
from ...training.rewards import RewardAssignment
from ...types import (
    ActionParseError,
    AgentContractError,
    AgentTurn,
    Generation,
    GenerationEvent,
    EpisodeScorer,
    Interaction,
    ToolCallEvent,
    ToolResult,
    ToolSystemError,
)
from .backend import PyTRIOBackend, _EpisodeLimitReached
from .native import OpenAINativeActionParser
from .session import OpenAISession
from .workflow import OpenAIWorkflow


class OpenAIEpisodeExecutor(EpisodeRunner):
    """Run a candidate-owned SDK workflow under a collector-owned snapshot.

    ``workflow_factory(snapshot)`` must return a fresh OpenAIWorkflow and fresh
    PyTRIOBackend bound to that exact snapshot. This avoids mutating a shared
    Agent/client when candidates overlap or PPO publishes the next policy.
    The factory transfers workflow lifecycle ownership, never sampler ownership.

    ``scorer`` and optional interaction_scorer use the native EpisodeRunner
    contracts, including access to the original row's ground truth. SDK reward
    submissions are additional explicit contributions; avoid submitting the
    same outcome through both scorers. PPO alone owns shaping/normalization,
    GAE, critic targets, and physical-row weights.
    """

    def __init__(
        self,
        workflow_factory: Callable[[SamplingSnapshot], OpenAIWorkflow],
        *,
        scorer: EpisodeScorer,
        execute=None,
        **kwargs,
    ):
        if not callable(workflow_factory):
            raise TypeError("workflow_factory must be callable.")
        self.workflow_factory = workflow_factory
        self.execute = execute
        self._workflows, self._backends = WeakSet(), WeakSet()
        self._clients, self._agents = WeakSet(), WeakSet()
        # The SDK owns rendering, parsing, and tool execution. The base runner
        # retains resource, timeout, scoring, reward-commit, and export behavior.
        super().__init__(
            parser=None, tools=None, observations=None, scorer=scorer, **kwargs
        )

    async def _execute(self, episode, messages, snapshot, renderer, seed):
        del renderer
        workflow = self.workflow_factory(snapshot)
        if not isinstance(workflow, OpenAIWorkflow):
            raise AgentContractError("workflow_factory must return OpenAIWorkflow.")
        backend = workflow.client.backend
        if not isinstance(backend, PyTRIOBackend) or backend.snapshot is not snapshot:
            raise AgentContractError(
                "SDK training requires PyTRIOBackend bound to the supplied snapshot."
            )
        resources = [(workflow, self._workflows), (backend, self._backends),
                     (workflow.client, self._clients)]
        if workflow.agent is not None:
            resources.append((workflow.agent, self._agents))
        if any(resource in used for resource, used in resources):
            raise AgentContractError(
                "SDK candidates require fresh workflows, agents, clients, and backends."
            )
        for resource, used in resources:
            used.add(resource)
        backend.episode_limits, backend.episode_seed = self.limits, seed
        session = OpenAISession(
            context=episode.context,
            training=True,
            max_generated_tokens=self.limits.max_generated_tokens,
            episode_timeout=self.limits.episode_timeout,
        )
        if workflow.agent is not None:
            # These objects belong to this candidate, not to another rollout.
            workflow.agent.max_turns = min(
                workflow.agent.max_turns, self.limits.max_turns
            )
            workflow.agent.max_generated_tokens = min(
                workflow.agent.max_generated_tokens or self.limits.max_generated_tokens,
                self.limits.max_generated_tokens,
            )
            workflow.agent.episode_timeout = min(
                workflow.agent.episode_timeout or self.limits.episode_timeout,
                self.limits.episode_timeout,
            )
            workflow.agent.tool_timeout = min(
                workflow.agent.tool_timeout, self.limits.tool_timeout
            )
        error = None
        try:
            result = await workflow.run(
                copy.deepcopy(
                    messages[0]["content"]
                    if len(messages) == 1 and messages[0]["role"] == "user"
                    else messages
                ),
                session=session,
                snapshot=snapshot,
                execute=self.execute,
            )
            if not result.accepted:
                episode.status = "rejected"
            elif session.status != "completed":
                episode.status = (
                    "token_limit"
                    if any(
                        item.stop_reason in {"length", "max_length", "max_tokens"}
                        for item in session.interactions
                    )
                    else "sampling_failure"
                )
        except asyncio.CancelledError as exc:
            error = exc
            raise
        except _EpisodeLimitReached as exc:
            episode.status, episode.error = exc.status, str(exc)
        except ActionParseError as exc:
            episode.status = (
                "token_limit"
                if session.interactions and session.interactions[-1].stop_reason
                in {"length", "max_length", "max_tokens"}
                else "invalid_action"
            )
            episode.error = str(exc)
        except (AgentContractError, ToolSystemError) as exc:
            error = exc
            raise
        except Exception as exc:
            error = exc
            # Budget exits are model-side partial episodes. Infrastructure or
            # judge failures retain evidence but never become zero-reward rows.
            if session.remaining_generated_tokens() == 0:
                episode.status = "token_limit"
            elif (
                workflow.agent is not None
                and len(session.interactions) == workflow.agent.max_turns
                and "max_turns" in str(exc)
            ):
                episode.status = "max_turns"
            elif isinstance(exc, TimeoutError):
                raise
            else:
                stage = "scoring" if session.accepted is False else "sampling"
                classified = self._classify(stage, exc, episode)
                episode.status, episode.error = classified or (
                    f"{stage}_failure",
                    str(exc),
                )
        finally:
            # Convert while unwinding as well: a later tool/API failure must
            # not erase successful generations or already completed tools.
            try:
                self._record(episode, session, snapshot)
            finally:
                try:
                    await workflow.close()
                    if workflow.agent is not None:
                        await workflow.agent.close()
                    else:
                        await workflow.client.close()
                except Exception as cleanup:
                    if error is None:
                        raise
                    error.add_note(f"SDK workflow cleanup also failed: {cleanup}")

    @staticmethod
    def _record(episode, session, snapshot):
        """Translate recorded IDs directly; decoded text is only a parser view."""
        session.validate_identity(episode.context)
        parser = OpenAINativeActionParser()
        for item in session.interactions:
            turn = None
            if item.output_ids is not None:
                evidence = item.response.get("_triotune_sampling", {})
                if (
                    item.sampling_source != "pytrio"
                    or tuple(evidence.get("input_ids", ())) != item.input_ids
                    or tuple(evidence.get("output_ids", ())) != item.output_ids
                    or tuple(evidence.get("logprobs", ())) != item.logprobs
                    or evidence.get("snapshot_id") != snapshot.snapshot_id
                    or evidence.get("policy_version") != snapshot.policy_version
                ):
                    raise AgentContractError(
                        "SDK sampling evidence differs from its fixed-policy record."
                    )
                generation = Generation(
                    item.output_ids,
                    item.logprobs,
                    snapshot.tokenizer.decode(
                        list(item.output_ids), skip_special_tokens=True
                    ),
                    item.stop_reason,
                    item.input_tokens or 0,
                    item.output_tokens or 0,
                )
                action = None
                if not evidence.get("parse_error") and item.stop_reason not in {
                    "length",
                    "max_length",
                    "max_tokens",
                }:
                    try:
                        action = (
                            parser.parse_chat_completion(item.response)
                            if item.endpoint == "/v1/chat/completions"
                            else parser.parse_response(item.response)
                        )
                    except ActionParseError as exc:
                        episode.status, episode.error = "invalid_action", str(exc)
                turn = AgentTurn(
                    item.input_ids,
                    generation,
                    action,
                    [ToolResult(**value) for value in item.tool_results],
                )
                episode.turns.append(turn)
                timestamp = item.started_at + (item.latency or 0.0)
                episode.events.append(
                    GenerationEvent(
                        context=item.context, generation=generation, timestamp=timestamp
                    )
                )
                if action is not None:
                    for call in action.tool_calls:
                        if not any(
                            isinstance(event, ToolCallEvent)
                            and event.context == item.context
                            and event.call.call_id == call.call_id
                            for event in session.events
                        ):
                            episode.events.append(
                                ToolCallEvent(
                                    context=item.context, call=call, timestamp=timestamp
                                )
                            )
                    if action.final_text is not None:
                        episode.final_text = action.final_text
            elif item.status == "completed":
                raise AgentContractError(
                    "SDK training requires original IDs and behavior logprobs for every completed request."
                )
            count = len(item.input_ids or ())
            episode.interactions.append(
                Interaction(
                    context=item.context,
                    input_token_span=(0, count),
                    output_token_span=(
                        (count, count + len(item.output_ids)) if turn else None
                    ),
                    turn=turn,
                    latency=item.latency
                    or max(0.0, time.monotonic() - item.started_at),
                    status=item.status,
                    resource_owner=item.context.resource_owner,
                )
            )
        episode.events.extend(copy.deepcopy(session.events))
        episode.events.sort(key=lambda event: event.timestamp)
        episode.retries.extend(copy.deepcopy(session.retries))
        for index, interaction in enumerate(episode.interactions):
            interaction.retries = [
                event for event in episode.retries if event.turn == index
            ]
        # Keep JSON-safe source mappings rather than a live SDK/session object
        # in EpisodeResult; normal JSONL logging and wheel imports stay usable.
        episode.export_metadata["sdk"] = {
            "session_id": session.session_id,
            "status": session.status,
            "error": session.error,
            "accepted": session.accepted,
            "failure_classification": session.failure_classification,
            "requests": [
                {
                    "request_id": item.request_id,
                    "response_id": item.response_id,
                    "interaction_id": item.interaction_id,
                }
                for item in session.interactions
            ],
        }
        episode.export_metadata["sdk_reward_bindings"] = [
            copy.deepcopy(binding)
            for event in session.events
            if event.kind == "reward_submission"
            for binding in event.details["bindings"]
        ]
        episode.__post_init__()

    async def _score_episode(self, row, episode):
        """Merge delayed SDK contributions once before the native reward commit."""
        score = await super()._score_episode(row, episode)
        assignments = list(score.assignments)
        turns = {
            item.interaction_id: index
            for index, item in enumerate(
                value for value in episode.interactions if value.turn is not None
            )
        }
        for binding in episode.export_metadata.get("sdk_reward_bindings", ()):
            value = binding["reward"]
            if isinstance(value, dict) and "target" in value:
                value = dict(value)
                # Keep SDK auxiliary scorers distinct from the native primary
                # scorer, like CompositeScorer's term namespaces. Otherwise two
                # independent judges using the default source could accidentally
                # share a mean/last population or disagree on aggregation mode.
                value["source"] = f"sdk/assignment/{value['source']}"
                if value["target"] == "token":
                    value["position"] = tuple(value["position"])
                if value["target"] != "trajectory":
                    index = turns[binding["interaction_id"]]
                    value["position"] = (
                        (index, value["position"][1])
                        if value["target"] == "token"
                        else index
                    )
                assignments.append(RewardAssignment(**value))
            elif isinstance(value, dict):
                assignments.append(
                    RewardAssignment(
                        "trajectory", value["reward"], source="sdk/episode-outcome"
                    )
                )
            else:
                assignments.append(
                    RewardAssignment(
                        "interaction",
                        value,
                        source="sdk/scalar",
                        position=turns[binding["interaction_id"]],
                    )
                )
        return replace(score, assignments=tuple(assignments))
