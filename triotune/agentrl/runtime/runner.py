"""Run the bounded model-action-observation loop for one agent episode.

The runner coordinates rendering, sampling, parsing, tools, and terminal
scoring through their interfaces. It records what happened before deciding
whether the outcome is a model failure or an infrastructure failure. Group
collection, persistent logging, trajectory export, and optimizer updates remain
the responsibility of their respective orchestration layers.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any
from uuid import uuid4

from pytrio import PyTrioError

from ..config import RetryPolicy
from ..rollout.trajectory import EpisodeTrajectoryBuilder
from ..training.rewards import RewardAssignment, RewardLedger
from ..types import (
    SYSTEM_STATUSES,
    ActionParseError,
    ActionParser,
    AgentContractError,
    AgentEvent,
    AgentTurn,
    ConversationRenderer,
    EpisodeLimits,
    EpisodeResult,
    EpisodeScore,
    EpisodeScorer,
    ExecutionContext,
    FailureClassifier,
    GenerationEvent,
    Interaction,
    InteractionScorer,
    ObservationFormatter,
    RetryEvent,
    RewardEvent,
    ToolCallEvent,
    ToolExecutor,
    ToolResultEvent,
    ToolSystemError,
    TrajectoryExporter,
    _execution_context,
    validate_tokens,
)
from .sampling import SamplingSnapshot


class RuntimeBoundary:
    """Share retries, failure classification, and owned-resource lifecycle.

    EpisodeRunner and SDK execution use the same boundary semantics without
    sharing model/tool-loop implementations. A record must expose ``context``
    and ``events``; optional ``retries`` and ``export_metadata`` fields receive
    the richer EpisodeResult evidence when available. Cancellation is never
    caught as a retryable failure.
    """

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        failure_classifier: FailureClassifier | None = None,
        resource_owner: str = "runner",
    ) -> None:
        if not isinstance(resource_owner, str) or not resource_owner:
            raise ValueError("resource_owner must be a nonempty string.")
        self.retry = retry or RetryPolicy()
        self.failure_classifier = failure_classifier
        self.resource_owner = resource_owner
        self.started = False
        self._lock = asyncio.Lock()
        self._resources = []

    async def start(self, *resources) -> None:
        """Start each owned resource once in declaration order."""
        async with self._lock:
            try:
                for resource in resources:
                    if resource is None or any(
                        resource is current for current in self._resources
                    ):
                        continue
                    if (
                        getattr(resource, "resource_owner", self.resource_owner)
                        == self.resource_owner
                    ):
                        method = getattr(resource, "start", None)
                        if callable(method):
                            result = method()
                            if inspect.isawaitable(result):
                                await result
                    self._resources.append(resource)
                self.started = True
            except BaseException as error:
                # Partial startup must not leak the resources that succeeded
                # before a later resource failed. Preserve the startup error;
                # cleanup errors are attached as notes for diagnostics.
                failure = None
                for resource in reversed(self._resources):
                    if (
                        getattr(resource, "resource_owner", self.resource_owner)
                        != self.resource_owner
                    ):
                        continue
                    method = getattr(resource, "close", None) or getattr(
                        resource, "aclose", None
                    )
                    if callable(method):
                        try:
                            result = method()
                            if inspect.isawaitable(result):
                                await result
                        except Exception as exc:  # noqa: BLE001 -- preserve startup error
                            failure = exc
                self._resources.clear()
                self.started = False
                if failure is not None:
                    error.add_note(f"Runtime resource cleanup also failed: {failure}")
                raise

    async def close(self) -> None:
        """Close owned resources once in reverse declaration order."""
        async with self._lock:
            if not self.started:
                return
            for resource in reversed(self._resources):
                if (
                    getattr(resource, "resource_owner", self.resource_owner)
                    != self.resource_owner
                ):
                    continue
                method = getattr(resource, "close", None) or getattr(
                    resource, "aclose", None
                )
                if callable(method):
                    result = method()
                    if inspect.isawaitable(result):
                        await result
            self._resources.clear()
            self.started = False

    def classify(self, stage, error, record):
        """Apply one failure hook and preserve its status as diagnostic data."""
        if self.failure_classifier is None:
            return None
        method = getattr(self.failure_classifier, "classify", self.failure_classifier)
        result = method(stage, error, record)
        if isinstance(result, tuple):
            status, message = result
        else:
            status, message = result, str(error)
        if not isinstance(status, str) or not status:
            raise AgentContractError("Failure classifier must return a status string.")
        metadata = getattr(record, "export_metadata", None)
        if isinstance(metadata, dict):
            metadata["system_failure"] = True
        return status, str(message)

    async def operation(
        self,
        record,
        stage,
        turn,
        factory,
        *,
        timeout=None,
        retry_safe=True,
        retry_policy=None,
    ):
        """Retry transient operations without retrying cancellation or actions."""
        policy = retry_policy or self.retry
        for attempt in range(1, policy.max_attempts + 1):
            try:
                remaining = (
                    record.remaining_timeout()
                    if callable(getattr(record, "remaining_timeout", None))
                    else None
                )
                if remaining is not None:
                    if remaining <= 0:
                        raise TimeoutError("Total execution timeout exceeded.")
                    timeout = remaining if timeout is None else min(timeout, remaining)
                if timeout is None:
                    return await factory()
                return await asyncio.wait_for(factory(), timeout)
            except Exception as exc:
                transient = (
                    exc.retryable
                    if isinstance(exc, (ToolSystemError, PyTrioError))
                    else isinstance(exc, (ConnectionError, TimeoutError, OSError))
                )
                if not transient:
                    raise
                again = retry_safe and attempt < policy.max_attempts
                retry = RetryEvent(
                    stage, turn, attempt, type(exc).__name__, str(exc), again
                )
                retries = getattr(record, "retries", None)
                if isinstance(retries, list):
                    retries.append(retry)
                record.events.append(
                    AgentEvent(
                        context=_execution_context.get() or record.context,
                        kind="retry",
                        status="retrying" if again else "exhausted",
                        details={
                            "stage": stage,
                            "turn": turn,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                )
                if not again:
                    raise
                delay = policy.delay(attempt)
                remaining = (
                    record.remaining_timeout()
                    if callable(getattr(record, "remaining_timeout", None))
                    else None
                )
                if remaining is not None and delay >= remaining:
                    raise TimeoutError("Total execution timeout exceeded.") from exc
                await asyncio.sleep(delay)


class EpisodeRunner:
    """Own the model/tool loop and terminal status, but not optimizer state.

    Scorers and tools may run concurrently for different episodes and must not
    keep unkeyed per-episode mutable state. asyncio cancellation propagates.

    A runner can be reused for training and evaluation. The ``is_eval`` flag
    labels the result; it does not switch to a separate agent implementation.
    """

    def __init__(
        self,
        *,
        parser: ActionParser,
        tools: ToolExecutor,
        scorer: EpisodeScorer,
        observations: ObservationFormatter,
        renderer: ConversationRenderer | None = None,
        limits: EpisodeLimits | None = None,
        prompt_key: str = "messages",
        trajectory_builder: TrajectoryExporter | None = None,
        retry: RetryPolicy | None = None,
        interaction_scorer: InteractionScorer | None = None,
        failure_classifier: FailureClassifier | None = None,
        scorer_timeout: float | None = None,
        scorer_retry: RetryPolicy | None = None,
        score_partial: bool = True,
        reward_ledger: RewardLedger | None = None,
        resource_owner: str = "runner",
    ):
        """Compose execution adapters and optional model-specific serialization.

        ``prompt_key`` selects the row field containing text or chat messages.
        If no renderer is provided, each run uses the snapshot's tokenizer.
        The exporter validates continuity during execution and is later reused
        by the collector to produce PPO trajectories.
        """
        self.parser, self.tools, self.scorer = parser, tools, scorer
        self.interaction_scorer = interaction_scorer
        self.failure_classifier = failure_classifier
        self.observations, self.renderer = observations, renderer
        self.limits, self.prompt_key = limits or EpisodeLimits(), prompt_key
        self.trajectory_builder = trajectory_builder or EpisodeTrajectoryBuilder()
        self.retry = retry or RetryPolicy()
        # ``None`` follows the mutable runner retry policy for compatibility;
        # callers can provide a separate policy for expensive PRM calls.
        self.scorer_retry = scorer_retry
        self.scorer_timeout = scorer_timeout
        if scorer_timeout is not None and scorer_timeout <= 0:
            raise ValueError("scorer_timeout must be positive when provided.")
        if not isinstance(score_partial, bool):
            raise TypeError("score_partial must be boolean.")
        self.score_partial = score_partial
        self.reward_ledger = reward_ledger or RewardLedger()
        if not isinstance(resource_owner, str) or not resource_owner:
            raise ValueError("resource_owner must be a nonempty string.")
        self.resource_owner = resource_owner
        self._boundary = RuntimeBoundary(
            retry=self.retry,
            failure_classifier=self.failure_classifier,
            resource_owner=self.resource_owner,
        )
        self._resource_started = False

    async def start(self) -> None:
        """Start scorer/tool resources once for the runner's session."""
        await self._boundary.start(self.tools, self.scorer, self.interaction_scorer)
        self._resource_started = self._boundary.started

    async def close(self) -> None:
        """Release resources owned by this runner, preserving shared clients."""
        await self._boundary.close()
        self._resource_started = self._boundary.started

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

    def _classify(self, stage, error, episode):
        """Apply an optional failure hook while retaining built-in statuses."""
        self._boundary.failure_classifier = self.failure_classifier
        return self._boundary.classify(stage, error, episode)

    async def submit_reward(self, episode, score=None):
        """Submit a delayed score exactly once and emit interaction evidence."""
        if score is None and episode.export_metadata.get("reward_submitted"):
            return episode.score
        if score is None:
            score = await self.scorer.score({}, episode)
        score.__post_init__()
        self.reward_ledger.submit(
            score.assignments, logical_rollout_id=episode.rollout_id
        )
        episode.score = score
        episode.export_metadata["reward_submitted"] = True
        for index, interaction in enumerate(episode.interactions):
            interaction.reward = (
                score.reward if index == len(episode.interactions) - 1 else 0.0
            )
            interaction.process_rewards = score.process_rewards.get(index, ())
            episode.events.append(
                RewardEvent(
                    context=interaction.context,
                    reward=interaction.reward,
                    process_rewards=interaction.process_rewards,
                )
            )
        return score

    async def _operation(
        self,
        episode,
        stage,
        turn,
        factory,
        *,
        timeout=None,
        retry_safe=True,
        retry_policy=None,
    ):
        """Retry transient transport errors within the existing episode deadline."""
        return await self._boundary.operation(
            episode,
            stage,
            turn,
            factory,
            timeout=timeout,
            retry_safe=retry_safe,
            retry_policy=retry_policy or self.retry,
        )

    async def _score_interactions(self, row, episode) -> None:
        """Run optional interaction scorers with their own retry/deadline."""
        if self.interaction_scorer is None:
            return
        assignments = []
        for index, interaction in enumerate(episode.interactions):
            method = getattr(self.interaction_scorer, "score", self.interaction_scorer)

            async def score_one(method=method, interaction=interaction):
                try:
                    return await method(row, episode, interaction)
                except TypeError:
                    return await method(row, interaction)

            result = await self._operation(
                episode,
                "interaction_scoring",
                index,
                score_one,
                timeout=self.scorer_timeout,
                retry_policy=self.scorer_retry or self.retry,
            )
            if result is None:
                continue
            if isinstance(result, EpisodeScore):
                assignments.extend(
                    replace(item, position=index)
                    if item.target in {"interaction", "turn"} and item.position is None
                    else item
                    for item in result.assignments
                )
                if result.reward:
                    assignments.append(
                        RewardAssignment(
                            target="interaction",
                            value=result.reward,
                            source="interaction_scorer",
                            position=index,
                        )
                    )
                for turn_index, values in result.process_rewards.items():
                    if turn_index not in {0, index}:
                        raise AgentContractError(
                            "Interaction scorer process rewards must target its interaction."
                        )
                    if interaction.turn is None or len(values) != len(
                        interaction.turn.generation.tokens
                    ):
                        raise AgentContractError(
                            "Interaction scorer process rewards must align with output tokens."
                        )
                    for token_index, value in enumerate(values):
                        assignments.append(
                            RewardAssignment(
                                target="token",
                                value=value,
                                source="interaction_scorer",
                                position=(index, token_index),
                            )
                        )
                interaction.reward = result.reward
                continue
            if not isinstance(result, (tuple, list)):
                result = (result,)
            for assignment in result:
                if not hasattr(assignment, "target"):
                    raise AgentContractError(
                        "Interaction scorer must return RewardAssignment objects."
                    )
                if (
                    assignment.target in {"interaction", "turn"}
                    and assignment.position is None
                ):
                    assignment = replace(assignment, position=index)
                assignments.append(assignment)
        episode.export_metadata["interaction_assignments"] = tuple(assignments)

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
    ) -> EpisodeResult:
        """Execute and score one task while preserving an inspectable record.

        Args:
            row: Task input plus optional ground truth used only by the scorer.
            snapshot: Fixed policy and tokenizer for every generated turn.
            group_id: Collector-owned identity shared by attempts of one task.
            rollout_id: Identity of the logical task execution. It may later
                produce more than one physical training row.
            sample_index: Candidate index within the logical rollout group.
            context: Collector-owned immutable identity shared with adapters.
                Direct callers can omit it to create a standalone candidate.
            seed: First turn's seed; following turns increment this value.
            is_eval: Provenance flag stored in the returned episode.

        Ordinary model and system errors become explicit terminal statuses.
        System failures do not receive a task score. Cancellation is allowed to
        escape so callers can stop collection without creating failed samples.
        """
        from .conversation import ChatTemplateRenderer

        await self.start()

        if context is None:
            context = ExecutionContext(
                rollout_id=rollout_id or uuid4().hex,
                episode_id=uuid4().hex,
                group_id=group_id,
                sample_index=sample_index,
                group_size=sample_index + 1,
                snapshot_id=snapshot.snapshot_id,
                policy_version=snapshot.policy_version,
                task_id=str(row["task_id"]) if "task_id" in row else None,
                resource_owner=self.resource_owner,
            )
        if (
            context.group_id != group_id
            or context.sample_index != sample_index
            or context.snapshot_id != snapshot.snapshot_id
            or context.policy_version != snapshot.policy_version
            or context.resource_owner != self.resource_owner
            or context.interaction_id is not None
            or rollout_id is not None
            and context.rollout_id != rollout_id
        ):
            raise AgentContractError(
                "Execution context does not match the supplied candidate."
            )
        episode = EpisodeResult(
            context.episode_id,
            group_id,
            snapshot.snapshot_id,
            is_eval,
            rollout_id=context.rollout_id,
            policy_version=snapshot.policy_version,
            sample_index=sample_index,
            context=context,
            task_id=context.task_id,
        )
        # Context remains task-local across awaits and synchronous adapters
        # dispatched through asyncio.to_thread. The matching reset in finally
        # prevents identities from leaking into the next candidate or caller.
        context_token = _execution_context.set(context)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.limits.episode_timeout):
                # Each attempt gets its own conversation. Tool observations
                # must never leak into another sample or mutate dataset rows.
                prompt = copy.deepcopy(row[self.prompt_key])
                messages = (
                    [{"role": "user", "content": prompt}]
                    if isinstance(prompt, str)
                    else list(prompt)
                )
                if not messages or any(
                    not isinstance(m, Mapping) or "role" not in m for m in messages
                ):
                    raise AgentContractError(
                        "Agent input must be text or nonempty chat messages."
                    )
                renderer = self.renderer or ChatTemplateRenderer(snapshot.tokenizer)
                await self._execute(episode, messages, snapshot, renderer, seed)
                _execution_context.set(context)
                if episode.status not in SYSTEM_STATUSES and (
                    self.score_partial or episode.status == "completed"
                ):
                    # Invalid actions and budget exits can still carry useful
                    # negative task signals. Infrastructure failures cannot.
                    try:
                        episode.score = await self._operation(
                            episode,
                            "scoring",
                            len(episode.turns) - 1,
                            lambda: self.scorer.score(row, episode),
                            timeout=self.scorer_timeout,
                            retry_policy=self.scorer_retry or self.retry,
                        )
                        # Validate the scorer boundary even if it returns a custom object.
                        episode.score.__post_init__()
                        await self._score_interactions(row, episode)
                        interaction_assignments = episode.export_metadata.get(
                            "interaction_assignments", ()
                        )
                        if interaction_assignments:
                            episode.score = replace(
                                episode.score,
                                assignments=tuple(episode.score.assignments)
                                + tuple(interaction_assignments),
                            )
                            episode.score.__post_init__()
                        if episode.status != "completed" and episode.score.success:
                            raise AgentContractError(
                                "An unfinished episode cannot be scored successful."
                            )
                        await self.submit_reward(episode, episode.score)
                    except Exception as exc:  # noqa: BLE001 -- isolate arbitrary scorer implementations
                        classified = self._classify("scoring", exc, episode)
                        episode.status, episode.error = classified or (
                            "scoring_failure",
                            str(exc),
                        )
                        episode.score = None
        except asyncio.CancelledError:
            episode.status, episode.score = "cancelled", None
            raise
        except TimeoutError:
            episode.status, episode.error = (
                "timeout",
                "Episode, generation or tool deadline exceeded.",
            )
            episode.score = None
        except AgentContractError as exc:
            classified = self._classify("contract", exc, episode)
            episode.status, episode.error = classified or ("contract_error", str(exc))
        except ToolSystemError as exc:
            classified = self._classify("tool", exc, episode)
            episode.status, episode.error = classified or ("tool_failure", str(exc))
        except Exception as exc:  # noqa: BLE001 -- preserve failed adapter episodes for audit
            classified = self._classify("episode", exc, episode)
            episode.status, episode.error = classified or (
                "contract_error",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            episode.elapsed_seconds = time.perf_counter() - started
            episode.events.append(
                AgentEvent(
                    context=context,
                    kind="episode_end",
                    status=episode.status,
                    latency=episode.elapsed_seconds,
                )
            )
            _execution_context.reset(context_token)
        return episode

    async def _execute(self, episode, messages, snapshot, renderer, seed) -> None:
        """Advance the conversation until an answer, budget exit, or failure.

        This helper records generations and tool outcomes but leaves final task
        scoring to run(). Every generation is appended before parsing, so even
        malformed actions retain the tokens needed for a valid failure sample.
        """
        limits = self.limits
        generated_tokens = 0
        for index in range(limits.max_turns):
            # Tokenization and user renderers can be synchronous CPU work.
            if episode.turns and hasattr(renderer, "render_continuation"):
                prompt_tokens = await asyncio.to_thread(
                    renderer.render_continuation, episode, messages, self.tools.schemas
                )
            else:
                prompt_tokens = await asyncio.to_thread(
                    renderer.render, messages, self.tools.schemas
                )
            validate_tokens(prompt_tokens, name="rendered prompt")
            self.trajectory_builder.validate_prefix(episode, prompt_tokens)
            # Context capacity includes rendered history and observations,
            # whereas the episode output budget counts model generations only.
            context_budget = limits.max_context_tokens - len(prompt_tokens)
            if context_budget <= 0:
                episode.status = "context_limit"
                return
            remaining = limits.max_generated_tokens - generated_tokens
            if remaining <= 0:
                episode.status = "token_limit"
                return
            budget = min(limits.max_tokens_per_turn, remaining, context_budget)
            context = replace(
                episode.context,
                interaction_id=uuid4().hex,
                parent_interaction_id=(
                    episode.interactions[-1].interaction_id
                    if episode.interactions
                    else None
                ),
            )
            interaction = Interaction(
                context=context,
                input_token_span=(0, len(prompt_tokens)),
                resource_owner=context.resource_owner,
            )
            episode.interactions.append(interaction)
            context_token = _execution_context.set(context)
            started = time.perf_counter()
            try:
                await self._execute_turn(
                    episode,
                    interaction,
                    messages,
                    snapshot,
                    prompt_tokens,
                    budget,
                    None if seed is None else seed + index,
                    index,
                )
            except BaseException as exc:
                # Cancellation has no task reward. Its evidence remains
                # available without converting it into a model-side failure.
                interaction.status = (
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "timeout"
                    if isinstance(exc, TimeoutError)
                    else "tool_failure"
                    if isinstance(exc, ToolSystemError)
                    else "contract_error"
                )
                raise
            finally:
                interaction.latency = time.perf_counter() - started
                interaction.retries = [r for r in episode.retries if r.turn == index]
                episode.events.append(
                    AgentEvent(
                        context=context,
                        kind="interaction_end",
                        status=interaction.status,
                        latency=interaction.latency,
                    )
                )
                _execution_context.reset(context_token)
            if interaction.turn is not None:
                generated_tokens += len(interaction.turn.generation.tokens)
            if interaction.status != "tool_results":
                return
        # A tool observation alone does not terminate a task successfully.
        episode.status = "max_turns"

    async def _execute_turn(
        self,
        episode,
        interaction,
        messages,
        snapshot,
        prompt_tokens,
        budget,
        seed,
        index,
    ):
        """Execute one request while retaining partial evidence on every exit.

        The interaction exists before sampling, so timeouts and backend failures
        have identities even when no generation is available. Successful output
        is added to the episode's training turns before parsing. A malformed
        action therefore remains a scoreable model failure rather than being
        mistaken for missing infrastructure data.
        """
        limits = self.limits
        context = interaction.context
        started = time.perf_counter()
        try:
            generation = await self._operation(
                episode,
                "sampling",
                index,
                partial(
                    snapshot.sampler.generate,
                    prompt_tokens,
                    max_tokens=budget,
                    seed=seed,
                ),
                timeout=limits.sampling_timeout,
            )
        except (AgentContractError, TimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 -- classify sampling adapter failures
            classified = self._classify("sampling", exc, episode)
            episode.status, episode.error = classified or (
                "sampling_failure",
                str(exc),
            )
            interaction.status = "sampling_failure"
            return
        generation.__post_init__()
        if len(generation.tokens) > budget:
            raise AgentContractError("Sampler exceeded the requested output budget.")
        turn = AgentTurn(tuple(prompt_tokens), generation)
        interaction.turn = turn
        interaction.output_token_span = (
            len(prompt_tokens),
            len(prompt_tokens) + len(generation.tokens),
        )
        episode.turns.append(turn)
        episode.events.append(
            GenerationEvent(
                context=context,
                generation=generation,
                latency=time.perf_counter() - started,
            )
        )
        if generation.stop_reason in {"length", "max_tokens", "max_length"}:
            episode.status = interaction.status = "token_limit"
            return
        try:
            turn.action = self.parser.parse(generation)
            turn.action.__post_init__()
        except ActionParseError as exc:
            episode.status, episode.error = "invalid_action", str(exc)
            interaction.status = "invalid_action"
            return
        messages.append(turn.action.assistant_message)
        if turn.action.final_text is not None:
            episode.status, episode.final_text = "completed", turn.action.final_text
            interaction.status = "completed"
            return
        for call in turn.action.tool_calls:
            episode.events.append(
                ToolCallEvent(context=context, call=call, status="requested")
            )
            started = time.perf_counter()
            result = await self._operation(
                episode,
                "tool",
                index,
                partial(self.tools.execute, call),
                timeout=limits.tool_timeout,
                retry_safe=self.retry.retry_tools
                and bool(
                    getattr(self.tools, "is_retry_safe", lambda call: False)(call)
                ),
            )
            result.__post_init__()
            if result.call_id != call.call_id or result.name != call.name:
                raise AgentContractError(
                    "Tool result does not match its requested call."
                )
            turn.tool_results.append(result)
            episode.events.append(
                ToolResultEvent(
                    context=context,
                    result=result,
                    latency=time.perf_counter() - started,
                    status="action_error" if result.is_error else "completed",
                )
            )
        messages.extend(self.observations.format(turn.tool_results))
        interaction.status = "tool_results"
