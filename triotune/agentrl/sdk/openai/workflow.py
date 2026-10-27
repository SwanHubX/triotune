"""Small workflow façade for context-bound OpenAI agent executions.

The workflow does not allocate rollout identities, refresh policies, or directly
invoke a sampler. A collector supplies those objects and this façade checks
their relationship before delegating to ``Agent``.  Keeping reward submission
and trace export as explicit methods also makes the delayed lifecycle visible
to callers instead of hiding it inside a request callback.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...config import RetryPolicy
from ...runtime.runner import RuntimeBoundary
from ...runtime.sampling import SamplingSnapshot
from ...training.rewards import RewardAssignment
from ...types import (
    AgentContractError,
    AgentEvent,
    EpisodeScore,
    ExecutionContext,
    FailureClassifier,
    _execution_context,
)
from .agent import Agent, AgentRunResult
from .client import OpenAIClient
from .session import OpenAISession, OpenAITrajectoryExport


@dataclass(frozen=True, slots=True)
class SDKRunResult:
    """Keep a third-party function's arbitrary output beside its session trace.

    Unlike AgentRunResult, the output need not be final-answer text: third-party
    frameworks may return their own typed result objects. Reward submission and
    export still use the exact same terminal session and collector provenance.
    """

    output: Any
    session: OpenAISession
    accepted: bool = True

    @property
    def context(self) -> ExecutionContext:
        """Expose immutable candidate identity to generic lifecycle hooks."""
        return self.session.context

    @property
    def status(self) -> str:
        """Expose terminal SDK status without claiming EpisodeResult semantics."""
        return self.session.status

    @property
    def interactions(self):
        """Expose SDK interactions as a read-only tuple."""
        return tuple(self.session.interactions)

    @property
    def trainable(self) -> bool:
        """Remain false until an EpisodeResult and RolloutExport are produced."""
        return False

    @property
    def session_id(self) -> str:
        """Return the independent SDK conversation identity."""
        return self.session.session_id

    def export_trajectory(self) -> OpenAITrajectoryExport:
        """Export recorded evidence without claiming it is a PPO RolloutExport."""
        return self.session.export_trajectory()


class OpenAIWorkflow:
    """Run an agent while preserving collector identity and session isolation."""

    def __init__(
        self,
        agent: Agent | None = None,
        *,
        client: OpenAIClient | None = None,
        scorer: Any | None = None,
        interaction_scorer: Any | None = None,
        accept_sample: Any | None = None,
        retry: RetryPolicy | None = None,
        scorer_retry: RetryPolicy | None = None,
        scorer_timeout: float | None = None,
        failure_classifier: FailureClassifier | None = None,
        resource_owner: str = "openai-workflow",
    ):
        """Bind either a built-in agent or a client for external async functions."""
        if (agent is None) == (client is None):
            raise AgentContractError(
                "Supply one Agent or OpenAIClient to the workflow."
            )
        self.agent = agent
        self.client = client if client is not None else agent.client
        if not isinstance(self.client, OpenAIClient):
            raise AgentContractError("OpenAIWorkflow requires OpenAIClient.")
        if scorer_timeout is not None and (
            isinstance(scorer_timeout, bool)
            or not isinstance(scorer_timeout, (int, float))
            or scorer_timeout <= 0
        ):
            raise AgentContractError("scorer_timeout must be positive or None.")
        self.scorer = scorer
        self.interaction_scorer = interaction_scorer
        self.accept_sample = accept_sample
        self.scorer_retry = scorer_retry
        self.scorer_timeout = scorer_timeout
        self.boundary = RuntimeBoundary(
            retry=retry,
            failure_classifier=failure_classifier,
            resource_owner=resource_owner,
        )

    async def start(self, *resources) -> None:
        """Start owned scorers, acceptance hooks, and callback resources once."""
        await self.boundary.start(
            self.scorer, self.interaction_scorer, self.accept_sample, *resources
        )

    async def close(self) -> None:
        """Close workflow-owned hooks without closing an injected SDK client."""
        await self.boundary.close()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()

    async def run(
        self,
        input: str | list[dict[str, Any]],
        *,
        context: ExecutionContext | None = None,
        session: OpenAISession | None = None,
        training: bool = False,
        snapshot: SamplingSnapshot | None = None,
        execute: Callable[[Any, Any], Awaitable[Any]] | None = None,
        client_options: Mapping[str, Any] | None = None,
        episode_timeout: float | None = None,
        max_generated_tokens: int | None = None,
        **kwargs,
    ) -> AgentRunResult | SDKRunResult:
        """Bind a collector candidate and optionally verify its sampling snapshot.

        The snapshot argument validates provenance only; model execution is
        delegated to the agent's configured backend. A PyTRIO backend binds its
        own fixed snapshot, whereas an external backend calls its provider.
        This facade exports diagnostic traces. OpenAIEpisodeExecutor wraps a
        snapshot-bound workflow factory to produce native PPO EpisodeResults
        without changing this API's return type or external-provider semantics.

        Supplying ``snapshot`` turns provenance checking on: the supplied root
        context must carry the same snapshot ID and policy version, and the
        delegated run is forced into training-session validation.  The snapshot
        object itself is never mutated or refreshed by this façade.

        With execute supplied, await execute(official_async_client, input)
        instead of running the built-in Agent. The callback owns its tool loop;
        this workflow owns session finalization and the local SDK transport.
        It never closes the injected upstream adapter or trainer snapshot.
        The callback can inspect the task-local root ExecutionContext; API
        calls run under child contexts and captured tool-result events refer
        to the interaction that originally requested each tool.
        """
        if type(training) is not bool:
            raise AgentContractError("training must be boolean.")
        await self.start(execute)
        if snapshot is not None:
            expected = (
                context
                if context is not None
                else session.context
                if session is not None
                else None
            )
            if not isinstance(snapshot, SamplingSnapshot) or not isinstance(
                expected, ExecutionContext
            ):
                raise AgentContractError(
                    "Snapshot binding requires SamplingSnapshot and collector context."
                )
            if (
                expected.snapshot_id != snapshot.snapshot_id
                or expected.policy_version != snapshot.policy_version
            ):
                raise AgentContractError(
                    "ExecutionContext does not match the supplied sampling snapshot."
                )
            training = True
        if execute is not None:
            if kwargs:
                raise AgentContractError(
                    "Third-party callbacks own their request options."
                )
            result = await self._run_function(
                input,
                execute,
                session=session,
                context=context,
                training=training,
                client_options=client_options,
                episode_timeout=episode_timeout,
                max_generated_tokens=max_generated_tokens,
            )
            return await self._postprocess(input, result)
        if self.agent is None or client_options is not None:
            raise AgentContractError(
                "Client-only workflows require an execute callback."
            )
        result = await self.agent.run(
            input, context=context, session=session, training=training, **kwargs
        )
        return await self._postprocess(input, result)

    async def _run_function(
        self,
        input,
        execute,
        *,
        session,
        context,
        training,
        client_options,
        episode_timeout,
        max_generated_tokens,
    ):
        """Isolate identity, transport ownership, and cleanup around one callback.

        Request failures and cancellation remain exceptions, never zero rewards.
        Any completed tool observations captured before a later failure remain
        in the trace. Restore the outer context even if a callback fails before
        making its first request or the final session check reports partial work.
        """
        if not callable(execute):
            raise AgentContractError("execute must be an asynchronous callable.")
        session = (
            session
            if session is not None
            else OpenAISession(context=context, training=training)
        )
        session.validate_identity(context)
        if training and not session.training:
            raise AgentContractError(
                "A training callback requires a training-bound session."
            )
        if (
            session.running
            or session.interactions
            or session.status != "open"
            or session.released
        ):
            raise AgentContractError("Third-party runs require a fresh session.")
        if episode_timeout is not None or max_generated_tokens is not None:
            session.configure_limits(
                episode_timeout=episode_timeout,
                max_generated_tokens=max_generated_tokens,
            )
        session.begin_execution()
        session.running = True
        token = _execution_context.set(session.context)
        status, error, failure = "completed", None, None

        async def invoke():
            async with self.client.as_async_openai(
                session=session, **dict(client_options or {})
            ) as sdk:
                pending = execute(sdk, input)
                if not inspect.isawaitable(pending):
                    raise AgentContractError("execute must return an awaitable.")
                return await pending

        try:
            remaining = session.remaining_timeout()
            output = (
                await asyncio.wait_for(invoke(), remaining)
                if remaining is not None
                else await invoke()
            )
            if not session.interactions or any(
                item.status != "completed" for item in session.interactions
            ):
                raise AgentContractError(
                    "A completed callback requires completed request evidence."
                )
            return SDKRunResult(output, session)
        except BaseException as exc:
            failure = exc
            status = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, asyncio.CancelledError):
                classified = self.boundary.classify("execution", exc, session)
                if classified is not None:
                    session.failure_classification = classified
                    error = classified[1]
            raise
        finally:
            session.running = False
            try:
                session.finish(status, error=error)
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Session finalization also failed: {cleanup_error}")
            finally:
                _execution_context.reset(token)

    async def _postprocess(self, input, result):
        """Apply existing scoring and acceptance semantics to one SDK trace.

        SDK traces remain non-trainable; OpenAIEpisodeExecutor separately builds
        EpisodeResult and RolloutExport. A scoring or acceptance failure therefore
        marks this result unaccepted and raises; it cannot silently enter PPO.
        """
        session = result.session
        # Agent/callback execution has already restored its caller's context.
        # Establish this session's root for terminal scoring and acceptance;
        # interaction scoring temporarily narrows it to the matching child.
        # The final reset also preserves nested callers on failure/cancellation.
        context_token = _execution_context.set(session.context)
        try:
            if self.interaction_scorer is not None:
                method = getattr(
                    self.interaction_scorer, "score", self.interaction_scorer
                )
                for index, interaction in enumerate(session.interactions):
                    # Keep PRM retries, timeout tasks, and reward submission in
                    # one child scope. ContextVars remain task-local and are
                    # copied by call_async's synchronous worker fallback; no
                    # shared scorer or sampling-client identity is mutated.
                    interaction_token = _execution_context.set(interaction.context)
                    try:
                        value = await self.boundary.operation(
                            session,
                            "interaction_scoring",
                            index,
                            lambda method=method, interaction=interaction: method(
                                input, result, interaction
                            ),
                            timeout=self.scorer_timeout,
                            retry_policy=self.scorer_retry,
                        )
                        await self._submit_interaction_score(
                            session, interaction, index, value
                        )
                    finally:
                        _execution_context.reset(interaction_token)
            if self.scorer is not None:
                method = getattr(self.scorer, "score", self.scorer)
                score = await self.boundary.operation(
                    session,
                    "scoring",
                    len(session.interactions) - 1,
                    lambda: method(input, result),
                    timeout=self.scorer_timeout,
                    retry_policy=self.scorer_retry,
                )
                if score is not None:
                    session.submit_reward(
                        score, submission_id=f"workflow-score:{session.session_id}"
                    )
            accepted = True
            if self.accept_sample is not None:
                # P0/P1 SampleAcceptance accepts export=None for records that
                # have no training export. Passing OpenAITrajectoryExport here
                # would falsely imply the P2.5 EpisodeExecutor bridge exists.
                accepted = self.accept_sample(result, None)
                if inspect.isawaitable(accepted):
                    remaining = session.remaining_timeout()
                    accepted = (
                        await asyncio.wait_for(accepted, remaining)
                        if remaining is not None
                        else await accepted
                    )
                if type(accepted) is not bool:
                    raise AgentContractError(
                        "Sample acceptance hook must return a boolean."
                    )
            session.accepted = accepted
            return replace(result, accepted=accepted)
        except Exception as exc:
            session.accepted = False
            stage = "acceptance" if self.accept_sample is not None else "scoring"
            classified = self.boundary.classify(stage, exc, session)
            if classified is not None:
                session.failure_classification = classified
            session.events.append(
                AgentEvent(
                    context=session.context,
                    kind="postprocess_failure",
                    status="failed",
                    details={"stage": stage, "error": f"{type(exc).__name__}: {exc}"},
                )
            )
            raise
        finally:
            _execution_context.reset(context_token)

    async def _submit_interaction_score(
        self, session, interaction, index, value
    ) -> None:
        """Map interaction scorer output through the P1 RewardAssignment contract."""
        if value is None:
            return
        assignments = []
        if isinstance(value, EpisodeScore):
            if value.reward:
                assignments.append(
                    RewardAssignment(
                        "interaction",
                        value.reward,
                        source="interaction_scorer",
                        position=index,
                    )
                )
            for turn, rewards in value.process_rewards.items():
                if turn not in {0, index}:
                    raise AgentContractError(
                        "Interaction scorer process rewards target another interaction."
                    )
                for offset, reward in enumerate(rewards):
                    assignments.append(
                        RewardAssignment(
                            "token",
                            reward,
                            source="interaction_scorer",
                            position=(index, offset),
                        )
                    )
            assignments.extend(value.assignments)
        elif isinstance(value, (tuple, list)):
            assignments.extend(value)
        elif isinstance(value, RewardAssignment):
            assignments.append(value)
        else:
            session.submit_reward(
                value,
                interaction_id=interaction.interaction_id,
                submission_id=f"interaction-score:{interaction.interaction_id}",
            )
            return
        for number, assignment in enumerate(assignments):
            if not isinstance(assignment, RewardAssignment):
                raise AgentContractError(
                    "Interaction scorer must return RewardAssignment objects."
                )
            if (
                assignment.target in {"interaction", "turn"}
                and assignment.position is None
            ):
                assignment = replace(assignment, position=index)
            session.submit_reward(
                assignment,
                submission_id=f"interaction-score:{interaction.interaction_id}:{number}",
            )

    async def submit_reward(
        self, result: AgentRunResult | SDKRunResult, reward: Any, **kwargs
    ) -> Any:
        """Submit one delayed reward through the result's session owner."""
        return result.session.submit_reward(reward, **kwargs)

    def export_trajectory(self, result: AgentRunResult | SDKRunResult):
        """Export a terminal result trace after session quiescence checks."""
        return result.export_trajectory()
