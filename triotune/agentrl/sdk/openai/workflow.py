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
from dataclasses import dataclass
from typing import Any

from ...runtime.sampling import SamplingSnapshot
from ...types import AgentContractError, ExecutionContext, _execution_context
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
        self, agent: Agent | None = None, *, client: OpenAIClient | None = None
    ):
        """Bind either a built-in agent or a client for external async functions."""
        if (agent is None) == (client is None):
            raise AgentContractError(
                "Supply one Agent or OpenAIClient to the workflow."
            )
        self.agent = agent
        self.client = client if client is not None else agent.client

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
        **kwargs,
    ) -> AgentRunResult | SDKRunResult:
        """Bind a collector candidate and optionally verify its sampling snapshot.

        The snapshot argument validates provenance only; model execution is
        delegated to the agent's configured backend. A PyTRIO backend binds its
        own fixed snapshot, whereas an external backend calls its provider.
        This facade does not implement EpisodeExecutor or produce a PPO
        RolloutExport. Its exported trace remains untrainable.

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
            return await self._run_function(
                input,
                execute,
                session=session,
                context=context,
                training=training,
                client_options=client_options,
            )
        if self.agent is None or client_options is not None:
            raise AgentContractError(
                "Client-only workflows require an execute callback."
            )
        return await self.agent.run(
            input, context=context, session=session, training=training, **kwargs
        )

    async def _run_function(
        self, input, execute, *, session, context, training, client_options
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
        session.running = True
        token = _execution_context.set(session.context)
        status, error, failure = "completed", None, None
        try:
            async with self.client.as_async_openai(
                session=session, **dict(client_options or {})
            ) as sdk:
                pending = execute(sdk, input)
                if not inspect.isawaitable(pending):
                    raise AgentContractError("execute must return an awaitable.")
                output = await pending
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

    async def submit_reward(
        self, result: AgentRunResult | SDKRunResult, reward: Any, **kwargs
    ) -> Any:
        """Submit one delayed reward through the result's session owner."""
        return result.session.submit_reward(reward, **kwargs)

    def export_trajectory(self, result: AgentRunResult | SDKRunResult):
        """Export a terminal result trace after session quiescence checks."""
        return result.export_trajectory()
