"""Workflow façade for collecting OpenAI agent sessions."""

from __future__ import annotations

from typing import Any

from ...runtime.sampling import SamplingSnapshot
from ...types import AgentContractError, ExecutionContext
from .agent import Agent, AgentRunResult
from .session import OpenAISession


class OpenAIWorkflow:
    """Run one or more agents while preserving each session independently."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def run(
        self,
        input: str | list[dict[str, Any]],
        *,
        context: ExecutionContext | None = None,
        session: OpenAISession | None = None,
        training: bool = False,
        snapshot: SamplingSnapshot | None = None,
        **kwargs,
    ) -> AgentRunResult:
        """Bind a collector candidate and optionally verify its sampling snapshot.

        Snapshot validation checks provenance only. This external OpenAI
        workflow does not invoke the snapshot sampler, implement EpisodeExecutor
        or produce a PPO RolloutExport. Its exported trace remains untrainable.
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
        return await self.agent.run(
            input, context=context, session=session, training=training, **kwargs
        )

    async def submit_reward(self, result: AgentRunResult, reward: Any, **kwargs) -> Any:
        return result.session.submit_reward(reward, **kwargs)

    def export_trajectory(self, result: AgentRunResult):
        return result.export_trajectory()
