"""Small workflow façade for context-bound OpenAI agent executions.

The workflow does not allocate rollout identities, refresh policies, or directly
invoke a sampler. A collector supplies those objects and this façade checks
their relationship before delegating to ``Agent``.  Keeping reward submission
and trace export as explicit methods also makes the delayed lifecycle visible
to callers instead of hiding it inside a request callback.
"""

from __future__ import annotations

from typing import Any

from ...runtime.sampling import SamplingSnapshot
from ...types import AgentContractError, ExecutionContext
from .agent import Agent, AgentRunResult
from .session import OpenAISession


class OpenAIWorkflow:
    """Run an agent while preserving collector identity and session isolation."""

    def __init__(self, agent: Agent):
        """Bind an already configured agent; no network or sampler starts here."""
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

        The snapshot argument validates provenance only; model execution is
        delegated to the agent's configured backend. A PyTRIO backend binds its
        own fixed snapshot, whereas an external backend calls its provider.
        This facade does not implement EpisodeExecutor or produce a PPO
        RolloutExport. Its exported trace remains untrainable.

        Supplying ``snapshot`` turns provenance checking on: the supplied root
        context must carry the same snapshot ID and policy version, and the
        delegated run is forced into training-session validation.  The snapshot
        object itself is never mutated or refreshed by this façade.
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
        """Submit one delayed reward through the result's session owner."""
        return result.session.submit_reward(reward, **kwargs)

    def export_trajectory(self, result: AgentRunResult):
        """Export a terminal result trace after session quiescence checks."""
        return result.export_trajectory()
