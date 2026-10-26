"""Workflow façade for collecting OpenAI agent sessions."""

from __future__ import annotations

from typing import Any

from .agent import Agent, AgentRunResult


class OpenAIWorkflow:
    """Run one or more agents while preserving each session independently."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def run(self, input: str | list[dict[str, Any]], **kwargs) -> AgentRunResult:
        return await self.agent.run(input, **kwargs)

    async def submit_reward(self, result: AgentRunResult, reward: Any, **kwargs) -> Any:
        return result.session.submit_reward(reward, **kwargs)

    def export_trajectory(self, result: AgentRunResult):
        return result.export_trajectory()
