"""SDK lifecycle budgets, failure stages, and trace-only replay APIs."""

import asyncio
import unittest
from types import SimpleNamespace

from test_openai_sdk import _FakeChat, add

from triotune.agentrl import (
    Agent,
    AgentContractError,
    ExecutionContext,
    OpenAIClient,
    OpenAISession,
    RetryPolicy,
    ToolResult,
    current_execution_context,
    function_tool,
)
from triotune.agentrl.integrations.openai import OpenAIWorkflow
from triotune.agentrl.runtime.tools import FunctionToolExecutor


def workflow(chat=None, **options):
    chat = chat or _FakeChat()
    client = OpenAIClient(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)), model="fake"
    )
    return (
        OpenAIWorkflow(
            Agent(
                name="calculator", instructions="Use tools.", tools=[add], client=client
            ),
            **options,
        ),
        chat,
    )


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_budgets_reach_builtin_agent(self):
        run, chat = workflow()
        async with run:
            with self.assertRaises(TimeoutError):
                await run.run("2+3", max_generated_tokens=2)
        self.assertEqual(len(chat.calls), 1)

    async def test_workflow_timeout_covers_builtin_api_call(self):
        class Slow(_FakeChat):
            async def create(self, **kwargs):
                await asyncio.sleep(0.03)
                return await super().create(**kwargs)

        run, _ = workflow(Slow())
        async with run:
            with self.assertRaises(TimeoutError):
                await run.run("2+3", episode_timeout=0.001)

    async def test_failure_classifier_reports_actual_postprocess_stage(self):
        observed = []

        async def broken_score(input, result):
            raise ConnectionError("judge offline")

        def classify(stage, error, record):
            observed.append(stage)
            return "failed", str(error)

        run, _ = workflow(
            scorer=broken_score,
            accept_sample=lambda result, export: True,
            failure_classifier=classify,
        )
        async with run:
            with self.assertRaises(ConnectionError):
                await run.run("2+3")
        self.assertEqual(observed, ["scoring"])

    async def test_acceptance_failure_is_classified_as_acceptance(self):
        observed = []

        def classify(stage, error, record):
            observed.append(stage)
            return "failed", str(error)

        run, _ = workflow(
            accept_sample=lambda result, export: (_ for _ in ()).throw(
                RuntimeError("reject")
            ),
            failure_classifier=classify,
        )
        async with run:
            with self.assertRaises(RuntimeError):
                await run.run("2+3")
        self.assertEqual(observed, ["acceptance"])


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_rescore_branch_and_compare_are_trace_only(self):
        fake = _FakeChat()
        client = OpenAIClient(
            SimpleNamespace(chat=SimpleNamespace(completions=fake)), model="fake"
        )
        left = OpenAISession()
        right = OpenAISession()
        for session in (left, right):
            await client.create_chat_completion(
                messages=[{"role": "user", "content": "compute"}], session=session
            )
            session.finish("completed")
        before = len(left.interactions)
        observations = await left.replay_tools()
        self.assertEqual(len(observations), 1)
        score = await left.rescore(lambda row, trace: 3.0, {"answer": "5"})
        self.assertEqual(score, 3.0)
        branch = left.branch(left.interactions[0].interaction_id)
        self.assertEqual(branch.metadata["trainable"], False)
        self.assertEqual(len(branch.interactions), 1)
        compared = OpenAISession.compare(left, right)
        self.assertEqual(compared["lengths"], (1, 1))
        self.assertEqual(len(left.interactions), before)
        self.assertEqual(left.rewards, [])

    async def test_tool_replay_uses_new_executor_without_mutating_trace(self):
        calls = []

        @function_tool
        def remember(value: str) -> str:
            calls.append(value)
            return value.upper()

        session = OpenAISession()
        interaction = session.start_request(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "remember"}]},
        )
        interaction.status = "completed"
        interaction.tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "remember", "arguments": {"value": "x"}},
            }
        ]
        session.finish("completed")
        executor = FunctionToolExecutor([remember])
        result = await session.replay_tools(executor)
        self.assertEqual(calls, ["x"])
        self.assertEqual(result[0][1][0].content, "X")
        self.assertEqual(session.interactions[0].tool_results, [])

    async def test_recover_inflight_preserves_partial_trace_without_resampling(self):
        session = OpenAISession()
        interaction = session.start_request(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "slow"}]},
        )
        interaction.status = "completed"
        interaction.tool_results.append(
            {"call_id": "call", "name": "remember", "content": "done"}
        )
        pending = session.start_request(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "continue"}]},
            parent_interaction_id=interaction.interaction_id,
        )
        export = session.recover_inflight(error="cancelled by owner")
        self.assertEqual(session.status, "failed")
        self.assertEqual(export.metadata["trainable"], False)
        self.assertEqual(session.interactions[0].tool_results[0]["content"], "done")
        self.assertEqual(pending.status, "cancelled")
        with self.assertRaises(AgentContractError):
            session.submit_reward(1.0)


if __name__ == "__main__":
    unittest.main()
