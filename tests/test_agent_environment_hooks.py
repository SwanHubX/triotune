"""Environment-owned termination keeps complete tool evidence and strict types."""

import unittest

from agent_test_support import AgentTokenizer
from recipes.calculator_agent import CalculatorScorer, calculator_rows, calculator_tools
from test_agent_episode_runner import snapshot

from triotune.agentrl import AgentConfig, AgentEvaluationConfig


class EnvironmentHookTests(unittest.IsolatedAsyncioTestCase):
    def workflow(self, hook):
        return AgentConfig().build_workflow(
            tokenizer=AgentTokenizer(),
            tools=calculator_tools(),
            scorer=CalculatorScorer(),
            post_tool_stop=hook,
        )

    async def test_default_and_sync_async_stops_preserve_recorded_tool_results(self):
        async def stop_async(episode, turn):
            self.assertEqual(len(turn.tool_results), 1)
            return True

        for hook, expected_turns in (
            (None, 2),
            (lambda episode, turn: True, 1),
            (stop_async, 1),
        ):
            with self.subTest(hook=hook):
                workflow = self.workflow(hook)
                policy, client = snapshot()
                batch = await workflow.collect(calculator_rows()[:1], policy)
                episode = batch.episodes[0]
                self.assertEqual(episode.status, "completed")
                self.assertEqual(len(episode.turns), expected_turns)
                self.assertEqual(len(episode.turns[0].tool_results), 1)
                self.assertEqual(len(client.requests), expected_turns)
                await workflow.close()

    async def test_invalid_stop_result_is_contract_failure_and_not_training_data(self):
        with self.assertRaisesRegex(TypeError, "post_tool_stop"):
            self.workflow(True)
        workflow = self.workflow(lambda episode, turn: 1)
        policy, _ = snapshot()
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.trajectories, [])
        self.assertIn("post_tool_stop must return a boolean", batch.episodes[0].error)
        await workflow.close()

    def test_evaluation_distribution_validation(self):
        self.assertEqual(AgentEvaluationConfig(temperature=1.5).temperature, 1.5)
        for options in (
            {"temperature": 0},
            {"temperature": True},
            {"temperature": float("nan")},
            {"top_p": 0},
            {"top_p": 1.1},
            {"top_p": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AgentEvaluationConfig(**options)
