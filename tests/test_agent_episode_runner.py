"""Calculator execution, model recovery, infrastructure errors and limits."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agent_test_support import AgentTokenizer, CalculatorSampler
from recipes.calculator_agent import (
    calculator_rows,
    calculator_tools,
    calculator_workflow,
)

from triotune.agentrl import (
    EpisodeCollectionError,
    EpisodeLimits,
    FunctionTool,
    FunctionToolExecutor,
    JsonlEpisodeLogger,
    PyTRIOAgentSampler,
    SamplingSnapshot,
    ToolCall,
    ToolSystemError,
)
from triotune.agentrl import EpisodeScore


def snapshot(*, output=None, failure=None):
    tokenizer = AgentTokenizer()
    client = CalculatorSampler(tokenizer, [], 0, output=output, failure=failure)
    return SamplingSnapshot(
        "snapshot-0", PyTRIOAgentSampler(client, tokenizer), tokenizer
    ), client


class AgentEpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_identity_is_available_during_scoring(self):
        policy, _ = snapshot()
        observed = []

        class IdentityScorer:
            async def score(self, row, episode):
                observed.append(
                    (
                        episode.rollout_id,
                        episode.group_id,
                        episode.sample_index,
                        episode.snapshot_id,
                        episode.policy_version,
                    )
                )
                return EpisodeScore(0.0, False)

        workflow = calculator_workflow()
        workflow.runner.scorer = IdentityScorer()
        batch = await workflow.collect(
            calculator_rows()[:1], policy, n_samples=3, strict=False
        )

        self.assertEqual([item[2] for item in observed], [0, 1, 2])
        self.assertEqual(len({item[0] for item in observed}), 1)
        self.assertEqual(len({item[1] for item in observed}), 1)
        self.assertNotEqual(observed[0][0], observed[0][1])
        self.assertEqual({item[3] for item in observed}, {"snapshot-0"})
        self.assertEqual({item[4] for item in observed}, {None})
        self.assertEqual([e.sample_index for e in batch.episodes], [0, 1, 2])

    async def test_wrong_answer_is_a_valid_zero_reward_episode(self):
        policy, _ = snapshot(output='{"final":"99"}')
        batch = await calculator_workflow().collect(calculator_rows()[:1], policy)
        self.assertEqual(batch.episodes[0].status, "completed")
        self.assertEqual(batch.trajectories[0].reward, 0)
        self.assertFalse(batch.episodes[0].score.success)

    async def test_calculator_success_uses_original_ids_and_records_all_interactions(
        self,
    ):
        policy, client = snapshot()
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlEpisodeLogger(Path(directory) / "episodes.jsonl")
            batch = await calculator_workflow(logger=logger).collect(
                calculator_rows()[:1], policy, n_samples=2
            )
            self.assertEqual(len(batch.trajectories), 2)
            self.assertEqual(len(client.requests), 4)
            for trajectory in batch.trajectories:
                self.assertEqual(set(trajectory.turn_ids), {0, -1, 1})
                self.assertEqual(trajectory.reward, 1)
                self.assertNotEqual(trajectory.rollout_id, trajectory.group_id)
                self.assertEqual(trajectory.snapshot_id, "snapshot-0")
                self.assertEqual(trajectory.policy_version, None)
                self.assertEqual(trajectory.episode_id is not None, True)
                self.assertIn(trajectory.sample_index, (0, 1))
            self.assertEqual(len({e.group_id for e in batch.episodes}), 1)
            records = [
                json.loads(line) for line in logger.path.read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["snapshot_id"], "snapshot-0")
            self.assertEqual(records[0]["turns"][0]["tool_results"][0]["content"], "5")
            self.assertNotIn("untrusted", records[0]["turns"][0]["generation"]["text"])

    async def test_invalid_tool_name_is_observation_and_model_can_recover(self):
        result = await calculator_tools().execute(ToolCall("1", "missing", {}))
        self.assertTrue(result.is_error)
        policy, _ = snapshot(
            output='{"tool_calls":[{"name":"missing","arguments":{}}]}'
        )
        batch = await calculator_workflow(limits=EpisodeLimits(max_turns=2)).collect(
            calculator_rows()[:1], policy
        )
        episode = batch.episodes[0]
        self.assertEqual(episode.status, "max_turns")
        self.assertTrue(episode.trainable)
        self.assertTrue(episode.turns[0].tool_results[0].is_error)

    async def test_model_argument_errors_are_recoverable_but_tool_crash_is_system_failure(
        self,
    ):
        tools = calculator_tools()
        for arguments in (
            {},
            {"operation": "divide", "a": 1, "b": 0},
            {"operation": "add", "a": "x", "b": 1},
        ):
            self.assertTrue(
                (await tools.execute(ToolCall("0", "calculate", arguments))).is_error
            )

        def broken():
            raise OSError("unavailable")

        executor = FunctionToolExecutor(
            [FunctionTool("broken", "", {"type": "object"}, broken)]
        )
        with self.assertRaises(ToolSystemError):
            await executor.execute(ToolCall("0", "broken", {}))

    async def test_sampling_failure_is_logged_and_never_becomes_reward_zero(self):
        policy, _ = snapshot(failure=ConnectionError("offline"))
        workflow = calculator_workflow()
        with self.assertRaises(EpisodeCollectionError) as context:
            await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(context.exception.batch.trajectories, [])
        episode = context.exception.batch.episodes[0]
        self.assertEqual(episode.status, "sampling_failure")
        self.assertIsNone(episode.score)

    async def test_invalid_json_is_model_failure_with_trainable_actions(self):
        policy, _ = snapshot(output='{"tool_calls":')
        batch = await calculator_workflow().collect(calculator_rows()[:1], policy)
        self.assertEqual(batch.episodes[0].status, "invalid_action")
        self.assertEqual(batch.trajectories[0].reward, 0)

    async def test_output_and_context_limits_have_explicit_terminal_status(self):
        policy, _ = snapshot()
        workflow = calculator_workflow(limits=EpisodeLimits(max_tokens_per_turn=2))
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(batch.episodes[0].status, "token_limit")
        self.assertTrue(batch.trajectories[0].truncated)
        workflow = calculator_workflow(limits=EpisodeLimits(max_context_tokens=1))
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "context_limit")
        self.assertFalse(batch.trajectories)

    async def test_tool_and_episode_timeouts_reject_training(self):
        async def slow(**kwargs):
            await asyncio.sleep(1)

        policy, _ = snapshot()
        workflow = calculator_workflow(limits=EpisodeLimits(tool_timeout=0.01))
        workflow.runner.tools = FunctionToolExecutor(
            [FunctionTool("calculate", "", {"type": "object"}, slow)]
        )
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "timeout")
        self.assertFalse(batch.trajectories)
        self.assertIsNone(batch.episodes[0].score)

    async def test_template_rewriting_rejects_even_an_otherwise_successful_episode(
        self,
    ):
        policy, _ = snapshot()
        from triotune.agentrl import ChatTemplateRenderer

        class RewritingRenderer(ChatTemplateRenderer):
            def render(self, messages, tools):
                tokens = super().render(messages, tools)
                if len(messages) > 2:
                    tokens[0] += 1
                return tokens

        workflow = calculator_workflow(renderer=RewritingRenderer(policy.tokenizer))
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "contract_error")
        self.assertFalse(batch.trajectories)

    async def test_scorer_failure_and_cancellation_do_not_update_policy(self):
        policy, _ = snapshot()

        class BrokenScorer:
            async def score(self, row, episode):
                raise OSError("judge offline")

        workflow = calculator_workflow()
        workflow.runner.scorer = BrokenScorer()
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "scoring_failure")
        self.assertFalse(batch.trajectories)

        class CancelledScorer:
            async def score(self, row, episode):
                raise asyncio.CancelledError()

        workflow.runner.scorer = CancelledScorer()
        with self.assertRaises(asyncio.CancelledError):
            await workflow.collect(calculator_rows()[:1], policy)

    async def test_one_system_failure_rejects_the_entire_prompt_group(self):
        policy, client = snapshot()
        original = client.sample_async

        async def fail_one(**kwargs):
            if kwargs["sampling_params"].seed == 8:
                raise ConnectionError("one replica failed")
            return await original(**kwargs)

        client.sample_async = fail_one
        batch = await calculator_workflow().collect(
            calculator_rows()[:1],
            policy,
            n_samples=2,
            seed=0,
            strict=False,
        )
        self.assertEqual(len([e for e in batch.episodes if e.trainable]), 1)
        self.assertEqual(len(batch.trajectories), 0)
        self.assertEqual(batch.metrics()["agent/system_failure_rate"], 0.5)

    async def test_collection_bounds_concurrency_and_scopes_group_ids(self):
        policy, client = snapshot()
        original = client.sample_async
        running = peak = 0

        async def delayed(**kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await asyncio.sleep(0.005)
                return await original(**kwargs)
            finally:
                running -= 1

        client.sample_async = delayed
        workflow = calculator_workflow()
        workflow.max_concurrency = 2
        first = await workflow.collect([calculator_rows()[0]] * 2, policy, n_samples=3)
        second = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(peak, 2)
        first_groups = {e.group_id for e in first.episodes}
        self.assertEqual(len(first_groups), 2)
        self.assertTrue(first_groups.isdisjoint({e.group_id for e in second.episodes}))

    async def test_sampling_timeout_is_not_a_model_failure(self):
        policy, client = snapshot()

        async def slow(**kwargs):
            await asyncio.sleep(1)

        client.sample_async = slow
        batch = await calculator_workflow(
            limits=EpisodeLimits(sampling_timeout=0.01)
        ).collect(
            calculator_rows()[:1],
            policy,
            strict=False,
        )
        self.assertEqual(batch.episodes[0].status, "timeout")
        self.assertIsNone(batch.episodes[0].score)

    async def test_missing_behavior_logprobs_rejects_episode(self):
        policy, client = snapshot()
        original = client.sample_async

        async def missing(**kwargs):
            result = await original(**kwargs)
            result.sequences[0].logprobs = None
            return result

        client.sample_async = missing
        batch = await calculator_workflow().collect(
            calculator_rows()[:1], policy, strict=False
        )
        self.assertEqual(batch.episodes[0].status, "contract_error")
        self.assertFalse(batch.trajectories)
