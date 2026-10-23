"""Transient retries never replace model failures or repeat successful siblings."""

import asyncio
import unittest

from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentContractError,
    AgentTrainingLoop,
    EpisodeLimits,
    FunctionTool,
    FunctionToolExecutor,
    RetryPolicy,
)


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def retry(self, **kwargs):
        return RetryPolicy(max_attempts=3, initial_delay=0, max_delay=0, **kwargs)

    async def test_sampling_retries_same_prompt_seed_and_snapshot(self):
        policy, client = snapshot()
        original, attempts = client.sample_async, []

        async def unstable(**kwargs):
            attempts.append(
                (kwargs["prompt"].to_ints(), kwargs["sampling_params"].seed)
            )
            if len(attempts) == 1:
                raise ConnectionError("temporary")
            return await original(**kwargs)

        client.sample_async = unstable
        workflow = calculator_workflow()
        workflow.runner.retry = self.retry()
        batch = await workflow.collect(
            calculator_rows()[:1], policy, n_samples=2, seed=0
        )
        self.assertEqual(len(batch.trajectories), 2)
        retried = [attempt for attempt in attempts if attempt[1] == 0]
        self.assertEqual(len(retried), 2)
        self.assertEqual(retried[0], retried[1])
        self.assertTrue(
            all(e.snapshot_id == policy.snapshot_id for e in batch.episodes)
        )
        self.assertEqual(batch.metrics()["agent/retries/sampling"], 1)
        # Two successful turns plus the model's one-turn incorrect sibling.
        self.assertEqual(len(attempts), 4)

    async def test_contract_and_model_failures_are_not_retried(self):
        policy, client = snapshot(output='{"tool_calls":')
        workflow = calculator_workflow()
        workflow.runner.retry = self.retry()
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(batch.episodes[0].status, "invalid_action")
        self.assertEqual(batch.metrics()["agent/retry_count"], 0)
        policy, _ = snapshot(failure=AgentContractError("broken alignment"))
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "contract_error")
        self.assertEqual(batch.episodes[0].retries, [])

    async def test_pytrio_explicit_retryability_is_respected(self):
        import pytrio

        for retryable in (True, False):
            policy, _ = snapshot(
                failure=pytrio.PyTrioError("service failure", retryable=retryable)
            )
            workflow = calculator_workflow()
            workflow.runner.retry = self.retry()
            batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
            self.assertEqual(batch.episodes[0].status, "sampling_failure")
            self.assertEqual(len(batch.episodes[0].retries), 3 if retryable else 0)

    async def test_retry_exhaustion_removes_whole_group_and_logs_attempt_count(self):
        policy, _ = snapshot(failure=ConnectionError("offline"))
        workflow = calculator_workflow()
        workflow.runner.retry = self.retry()
        batch = await workflow.collect(
            calculator_rows()[:1], policy, n_samples=2, strict=False
        )
        self.assertEqual(batch.trajectories, [])
        self.assertEqual(batch.dropped_groups, 1)
        self.assertEqual(len(batch.episodes[0].retries), 3)
        self.assertFalse(batch.episodes[0].retries[-1].retried)
        self.assertIsNone(batch.episodes[0].score)

    async def test_tool_retry_requires_both_policy_and_idempotency_opt_in(self):
        for retry_tools, retry_safe in ((True, True), (True, False), (False, True)):
            with self.subTest(retry_tools=retry_tools, retry_safe=retry_safe):
                calls = 0

                async def tool(**kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise ConnectionError("temporary tool outage")
                    return 5

                policy, _ = snapshot()
                workflow = calculator_workflow()
                workflow.runner.retry = self.retry(retry_tools=retry_tools)
                workflow.runner.tools = FunctionToolExecutor(
                    [
                        FunctionTool(
                            "calculate",
                            "",
                            {"type": "object"},
                            tool,
                            retry_safe=retry_safe,
                        )
                    ]
                )
                batch = await workflow.collect(
                    calculator_rows()[:1], policy, strict=False
                )
                self.assertEqual(calls, 2 if retry_tools and retry_safe else 1)
                self.assertEqual(bool(batch.trajectories), retry_tools and retry_safe)

    async def test_scoring_retry_keeps_completed_actions_and_cancellation_propagates(
        self,
    ):
        policy, client = snapshot()
        workflow = calculator_workflow()
        original, calls = workflow.runner.scorer, 0

        class Scorer:
            async def score(self, row, episode):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ConnectionError("judge offline")
                return await original.score(row, episode)

        workflow.runner.scorer = Scorer()
        workflow.runner.retry = self.retry()
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertTrue(batch.episodes[0].score.success)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(calls, 2)

        async def cancelled(**kwargs):
            raise asyncio.CancelledError()

        client.sample_async = cancelled
        with self.assertRaises(asyncio.CancelledError):
            await workflow.collect(calculator_rows()[:1], policy)

    async def test_episode_deadline_bounds_retries(self):
        policy, _ = snapshot(failure=ConnectionError("offline"))
        workflow = calculator_workflow(limits=EpisodeLimits(episode_timeout=0.02))
        workflow.runner.retry = RetryPolicy(
            max_attempts=100, initial_delay=1, max_delay=1
        )
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "timeout")
        self.assertEqual(len(batch.episodes[0].retries), 1)
        self.assertLess(batch.episodes[0].elapsed_seconds, 0.5)

    async def test_dropped_group_is_refilled_under_one_snapshot(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm(
                "grpo", batch_size=1, num_generations=2, seed=0
            ),
            training_client=client,
        )
        workflow = calculator_workflow()
        original = workflow.runner.run
        calls = 0

        async def fail_first_group(*args, **kwargs):
            nonlocal calls
            calls += 1
            episode = await original(*args, **kwargs)
            if calls <= 2:
                episode.status, episode.score = "sampling_failure", None
            return episode

        workflow.runner.run = fail_first_group
        loop = AgentTrainingLoop(
            trainer, workflow, config=AgentConfig(on_system_failure="drop_group")
        )
        batch = await loop._collect(iter([calculator_rows()[0]] * 2))
        self.assertEqual(len(client.snapshots), 1)
        self.assertEqual(len(batch.trajectories), 2)
        self.assertEqual(batch.dropped_groups, 1)
        self.assertEqual(batch.refill_batches, 1)
        self.assertEqual(len(batch.episodes), 4)
        await trainer.close()

    async def test_refill_limit_rejects_without_optimizer(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm("grpo", batch_size=1, num_generations=2),
            training_client=client,
        )
        workflow = calculator_workflow()
        original = workflow.runner.run

        async def reject(*args, **kwargs):
            episode = await original(*args, **kwargs)
            episode.status, episode.score = "tool_failure", None
            return episode

        workflow.runner.run = reject
        loop = AgentTrainingLoop(
            trainer,
            workflow,
            config=AgentConfig(on_system_failure="drop_group", max_refill_batches=2),
        )
        with self.assertRaisesRegex(RuntimeError, "max_refill_batches"):
            await loop._collect(iter([calculator_rows()[0]] * 10))
        self.assertNotIn("submit:optim", client.trace)
        await trainer.close()
