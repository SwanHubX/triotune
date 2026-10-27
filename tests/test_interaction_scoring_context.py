"""Interaction identity across scoring, PRM calls, retries, and cleanup."""

import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace

from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot
from test_openai_sdk import add
from test_ppo_family import raw_config

from triotune import PPOTrainer
from triotune.agentrl import (
    EpisodeScore,
    ExecutionContext,
    RetryPolicy,
    RewardAssignment,
    current_execution_context,
)
from triotune.agentrl.integrations.openai import (
    Agent,
    OpenAIClient,
    OpenAISession,
    OpenAIWorkflow,
)
from triotune.agentrl.types import _execution_context
from triotune.api.client_api import call_async


def context(index=0):
    return ExecutionContext(
        f"rollout-{index}",
        f"episode-{index}",
        "group",
        index,
        "snapshot-0",
        policy_version=0,
        group_size=3,
        resource_owner="runner",
    )


def sdk_workflow(kind, **options):
    """Use stateless responses so one workflow can serve concurrent candidates."""

    async def create(**payload):
        identity = current_execution_context().interaction_id
        await asyncio.sleep(0)
        final = any(item.get("role") == "tool" for item in payload["messages"])
        message = (
            {"role": "assistant", "content": "5"}
            if final
            else {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{identity}",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                    }
                ],
            }
        )
        return {
            "id": f"response-{identity}",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }

    adapter = OpenAIClient(
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="fake",
    )
    if kind == "sdk-callback":
        return OpenAIWorkflow(client=adapter, **options)
    return OpenAIWorkflow(
        Agent(
            name="calculator", instructions="Use tools.", tools=[add], client=adapter
        ),
        **options,
    )


async def external_agent(sdk, question):
    """Execute tools using only standard AsyncOpenAI messages."""
    messages = [{"role": "user", "content": question}]
    response = await sdk.chat.completions.create(model="fake", messages=messages)
    message = response.choices[0].message
    call = message.tool_calls[0]
    final = await sdk.chat.completions.create(
        model="fake",
        messages=messages
        + [message.model_dump(exclude_none=True)]
        + [{"role": "tool", "tool_call_id": call.id, "content": "5"}],
    )
    return final.choices[0].message.content


class ScoringContextTests(unittest.IsolatedAsyncioTestCase):
    kinds = ("runner", "sdk-agent", "sdk-callback")

    def workflow(self, kind, scorer, **options):
        if kind != "runner":
            return sdk_workflow(kind, interaction_scorer=scorer, **options)
        workflow = calculator_workflow()
        workflow.runner.interaction_scorer = scorer
        for name, value in options.items():
            setattr(workflow.runner, name, value)
        return workflow

    async def run_candidate(self, kind, workflow, identity):
        if kind == "runner":
            policy, _ = snapshot()
            policy = replace(policy, policy_version=identity.policy_version)
            return await workflow.runner.run(
                calculator_rows()[0],
                policy,
                group_id=identity.group_id,
                sample_index=identity.sample_index,
                context=identity,
            )
        session = OpenAISession(context=identity, training=True)
        return await workflow.run(
            "2+3",
            session=session,
            **({"execute": external_agent} if kind == "sdk-callback" else {}),
        )

    async def test_concurrent_candidates_keep_child_context_in_logs_and_pytrio_prms(
        self,
    ):
        for kind in self.kinds:
            with self.subTest(kind=kind):
                observations, roots = [], []

                async def score(row, result, interaction):
                    expected = interaction.context
                    observations.append((current_execution_context(), expected))
                    await asyncio.sleep(0)
                    observations.append((current_execution_context(), expected))

                    # call_async exercises both PyTRIO-style asynchronous
                    # result stages and the synchronous worker-thread fallback.
                    def sync_score():
                        observations.append((current_execution_context(), expected))
                        return 0.25

                    async def async_score():
                        observations.append((current_execution_context(), expected))

                        async def resolve():
                            await asyncio.sleep(0)
                            observations.append((current_execution_context(), expected))
                            return 0.25

                        return resolve()

                    await call_async(SimpleNamespace(score=sync_score), "score")
                    await call_async(SimpleNamespace(score_async=async_score), "score")
                    return RewardAssignment("interaction", 0.25)

                async def outcome(row, result):
                    roots.append((current_execution_context(), result.context))
                    return EpisodeScore(1.0, True)

                workflow = self.workflow(kind, score, scorer_timeout=1.0)
                if kind == "runner":
                    workflow.runner.scorer = SimpleNamespace(score=outcome)
                    original = workflow.runner.submit_reward

                    async def submit(result, score=None):
                        roots.append((current_execution_context(), result.context))
                        return await original(result, score)

                    workflow.runner.submit_reward = submit
                else:
                    workflow.scorer = outcome

                    def accept(result, exported):
                        roots.append((current_execution_context(), result.context))
                        return True

                    workflow.accept_sample = accept

                outer = replace(
                    context(2),
                    interaction_id="enclosing-interaction",
                    parent_interaction_id="enclosing-parent",
                )
                token = _execution_context.set(outer)
                try:
                    async with workflow:
                        results = await asyncio.gather(
                            *(
                                self.run_candidate(kind, workflow, context(index))
                                for index in range(3)
                            )
                        )
                    self.assertIs(current_execution_context(), outer)
                finally:
                    _execution_context.reset(token)
                self.assertTrue(observations)
                self.assertEqual(
                    {actual.sample_index for actual, _ in observations}, {0, 1, 2}
                )
                for actual, expected in observations + roots:
                    self.assertEqual(actual, expected)
                for result in results:
                    first, second = result.interactions
                    self.assertEqual(
                        second.context.parent_interaction_id,
                        first.context.interaction_id,
                    )
                self.assertIsNone(current_execution_context())

    async def test_retry_attempts_and_retry_events_keep_interaction_identity(self):
        for kind in self.kinds:
            with self.subTest(kind=kind):
                attempts, contexts = {}, []

                async def score(row, result, interaction):
                    contexts.append((current_execution_context(), interaction.context))
                    key = interaction.context.interaction_id
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] == 1:
                        raise ConnectionError("PRM temporarily unavailable")
                    return RewardAssignment("interaction", 0.25)

                workflow = self.workflow(
                    kind,
                    score,
                    scorer_timeout=1.0,
                    scorer_retry=RetryPolicy(
                        max_attempts=2, initial_delay=0, max_delay=0
                    ),
                )
                async with workflow:
                    result = await self.run_candidate(kind, workflow, context())
                record = result if kind == "runner" else result.session
                events = [event for event in record.events if event.kind == "retry"]
                self.assertEqual(len(events), 2)
                self.assertEqual(list(attempts.values()), [2, 2])
                by_id = {
                    item.context.interaction_id: item.context
                    for item in result.interactions
                }
                for event in events:
                    self.assertEqual(event.details["stage"], "interaction_scoring")
                    self.assertIn(event.context.interaction_id, by_id)
                    self.assertEqual(event.context, by_id[event.context.interaction_id])
                self.assertTrue(
                    all(actual == expected for actual, expected in contexts)
                )
                self.assertIsNone(current_execution_context())

    async def test_failure_timeout_and_cancel_restore_exact_enclosing_context(self):
        for kind in self.kinds:
            for failure in ("error", "timeout", "cancel"):
                with self.subTest(kind=kind, failure=failure):
                    entered = asyncio.Event()
                    observations, restored, records = [], [], []

                    async def score(row, result, interaction):
                        records.append(result)
                        entered.set()
                        try:
                            if failure == "error":
                                raise RuntimeError("judge failure")
                            await asyncio.Event().wait()
                        finally:
                            observations.append(
                                (current_execution_context(), interaction.context)
                            )

                    workflow = self.workflow(
                        kind,
                        score,
                        scorer_timeout=0.02 if failure == "timeout" else None,
                    )
                    outer = replace(
                        context(2),
                        interaction_id="enclosing-interaction",
                        parent_interaction_id="enclosing-parent",
                    )
                    token = _execution_context.set(outer)

                    async def invoke():
                        try:
                            return await self.run_candidate(kind, workflow, context())
                        finally:
                            restored.append(current_execution_context())

                    try:
                        async with workflow:
                            task = asyncio.create_task(invoke())
                            await asyncio.wait_for(entered.wait(), 5.0)
                            if failure == "cancel":
                                task.cancel()
                            if failure == "cancel":
                                with self.assertRaises(asyncio.CancelledError):
                                    await task
                            elif kind == "runner":
                                result = await task
                                self.assertEqual(result.status, "scoring_failure")
                                self.assertIsNone(result.score)
                            else:
                                with self.assertRaises(
                                    RuntimeError if failure == "error" else TimeoutError
                                ):
                                    await task
                        self.assertEqual(restored, [outer])
                        self.assertIs(current_execution_context(), outer)
                    finally:
                        _execution_context.reset(token)
                    self.assertTrue(
                        all(actual == expected for actual, expected in observations)
                    )
                    self.assertTrue(records[0].interactions[0].tool_results)
                    if failure == "timeout":
                        record = records[0] if kind == "runner" else records[0].session
                        retries = [
                            event for event in record.events if event.kind == "retry"
                        ]
                        self.assertEqual(len(retries), 1)
                        self.assertEqual(
                            retries[0].context, records[0].interactions[0].context
                        )
                    self.assertIsNone(current_execution_context())

    async def test_legacy_two_argument_runner_scorer_receives_child_context(self):
        observed = []

        async def score(row, interaction):
            observed.append((current_execution_context(), interaction.context))
            return None

        workflow = self.workflow("runner", score)
        async with workflow:
            await self.run_candidate("runner", workflow, context())
        self.assertTrue(observed)
        self.assertTrue(all(actual == expected for actual, expected in observed))

    async def test_context_aware_interaction_rewards_reach_existing_ppo_update(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer("fake", args=raw_config(), training_client=client)
        observed = []

        async def score(row, result, interaction):
            current = current_execution_context()
            observed.append((current, interaction.context))
            return RewardAssignment("interaction", 0.25 * (current.sample_index + 1))

        workflow = self.workflow("runner", score)
        try:
            policy = await trainer.create_sampling_snapshot()
            async with workflow:
                batch = await workflow.collect(
                    calculator_rows()[:1], policy, n_samples=2
                )
            for episode, row in zip(batch.episodes, batch.trajectories):
                self.assertEqual(
                    sum(row.token_rewards), 0.5 * (episode.sample_index + 1)
                )
                self.assertEqual(row.snapshot_id, policy.snapshot_id)
                self.assertEqual(row.policy_version, policy.policy_version)
            self.assertTrue(all(actual == expected for actual, expected in observed))
            await trainer.step(batch.trajectories)
            self.assertEqual(trainer.state.global_step, 1)
        finally:
            await trainer.close()


if __name__ == "__main__":
    unittest.main()
