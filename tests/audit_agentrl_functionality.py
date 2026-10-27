"""Requirement-level audit of AgentRL extension boundaries.

Run explicitly with::

    PYTHONPATH=.:tests uv run --with pytest pytest -q tests/audit_agentrl_functionality.py

These assertions describe the requested behavior, including currently missing
behavior. The filename deliberately keeps this diagnostic audit separate from
the ordinary passing regression suite. No remote inference service is used.
"""

import asyncio
import unittest
from types import SimpleNamespace

from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot
from test_agent_trajectory_builder import episode
from test_openai_native import FakeSamplingClient, NativeTokenizer, xml_call
from test_openai_sdk import _FakeChat, add

from triotune.agentrl import (
    EpisodeResult,
    EpisodeScore,
    EpisodeTrajectoryBuilder,
    OpenAIEpisodeExecutor,
    ExecutionContext,
    RewardAssignment,
    RewardLedger,
    current_execution_context,
)
from triotune.agentrl.runtime.sampling import PyTRIOAgentSampler, SamplingSnapshot
from triotune.agentrl.sdk.openai import (
    Agent,
    OpenAIClient,
    OpenAIWorkflow,
    PyTRIOBackend,
)
from triotune.trainer.ppo import NormConfig, PPOAlgorithmPlan, PPOConfig, Trajectory
from triotune.trainer.ppo import prepare_advantages
from triotune.trainer.ppo import PPOTrainer, compute_gae
from triotune.trainer.ppo.loss import make_loss_fn


def config(**kwargs):
    return PPOConfig(use_critic=False, reward_norm=None, adv_norm=None, **kwargs)


def sdk_workflow(chat=None, **options):
    chat = chat or _FakeChat()
    client = OpenAIClient(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)), model="fake"
    )
    agent = Agent(
        name="calculator", instructions="Use tools.", tools=[add], client=client
    )
    return OpenAIWorkflow(agent, **options), chat


class PPOPluginAudit(unittest.TestCase):
    def test_separate_outcome_and_process_shapers_both_reach_returns(self):
        row = Trajectory([1], [2], [-1.0], 1.0, "g", token_rewards=[1.0])
        batch = prepare_advantages(
            [row],
            config(),
            PPOAlgorithmPlan(
                reward_shaper=lambda trajectory, reward: reward * 2,
                process_shaper=lambda trajectory, rewards: [rewards[0] * 3],
            ),
        )
        self.assertEqual(batch.returns, [[5.0]])
        self.assertEqual(batch.advantages, [[5.0]])

    def test_return_plugin_can_change_critic_targets_independently(self):
        row = Trajectory([1], [2], [-1.0], 1.0, "g")
        batch = prepare_advantages(
            [row],
            config(),
            PPOAlgorithmPlan(return_estimator=lambda *args, **kwargs: [7.0]),
        )
        self.assertEqual(batch.returns, [[7.0]])
        self.assertEqual(batch.advantages, [[1.0]])

    def test_estimator_can_select_dynamic_lambda(self):
        row = Trajectory([1], [2, 3], [-1.0, -1.0], 1.0, "g")

        def estimate(rewards, values, mask, **kwargs):
            kwargs["gae_lambda"] = 0.0 if len(rewards) > 1 else 1.0
            return compute_gae(rewards, values, mask, **kwargs)

        batch = prepare_advantages(
            [row], config(), PPOAlgorithmPlan(advantage_estimator=estimate)
        )
        self.assertEqual(batch.advantages, [[0.0, 1.0]])

    def test_documented_normalizer_receives_masks_not_config(self):
        observed = []

        class Normalizer:
            # Match RewardNormalizer.normalize exactly.
            def normalize(self, rows, group_ids, masks=None):
                observed.append(masks)
                return rows

        row = Trajectory([1], [2], [-1.0], 1.0, "g")
        prepare_advantages(
            [row],
            PPOConfig(use_critic=False, reward_norm=NormConfig("group", None)),
            PPOAlgorithmPlan(normalizer=Normalizer()),
        )
        self.assertEqual(observed, [None])

    def test_reward_shaper_protocol_applies_process_method(self):
        class Shaper:
            def shape(self, trajectory, reward):
                return reward

            def shape_process(self, trajectory, rewards):
                return [value + 2.0 for value in rewards]

        row = Trajectory([1], [2], [-1.0], 0.0, "g", token_rewards=[1.0])
        batch = prepare_advantages(
            [row], config(), PPOAlgorithmPlan(reward_shaper=Shaper())
        )
        self.assertEqual(batch.returns, [[3.0]])

    def test_plugin_typeerror_propagates_without_signature_retry(self):
        def broken_shaper(trajectory, reward):
            raise TypeError("failure inside the plugin")

        row = Trajectory([1], [2], [-1.0], 0.0, "g")
        with self.assertRaisesRegex(TypeError, "failure inside the plugin"):
            prepare_advantages(
                [row], config(), PPOAlgorithmPlan(reward_shaper=broken_shaper)
            )


class RewardAudit(unittest.TestCase):
    def test_token_advantage_changes_policy_without_changing_critic_return(self):
        row = Trajectory([1], [2], [-1.0], 0.0, "g", token_advantages=[3.0])
        batch = prepare_advantages([row], config())
        self.assertEqual(batch.advantages, [[3.0]])
        self.assertEqual(batch.returns, [[0.0]])

    def assert_aggregation(self, aggregation, expected):
        ledger = RewardLedger()
        ledger.submit(
            [
                RewardAssignment("trajectory", value, aggregation=aggregation)
                for value in (2.0, 4.0)
            ],
            logical_rollout_id="rollout",
        )
        self.assertEqual(ledger.aggregate()[0], expected)

    def test_one_submission_preserves_trajectory_sum(self):
        self.assert_aggregation("sum", 6.0)

    def test_one_submission_preserves_trajectory_mean(self):
        self.assert_aggregation("mean", 3.0)

    def test_one_submission_preserves_trajectory_last(self):
        self.assert_aggregation("last", 4.0)

    def test_trajectory_advantage_assignment_is_not_silently_lost(self):
        value = episode()
        value.score = EpisodeScore(
            0.0,
            False,
            assignments=(RewardAssignment("trajectory", 5.0, mode="advantage"),),
        )
        row = EpisodeTrajectoryBuilder().build(value)
        batch = prepare_advantages([row], config())
        self.assertTrue(
            any(batch.advantages[0]), "Accepted direct advantage was dropped"
        )
        self.assertEqual(batch.returns, [[0.0] * len(row.completion_tokens)])


class RuntimeAudit(unittest.IsolatedAsyncioTestCase):
    async def test_custom_objective_and_row_weight_reach_pytrio_update(self):
        observed = []

        class CustomObjective:
            def make_loss(self, batch, configuration, diagnostics=None):
                observed.append(tuple(batch.loss_weights))
                return make_loss_fn(batch, configuration, diagnostics)

        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake",
            args=config(),
            training_client=client,
            plan=PPOAlgorithmPlan(
                objective=CustomObjective(),
                loss_weight_policy=lambda trajectory, mask: 2.0,
            ),
        )
        try:
            await trainer.step([Trajectory([1], [2], [-1.0], 1.0, "g")])
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(observed, [(2.0,)])
        finally:
            await trainer.close()

    async def test_interaction_scorer_runs_under_its_child_context(self):
        policy, _ = snapshot()
        observed = []

        class Scorer:
            async def score(self, row, result, interaction):
                observed.append((current_execution_context(), interaction.context))
                return EpisodeScore(0.0, False)

        workflow = calculator_workflow()
        workflow.runner.interaction_scorer = Scorer()
        async with workflow:
            await workflow.collect(calculator_rows()[:1], policy)
        self.assertTrue(observed)
        for actual, expected in observed:
            self.assertEqual(actual, expected)

    async def test_interaction_reward_evidence_matches_exported_process_reward(self):
        policy, _ = snapshot()

        class Scorer:
            async def score(self, row, result, interaction):
                return EpisodeScore(
                    2.0 if interaction.parent_interaction_id is None else 0.0, False
                )

        workflow = calculator_workflow()
        workflow.runner.interaction_scorer = Scorer()
        async with workflow:
            batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(sum(batch.trajectories[0].token_rewards), 2.0)
        self.assertEqual(batch.episodes[0].interactions[0].reward, 2.0)


class SDKWorkflowAudit(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_workflow_honors_total_token_budget(self):
        workflow, chat = sdk_workflow()
        async with workflow:
            with self.assertRaises(TimeoutError):
                await workflow.run("2+3", max_generated_tokens=2)
        self.assertEqual(
            len(chat.calls), 1, "Workflow discarded the total token budget"
        )

    async def test_builtin_workflow_honors_episode_timeout(self):
        class SlowChat(_FakeChat):
            async def create(self, **kwargs):
                await asyncio.sleep(0.03)
                return await super().create(**kwargs)

        workflow, _ = sdk_workflow(SlowChat())
        async with workflow:
            with self.assertRaises(TimeoutError):
                await workflow.run("2+3", episode_timeout=0.001)

    async def test_workflow_scorer_can_read_candidate_identity(self):
        observed = []

        class Scorer:
            async def score(self, input, result):
                observed.append((current_execution_context(), result.context))
                return EpisodeScore(1.0, True)

        workflow, _ = sdk_workflow(scorer=Scorer())
        async with workflow:
            await workflow.run("2+3")
        self.assertTrue(observed)
        self.assertEqual(observed[0][0], observed[0][1])

    async def test_scoring_failure_keeps_stage_when_acceptance_hook_is_present(self):
        observed = []

        async def broken_score(input, result):
            raise ConnectionError("judge offline")

        def classify(stage, error, record):
            observed.append(stage)
            return "scoring_failure", str(error)

        workflow, _ = sdk_workflow(
            scorer=broken_score,
            accept_sample=lambda result, exported: True,
            failure_classifier=classify,
        )
        async with workflow:
            with self.assertRaises(ConnectionError):
                await workflow.run("2+3")
        self.assertEqual(observed, ["scoring"])

    async def test_pytrio_sdk_workflow_returns_collector_episode(self):
        tokenizer = NativeTokenizer()
        sampling = FakeSamplingClient(
            tokenizer,
            ["</think>\n" + xml_call() + "<|im_end|>", "</think>\n5<|im_end|>"],
            noncanonical=True,
        )
        policy = SamplingSnapshot(
            "snapshot",
            PyTRIOAgentSampler(sampling, tokenizer),
            tokenizer,
            policy_version=0,
        )
        context = ExecutionContext(
            "rollout", "episode", "group", 0, "snapshot", policy_version=0
        )
        workflow = OpenAIWorkflow(
            Agent(
                name="calculator",
                instructions="Solve the problem.",
                tools=[add],
                model="Qwen/Qwen3.5-4B",
                client=OpenAIClient(
                    backend=PyTRIOBackend(policy, model="Qwen/Qwen3.5-4B")
                ),
            ),
        )

        # Use the same async scorer boundary as normal runtime execution.
        async def score(input, result):
            return EpisodeScore(1.0, True)

        # The trace facade remains backward compatible. Its explicit episode
        # adapter supplies the collector contract and standard PPO exporter.
        runner = OpenAIEpisodeExecutor(lambda snapshot: workflow, scorer=SimpleNamespace(score=score))
        async with runner:
            result = await runner.run({"messages": "2+3"}, policy, group_id=context.group_id, context=context)
        self.assertEqual(len(sampling.calls), 2)
        self.assertTrue(result.interactions[0].tool_results)
        self.assertEqual(
            result.interactions[0].turn.generation.tokens, tuple(sampling.generated[0])
        )
        self.assertEqual(result.snapshot_id, policy.snapshot_id)
        self.assertTrue(
            isinstance(result, EpisodeResult),
            f"Returned {type(result).__name__}, not a collector EpisodeResult",
        )
        exported = EpisodeTrajectoryBuilder().export(result)
        self.assertTrue(exported.trajectories)


if __name__ == "__main__":
    unittest.main()
