"""Reward composition preserves success and aligns costs/PRMs to action IDs."""

import unittest

from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot

from triotune import PPOConfig
from triotune.agentrl import (
    AgentContractError,
    CompositeScorer,
    CostConfig,
    EpisodeScore,
    RewardTerm,
    ToolCostScorer,
    TurnRewardScorer,
)
from triotune.trainer.ppo import prepare_advantages


class RewardTests(unittest.IsolatedAsyncioTestCase):
    async def test_costs_land_on_responsible_turn_and_reach_gae(self):
        policy, _ = snapshot()
        workflow = calculator_workflow()
        workflow.runner.scorer = CompositeScorer(
            workflow.runner.scorer,
            [
                RewardTerm(
                    "cost",
                    ToolCostScorer(CostConfig(tool_call=0.1, format_reward=0.02)),
                    2,
                )
            ],
        )
        batch = await workflow.collect(calculator_rows()[:1], policy)
        episode, trajectory = batch.episodes[0], batch.trajectories[0]
        self.assertTrue(episode.score.success)
        self.assertEqual(episode.score.reward, 1)
        self.assertAlmostEqual(sum(episode.score.process_rewards[0]), -0.2)
        self.assertAlmostEqual(sum(episode.score.process_rewards[1]), 0.04)
        self.assertTrue(
            all(
                reward == 0
                for reward, active in zip(
                    trajectory.token_rewards, trajectory.loss_mask
                )
                if not active
            )
        )
        prepared = prepare_advantages(
            [trajectory],
            PPOConfig.for_algorithm(
                "grpo", reward_norm=None, adv_norm=None, gae_timestep_unit="turn"
            ),
        )
        self.assertAlmostEqual(prepared.advantages[0][0], 0.84)

    async def test_turn_adapter_discount_and_token_prm_combine_without_mutation(self):
        policy, _ = snapshot()
        workflow = calculator_workflow()
        episode = (await workflow.collect(calculator_rows()[:1], policy)).episodes[0]
        n = len(episode.turns[0].generation.tokens)
        original = EpisodeScore(0, True, process_rewards={0: tuple([0.1] * n)})

        class PRM:
            async def score(self, row, episode):
                return original

        scorer = CompositeScorer(
            workflow.runner.scorer,
            [
                RewardTerm(
                    "turn",
                    TurnRewardScorer(lambda row, episode: {1: 0.5}, discount=0.5),
                ),
                RewardTerm("prm", PRM(), 0.1),
            ],
        )
        result = await scorer.score(calculator_rows()[0], episode)
        self.assertAlmostEqual(sum(result.process_rewards[0]), 0.01 * n)
        self.assertAlmostEqual(sum(result.process_rewards[1]), 0.25)
        self.assertEqual(original.process_rewards[0], tuple([0.1] * n))
        self.assertTrue(result.success)
        self.assertEqual(result.reward, 1)

    async def test_auxiliary_success_cannot_turn_wrong_answer_into_success(self):
        policy, _ = snapshot(output='{"final":"wrong"}')
        workflow = calculator_workflow()

        class Bonus:
            async def score(self, row, episode):
                return EpisodeScore(5, True)

        workflow.runner.scorer = CompositeScorer(
            workflow.runner.scorer, [RewardTerm("bonus", Bonus())]
        )
        result = (
            (await workflow.collect(calculator_rows()[:1], policy)).episodes[0].score
        )
        self.assertFalse(result.success)
        self.assertEqual(result.reward, 5)

    async def test_bad_prm_shapes_unknown_turns_and_nan_are_rejected(self):
        policy, _ = snapshot()
        workflow = calculator_workflow()
        episode = (await workflow.collect(calculator_rows()[:1], policy)).episodes[0]
        for value in ({9: 1}, {0: float("nan")}, {True: 1}):
            with self.subTest(value=value), self.assertRaises(AgentContractError):
                await TurnRewardScorer(lambda row, episode, value=value: value).score(
                    {}, episode
                )

        class BadPRM:
            async def score(self, row, episode):
                return EpisodeScore(0, False, process_rewards={0: (1,)})

        with self.assertRaises(AgentContractError):
            await CompositeScorer(
                workflow.runner.scorer, [RewardTerm("prm", BadPRM())]
            ).score(calculator_rows()[0], episode)

    def test_invalid_costs_and_duplicate_names_fail_at_construction(self):
        for kwargs in (
            {"tool_call": -1},
            {"turn_discount": 0},
            {"generated_token": True},
        ):
            with self.assertRaises(ValueError):
                CostConfig(**kwargs)
        scorer = ToolCostScorer()
        with self.assertRaises(ValueError):
            CompositeScorer(
                scorer, [RewardTerm("cost", scorer), RewardTerm("cost", scorer)]
            )

    async def test_dynamic_filter_keeps_equal_outcomes_with_process_rewards(self):
        from triotune.agentrl.training.loop import filter_agent_groups

        policy, _ = snapshot()
        batch = await calculator_workflow(costs=CostConfig(tool_call=0.1)).collect(
            calculator_rows()[:1], policy, n_samples=2
        )
        self.assertEqual([t.reward for t in batch.trajectories], [1, 1])
        self.assertEqual(len(filter_agent_groups(batch.trajectories)), 2)
        for t in batch.trajectories:
            t.token_rewards = None
        self.assertEqual(filter_agent_groups(batch.trajectories), [])

    def test_overlong_penalty_counts_actions_and_excludes_tool_context(self):
        from triotune.trainer.ppo import Trajectory

        config = PPOConfig.for_algorithm(
            "reinforce",
            max_tokens=4,
            overlong_tokens=2,
            overlong_penalty_factor=1,
            adv_norm=None,
        )
        for n_actions, expected in ((2, 1.0), (3, 0.5), (4, 0.0)):
            mask = [True] * n_actions + [False] * 20
            trajectory = Trajectory(
                [1],
                list(range(30, 30 + len(mask))),
                [-1.0] * len(mask),
                1,
                "g",
                loss_mask=mask,
            )
            prepared = prepare_advantages([trajectory], config)
            self.assertAlmostEqual(prepared.advantages[0][0], expected)
