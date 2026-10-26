"""Contracts for structured rewards, PPO plans and runner lifecycle hooks."""

import unittest

from triotune.agentrl import EpisodeScore, RewardAssignment, RewardLedger
from triotune.agentrl.runtime.runner import EpisodeRunner
from triotune.trainer.ppo import (
    PPOAlgorithmPlan,
    Trajectory,
    compute_trajectory_level_gae,
    prepare_advantages,
)
from triotune import PPOConfig


class StructuredRewardTests(unittest.TestCase):
    def test_ledger_deduplicates_one_logical_outcome(self):
        ledger = RewardLedger()
        outcome = RewardAssignment("trajectory", 2.0, source="outcome")
        self.assertEqual(len(ledger.submit(outcome, logical_rollout_id="r")), 1)
        self.assertEqual(len(ledger.submit(outcome, logical_rollout_id="r")), 0)
        self.assertEqual(ledger.aggregate()[0], 2.0)

    def test_episode_score_keeps_advantage_separate(self):
        score = EpisodeScore(
            0.0,
            True,
            assignments=(
                RewardAssignment("token", 0.5, mode="advantage", position=(0, 1)),
            ),
        )
        self.assertEqual(score.assignments[0].mode, "advantage")


class PPOPlanTests(unittest.TestCase):
    def test_trajectory_estimator_is_public(self):
        advantages, returns = compute_trajectory_level_gae(
            [1.0, 2.0, 0.0], [0.5, 0.0, 0.0, 3.0], [True, True, False], truncated=True
        )
        self.assertEqual(advantages, [5.5, 5.5, 0.0])
        self.assertEqual(returns, [6.0, 6.0, 0.0])

    def test_custom_row_weight_and_advantage_stage_are_public(self):
        trajectory = Trajectory([1], [2, 3], [-1.0, -1.0], 1.0, "g")

        class Estimator:
            def estimate(self, rewards, values, mask, **kwargs):
                return [4.0, 4.0], [7.0, 7.0]

        plan = PPOAlgorithmPlan(
            advantage_estimator=Estimator(),
            loss_weight_policy=lambda trajectory, mask: 2.0,
        )
        batch = prepare_advantages(
            [trajectory],
            PPOConfig.for_algorithm("reinforce", reward_norm=None, adv_norm=None),
            plan,
        )
        self.assertEqual(batch.advantages, [[4.0, 4.0]])
        self.assertEqual(batch.returns, [[7.0, 7.0]])
        self.assertEqual(batch.loss_weights, [2.0])


class RunnerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_scorer_resources_start_and_close_once(self):
        class Resource:
            schemas = ()

            def __init__(self):
                self.started = self.closed = 0

            async def start(self):
                self.started += 1

            async def close(self):
                self.closed += 1

        parser = object()
        tools = Resource()
        scorer = Resource()
        observations = object()
        runner = EpisodeRunner(
            parser=parser,
            tools=tools,
            scorer=scorer,
            observations=observations,
        )
        await runner.start()
        await runner.start()
        await runner.close()
        self.assertEqual((tools.started, tools.closed), (1, 1))
        self.assertEqual((scorer.started, scorer.closed), (1, 1))


if __name__ == "__main__":
    unittest.main()
