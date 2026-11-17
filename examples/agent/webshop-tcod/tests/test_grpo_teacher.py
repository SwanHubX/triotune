"""Configuration contracts for the isolated WebShop GRPO teacher experiment."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.grpo_teacher import DEFAULT_CONFIG, load_experiment
from triotune.trainer import PPOTrainer
from webshop_env.environment import WebShopScorer


class GRPOTeacherConfigurationTests(unittest.TestCase):
    def test_complete_config_uses_grpo_grouping_and_pass1_evaluation(self):
        experiment = load_experiment(DEFAULT_CONFIG)
        trainer = experiment.trainer
        agent = experiment.agent

        self.assertEqual(trainer.model.name, "Qwen/Qwen3.5-4B")
        self.assertFalse(trainer.use_critic)
        self.assertEqual(trainer.reward_norm.mean_level, "group")
        self.assertEqual(trainer.reward_norm.std_level, "group")
        self.assertEqual(trainer.batch_size, 8)
        self.assertEqual(trainer.num_generations, 8)
        self.assertEqual(trainer.batch_size * trainer.num_generations, 64)
        self.assertEqual(trainer.learning_rate, 1e-6)
        self.assertEqual(trainer.reference_kl_coef, 0.01)
        self.assertEqual(trainer.update_epochs, 1)
        self.assertEqual(experiment.exact_success_alpha, 0.4)
        self.assertEqual(agent.limits.max_turns, 15)
        self.assertEqual(agent.evaluation.num_generations, 1)
        self.assertEqual(agent.evaluation.pass_ks, (1,))
        self.assertEqual(agent.evaluation.temperature, 0.4)
        self.assertEqual(
            Path(inspect.getfile(PPOTrainer)).resolve().parent,
            Path(__file__).resolve().parents[4] / "triotune/trainer",
        )

    def test_step_override_does_not_mutate_grpo_yaml(self):
        short = load_experiment(DEFAULT_CONFIG, steps=2)
        original = load_experiment(DEFAULT_CONFIG)
        self.assertEqual(short.trainer.max_steps, 2)
        self.assertEqual(original.trainer.max_steps, 200)

    def test_exact_success_mixture_preserves_endpoints_and_reduces_partial_credit(self):
        # The pool is not consulted by this pure reward transformation.  A
        # sentinel keeps the unit test independent of WebShop assets and Java.
        scorer = WebShopScorer(object(), exact_success_alpha=0.4)

        self.assertEqual(scorer.training_reward(0.0), 0.0)
        self.assertEqual(scorer.training_reward(1.0), 1.0)
        self.assertAlmostEqual(scorer.training_reward(0.6), 0.36)
        self.assertAlmostEqual(
            WebShopScorer(object()).training_reward(0.6),
            0.6,
        )

    def test_exact_success_mixture_rejects_invalid_alpha(self):
        for alpha in (-0.1, 1.1, True, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                WebShopScorer(object(), exact_success_alpha=alpha)


class WebShopRewardBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reward_shaping_changes_training_but_not_evaluation(self):
        class PartialPurchasePool:
            async def outcome(self, rollout_id):
                self.rollout_id = rollout_id
                return True, 0.6

        pool = PartialPurchasePool()
        scorer = WebShopScorer(pool, exact_success_alpha=0.4)
        training_episode = SimpleNamespace(status="completed", is_eval=False)
        evaluation_episode = SimpleNamespace(status="completed", is_eval=True)

        # The production runner supplies this identity through its execution
        # context.  Patching only the accessor keeps this test focused on the
        # scorer boundary rather than rebuilding a complete agent trajectory.
        with patch(
            "webshop_env.environment.current_rollout_id",
            return_value="reward-test-rollout",
        ):
            training_score = await scorer.score({}, training_episode)
            evaluation_score = await scorer.score({}, evaluation_episode)

        self.assertAlmostEqual(training_score.reward, 0.36)
        self.assertAlmostEqual(evaluation_score.reward, 0.6)
        self.assertEqual(training_score.components["task_score"], 0.6)
        self.assertFalse(training_score.success)
        self.assertEqual(pool.rollout_id, "reward-test-rollout")


if __name__ == "__main__":
    unittest.main()
