"""Contracts for the strict-reward GRPO teacher continuation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from train.resume_grpo_teacher import (
    DEFAULT_CONFIG,
    PrecisionRewardConfig,
    PrecisionWebShopScorer,
    apply_remote_continuation,
    build_resume_scorer,
    load_experiment,
)
from train.resume_grpo_teacher_native import DEFAULT_CONFIG as NATIVE_CONFIG
from triotune.agentrl import AgentTrainingLoop, EpisodeScore
from triotune.agentrl.rollout.workflow import AgentRolloutBatch
from webshop_env.metrics import WebShopMetricsAgentTrainingLoop, purchase_metrics


class PrecisionResumeConfigurationTests(unittest.TestCase):
    def test_default_continuation_uses_the_remote_train_checkpoint(self):
        experiment = load_experiment(DEFAULT_CONFIG)

        self.assertEqual(experiment.resume_step, 200)
        self.assertEqual(
            experiment.checkpoint,
            "trio://run_n9e12csgxlof/weights/ppo-step-00000200",
        )
        self.assertEqual(experiment.trainer.max_steps, 400)
        self.assertEqual(experiment.trainer.batch_size, 8)
        self.assertEqual(experiment.trainer.num_generations, 8)
        self.assertTrue(experiment.trainer.checkpoint.sampler.enabled)
        self.assertEqual(experiment.trainer.checkpoint.save_every, 20)
        self.assertIsNone(experiment.trainer.checkpoint.resume_from)

    def test_step_override_counts_additional_updates_after_the_restored_step(self):
        experiment = load_experiment(DEFAULT_CONFIG, steps=20)
        self.assertEqual(experiment.resume_step, 200)
        self.assertEqual(experiment.trainer.max_steps, 220)
        with self.assertRaisesRegex(ValueError, "positive number of additional"):
            load_experiment(DEFAULT_CONFIG, steps=0)

    def test_remote_continuation_reconstructs_local_progress(self):
        experiment = load_experiment(DEFAULT_CONFIG)
        trainer = SimpleNamespace(
            state=SimpleNamespace(
                global_step=0,
                optimizer_steps=0,
                data_cursor=0,
                checkpoint_path=None,
            ),
            _actor=SimpleNamespace(optimizer_steps=0),
            _checkpoint=SimpleNamespace(saved_step=None),
        )
        continuation = apply_remote_continuation(
            experiment, trainer, [None] * 2000
        )

        self.assertEqual(trainer.state.global_step, 200)
        self.assertEqual(trainer.state.optimizer_steps, 200)
        self.assertEqual(trainer.state.data_cursor, 1600)
        self.assertEqual(trainer.state.checkpoint_path, experiment.checkpoint)
        self.assertEqual(trainer._actor.optimizer_steps, 200)
        self.assertEqual(continuation["restored_seed_offset"], 192000)
        self.assertFalse(continuation["exact_resume"])

    def test_native_control_keeps_the_original_grpo_reward(self):
        experiment = load_experiment(NATIVE_CONFIG)
        self.assertEqual(experiment.reward_mode, "native")
        self.assertIsNone(experiment.reward)
        self.assertEqual(experiment.resume_step, 200)
        self.assertEqual(experiment.trainer.max_steps, 400)
        self.assertEqual(
            experiment.checkpoint,
            "trio://run_n9e12csgxlof/weights/ppo-step-00000200",
        )
        scorer = build_resume_scorer(experiment, object())
        self.assertAlmostEqual(scorer.training_reward(0.6), 0.6)

    def test_reward_configuration_enforces_precision_ordering(self):
        reward = PrecisionRewardConfig()
        self.assertEqual(reward.training_reward(
            task_score=1.0, purchased=True, success=True, status="completed"
        ), 1.0)
        self.assertAlmostEqual(reward.training_reward(
            task_score=0.75, purchased=True, success=False, status="completed"
        ), -0.4)
        self.assertEqual(reward.training_reward(
            task_score=0.0, purchased=False, success=False, status="max_turns"
        ), -0.2)
        self.assertEqual(reward.training_reward(
            task_score=0.0, purchased=False, success=False, status="invalid_action"
        ), -0.4)
        # Buying exactly on the final budgeted turn is not protocol-complete.
        self.assertEqual(reward.training_reward(
            task_score=1.0, purchased=True, success=False, status="max_turns"
        ), -0.3)

        for values in (
            {"mismatch_penalty": -0.1},
            {"incorrect_purchase_base": -0.1},
            {"no_purchase_reward": 2.0},
            {"invalid_action_reward": 0.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PrecisionRewardConfig(**values)


class PrecisionRewardBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_is_strict_while_evaluation_keeps_native_score(self):
        class Pool:
            async def outcome(self, rollout_id):
                self.rollout_id = rollout_id
                return True, 0.75

        pool = Pool()
        scorer = PrecisionWebShopScorer(pool, PrecisionRewardConfig())
        with patch(
            "train.resume_grpo_teacher.current_rollout_id",
            return_value="precision-rollout",
        ):
            training = await scorer.score(
                {}, SimpleNamespace(status="completed", is_eval=False)
            )
            evaluation = await scorer.score(
                {}, SimpleNamespace(status="completed", is_eval=True)
            )

        self.assertAlmostEqual(training.reward, -0.4)
        self.assertAlmostEqual(evaluation.reward, 0.75)
        self.assertFalse(training.success)
        self.assertEqual(training.components["purchased"], 1.0)
        self.assertEqual(training.components["incorrect_purchase"], 1.0)
        self.assertEqual(pool.rollout_id, "precision-rollout")

    def test_agent_metrics_expose_purchase_coverage_and_precision(self):
        def episode(index, *, purchased, success):
            return SimpleNamespace(
                trainable=True,
                score=EpisodeScore(
                    float(success),
                    success,
                    {"purchased": float(purchased)},
                ),
                status="completed",
                turns=[],
                retries=[],
                compactions=[],
                group_id=f"group-{index}",
                task_id=f"task-{index}",
                elapsed_seconds=0.0,
            )

        episodes = [
            episode(0, purchased=True, success=True),
            episode(1, purchased=True, success=False),
            episode(2, purchased=False, success=False),
        ]
        generic = AgentRolloutBatch(episodes, []).metrics(prefix="eval/agent")
        self.assertNotIn("eval/agent/purchase_rate", generic)
        metrics = purchase_metrics(episodes, prefix="eval/agent")
        self.assertEqual(metrics["eval/agent/purchase_count"], 2.0)
        self.assertEqual(metrics["eval/agent/successful_purchase_count"], 1.0)
        self.assertAlmostEqual(metrics["eval/agent/purchase_rate"], 2 / 3)
        self.assertEqual(metrics["eval/agent/purchase_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["eval/agent/incorrect_purchase_rate"], 1 / 3)

    async def test_external_loop_publishes_each_training_and_evaluation_batch(self):
        episodes = [
            SimpleNamespace(
                trainable=True,
                score=EpisodeScore(1.0, True, {"purchased": 1.0}),
            ),
            SimpleNamespace(
                trainable=True,
                score=EpisodeScore(0.0, False, {"purchased": 1.0}),
            ),
        ]
        publish = AsyncMock(side_effect=lambda values: values)
        loop = object.__new__(WebShopMetricsAgentTrainingLoop)
        loop.trainer = SimpleNamespace(publish_metrics=publish)
        loop.last_eval_batch = SimpleNamespace(episodes=episodes)
        batch = SimpleNamespace(
            episodes=episodes,
            metrics=lambda *, prefix="agent": {f"{prefix}/base": 1.0},
        )

        async def update_batch(wrapped, snapshot, progress, **kwargs):
            await publish(wrapped.metrics())

        with patch.object(
            AgentTrainingLoop,
            "_update",
            new=AsyncMock(side_effect=update_batch),
        ) as update:
            await loop._update(
                batch,
                object(),
                object(),
            )
        update.assert_awaited_once()
        publish.assert_awaited_with(
            {
                "agent/base": 1.0,
                "agent/purchase_count": 2.0,
                "agent/successful_purchase_count": 1.0,
                "agent/purchase_rate": 1.0,
                "agent/purchase_success_rate": 0.5,
                "agent/incorrect_purchase_rate": 0.5,
            }
        )

        publish.reset_mock()
        with patch.object(
            AgentTrainingLoop,
            "evaluate",
            new=AsyncMock(return_value={"eval/agent/pass@1": 0.5}),
        ):
            metrics = await loop.evaluate()
        self.assertEqual(metrics["eval/agent/purchase_success_rate"], 0.5)
        publish.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
