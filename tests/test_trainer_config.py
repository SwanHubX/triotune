"""Trainer configuration classes share one public contract layer."""

from __future__ import annotations

import unittest

from trio_cookbook.trainer import (
    BaseTrainerConfig,
    DPOConfig,
    EvaluationConfig,
    GRPOConfig,
    OPDConfig,
    SFTConfig,
)


class TrainerConfigTests(unittest.TestCase):
    def test_algorithm_configs_inherit_common_controls_and_keep_defaults(self) -> None:
        expected = (
            (GRPOConfig, 1, 4e-5, "trio-cookbook-grpo"),
            (DPOConfig, 1, 5e-6, "trio-cookbook-dpo"),
            (SFTConfig, 1, 2e-5, "trio-cookbook-sft"),
            (OPDConfig, 4, 1e-5, "trio-cookbook-opd"),
        )
        for config_class, batch_size, learning_rate, project in expected:
            with self.subTest(config=config_class.__name__):
                config = config_class()
                self.assertIsInstance(config, BaseTrainerConfig)
                self.assertEqual(config.batch_size, batch_size)
                self.assertEqual(config.learning_rate, learning_rate)
                self.assertEqual(config.project, project)
                self.assertEqual(
                    config.optimizer_params(),
                    {
                        "learning_rate": learning_rate,
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 0.0,
                    },
                )

    def test_common_controls_have_one_validation_contract(self) -> None:
        for config_class in (GRPOConfig, DPOConfig, SFTConfig, OPDConfig):
            with self.subTest(config=config_class.__name__, field="batch_size"):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    config_class(batch_size=0)
            with self.subTest(config=config_class.__name__, field="workspace"):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    config_class(workspace=" ")
            with self.subTest(config=config_class.__name__, field="tags"):
                with self.assertRaisesRegex(ValueError, "tags"):
                    config_class(tags=[""])

    def test_optimizer_contract_exposes_decay_and_gradient_clipping(self) -> None:
        config = DPOConfig(weight_decay=0.01, grad_clip_norm=1.0)

        self.assertEqual(config.optimizer_params()["weight_decay"], 0.01)
        self.assertEqual(config.optimizer_params()["grad_clip_norm"], 1.0)
        for field in ("weight_decay", "grad_clip_norm"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                DPOConfig(**{field: -1.0})

    def test_evaluation_schedule_rejects_invalid_limits(self) -> None:
        for kwargs in (
            {"every_steps": -1},
            {"every_steps": True},
            {"batch_size": 0},
            {"max_batches": 0},
            {"before_train": "yes"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationConfig(**kwargs)

    def test_algorithm_config_accepts_nested_evaluation_mapping(self) -> None:
        config = SFTConfig(
            evaluation={"before_train": True, "every_steps": 5, "max_batches": 2}
        )

        self.assertIsInstance(config.evaluation, EvaluationConfig)
        self.assertTrue(config.evaluation.before_train)
        self.assertEqual(config.evaluation.every_steps, 5)
        self.assertEqual(config.evaluation.max_batches, 2)



if __name__ == "__main__":
    unittest.main()
