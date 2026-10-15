"""Trainer configuration classes share one public contract layer."""

from __future__ import annotations

import unittest

from trio_cookbook.trainer import (
    BaseTrainerConfig,
    DPOConfig,
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


if __name__ == "__main__":
    unittest.main()
