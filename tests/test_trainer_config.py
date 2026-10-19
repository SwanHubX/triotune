"""Trainer configuration classes share one public contract layer."""

from __future__ import annotations

import tomllib
import unittest
from dataclasses import asdict

from trio_cookbook.api import NormConfig as APINormConfig
from trio_cookbook.api import PPOConfig as APIPPOConfig
from trio_cookbook.trainer import (
    BaseTrainerConfig,
    DPOConfig,
    OPDConfig,
    PPOConfig,
    SFTConfig,
)


class TrainerConfigTests(unittest.TestCase):
    def test_algorithm_configs_inherit_common_controls_and_keep_defaults(self) -> None:
        expected = (
            (PPOConfig, 1, 1e-5, "trio-cookbook"),
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
        for config_class in (PPOConfig, DPOConfig, SFTConfig, OPDConfig):
            with self.subTest(config=config_class.__name__, field="batch_size"):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    config_class(batch_size=0)
            with self.subTest(config=config_class.__name__, field="workspace"):
                with self.assertRaisesRegex(ValueError, "workspace"):
                    config_class(workspace=" ")
            with self.subTest(config=config_class.__name__, field="tags"):
                with self.assertRaisesRegex(ValueError, "tags"):
                    config_class(tags=[""])

    def test_ppo_config_is_defined_by_the_public_api_contract(self) -> None:
        from trio_cookbook.trainer.ppo import NormConfig, PPOConfig

        self.assertIs(PPOConfig, APIPPOConfig)
        self.assertIs(NormConfig, APINormConfig)
        self.assertIsInstance(PPOConfig(), BaseTrainerConfig)
        self.assertIs(
            __import__("trio_cookbook.trainer.ppo.config", fromlist=["PPOConfig"]),
            __import__("trio_cookbook.api.cli_args", fromlist=["PPOConfig"]),
        )
        self.assertFalse(PPOConfig.for_algorithm("grpo").use_critic)
        self.assertTrue(PPOConfig.for_algorithm("ppo").use_critic)

    def test_optimizer_contract_exposes_decay_and_gradient_clipping(self) -> None:
        config = DPOConfig(weight_decay=0.01, grad_clip_norm=1.0)

        self.assertEqual(config.optimizer_params()["weight_decay"], 0.01)
        self.assertEqual(config.optimizer_params()["grad_clip_norm"], 1.0)
        for field in ("weight_decay", "grad_clip_norm"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                DPOConfig(**{field: -1.0})

    def test_evaluation_schedule_rejects_invalid_limits(self) -> None:
        for config_class in (PPOConfig, DPOConfig, SFTConfig, OPDConfig):
            for kwargs in (
                {"eval_steps": -1},
                {"eval_steps": True},
                {"eval_steps": 1.5},
                {"eval_batch_size": 0},
                {"eval_batch_size": True},
                {"eval_max_batches": 0},
                {"eval_before_train": "yes"},
                {"eval_after_train": 1},
            ):
                with self.subTest(config=config_class.__name__, kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        config_class(**kwargs)

    def test_flat_eval_settings_roundtrip_through_toml_and_config(self) -> None:
        settings = tomllib.loads(
            '[sft]\neval_steps = 50\neval_batch_size = 2\n'
            'eval_max_batches = 20\neval_after_train = false\n'
        )
        config = SFTConfig(**settings["sft"])

        self.assertEqual(config.eval_steps, 50)
        self.assertEqual(config.eval_batch_size, 2)
        self.assertEqual(config.eval_max_batches, 20)
        self.assertFalse(config.eval_before_train)
        self.assertFalse(config.eval_after_train)
        serialized = asdict(config)
        self.assertNotIn("evaluation", serialized)
        self.assertEqual(SFTConfig(**serialized), config)

if __name__ == "__main__":
    unittest.main()
