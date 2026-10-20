"""Trainer configuration classes share one public contract layer."""

from __future__ import annotations

import tomllib
import unittest
from dataclasses import asdict

from triotune.api import NormConfig as APINormConfig
from triotune.api import PPOConfig as APIPPOConfig
from triotune.trainer import (
    BaseTrainerConfig,
    CheckpointConfig,
    DatasetConfig,
    DPOConfig,
    EvaluationConfig,
    ModelConfig,
    OPDConfig,
    OptimizerConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    PPORolloutConfig,
    ReportingConfig,
    ReferenceConfig,
    SFTConfig,
    TeacherConfig,
    TrainingConfig,
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
        from triotune.trainer.ppo import NormConfig, PPOConfig

        self.assertIs(PPOConfig, APIPPOConfig)
        self.assertIs(NormConfig, APINormConfig)
        self.assertIsInstance(PPOConfig(), BaseTrainerConfig)
        self.assertIs(
            __import__("triotune.trainer.ppo.config", fromlist=["PPOConfig"]),
            __import__("triotune.api.cli_args", fromlist=["PPOConfig"]),
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

    def test_flat_eval_settings_serialize_as_a_functional_module(self) -> None:
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
        self.assertEqual(
            serialized["evaluation"],
            {
                "steps": 50,
                "before_train": False,
                "after_train": False,
                "batch_size": 2,
                "max_batches": 20,
            },
        )
        self.assertEqual(SFTConfig(**serialized), config)

    def test_ppo_composes_functional_modules(self) -> None:
        config = PPOConfig(
            training=TrainingConfig(batch_size=4, max_steps=10, seed=7),
            optimizer=OptimizerConfig(learning_rate=4e-5),
            checkpoint=CheckpointConfig(save_every=5),
            evaluation=EvaluationConfig(after_train=False),
            reporting=ReportingConfig(project="modular-config"),
            rollout=PPORolloutConfig(num_generations=8, max_tokens=256),
            actor=PPOActorConfig(eps_clip=0.1),
            critic=PPOCriticConfig(
                optimizer=OptimizerConfig(learning_rate=2e-5, beta2=0.99)
            ),
        )

        self.assertEqual(config.training.batch_size, 4)
        self.assertEqual(config.rollout.num_generations, 8)
        self.assertEqual(config.actor.eps_clip, 0.1)
        self.assertEqual(config.optimizer.learning_rate, 4e-5)
        self.assertEqual(config.critic.optimizer.learning_rate, 2e-5)
        self.assertEqual(config.critic.optimizer.beta2, 0.99)

    def test_flat_compatibility_routes_to_modules(self) -> None:
        config = PPOConfig(
            batch_size=3,
            learning_rate=3e-5,
            num_generations=6,
            critic_learning_rate=1e-5,
        )

        self.assertEqual(config.training.batch_size, 3)
        self.assertEqual(config.optimizer.learning_rate, 3e-5)
        self.assertEqual(config.rollout.num_generations, 6)
        self.assertEqual(config.critic.optimizer.learning_rate, 1e-5)
        config.eps_clip = 0.15
        self.assertEqual(config.actor.eps_clip, 0.15)

        inherited = PPOConfig(learning_rate=2e-4)
        self.assertEqual(inherited.critic.optimizer.learning_rate, 2e-4)

    def test_algorithm_yaml_adapters_build_functional_modules(self) -> None:
        dpo = DPOConfig.from_modules(
            {
                "training": {"batch_size": 2, "max_steps": 3},
                "optimizer": {"lr": 2e-5},
                "data": {"max_length": 256},
                "objective": {"beta": 0.2},
            }
        )
        sft = SFTConfig.from_modules(
            {
                "training": {"batch_size": 4},
                "data": {"max_length": 512},
                "execution": {"max_in_flight": 8},
            }
        )
        opd = OPDConfig.from_modules(
            {
                "rollout": {"max_completion_length": 128},
                "execution": {"max_concurrent_scoring": 4},
                "objective": {"kl_coef": 0.5},
            }
        )

        self.assertEqual(dpo.training.max_steps, 3)
        self.assertEqual(dpo.optimizer.learning_rate, 2e-5)
        self.assertEqual(dpo.objective.beta, 0.2)
        self.assertEqual(sft.data.max_length, 512)
        self.assertEqual(sft.execution.max_in_flight, 8)
        self.assertEqual(opd.rollout.max_completion_length, 128)
        self.assertEqual(opd.execution.max_concurrent_scoring, 4)
        self.assertEqual(opd.objective.kl_coef, 0.5)

    def test_algorithm_configs_can_describe_complete_experiments(self) -> None:
        model = ModelConfig(name="student", lora_rank=16)
        train_data = DatasetConfig(path="data/train.jsonl", max_examples=32)
        valid_data = DatasetConfig(path="data/valid.jsonl", split="test")

        ppo = PPOConfig(
            model=model,
            train_dataset=train_data,
            valid_dataset=valid_data,
            reference=ReferenceConfig(name="reference"),
        )
        dpo = DPOConfig(
            model=model,
            train_dataset=train_data,
            valid_dataset=valid_data,
            reference={"name": "reference"},
        )
        sft = SFTConfig(
            model=model,
            train_dataset=train_data,
            valid_dataset=valid_data,
        )
        opd = OPDConfig(
            model=model,
            train_dataset=train_data,
            valid_dataset=valid_data,
            teacher=TeacherConfig(name="teacher", model_path="trio://teacher"),
        )

        for config in (ppo, dpo, sft, opd):
            with self.subTest(config=type(config).__name__):
                self.assertEqual(config.model.name, "student")
                self.assertEqual(config.train_dataset.max_examples, 32)
                self.assertEqual(config.valid_dataset.split, "test")
        self.assertEqual(ppo.reference.name, "reference")
        self.assertEqual(dpo.reference.name, "reference")
        self.assertEqual(opd.teacher.model_path, "trio://teacher")

if __name__ == "__main__":
    unittest.main()
