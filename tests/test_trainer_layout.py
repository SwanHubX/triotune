"""Compatibility checks for reorganized trainer packages."""

import importlib
import pickle
import unittest
from unittest.mock import patch

import trio_cookbook.trainer as trainers


class TrainerLayoutTests(unittest.TestCase):
    def test_legacy_imports_are_the_actual_modules(self):
        aliases = {
            "grpo_trainer": "grpo.trainer",
            "grpo_config": "grpo.config",
            "dpo_trainer": "dpo.trainer",
            "dpo_config": "dpo.config",
            "sft_trainer": "sft.trainer",
            "sft_config": "sft.config",
            "opd_trainer": "opd.trainer",
            "opd_config": "opd.config",
            "opd_distillation": "opd.distillation",
            "advantages": "grpo.advantages",
            "datum": "grpo.datum",
            "rewards": "grpo.rewards",
            "rollout": "grpo.rollout",
            "types": "grpo.types",
            "prompts": "dataset.prompt",
        }
        for old, new in aliases.items():
            with self.subTest(module=old):
                legacy = importlib.import_module(f"trio_cookbook.trainer.{old}")
                actual = importlib.import_module(f"trio_cookbook.trainer.{new}")
                self.assertIs(legacy, actual)
                self.assertIs(getattr(trainers, old), actual)

    def test_algorithm_packages_export_the_same_public_objects(self):
        for algorithm in ("grpo", "dpo", "sft", "opd"):
            package = importlib.import_module(f"trio_cookbook.trainer.{algorithm}")
            for name in package.__all__:
                with self.subTest(algorithm=algorithm, name=name):
                    self.assertIs(getattr(package, name), getattr(trainers, name))

    def test_patching_a_legacy_module_updates_the_actual_module(self):
        actual = importlib.import_module("trio_cookbook.trainer.dpo.trainer")
        with patch("trio_cookbook.trainer.dpo_trainer.tqdm") as progress:
            self.assertIs(actual.tqdm, progress)

    def test_state_pickle_roundtrips_and_historical_globals_resolve(self):
        for name in ("TrainerState", "DPOState", "SFTState", "OPDState"):
            state = getattr(trainers, name)(global_step=3, checkpoint_path="trio://saved")
            restored = pickle.loads(pickle.dumps(state))
            self.assertEqual(restored, state)
            self.assertIs(type(restored), type(state))
        # Protocol 0 GLOBAL records use module paths embedded in older pickles.
        for module, name in (
            ("grpo_trainer", "TrainerState"),
            ("dpo_trainer", "DPOState"),
            ("sft_trainer", "SFTState"),
            ("opd_trainer", "OPDState"),
        ):
            record = f"ctrio_cookbook.trainer.{module}\n{name}\n."
            self.assertIs(pickle.loads(record.encode()), getattr(trainers, name))
