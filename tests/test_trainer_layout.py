"""Compatibility checks for reorganized trainer packages."""

import importlib
import pickle
import unittest
from unittest.mock import patch

import trio_cookbook.trainer as trainers


class TrainerLayoutTests(unittest.TestCase):
    def test_legacy_imports_are_the_actual_modules(self):
        aliases = {
            "grpo_trainer": "grpo_trainer",
            "grpo_config": "common.config",
            "dpo_trainer": "dpo_trainer",
            "dpo_config": "common.config",
            "sft_trainer": "sft_trainer",
            "sft_config": "common.config",
            "opd_trainer": "opd_trainer",
            "opd_config": "common.config",
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

    def test_algorithm_config_paths_share_the_common_contract_module(self):
        common = importlib.import_module("trio_cookbook.trainer.common.config")
        for algorithm in ("grpo", "dpo", "sft", "opd"):
            with self.subTest(algorithm=algorithm):
                module = importlib.import_module(
                    f"trio_cookbook.trainer.{algorithm}.config"
                )
                self.assertIs(module, common)

    def test_algorithm_trainer_paths_alias_root_modules(self):
        for algorithm in ("grpo", "dpo", "sft", "opd"):
            with self.subTest(algorithm=algorithm):
                root = importlib.import_module(
                    f"trio_cookbook.trainer.{algorithm}_trainer"
                )
                nested = importlib.import_module(
                    f"trio_cookbook.trainer.{algorithm}.trainer"
                )
                self.assertIs(nested, root)

    def test_patching_a_legacy_module_updates_the_actual_module(self):
        actual = importlib.import_module("trio_cookbook.trainer.dpo_trainer")
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
