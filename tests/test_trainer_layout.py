"""Compatibility checks for reorganized trainer packages."""

import importlib
import pickle
import unittest
from unittest.mock import patch

import triotune.trainer as trainers


class TrainerLayoutTests(unittest.TestCase):
    def test_legacy_imports_are_the_actual_modules(self):
        aliases = {
            "dpo_trainer": "triotune.trainer.dpo_trainer",
            "dpo_config": "triotune.api.cli_args",
            "sft_trainer": "triotune.trainer.sft_trainer",
            "sft_config": "triotune.api.cli_args",
            "opd_trainer": "triotune.trainer.opd_trainer",
            "opd_config": "triotune.api.cli_args",
            "opd_distillation": "triotune.trainer.opd",
            "prompts": "triotune.dataset.prompt",
        }
        for old, new in aliases.items():
            with self.subTest(module=old):
                legacy = importlib.import_module(f"triotune.trainer.{old}")
                actual = importlib.import_module(new)
                self.assertIs(legacy, actual)
                self.assertIs(getattr(trainers, old), actual)

    def test_algorithm_packages_export_the_same_public_objects(self):
        for algorithm in ("dpo", "sft", "opd"):
            package = importlib.import_module(f"triotune.trainer.{algorithm}")
            for name in package.__all__:
                with self.subTest(algorithm=algorithm, name=name):
                    self.assertIs(getattr(package, name), getattr(trainers, name))

    def test_algorithm_config_paths_share_the_api_contract_module(self):
        cli_args = importlib.import_module("triotune.api.cli_args")
        for algorithm in ("dpo", "sft", "opd"):
            with self.subTest(algorithm=algorithm):
                module = importlib.import_module(
                    f"triotune.trainer.{algorithm}.config"
                )
                self.assertIs(module, cli_args)

    def test_old_nested_packages_alias_new_sibling_packages(self):
        aliases = {
            "triotune.trainer.common": "triotune.api",
            "triotune.trainer.common.config": "triotune.api.cli_args",
            "triotune.trainer.common.client": "triotune.api.client_api",
            "triotune.trainer.common.checkpoint": "triotune.utils.checkpoint",
            "triotune.trainer.common.metrics_api": "triotune.utils.metrics",
            "triotune.trainer.common.logging": "triotune.utils.metrics",
            "triotune.trainer.dataset": "triotune.dataset",
            "triotune.trainer.dataset.swe_sft": "triotune.dataset.swe_sft",
        }
        for old, new in aliases.items():
            with self.subTest(module=old):
                self.assertIs(
                    importlib.import_module(old),
                    importlib.import_module(new),
                )

    def test_algorithm_trainer_paths_alias_root_modules(self):
        for algorithm in ("dpo", "sft", "opd"):
            with self.subTest(algorithm=algorithm):
                root = importlib.import_module(
                    f"triotune.trainer.{algorithm}_trainer"
                )
                nested = importlib.import_module(
                    f"triotune.trainer.{algorithm}.trainer"
                )
                self.assertIs(nested, root)

    def test_renamed_algorithm_modules_keep_legacy_imports(self):
        aliases = {
            "dpo.types": "dpo.state",
            "sft.types": "sft.state",
            "opd.types": "opd.contracts",
        }
        for old, new in aliases.items():
            with self.subTest(module=old):
                legacy = importlib.import_module(f"triotune.trainer.{old}")
                canonical = importlib.import_module(f"triotune.trainer.{new}")
                self.assertIs(legacy, canonical)

    def test_patching_a_legacy_module_updates_the_actual_module(self):
        actual = importlib.import_module("triotune.trainer.dpo_trainer")
        with patch("triotune.trainer.dpo_trainer.tqdm") as progress:
            self.assertIs(actual.tqdm, progress)

    def test_state_pickle_roundtrips_and_historical_globals_resolve(self):
        for name in ("DPOState", "SFTState", "OPDState"):
            state = getattr(trainers, name)(global_step=3, checkpoint_path="trio://saved")
            restored = pickle.loads(pickle.dumps(state))
            self.assertEqual(restored, state)
            self.assertIs(type(restored), type(state))
        # Protocol 0 GLOBAL records use module paths embedded in older pickles.
        for module, name in (
            ("dpo_trainer", "DPOState"),
            ("sft_trainer", "SFTState"),
            ("opd_trainer", "OPDState"),
        ):
            record = f"ctriotune.trainer.{module}\n{name}\n."
            self.assertIs(pickle.loads(record.encode()), getattr(trainers, name))
