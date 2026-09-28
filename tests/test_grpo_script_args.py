from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from datasets import Dataset


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "trio-cookbook" / "recipes" / "grpo_gms8k_trio.py"
    spec = importlib.util.spec_from_file_location("grpo_gms8k_trio_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


class ScriptArgumentsTests(unittest.TestCase):
    def test_prepared_gsm8k_rows_use_prompt_column(self) -> None:
        source = Dataset.from_list([{"question": "6 * 7?", "answer": "work #### 42"}])
        with TemporaryDirectory() as directory:
            data_file = Path(directory) / "train.jsonl"
            with patch.object(script, "load_dataset", return_value=source):
                script.prepare_gsm8k_dataset(data_file, Path(directory), rebuild=False)

            raw_row = json.loads(data_file.read_text(encoding="utf-8"))
            self.assertIn("prompt", raw_row)
            self.assertNotIn("messages", raw_row)
            self.assertEqual(script.load_training_rows(data_file, None), [raw_row])

            data_file.write_text(json.dumps({"prompt": [1, 2], "answer": "42"}))
            self.assertEqual(script.load_training_rows(data_file, None)[0]["prompt"], [1, 2])

            data_file.write_text(json.dumps({"messages": raw_row["prompt"], "answer": "42"}))
            with self.assertRaisesRegex(ValueError, "Invalid trainer row"):
                script.load_training_rows(data_file, None)

    def test_default_toml_contains_model_data_and_grpo_settings(self) -> None:
        settings = script.load_config(script.DEFAULT_CONFIG)

        self.assertEqual(settings["model"]["name"], "Qwen/Qwen3.5-9B")
        self.assertEqual(settings["data"]["file"], "data/gsm8k_train.jsonl")
        self.assertEqual(settings["grpo"]["max_steps"], 100)
        self.assertEqual(settings["grpo"]["chat_template_kwargs"], {"enable_thinking": False})

    def test_swanlab_options_default_to_none(self) -> None:
        with patch.object(sys, "argv", ["grpo_gms8k_trio.py"]):
            args = script.parse_args()

        self.assertIsNone(args.workspace)
        self.assertIsNone(args.tags)

    def test_accepts_workspace_and_repeated_tags(self) -> None:
        argv = [
            "grpo_gms8k_trio.py", "--workspace", "my-organization",
            "--tag", "grpo", "--tags", "gsm8k",
        ]
        with patch.object(sys, "argv", argv):
            args = script.parse_args()

        self.assertEqual(args.workspace, "my-organization")
        self.assertEqual(args.tags, ["grpo", "gsm8k"])

    def test_passes_swanlab_options_to_trainer_config(self) -> None:
        argv = [
            "grpo_gms8k_trio.py", "--mode", "online", "--workspace", "my-organization",
            "--tag", "grpo", "--tag", "gsm8k",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(script, "prepare_gsm8k_dataset"),
            patch.object(
                script, "load_training_rows",
                return_value=[{"prompt": [{"role": "user", "content": "6 * 7?"}], "answer": "42"}],
            ),
            patch.object(script, "GRPOTrainer") as trainer_class,
            patch.object(script, "print", create=True),
        ):
            script.main()

        config = trainer_class.call_args.kwargs["args"]
        self.assertEqual(config.workspace, "my-organization")
        self.assertEqual(config.tags, ["grpo", "gsm8k"])

    def test_cli_overrides_training_settings_and_final_save_interval(self) -> None:
        argv = [
            "grpo_gms8k_trio.py", "--steps", "2", "--batch-size", "1", "--group-size", "4",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(script, "prepare_gsm8k_dataset"),
            patch.object(script, "load_training_rows", return_value=[{"prompt": [1], "answer": "42"}]),
            patch.object(script, "GRPOTrainer") as trainer_class,
            patch.object(script, "print", create=True),
        ):
            script.main()

        config = trainer_class.call_args.kwargs["args"]
        self.assertEqual((config.max_steps, config.batch_size, config.num_generations), (2, 1, 4))
        self.assertEqual(config.save_every, 2)
        self.assertEqual(trainer_class.call_args.kwargs["model_name"], "Qwen/Qwen3.5-9B")

    def test_rejects_missing_config_section(self) -> None:
        settings = {
            "model": {"name": "model", "lora_rank": 32},
            "data": {"file": "train.jsonl", "cache_dir": ".cache"},
        }
        with patch.object(script.tomllib, "load", return_value=settings):
            with self.assertRaisesRegex(ValueError, r"\[grpo\]"):
                script.load_config(script.DEFAULT_CONFIG)

    def test_rejects_missing_config_field(self) -> None:
        with patch.object(script.tomllib, "load", return_value={"model": {}}):
            with self.assertRaisesRegex(ValueError, r"\[model\]\.name"):
                script.load_config(script.DEFAULT_CONFIG)

    def test_relative_data_paths_resolve_from_repository_root(self) -> None:
        expected = Path(__file__).parents[1] / "data" / "example.jsonl"
        self.assertEqual(script.resolve_path("data/example.jsonl"), expected.resolve())

    def test_rejects_invalid_configured_example_limit_before_data_preparation(self) -> None:
        settings = script.load_config(script.DEFAULT_CONFIG)
        settings["data"]["max_examples"] = 0
        with (
            patch.object(sys, "argv", ["grpo_gms8k_trio.py"]),
            patch.object(script, "load_config", return_value=settings),
            patch.object(script, "prepare_gsm8k_dataset") as prepare,
        ):
            with self.assertRaisesRegex(ValueError, "data.max_examples"):
                script.main()

        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
