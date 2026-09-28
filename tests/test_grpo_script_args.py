from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "trio-cookbook" / "scripts" / "grpo_gms8k_trio.py"
    spec = importlib.util.spec_from_file_location("grpo_gms8k_trio_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


class ScriptArgumentsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
