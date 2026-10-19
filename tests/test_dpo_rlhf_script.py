from __future__ import annotations

import gzip
import importlib.util
import itertools
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "triotune" / "recipes" / "dpo_rlhf_trio.py"
    spec = importlib.util.spec_from_file_location("dpo_rlhf_trio_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()
PROMPT = "\n\nHuman: Hello\n\nAssistant: Hi\n\nHuman: 2+2?"


class HHRecipeTests(unittest.TestCase):
    def test_extracts_shared_prompt_and_distinct_final_replies(self) -> None:
        pair = script.preference_row({
            "chosen": PROMPT + "\n\nAssistant: Four",
            "rejected": PROMPT + "\n\nAssistant: Five",
        })
        self.assertEqual(pair["messages"][-1], {"role": "user", "content": "2+2?"})
        self.assertEqual(pair["chosen"], [{"role": "assistant", "content": "Four"}])
        self.assertEqual(pair["rejected"], [{"role": "assistant", "content": "Five"}])
        self.assertIsNone(script.preference_row({
            "chosen": PROMPT + "\n\nAssistant: Four",
            "rejected": "\n\nHuman: Different\n\nAssistant: Five",
        }))

    def test_prepares_cached_modelscope_gzip_without_network(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            raw_file = base / "cache" / "helpful-base" / "train.jsonl.gz"
            raw_file.parent.mkdir(parents=True)
            with gzip.open(raw_file, "wt", encoding="utf-8") as output:
                for chosen, rejected in (("Four", "Five"), ("Yes", "No")):
                    output.write(json.dumps({
                        "chosen": PROMPT + "\n\nAssistant: " + chosen,
                        "rejected": PROMPT + "\n\nAssistant: " + rejected,
                    }) + "\n")
            data_file = base / "data" / "train.jsonl"
            settings = {"subsets": ["helpful-base"], "sample_size": 1, "seed": 42}
            with patch.object(script.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                script.prepare_dataset(data_file, base / "cache", settings, rebuild=False, force=False)
            rows = script.load_rows(data_file)
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {"messages", "chosen", "rejected"})

    def test_rejects_unmatched_or_empty_data(self) -> None:
        self.assertIsNone(script.preference_row({
            "chosen": PROMPT + "\n\nAssistant: Same",
            "rejected": PROMPT + "\n\nAssistant: Same",
        }))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid or empty"):
                script.load_rows(path)

    def test_default_config_and_cli_overrides(self) -> None:
        settings = script.load_config(script.DEFAULT_CONFIG)
        self.assertEqual(settings["data"]["cache_dir"], ".cache/modelscope/hh-rlhf")
        self.assertEqual(settings["dpo"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertGreaterEqual(settings["swanlab"]["logging_step"], 1)
        self.assertEqual(script.parse_args(["--steps", "2"]).steps, 2)
        self.assertIsNone(script.parse_args([]).report_to)

    def test_training_entry_passes_config_and_reuses_pairs(self) -> None:
        row = {"messages": [{"role": "user", "content": "P"}],
               "chosen": [{"role": "assistant", "content": "A"}],
               "rejected": [{"role": "assistant", "content": "B"}]}
        with (
            patch.object(script, "prepare_dataset"),
            patch.object(script, "load_rows", return_value=[row]),
            patch.object(script, "DPOTrainer") as trainer_class,
        ):
            trainer_class.return_value._preferences.return_value = [object()]
            script.main(["--steps", "2", "--batch-size", "1"])

        config = trainer_class.call_args.kwargs["args"]
        self.assertEqual((config.max_steps, config.batch_size), (2, 1))
        self.assertEqual(config.chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual(config.report_to, "swanlab")
        self.assertEqual(config.logging_step, script.load_config(script.DEFAULT_CONFIG)["swanlab"]["logging_step"])
        self.assertEqual(list(itertools.islice(trainer_class.return_value.train_dataset, 2)), [row, row])
        trainer_class.return_value.train.assert_called_once_with()

    def test_invalid_report_target_prints_error(self) -> None:
        for target in ("none", "other"):
            error = io.StringIO()
            with self.subTest(target=target), redirect_stderr(error), self.assertRaises(SystemExit) as exit_info:
                script.parse_args(["--report-to", target])
            self.assertEqual(exit_info.exception.code, 2)
            self.assertIn(f"invalid choice: '{target}'", error.getvalue())


if __name__ == "__main__":
    unittest.main()
