"""Exercise the DPO/OPD/SFT tutorial data and entry points without PyTRIO calls."""

import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from datasets import Dataset

from examples.dpo import gsm8k_dpo
from examples.opd import gsm8k_opd
from examples.sft import gsm8k_sft
from triotune.trainer import DPOConfig, OPDConfig, SFTConfig, DatasetConfig
from triotune.utils.config import load_yaml_config


CASES = (
    (gsm8k_dpo, "dpo", "gsm8k", DPOConfig, "DPOTrainer", "train_async"),
    (gsm8k_opd, "opd", "gsm8k", OPDConfig, "OPDTrainer", "train_async"),
    (gsm8k_sft, "sft", "gsm8k", SFTConfig, "SFTTrainer", "train"),
)
ROOT = gsm8k_dpo.ROOT


class TrainingExampleTests(unittest.TestCase):
    def test_configs_bind_the_intended_algorithm(self):
        for _, algorithm, task, config_type, _, _ in CASES:
            with self.subTest(algorithm=algorithm):
                config = config_type(**load_yaml_config(
                    ROOT / f"examples/{algorithm}/{task}_{algorithm}_args.yaml"
                ))
                self.assertEqual(config.model.name, "Qwen/Qwen3.5-4B")
                self.assertEqual(config.training.max_steps, 100)
                self.assertIsNotNone(config.valid_dataset)
                if algorithm == "dpo":
                    self.assertEqual(config.reference.name, config.model.name)
                    self.assertEqual(config.objective.beta, 0.1)
                elif algorithm == "opd":
                    self.assertEqual(config.teacher.name, "Qwen/Qwen3.8-27B")
                    self.assertEqual(config.objective.importance_ratio_cap, 5.0)
                else:
                    self.assertTrue(config.data.completion_only_loss)
                    self.assertFalse(config.data.assistant_only_loss)
                    self.assertEqual(config.execution.max_in_flight, 32)

    def test_jsonl_loader_applies_shuffle_before_limit_and_honors_paths(self):
        rows = [{"text": f"row {i}"} for i in range(8)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for example, algorithm, *_ in CASES:
                for dataset_seed in (None, 7):
                    with self.subTest(algorithm=algorithm, dataset_seed=dataset_seed):
                        source = DatasetConfig(
                            path="rows.jsonl", split="validation", max_examples=3,
                            shuffle=True, seed=dataset_seed,
                        )
                        with patch.object(example, "ROOT", root):
                            loaded = example.load_data(source, 42)
                        expected = Dataset.from_list(rows).shuffle(
                            seed=42 if dataset_seed is None else dataset_seed
                        ).select(range(3))
                        self.assertEqual(loaded.to_list(), expected.to_list())
                with self.subTest(algorithm=algorithm, absolute_path=True):
                    loaded = example.load_data(DatasetConfig(path=str(path), max_examples=99), 42)
                    self.assertEqual(loaded.to_list(), rows)

    def test_missing_and_empty_sources_fail_before_training(self):
        for example, algorithm, _, _, trainer_name, _ in CASES:
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, "path is required"):
                    example.load_data(DatasetConfig(), 42)
                with patch.object(example, "load_dataset", return_value=Dataset.from_list([])):
                    with self.assertRaisesRegex(ValueError, "at least one row"):
                        example.load_data(DatasetConfig(path="empty.jsonl"), 42)
                settings = {"model": {"name": "test-model"}}
                with (
                    patch("sys.argv", ["example", "--config", "unused.yaml"]),
                    patch.object(example, "load_yaml_config", return_value=settings),
                    patch.object(example, trainer_name) as trainer,
                ):
                    with self.assertRaisesRegex(ValueError, "train_dataset is required"):
                        example.main()
                trainer.assert_not_called()
                settings["train_dataset"] = {"path": "missing.jsonl"}
                with (
                    patch("sys.argv", ["example", "--config", "unused.yaml"]),
                    patch.object(example, "load_yaml_config", return_value=settings),
                    patch.object(example, "load_dataset", side_effect=FileNotFoundError("missing JSONL")),
                    patch.object(example, trainer_name) as trainer,
                ):
                    with self.assertRaisesRegex(FileNotFoundError, "missing JSONL"):
                        example.main()
                trainer.assert_not_called()

    def test_main_connects_loaded_rows_to_the_correct_async_method(self):
        for example, algorithm, task, _, trainer_name, method in CASES:
            for validation in (True, False):
                with self.subTest(algorithm=algorithm, validation=validation):
                    settings = load_yaml_config(
                        ROOT / f"examples/{algorithm}/{task}_{algorithm}_args.yaml"
                    )
                    settings["train_dataset"].update(shuffle=False, max_examples=2)
                    settings["valid_dataset"]["max_examples"] = 1
                    if not validation:
                        settings["valid_dataset"] = None
                    rows = Dataset.from_list([{"text": str(i)} for i in range(3)])
                    state = SimpleNamespace(global_step=1, checkpoint_path="trio://saved")
                    with (
                        patch("sys.argv", ["example", "--config", "unused.yaml"]),
                        patch.object(example, "load_yaml_config", return_value=settings),
                        patch.object(example, "load_dataset", return_value=rows) as load,
                        patch.object(example, trainer_name) as trainer,
                        patch("builtins.print"),
                    ):
                        run = AsyncMock(return_value=state)
                        setattr(trainer.return_value, method, run)
                        example.main()
                    run.assert_awaited_once_with()
                    self.assertEqual(load.call_count, 2 if validation else 1)
                    self.assertEqual(len(trainer.call_args.kwargs["train_dataset"]), 2)
                    valid = trainer.call_args.kwargs["eval_dataset"]
                    self.assertEqual(len(valid), 1) if validation else self.assertIsNone(valid)


class TutorialPreparationTests(unittest.TestCase):
    def test_gsm8k_preparation_uses_official_splits_and_algorithm_formats(self):
        fixtures = {
            "train": [
                {"question": "What is 1 + 1?", "answer": "1 + 1 = 2.\n#### 2"},
                {"question": "What is 10 - 12.5?", "answer": "Subtract.\n#### -2.5"},
                {"question": "What is 600 + 600?", "answer": "Add.\n#### 1,200"},
            ],
            "test": [{"question": "What is 2 + 2?", "answer": "2 + 2 = 4.\n#### 4"}],
        }
        for _, algorithm, _, _, _, _ in CASES:
            for language in ("", "_CN"):
                with self.subTest(algorithm=algorithm, language=language):
                    text = (ROOT / f"docs/algorithms/{algorithm}{language}.md").read_text()
                    reference = (ROOT / f"docs/algorithms/grpo_series{language}.md").read_text()
                    self.assertEqual(re.findall(r"^## .+$", text, re.M),
                                     re.findall(r"^## .+$", reference, re.M))
                    scripts = re.findall(r"```python\n(.*?)\n```", text, re.S)
                    self.assertEqual(len(scripts), 1)
                    prepare = scripts[0]
                    def dataset(path, name, *, split):
                        self.assertEqual((path, name), ("openai/gsm8k", "main"))
                        return fixtures[split]
                    with tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        def local_path(value):
                            return directory / Path(value).name
                        with (
                            patch("datasets.load_dataset", side_effect=dataset) as load,
                            patch("pathlib.Path", side_effect=local_path),
                        ):
                            exec(compile(prepare, "gsm8k-preparation", "exec"), {})
                        self.assertEqual([call.kwargs["split"] for call in load.call_args_list],
                                         ["train", "test"])
                        prefix = "gsm8k" + (f"_{algorithm}" if algorithm != "opd" else "")
                        for split, raw_rows in fixtures.items():
                            rows = [json.loads(line) for line in
                                    (directory / f"{prefix}_{split}.jsonl").read_text().splitlines()]
                            self.assertEqual(len(rows), len(raw_rows))
                            for raw, row in zip(raw_rows, rows, strict=True):
                                expected = raw["answer"].rsplit("####", 1)[1].strip().replace(",", "")
                                prompt = row["prompt"] if algorithm == "sft" else row["messages"]
                                self.assertTrue(prompt[0]["content"].startswith(raw["question"]))
                                if algorithm == "opd":
                                    self.assertEqual(row["answer"], expected)
                                else:
                                    solution = row["chosen"] if algorithm == "dpo" else row["completion"][0]["content"]
                                    self.assertTrue(solution.endswith(r"\boxed{" + expected + "}"))
                                    self.assertIn(raw["answer"].rsplit("####", 1)[0].strip(), solution)
                                    if algorithm == "dpo":
                                        from decimal import Decimal
                                        wrong = re.findall(r"\\boxed\{([^{}]+)\}", row["rejected"])[-1]
                                        self.assertEqual(Decimal(wrong), Decimal(expected) + 1)
                        with (
                            patch("datasets.load_dataset") as load,
                            patch("pathlib.Path", side_effect=local_path),
                        ):
                            exec(compile(prepare, "gsm8k-preparation", "exec"), {})
                        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
