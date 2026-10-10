"""Offline coverage of the framework DeepMath recipe and dataset adapter."""

from __future__ import annotations

import importlib.util
import io
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import urllib.parse

from datasets import Dataset


RECIPE = Path(__file__).parents[1] / "trio-cookbook/recipes/opd_deepmath_trio.py"
spec = importlib.util.spec_from_file_location("opd_deepmath_recipe_test", RECIPE)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class DeepMathDataTests(unittest.TestCase):
    def settings(self):
        return dict(script.load_config(script.DEFAULT_CONFIG)["data"])

    def test_modelscope_url_and_shard_boundaries(self):
        path = script.shard_name(9)
        self.assertEqual(path, "data/train-00009-of-00010.parquet")
        url = script.modelscope_file_url("AI-ModelScope/DeepMath-103K", "master", path)
        parsed = urllib.parse.urlsplit(url)
        self.assertEqual(parsed.netloc, "www.modelscope.cn")
        self.assertEqual(urllib.parse.parse_qs(parsed.query)["FilePath"], [path])
        self.assertEqual(urllib.parse.parse_qs(parsed.query)["Revision"], ["master"])
        data = self.settings()
        data["num_shards"] = 10
        script.validate_data_settings(data)

    def test_invalid_settings_fail_before_download(self):
        for key, value in (
            ("num_shards", 0), ("num_shards", 11), ("num_shards", True),
            ("sample_size", True), ("sample_size", "10"),
            ("seed", -1), ("seed", False), ("question_suffix", None),
            ("repo", "missing-namespace"), ("repo", "/empty"),
            ("revision", ""), ("dataset_dir", ""),
        ):
            data = self.settings()
            data[key] = value
            with self.subTest(key=key, value=value), patch.object(script, "download_if_needed") as download:
                with self.assertRaises(ValueError):
                    script.load_deepmath(data, force=False, timeout=600)
                download.assert_not_called()

    def test_download_reuses_cache_and_atomically_replaces_forced_download(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "data/test.parquet"
            path.parent.mkdir()
            path.write_bytes(b"old")
            with patch.object(script.urllib.request, "urlopen") as open_url:
                script.download_if_needed("https://example.com/data", path, force=False, timeout=60)
                open_url.assert_not_called()
            with patch.object(script.urllib.request, "urlopen", return_value=io.BytesIO(b"new")) as open_url:
                script.download_if_needed("https://example.com/data", path, force=True, timeout=60)
                self.assertEqual(open_url.call_args.kwargs["timeout"], 60)
            self.assertEqual(path.read_bytes(), b"new")
            self.assertFalse(path.with_suffix(".parquet.tmp").exists())

    def test_failed_download_preserves_cache_and_removes_partial_file(self):
        class BrokenResponse(io.BytesIO):
            def read(self, *args):
                raise OSError("connection lost")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.parquet"
            path.write_bytes(b"old")
            with patch.object(script.urllib.request, "urlopen", return_value=BrokenResponse()):
                with self.assertRaisesRegex(OSError, "connection lost"):
                    script.download_if_needed("https://example.com/data", path, force=True, timeout=60)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertFalse(path.with_suffix(".parquet.tmp").exists())

    def test_cached_parquet_sampling_is_repeatable_and_limited(self):
        with TemporaryDirectory() as directory:
            data = self.settings()
            data.update(dataset_dir=directory, sample_size=3)
            path = Path(directory) / script.shard_name(0)
            path.parent.mkdir()
            source = Dataset.from_dict({
                "question": [f"question {i}" for i in range(8)],
                "final_answer": ["not a training target"] * 8,
            })
            source.to_parquet(path)
            with patch.object(script.urllib.request, "urlopen") as open_url:
                a = script.load_deepmath(data, force=False, timeout=600)
                b = script.load_deepmath(data, force=False, timeout=600)
                self.assertEqual(a["question"], b["question"])
                self.assertEqual(len(a), 3)
                self.assertEqual(a.column_names, ["question"])
                data["sample_size"] = 100
                self.assertEqual(len(script.load_deepmath(data, force=False, timeout=600)), 8)
                data["sample_size"] = 0
                self.assertEqual(len(script.load_deepmath(data, force=False, timeout=600)), 8)
                open_url.assert_not_called()

    def test_multiple_shards_are_read_from_modelscope_paths(self):
        data = self.settings()
        data["num_shards"] = 2
        dataset = Dataset.from_dict({"question": ["q"]})
        with patch.object(script, "download_if_needed") as download, patch.object(
            script, "load_dataset", return_value=dataset,
        ) as load:
            script.load_deepmath(data, force=True, timeout=100)
        self.assertEqual(download.call_count, 2)
        for index, call in enumerate(download.call_args_list):
            self.assertIn("www.modelscope.cn", call.args[0])
            self.assertEqual(call.args[1].name, Path(script.shard_name(index)).name)
            self.assertEqual(call.kwargs, {"force": True, "timeout": 100})
        self.assertEqual(load.call_args.args, ("parquet",))
        self.assertEqual(load.call_args.kwargs["split"], "train")
        self.assertEqual(len(load.call_args.kwargs["data_files"]), 2)

    def test_invalid_parquet_schema_and_empty_datasets_fail(self):
        for dataset, error in (
            (Dataset.from_dict({"question": []}), ValueError),
            (Dataset.from_dict({"other": ["q"]}), ValueError),
            ([], TypeError),
        ):
            with self.subTest(dataset=dataset), patch.object(script, "download_if_needed"), patch.object(
                script, "load_dataset", return_value=dataset,
            ), self.assertRaises(error):
                script.load_deepmath(self.settings(), force=False, timeout=60)

    def test_rows_use_questions_only_and_preserve_suffix(self):
        dataset = Dataset.from_dict({"question": ["  What is 1+1?  "], "final_answer": ["2"], "r1_solution_1": ["ignore"]})
        self.assertEqual(script.prompt_rows(dataset, "suffix"), [
            {"messages": [{"role": "user", "content": "What is 1+1?\n\nsuffix"}]},
        ])
        self.assertEqual(script.prompt_rows(dataset, "")[0]["messages"][0]["content"], "What is 1+1?")

    def test_invalid_question_values_fail(self):
        for value in ("", "   ", None, 5):
            with self.subTest(question=value), self.assertRaisesRegex(ValueError, "non-empty string"):
                script.prompt_rows(Dataset.from_dict({"question": [value]}), "")


class DeepMathEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_only_does_not_configure_or_create_training_clients(self):
        dataset = Dataset.from_dict({"question": ["q"]})
        with patch.object(script, "load_deepmath", return_value=dataset), patch.object(
            script, "OPDTrainer",
        ) as trainer, patch.object(script.trio, "configure") as configure:
            await script.main(["--prepare-only"])
        trainer.assert_not_called()
        configure.assert_not_called()

    async def test_default_config_uses_async_framework_with_cyclic_prompts(self):
        dataset = Dataset.from_dict({"question": ["q1", "q2"]})
        with patch.object(script, "load_deepmath", return_value=dataset), patch.object(
            script, "OPDTrainer",
        ) as trainer, patch.object(script.trio, "configure") as configure:
            trainer.return_value.train_async = AsyncMock(return_value=SimpleNamespace(
                global_step=10, checkpoint_path="trio://saved", metrics={},
            ))
            await script.main([])
        kwargs = trainer.call_args.kwargs
        self.assertEqual(kwargs["model"], "Qwen/Qwen3.5-4B")
        self.assertEqual(kwargs["teacher_model"], "Qwen/Qwen3.8-27B")
        self.assertEqual(kwargs["lora_rank"], 32)
        config = kwargs["args"]
        self.assertEqual((config.max_steps, config.batch_size, config.num_generations), (50, 2, 2))
        self.assertEqual(config.chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual((config.kl_coef, config.discount_factor, config.report_to), (1.0, 0.0, "swanlab"))
        rows = list(itertools.islice(kwargs["train_dataset"], 5))
        self.assertEqual(rows[0], rows[2])
        self.assertEqual(rows[1], rows[3])
        self.assertNotEqual(rows[0], rows[1])
        trainer.return_value.train_async.assert_awaited_once_with()
        configure.assert_called_once_with(timeout=600)

    async def test_cli_overrides_data_models_training_and_logging(self):
        dataset = Dataset.from_dict({"question": ["q"]})
        with patch.object(script, "load_deepmath", return_value=dataset) as load, patch.object(
            script, "OPDTrainer",
        ) as trainer, patch.object(script.trio, "configure"):
            trainer.return_value.train_async = AsyncMock(return_value=SimpleNamespace(
                global_step=3, checkpoint_path="trio://saved", metrics={},
            ))
            await script.main([
                "--steps", "3", "--batch-size", "4", "--group-size", "1",
                "--max-tokens", "8", "--max-length", "4096", "--seed", "9",
                "--num-shards", "2", "--sample-size", "0", "--dataset-dir", "/tmp/deepmath-example",
                "--base-model", "student", "--teacher-base-model", "teacher",
                "--teacher-model-path", "trio://teacher", "--lora-rank", "8",
                "--enable-thinking", "--kl-penalty-coef", "0.5", "--learning-rate", "0.00001",
                "--max-concurrent-sampling", "2", "--max-concurrent-scoring", "3",
                "--save-every", "2", "--no-swanlab", "--question-suffix", "",
                "--project", "example", "--workspace", "team", "--run-name", "custom",
            ])
        data = load.call_args.args[0]
        self.assertEqual((data["seed"], data["num_shards"], data["sample_size"]), (9, 2, 0))
        self.assertEqual(data["dataset_dir"], "/tmp/deepmath-example")
        kwargs = trainer.call_args.kwargs
        self.assertEqual((kwargs["model"], kwargs["teacher_model"], kwargs["teacher_model_path"]), ("student", "teacher", "trio://teacher"))
        self.assertEqual(kwargs["lora_rank"], 8)
        config = kwargs["args"]
        self.assertEqual((config.max_steps, config.batch_size, config.num_generations), (3, 4, 1))
        self.assertEqual((config.max_completion_length, config.max_length, config.seed), (8, 4096, 9))
        self.assertEqual((config.max_concurrent_sampling, config.max_concurrent_scoring), (2, 3))
        self.assertEqual((config.kl_coef, config.learning_rate, config.save_every), (0.5, 1e-5, 2))
        self.assertIsNone(config.report_to)
        self.assertEqual(config.chat_template_kwargs, {"enable_thinking": True})
        self.assertEqual((config.project, config.workspace, config.run_name), ("example", "team", "custom"))
        self.assertEqual(next(kwargs["train_dataset"])["messages"][0]["content"], "q")

    async def test_bad_runtime_or_training_settings_fail_before_data_download(self):
        for section, key, value in (
            ("runtime", "timeout", 0), ("runtime", "timeout", float("inf")),
            ("opd", "max_steps", None), ("opd", "num_generations", 0),
            ("opd", "prompt_key", "question"), ("model", "lora_rank", 0),
            ("teacher", "name", ""), ("teacher", "model_path", ""),
        ):
            settings = script.load_config(script.DEFAULT_CONFIG)
            settings[section][key] = value
            with self.subTest(section=section, key=key), patch.object(
                script, "load_config", return_value=settings,
            ), patch.object(script, "load_deepmath") as load, self.assertRaises(ValueError):
                await script.main([])
            load.assert_not_called()

    async def test_empty_question_and_training_failures_propagate(self):
        with patch.object(script, "load_deepmath", return_value=Dataset.from_dict({"question": [" "]})), patch.object(
            script, "OPDTrainer",
        ) as trainer, self.assertRaisesRegex(ValueError, "non-empty string"):
            await script.main([])
        trainer.assert_not_called()
        with patch.object(script, "load_deepmath", return_value=Dataset.from_dict({"question": ["q"]})), patch.object(
            script, "OPDTrainer",
        ) as trainer, patch.object(script.trio, "configure"):
            trainer.return_value.train_async = AsyncMock(side_effect=RuntimeError("remote update failed"))
            with self.assertRaisesRegex(RuntimeError, "remote update failed"):
                await script.main([])

    def test_missing_config_sections_and_tables_fail(self):
        for settings in (
            {}, {"model": {}},
            {**script.load_config(script.DEFAULT_CONFIG), "runtime": []},
            {**script.load_config(script.DEFAULT_CONFIG), "swanlab": []},
        ):
            with self.subTest(settings=settings), patch.object(script.tomllib, "load", return_value=settings):
                with self.assertRaises(ValueError):
                    script.load_config(script.DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
