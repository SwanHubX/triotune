from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset

from trio_cookbook.trainer.datum import build_aligned_datum, to_client_datum
from trio_cookbook.trainer.grpo_trainer import _sample_train_kl
from trio_cookbook.trainer.types import SequenceSample


def _load_recipe() -> ModuleType:
    path = Path(__file__).parents[1] / "trio-cookbook" / "recipes" / "grpo_gsm8k.py"
    spec = importlib.util.spec_from_file_location("trio_cookbook_grpo_gsm8k_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recipe = _load_recipe()


class _Future:
    def __init__(
        self,
        events: list[str],
        event: str,
        value: object = None,
        error: Exception | None = None,
    ):
        self.events = events
        self.event = event
        self.value = value
        self.error = error

    def result(self) -> object:
        self.events.append(self.event)
        if self.error is not None:
            raise self.error
        return self.value


class _TrainingClient:
    def __init__(self, *, forward_error: Exception | None = None):
        self.events: list[str] = []
        self.forward_error = forward_error

    def forward_backward(self, datums: list[object], *, loss_fn: str) -> _Future:
        self.events.append("forward_submitted")
        return _Future(
            self.events,
            "forward_completed",
            SimpleNamespace(metrics={"loss_mean": 0.25}),
            self.forward_error,
        )

    def optim_step(self, adam_params: object) -> _Future:
        self.events.append("optimizer_submitted")
        return _Future(self.events, "optimizer_completed")


class OptimizationOrderTests(unittest.TestCase):
    def test_optimizer_is_submitted_after_forward_completes(self) -> None:
        client = _TrainingClient()

        metrics = recipe.run_optimization_step(
            client,
            [object()],
            loss_fn="importance_sampling",
            adam_params=object(),
        )

        self.assertEqual(metrics, {"loss_mean": 0.25})
        self.assertEqual(
            client.events,
            [
                "forward_submitted",
                "forward_completed",
                "optimizer_submitted",
                "optimizer_completed",
            ],
        )

    def test_forward_failure_does_not_submit_optimizer(self) -> None:
        client = _TrainingClient(forward_error=RuntimeError("forward failed"))

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            recipe.run_optimization_step(
                client,
                [object()],
                loss_fn="importance_sampling",
                adam_params=object(),
            )

        self.assertEqual(client.events, ["forward_submitted", "forward_completed"])


class PreparedDataTests(unittest.TestCase):
    def test_build_prompt_uses_tokenized_chat_template(self) -> None:
        messages = [{"role": "user", "content": "What is 6 times 7?"}]

        class Tokenizer:
            def apply_chat_template(self, received: object, **kwargs: object) -> list[int]:
                self.received = received
                self.kwargs = kwargs
                return [1, 2]

        tokenizer = Tokenizer()
        self.assertEqual(recipe.build_prompt(tokenizer, messages), [1, 2])
        self.assertEqual(tokenizer.received, messages)
        self.assertEqual(tokenizer.kwargs, {
            "tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
        })

    def test_loads_prepared_jsonl_and_normalized_answer(self) -> None:
        row = {
            "prompt": [{"role": "user", "content": "What is 6 times 7?"}],
            "answer": "42",
            "question": "What is 6 times 7?",
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = recipe.load_gsm8k_train(path)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(recipe.extract_gsm8k_answer(dataset[0]["answer"]), "42")

    def test_missing_data_points_to_separate_preparation_command(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "--prepare-only"):
                recipe.load_gsm8k_train(Path(directory) / "missing.jsonl")


class MetricParityTests(unittest.TestCase):
    def test_degenerate_group_logs_skip_without_loss_or_kl(self) -> None:
        samples = [
            recipe.RolloutSample([3], [-0.1], "\\boxed{0}", 0.0, 0.0),
            recipe.RolloutSample([4], [-0.2], "\\boxed{0}", 0.0, 0.0),
        ]
        metrics = recipe.build_step_metrics([samples], [], 1, {}, 4e-5)

        self.assertEqual(metrics["trainer/skipped"], 1.0)
        self.assertEqual(metrics["trainer/skipped_groups"], 1.0)
        self.assertNotIn("trainer/loss", metrics)
        self.assertNotIn("policy/kl_sample_train", metrics)

    def test_missing_forward_logprobs_omits_kl(self) -> None:
        sample = recipe.RolloutSample([3], [-0.1], "\\boxed{42}", 1.0, 0.5)
        datum = recipe.build_grpo_datum([1, 2], sample)
        forward = SimpleNamespace(loss_fn_outputs=[{}])
        self.assertIsNone(recipe.sample_train_kl([datum], forward))

    def test_datum_alignment_matches_trainer(self) -> None:
        sample = recipe.RolloutSample([3, 4], [-0.1, -0.2], "\\boxed{42}", 1.0, 0.5)
        actual = recipe.build_grpo_datum([1, 2], sample)
        expected = to_client_datum(
            build_aligned_datum(
                [1, 2], SequenceSample(sample.tokens, sample.logprobs, sample.text),
                sample.advantage, prompt_id="0", sample_id=0,
                model_input_builder=recipe.trio.ModelInput.from_ints,
            ),
            recipe.trio.Datum,
        )

        self.assertEqual(actual.model_input.to_ints(), expected.model_input.to_ints())
        for name in ("target_tokens", "logprobs", "advantages"):
            self.assertEqual(actual.loss_fn_inputs[name].data, expected.loss_fn_inputs[name].data)

    def test_kl_and_step_metrics_match_trainer_definitions(self) -> None:
        samples = [
            recipe.RolloutSample([3], [-0.1], "\\boxed{42}", 1.0, 0.5),
            recipe.RolloutSample([4, 5], [-0.2, -0.3], "\\boxed{0}", 0.0, -0.5),
        ]
        datums = [recipe.build_grpo_datum([1, 2], sample) for sample in samples]
        outputs = [
            {"logprobs": SimpleNamespace(data=[old - 0.1 for old in datum.loss_fn_inputs["logprobs"].data])}
            for datum in datums
        ]
        forward = SimpleNamespace(metrics={"loss_mean": 0.25}, loss_fn_outputs=outputs)
        optimizer = SimpleNamespace(metrics={"grad_norm": 1.5})

        class Client:
            def forward_backward(self, datums: object, *, loss_fn: str) -> _Future:
                return _Future([], "forward", forward)

            def optim_step(self, adam_params: object) -> _Future:
                return _Future([], "optimizer", optimizer)

        training = recipe.run_optimization_step(
            Client(), datums, loss_fn="importance_sampling", adam_params=object()
        )
        metrics = recipe.build_step_metrics([samples], datums, 0, training, 4e-5)

        self.assertAlmostEqual(metrics["policy/kl_sample_train"], _sample_train_kl(datums, forward))
        self.assertEqual(metrics["trainer/loss"], 0.25)
        self.assertEqual(metrics["optim/grad_norm"], 1.5)
        self.assertEqual(metrics["reward/mean"], 0.5)
        self.assertEqual(metrics["reward/std"], 0.5)
        self.assertAlmostEqual(metrics["policy/entropy"], 0.2)
        self.assertEqual(metrics["rollout/completion_length_mean"], 1.5)


class DirectTrainingTests(unittest.TestCase):
    def test_exp_name_only_changes_swanlab_experiment_name(self) -> None:
        with patch.object(sys, "argv", ["grpo_gsm8k.py", "--exp-name", "comparison-run"]):
            config = recipe.parse_args()
        with patch.object(recipe.swanlab, "init") as init:
            recipe.init_swanlab_run(config, effective_steps=1, dataset_size=1, run_name="weights-name")

        self.assertEqual(init.call_args.kwargs["name"], "comparison-run")
        self.assertEqual(init.call_args.kwargs["config"]["weights_name"], "weights-name")

        with patch.object(sys, "argv", ["grpo_gsm8k.py"]):
            default_config = recipe.parse_args()
        with patch.object(recipe.swanlab, "init") as init:
            recipe.init_swanlab_run(default_config, effective_steps=1, dataset_size=1, run_name="weights-name")
        self.assertEqual(init.call_args.kwargs["name"], "weights-name")

    def test_workspace_defaults_to_pytrio(self) -> None:
        with patch.object(sys, "argv", ["grpo_gsm8k.py"]):
            self.assertEqual(recipe.parse_args().workspace, "pytrio")

    def test_rejects_invalid_training_boundaries(self) -> None:
        with patch.object(sys, "argv", ["grpo_gsm8k.py", "--group-size", "1"]):
            with self.assertRaisesRegex(ValueError, "group_size"):
                recipe.parse_args()
        with patch.object(sys, "argv", ["grpo_gsm8k.py", "--top-p", "1.5"]):
            with self.assertRaisesRegex(ValueError, "top_p"):
                recipe.parse_args()
        with patch.object(sys, "argv", ["grpo_gsm8k.py", "--workspace", " "]):
            with self.assertRaisesRegex(ValueError, "workspace"):
                recipe.parse_args()
        with patch.object(sys, "argv", ["grpo_gsm8k.py", "--exp-name", " "]):
            with self.assertRaisesRegex(ValueError, "exp_name"):
                recipe.parse_args()

    def test_one_step_logs_comparable_metrics_and_finishes(self) -> None:
        dataset = Dataset.from_list([
            {"prompt": [{"role": "user", "content": "What is 6 times 7?"}], "answer": "42", "question": "6 * 7?"}
        ])
        events: list[str] = []

        class Tokenizer:
            def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
                self.template_kwargs = kwargs
                return [1, 2]

            def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
                return "\\boxed{42}"

        class Sampler:
            def sample(self, **kwargs: object) -> _Future:
                self.kwargs = kwargs
                sequences = [
                    SimpleNamespace(tokens=[3], logprobs=[-0.1], text="\\boxed{42}"),
                    SimpleNamespace(tokens=[4], logprobs=[-0.2], text="\\boxed{0}"),
                ]
                return _Future(events, "sampled", SimpleNamespace(sequences=sequences))

        sampler = Sampler()
        tokenizer = Tokenizer()

        class TrainingClient:
            def get_tokenizer(self) -> Tokenizer:
                return tokenizer

            def save_weights_and_get_sampling_client(self) -> Sampler:
                return sampler

            def forward_backward(self, datums: list[object], *, loss_fn: str) -> _Future:
                outputs = [
                    {"logprobs": SimpleNamespace(data=[0.0, -0.2])}
                    for _ in datums
                ]
                result = SimpleNamespace(metrics={"loss_mean": 0.25}, loss_fn_outputs=outputs)
                return _Future(events, "forward", result)

            def optim_step(self, adam_params: object) -> _Future:
                return _Future(events, "optimizer", SimpleNamespace(metrics={"grad_norm": 1.5}))

            def save_weights_for_sampler(self, *, name: str) -> _Future:
                return _Future(events, "saved", SimpleNamespace(path="trio://final-weights"))

        training_client = TrainingClient()

        class ServiceClient:
            workspace_name = "another-organization"

            def create_lora_training_client(self, **kwargs: object) -> TrainingClient:
                self.kwargs = kwargs
                return training_client

        service = ServiceClient()

        class Run:
            def __init__(self) -> None:
                self.logs: list[tuple[dict[str, float], int]] = []
                self.texts: dict[str, str] = {}
                self.finished = False

            def log(self, payload: dict[str, float], *, step: int) -> None:
                self.logs.append((payload, step))

            def log_text(self, *, key: str, data: str, step: int) -> None:
                self.texts[key] = data

            def finish(self, **_: object) -> None:
                self.finished = True

        run = Run()
        with patch.object(sys, "argv", [
            "grpo_gsm8k.py", "--steps", "1", "--batch-size", "1", "--group-size", "2",
            "--workspace", "swanlab-team",
        ]):
            config = recipe.parse_args()
        with (
            patch.object(recipe, "load_gsm8k_train", return_value=dataset),
            patch.object(recipe.trio, "ServiceClient", return_value=service),
            patch.object(recipe.swanlab, "init", return_value=run) as init,
            patch.object(recipe, "tqdm", side_effect=lambda rows, **_: rows),
            patch.object(recipe, "print", create=True),
        ):
            recipe.main(config)

        metrics, step = run.logs[0]
        self.assertEqual(step, 1)
        self.assertEqual(metrics["train/loss"], 0.25)
        self.assertEqual(metrics["train/learning_rate"], config.learning_rate)
        self.assertNotIn("optim/lr", metrics)
        self.assertFalse(any(key.startswith("trainer/") for key in metrics))
        self.assertTrue(all("/" in key for key in metrics))
        self.assertEqual(metrics["train/loss_mean"], 0.25)
        self.assertEqual(metrics["train/reward"], 0.5)
        self.assertIn("train/frac_degenerate", metrics)
        self.assertEqual(metrics["optim/grad_norm"], 1.5)
        self.assertAlmostEqual(metrics["policy/kl_sample_train"], 0.0025)
        self.assertEqual(metrics["reward/mean"], 0.5)
        self.assertTrue(all(metrics[key] >= 0 for key in ("time/rollout", "time/reward", "time/train", "time/step")))
        self.assertEqual(run.texts["weights/path"], "trio://final-weights")
        self.assertTrue(run.finished)
        self.assertEqual(init.call_args.kwargs["workspace"], "swanlab-team")
        self.assertEqual(service.kwargs["seed"], config.seed)
        self.assertEqual(sampler.kwargs["sampling_params"].seed, config.seed)
        self.assertIsNone(sampler.kwargs["sampling_params"].stop)
        self.assertEqual(tokenizer.template_kwargs["tokenize"], True)


if __name__ == "__main__":
    unittest.main()
