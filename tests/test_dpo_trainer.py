from __future__ import annotations

import asyncio
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trio_cookbook.trainer import DPOConfig, DPOTrainer
from trio_cookbook.trainer.dpo_trainer import _datum, _loss_fn, _reference_logprob


class _Tokenizer:
    eos_token_id = 255

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


class _ChatTokenizer(_Tokenizer):
    def __init__(self) -> None:
        self.thinking_values: list[bool] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = True,
    ) -> str | list[int]:
        self.thinking_values.append(enable_thinking)
        text = "".join(f"{message['role']}:{message['content']};" for message in messages)
        if add_generation_prompt:
            text += "assistant:"
        return self.encode(text) if tokenize else text


class _ReferenceClient:
    def __init__(self) -> None:
        self.requests: list[list[int]] = []

    def compute_logprobs(self, model_input: object) -> list[float | None]:
        tokens = model_input.to_ints()
        self.requests.append(tokens)
        return [None] + [-token / 1000 for token in tokens[1:]]

    async def compute_logprobs_async(self, model_input: object) -> list[float | None]:
        await asyncio.sleep(0)
        return self.compute_logprobs(model_input)


class _TrainingClient:
    def __init__(self) -> None:
        self.datums: list[object] = []
        self.optim_calls = 0
        self.save_calls = 0
        self.fail_forward = False

    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()

    def forward_backward_custom(self, datums: list[object], loss_fn: object) -> SimpleNamespace:
        self.datums = datums
        self.loss_fn = loss_fn
        if self.fail_forward:
            raise RuntimeError("forward failed")
        return SimpleNamespace(metrics={
            "train/loss": 0.5,
            "rewards/accuracies": 1.0,
            "rewards/margins": 0.3,
            "rewards/chosen": 0.2,
            "rewards/rejected": -0.1,
            "loss_sum": -1.0,
            "loss_mean": -0.25,
            "scalar/loss_mean": -0.25,
            "token_count": 4,
        })

    async def forward_backward_custom_async(self, datums: list[object], loss_fn: object) -> SimpleNamespace:
        await asyncio.sleep(0)
        return self.forward_backward_custom(datums, loss_fn)

    def optim_step(self, params: object) -> SimpleNamespace:
        self.optim_calls += 1
        return SimpleNamespace(metrics={"grad_norm": 0.2})

    async def optim_step_async(self, params: object) -> SimpleNamespace:
        await asyncio.sleep(0)
        return self.optim_step(params)

    def save_state(self, *, name: str) -> SimpleNamespace:
        self.save_calls += 1
        return SimpleNamespace(path=f"checkpoint://{name}")

    async def save_state_async(self, *, name: str) -> SimpleNamespace:
        await asyncio.sleep(0)
        return self.save_state(name=name)


class _ServiceClient:
    def __init__(self) -> None:
        self.training = _TrainingClient()
        self.reference = _ReferenceClient()

    def create_lora_training_client(self, **kwargs: object) -> _TrainingClient:
        return self.training

    def create_sampling_client(self, **kwargs: object) -> _ReferenceClient:
        return self.reference


class _SwanLabRun:
    def __init__(self) -> None:
        self.logs: list[tuple[object, int]] = []
        self.texts: list[tuple[str, str]] = []
        self.finished: str | None = None

    def log(self, metrics: object, *, step: int) -> None:
        self.logs.append((metrics, step))

    def log_text(self, *, key: str, data: str, step: int) -> None:
        self.texts.append((key, data))

    def finish(self, *, state: str, error: str | None = None) -> None:
        self.finished = state


class DPOTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _ServiceClient()
        self.run = _SwanLabRun()
        service_patch = patch("trio_cookbook.trainer.dpo_trainer.trio.ServiceClient", return_value=self.service)
        swanlab_patch = patch("trio_cookbook.trainer.dpo_trainer.swanlab.init", return_value=self.run)
        service_patch.start()
        swanlab_patch.start()
        self.addCleanup(service_patch.stop)
        self.addCleanup(swanlab_patch.stop)

    def trainer(self, **config: object) -> DPOTrainer:
        config.setdefault("report_to", "swanlab")
        return DPOTrainer("test-model", args=DPOConfig(**config))

    def test_none_report_to_uses_terminal_progress_without_swanlab(self) -> None:
        self.assertIsNone(DPOConfig().report_to)
        with (
            patch("trio_cookbook.trainer.dpo_trainer.swanlab.init") as init,
            patch("trio_cookbook.trainer.dpo_trainer.tqdm") as progress,
        ):
            trainer = self.trainer(report_to=None, max_steps=1)
            trainer.train_dataset = [{"messages": "P", "chosen": "A", "rejected": "B"}]
            trainer.train()

        init.assert_not_called()
        progress.return_value.set_postfix.assert_called_once()
        progress.return_value.update.assert_called_once_with(1)

    def test_rejects_unsupported_report_targets(self) -> None:
        for target in ("disabled", "online", "local", "offline"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "report_to"):
                DPOConfig(report_to=target)

    def test_logging_step_must_be_positive_integer(self) -> None:
        self.assertEqual(DPOConfig().logging_step, 1)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "logging_step"):
                DPOConfig(logging_step=value)

    def test_step_aligns_pair_and_scores_only_completion(self) -> None:
        trainer = self.trainer()
        result = trainer.step([{"messages": "P", "chosen": "A", "rejected": "B"}])

        self.assertEqual(result.step, 1)
        self.assertEqual(result.num_pairs, 1)
        self.assertEqual(self.service.reference.requests, [[80, 65, 255], [80, 66, 255]])
        self.assertEqual([datum.model_input.to_ints() for datum in self.service.training.datums], [[80, 65], [80, 66]])
        self.assertEqual([datum.loss_fn_inputs["target_tokens"].data for datum in self.service.training.datums], [[65, 255], [66, 255]])
        self.assertEqual(self.service.training.optim_calls, 1)
        self.assertEqual(self.run.logs[0][1], 1)

    def test_step_logs_dpo_loss_and_measured_work_without_surrogate_loss(self) -> None:
        result = self.trainer(save_every=1).step([{"messages": "P", "chosen": "A", "rejected": "B"}])
        metrics = result.metrics
        self.assertEqual(metrics["train/loss"], 0.5)
        self.assertEqual(metrics["data/tokens"], 4)
        self.assertEqual(metrics["optim/grad_norm"], 0.2)
        self.assertNotIn("loss_sum", metrics)
        self.assertNotIn("loss_mean", metrics)
        self.assertNotIn("scalar/loss_mean", metrics)
        payload = self.run.logs[0][0]
        self.assertEqual({name.split("/", 1)[0] for name in payload}, {"train", "reward", "time"})
        self.assertEqual(payload["train/loss"], 0.5)
        self.assertEqual(payload["reward/reward_accuracy"], 1.0)
        self.assertEqual(payload["reward/reward_margin"], 0.3)
        self.assertEqual(payload["reward/reward_chosen"], 0.2)
        self.assertEqual(payload["reward/reward_rejected"], -0.1)
        self.assertEqual(payload["train/learning_rate"], 5e-6)
        self.assertEqual(payload["train/grad_norm"], 0.2)
        self.assertEqual(payload["train/backend_token_count"], 4)
        self.assertEqual(payload["train/step"], 1)
        self.assertEqual(payload["train/pairs"], 1)
        self.assertNotIn("loss_sum", payload)
        for key in ("time/prepare", "time/reference", "time/forward_backward", "time/optim", "time/save"):
            self.assertGreaterEqual(metrics[key], 0)
        self.assertGreater(metrics["time/step"], 0)
        self.assertAlmostEqual(
            metrics["time/tokens_per_second"], metrics["data/tokens"] / metrics["time/step"]
        )
        self.assertAlmostEqual(metrics["time/pairs_per_second"], 1 / metrics["time/step"])

    def test_logging_step_averages_all_training_scalars_and_flushes_last_interval(self) -> None:
        trainer = self.trainer(logging_step=2, max_steps=3)
        trainer.train_dataset = [{"messages": "P", "chosen": "A", "rejected": "B"}] * 3
        forward = self.service.training.forward_backward_custom
        losses = iter((0.2, 0.4, 0.6))

        def forward_with_varying_loss(datums: list[object], loss_fn: object) -> SimpleNamespace:
            result = forward(datums, loss_fn)
            result.metrics["train/loss"] = next(losses)
            return result

        self.service.training.forward_backward_custom = forward_with_varying_loss
        trainer.train()

        self.assertEqual([step for _, step in self.run.logs], [2, 3, 3])
        first, final_interval, summary = [payload for payload, _ in self.run.logs]
        self.assertAlmostEqual(first["train/loss"], 0.3)
        self.assertEqual(first["reward/reward_accuracy"], 1.0)
        self.assertEqual(first["train/pairs"], 1.0)
        self.assertEqual(first["train/step"], 2)
        self.assertAlmostEqual(final_interval["train/loss"], 0.6)
        self.assertEqual(final_interval["train/step"], 3)
        self.assertIn("time/train_runtime", summary)

    def test_swanlab_mode_initializes_run_without_sdk_mode_override(self) -> None:
        with patch("trio_cookbook.trainer.dpo_trainer.swanlab.init") as init:
            self.trainer()
        self.assertNotIn("mode", init.call_args.kwargs)

    def test_forward_failure_does_not_optimize_or_advance_step(self) -> None:
        trainer = self.trainer()
        self.service.training.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            trainer.step([{"messages": "P", "chosen": "A", "rejected": "B"}])
        self.assertEqual(self.service.training.optim_calls, 0)
        self.assertEqual(trainer.state.global_step, 0)

    def test_rejects_missing_fields_and_overlong_pairs(self) -> None:
        trainer = self.trainer(max_length=2)
        with self.assertRaises(KeyError):
            trainer.step([{"messages": "P", "chosen": "A"}])
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            trainer.step([{"messages": "P", "chosen": "A", "rejected": "B"}])
        with self.assertRaisesRegex(KeyError, "messages"):
            trainer.step([{"prompt": "P", "chosen": "A", "rejected": "B"}])
        self.assertEqual(self.service.training.optim_calls, 0)

    def test_legacy_prompt_field_requires_explicit_configuration(self) -> None:
        trainer = self.trainer(prompt_key="prompt")
        result = trainer.step([{"prompt": "P", "chosen": "A", "rejected": "B"}])
        self.assertEqual(result.step, 1)

    def test_train_saves_final_checkpoint_and_finishes_swanlab(self) -> None:
        trainer = self.trainer()
        trainer.train_dataset = [{"messages": "P", "chosen": "A", "rejected": "B"}]
        state = trainer.train()
        self.assertEqual(state.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.service.training.save_calls, 1)
        self.assertEqual(self.run.finished, "success")
        self.assertEqual(self.run.texts, [("checkpoint/path", state.checkpoint_path)])
        self.assertEqual(
            {name.split("/", 1)[0] for name in self.run.logs[-1][0]}, {"train", "time"}
        )
        self.assertEqual(state.metrics["data/total_tokens"], 4)
        self.assertEqual(state.metrics["data/total_pairs"], 1)
        self.assertGreater(state.metrics["time/train_runtime"], 0)
        self.assertGreater(state.metrics["time/train_steps_per_second"], 0)
        self.assertGreater(state.metrics["time/train_pairs_per_second"], 0)

    def test_async_step_uses_async_clients(self) -> None:
        trainer = self.trainer(save_every=1)
        result = asyncio.run(trainer.step_async([{"messages": "P", "chosen": "A", "rejected": "B"}]))
        self.assertEqual(result.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.service.training.optim_calls, 1)
        self.assertEqual(result.metrics["data/tokens"], 4)
        self.assertIn("time/save", result.metrics)
        self.assertNotIn("loss_mean", result.metrics)
        self.assertEqual(
            {name.split("/", 1)[0] for name in self.run.logs[0][0]}, {"train", "reward", "time"}
        )

    def test_async_train_saves_final_checkpoint(self) -> None:
        trainer = self.trainer(logging_step=10)
        trainer.train_dataset = [{"messages": "P", "chosen": "A", "rejected": "B"}]
        state = asyncio.run(trainer.train_async())
        self.assertEqual(state.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.run.finished, "success")
        self.assertEqual(state.metrics["data/total_tokens"], 4)
        self.assertEqual([step for _, step in self.run.logs], [1, 1])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "DPO extra is not installed")
    def test_pairwise_loss_and_completion_gradients(self) -> None:
        import torch

        datums = [_datum([10, 11], [12]), _datum([10, 11], [13])]
        chosen = torch.tensor([0.0, -0.5], requires_grad=True)
        rejected = torch.tensor([0.0, -1.0], requires_grad=True)
        loss, metrics = _loss_fn([-0.5, -1.0], 0.1)(datums, [chosen, rejected])
        loss.backward()
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertAlmostEqual(chosen.grad[1].item(), -0.05, places=6)
        self.assertAlmostEqual(rejected.grad[1].item(), 0.05, places=6)
        self.assertEqual(chosen.grad[0].item(), 0.0)
        self.assertEqual(metrics["rewards/margins"], 0.0)

    def test_reference_logprobs_require_complete_finite_response(self) -> None:
        self.assertAlmostEqual(_reference_logprob([None, -1.0, -2.0], 1, 3), -3.0)
        with self.assertRaisesRegex(ValueError, "wrong number"):
            _reference_logprob([None], 1, 3)
        with self.assertRaisesRegex(ValueError, "missing"):
            _reference_logprob([None, -1.0, None], 1, 3)

    def test_datum_masks_prompt_and_shifts_completion(self) -> None:
        datum = _datum([10, 11], [12, 13])
        self.assertEqual(datum.model_input.to_ints(), [10, 11, 12])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [-100, 12, 13])

    def test_chat_completions_follow_rendered_prompt_prefix(self) -> None:
        trainer = DPOTrainer(
            "test-model",
            args=DPOConfig(chat_template_kwargs={"enable_thinking": False}),
        )
        tokenizer = _ChatTokenizer()
        trainer._tokenizer = tokenizer
        pairs = trainer._preferences([{
            "messages": [{"role": "user", "content": "P"}],
            "chosen": [{"role": "assistant", "content": "A"}],
            "rejected": [{"role": "assistant", "content": "B"}],
        }])
        self.assertEqual(pairs[0].chosen, [ord("A"), ord(";"), 255])
        self.assertEqual(pairs[0].rejected, [ord("B"), ord(";"), 255])
        self.assertEqual(tokenizer.thinking_values, [False] * 6)


if __name__ == "__main__":
    unittest.main()
