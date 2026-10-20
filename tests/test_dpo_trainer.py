from __future__ import annotations

import asyncio
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from triotune.trainer import DPOConfig, DPOTrainer
from triotune.trainer.dpo.loss import make_loss_fn as _loss_fn
from triotune.trainer.dpo.datum import build_dpo_datum


class _Tokenizer:
    eos_token_id = 255

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in tokens if not skip_special_tokens or token != 255)


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


class _SamplingClient:
    def __init__(self) -> None:
        self.requests: list[tuple[list[int], object]] = []
        self.async_calls = 0
        self.text: str | None = "generated answer"
        self.sequences: list[object] | None = None

    def sample(self, *, prompt: object, num_samples: int, sampling_params: object) -> SimpleNamespace:
        self.requests.append((prompt.to_ints(), sampling_params))
        assert num_samples == 1
        return SimpleNamespace(sequences=(
            self.sequences if self.sequences is not None
            else [SimpleNamespace(text=self.text, tokens=[65, 66])]
        ))

    async def sample_async(self, **kwargs: object) -> SimpleNamespace:
        self.async_calls += 1
        await asyncio.sleep(0)
        return self.sample(**kwargs)

    def compute_logprobs(self, model_input: object) -> list[float | None]:
        tokens = model_input.to_ints()
        return [None] + [-token / 2000 for token in tokens[1:]]

    async def compute_logprobs_async(
        self, model_input: object
    ) -> list[float | None]:
        await asyncio.sleep(0)
        return self.compute_logprobs(model_input)


class _TrainingClient:
    def __init__(self) -> None:
        self.datums: list[object] = []
        self.optim_calls = 0
        self.save_calls = 0
        self.fail_forward = False
        self.sampler = _SamplingClient()
        self.sampling_client_calls = 0
        self.async_sampling_client_calls = 0

    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()

    def forward_backward_custom(self, datums: list[object], loss_fn: object) -> SimpleNamespace:
        self.datums = datums
        self.loss_fn = loss_fn
        if self.fail_forward:
            raise RuntimeError("forward failed")
        return SimpleNamespace(metrics={
            "train/loss": 0.5,
            "reward/accuracy": 1.0,
            "reward/margin": 0.3,
            "reward/chosen": 0.2,
            "reward/rejected": -0.1,
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

    def save_weights_and_get_sampling_client(self) -> _SamplingClient:
        self.sampling_client_calls += 1
        return self.sampler

    async def save_weights_and_get_sampling_client_async(self) -> _SamplingClient:
        self.async_sampling_client_calls += 1
        await asyncio.sleep(0)
        return self.sampler


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
        self.texts: list[tuple[str, object]] = []
        self.finished: str | None = None

    def log(self, metrics: object, *, step: int) -> None:
        self.logs.append((metrics, step))

    def log_text(self, *, key: str, data: object, step: int) -> None:
        self.texts.append((key, data))

    def finish(self, *, state: str, error: str | None = None) -> None:
        self.finished = state


class DPOTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _ServiceClient()
        self.run = _SwanLabRun()
        service_patch = patch("triotune.trainer.dpo_trainer.trio.ServiceClient", return_value=self.service)
        swanlab_patch = patch(
            "triotune.utils.metrics.swanlab.init",
            return_value=self.run,
        )
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
            patch("triotune.utils.metrics.swanlab.init") as init,
            patch("triotune.trainer.dpo_trainer.tqdm") as progress,
        ):
            trainer = self.trainer(report_to=None, max_steps=1)
            trainer.train_dataset = [{"messages": "P", "chosen": "A", "rejected": "B"}]
            trainer.train()

        init.assert_not_called()
        self.assertEqual(self.service.training.sampling_client_calls, 0)
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
        self.assertEqual(self.service.training.sampling_client_calls, 0)

    def test_validation_scores_current_policy_without_optimizer(self) -> None:
        trainer = self.trainer()
        rows = [{"messages": "P", "chosen": "A", "rejected": "B"}]

        with patch(
            "triotune.trainer.dpo_trainer.tqdm",
            side_effect=lambda batches, **kwargs: batches,
        ) as progress:
            metrics = trainer.evaluate(rows)

        self.assertAlmostEqual(metrics["eval/loss"], math.log(2), places=3)
        self.assertEqual(metrics["eval/pairs"], 1.0)
        self.assertEqual(self.service.training.sampling_client_calls, 1)
        self.assertEqual(self.service.training.optim_calls, 0)
        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(
            progress.call_args.kwargs,
            {"total": 1, "desc": "DPO eval", "unit": "batch"},
        )

        with patch(
            "triotune.trainer.dpo_trainer.tqdm",
            side_effect=lambda batches, **kwargs: batches,
        ) as async_progress:
            asyncio.run(trainer.evaluate_async(rows))
        self.assertEqual(
            async_progress.call_args.kwargs,
            {"total": 1, "desc": "DPO eval", "unit": "batch"},
        )

    def test_eval_dataset_schedule_in_sync_and_async_training(self) -> None:
        rows = [{"messages": "P", "chosen": "A", "rejected": "B"}]
        for async_mode in (False, True):
            with self.subTest(async_mode=async_mode):
                trainer = DPOTrainer(
                    "test-model",
                    args=DPOConfig(
                        max_steps=4,
                        eval_steps=2,
                        eval_before_train=True,
                        eval_after_train=True,
                    ),
                    train_dataset=rows * 4,
                    eval_dataset=rows,
                )
                evaluated = []
                publish = trainer._publish_evaluation

                def track_evaluation(metrics):
                    evaluated.append(trainer.state.global_step)
                    return publish(metrics)

                trainer._publish_evaluation = track_evaluation
                if async_mode:
                    asyncio.run(trainer.train_async())
                else:
                    trainer.train()

                self.assertEqual(evaluated, [0, 2, 4])
                self.assertEqual(trainer.state.global_step, 4)

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
        self.assertEqual(
            {name.split("/", 1)[0] for name in payload},
            {"train", "reward", "time", "usage"},
        )
        self.assertEqual(payload["train/loss"], 0.5)
        self.assertEqual(payload["reward/accuracy"], 1.0)
        self.assertEqual(payload["reward/margin"], 0.3)
        self.assertEqual(payload["reward/chosen"], 0.2)
        self.assertEqual(payload["reward/rejected"], -0.1)
        self.assertEqual(payload["train/learning_rate"], 5e-6)
        self.assertEqual(payload["train/grad_norm"], 0.2)
        self.assertEqual(payload["train/backend_token_count"], 4)
        self.assertEqual(payload["train/step"], 1)
        self.assertEqual(payload["train/pairs"], 1)
        self.assertEqual(payload["usage/train_tokens"], 4)
        self.assertEqual(payload["usage/train_tokens_total"], 4)
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
        self.assertEqual(first["reward/accuracy"], 1.0)
        self.assertEqual(first["train/pairs"], 1.0)
        self.assertEqual(first["train/step"], 2)
        self.assertAlmostEqual(final_interval["train/loss"], 0.6)
        self.assertEqual(final_interval["train/step"], 3)
        self.assertIn("time/train_runtime", summary)

    def test_swanlab_mode_initializes_run_without_sdk_mode_override(self) -> None:
        with patch("triotune.utils.metrics.swanlab.init") as init:
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
        self.assertEqual(self.run.texts[0], ("checkpoint/path", state.checkpoint_path))
        self.assertEqual(self.run.texts[1][0], "sample/text")
        sample = self.run.texts[1][1][0]
        self.assertEqual((sample.caption, sample.content), ("P", "generated answer"))
        self.assertEqual(self.service.training.sampler.requests[0][0], [80])
        self.assertEqual(
            {name.split("/", 1)[0] for name in self.run.logs[-1][0]},
            {"train", "time", "usage"},
        )
        self.assertEqual(state.metrics["data/total_tokens"], 4)
        self.assertEqual(state.metrics["data/total_pairs"], 1)
        self.assertEqual(state.metrics["usage/train_tokens_total"], 4)
        self.assertEqual(state.metrics["usage/sample_input_tokens_total"], 1)
        self.assertEqual(state.metrics["usage/sample_output_tokens_total"], 2)
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
            {name.split("/", 1)[0] for name in self.run.logs[0][0]},
            {"train", "reward", "time", "usage"},
        )

    def test_async_train_saves_final_checkpoint(self) -> None:
        trainer = self.trainer(logging_step=10)
        trainer.train_dataset = [{"messages": "P", "chosen": "A", "rejected": "B"}]
        state = asyncio.run(trainer.train_async())
        self.assertEqual(state.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.run.finished, "success")
        self.assertEqual(state.metrics["data/total_tokens"], 4)
        self.assertEqual([step for _, step in self.run.logs], [1, 1])
        self.assertEqual(self.service.training.async_sampling_client_calls, 1)
        self.assertEqual(self.service.training.sampler.async_calls, 1)
        self.assertEqual(self.run.texts[1][0], "sample/text")

    def test_final_samples_cap_at_four_and_decode_missing_text(self) -> None:
        trainer = self.trainer(max_length=4)
        trainer.state.global_step = 5
        self.service.training.sampler.text = None
        trainer._log_final_samples([{"messages": f"P{i}"} for i in range(5)])

        requests = self.service.training.sampler.requests
        self.assertEqual(len(requests), 4)
        self.assertEqual([tokens for tokens, _ in requests], [[80, ord(str(i))] for i in range(4)])
        self.assertTrue(all(params.max_tokens == 2 for _, params in requests))
        texts = self.run.texts[0][1]
        self.assertEqual([(item.caption, item.content) for item in texts], [(f"P{i}", "AB") for i in range(4)])

    def test_final_samples_use_configured_chat_template(self) -> None:
        trainer = self.trainer(chat_template_kwargs={"enable_thinking": False})
        tokenizer = _ChatTokenizer()
        trainer._tokenizer = tokenizer
        trainer.state.global_step = 1

        trainer._log_final_samples([{"messages": [{"role": "user", "content": "P"}]}])

        self.assertEqual(tokenizer.thinking_values, [False, False])
        self.assertEqual(self.run.texts[0][1][0].caption, "user:P;assistant:")
        self.assertEqual(
            self.service.training.sampler.requests[0][0], tokenizer.encode("user:P;assistant:")
        )

    def test_final_samples_skip_empty_rows_and_reject_empty_response(self) -> None:
        trainer = self.trainer()
        trainer.train_dataset = []
        trainer.train()
        self.assertEqual(self.service.training.sampling_client_calls, 0)
        self.assertEqual(self.run.texts, [])

        trainer = self.trainer()
        self.service.training.sampler.sequences = []
        with self.assertRaisesRegex(ValueError, "no sequences"):
            trainer._log_final_samples([{"messages": "P"}])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "DPO extra is not installed")
    def test_pairwise_loss_and_completion_gradients(self) -> None:
        import torch

        datums = [
            build_dpo_datum([10, 11], [12]),
            build_dpo_datum([10, 11], [13]),
        ]
        chosen = torch.tensor([0.0, -0.5], requires_grad=True)
        rejected = torch.tensor([0.0, -1.0], requires_grad=True)
        loss, metrics = _loss_fn([-0.5, -1.0], 0.1)(datums, [chosen, rejected])
        loss.backward()
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertAlmostEqual(chosen.grad[1].item(), -0.05, places=6)
        self.assertAlmostEqual(rejected.grad[1].item(), 0.05, places=6)
        self.assertEqual(chosen.grad[0].item(), 0.0)
        self.assertEqual(metrics["reward/margin"], 0.0)

    def test_datum_masks_prompt_and_shifts_completion(self) -> None:
        datum = build_dpo_datum([10, 11], [12, 13])
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
