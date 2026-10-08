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
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str | list[int]:
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
        return SimpleNamespace(metrics={"train/loss": 0.5, "rewards/accuracies": 1.0})

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
        return DPOTrainer("test-model", args=DPOConfig(**config))

    def test_step_aligns_pair_and_scores_only_completion(self) -> None:
        trainer = self.trainer()
        result = trainer.step([{"prompt": "P", "chosen": "A", "rejected": "B"}])

        self.assertEqual(result.step, 1)
        self.assertEqual(result.num_pairs, 1)
        self.assertEqual(self.service.reference.requests, [[80, 65, 255], [80, 66, 255]])
        self.assertEqual([datum.model_input.to_ints() for datum in self.service.training.datums], [[80, 65], [80, 66]])
        self.assertEqual([datum.loss_fn_inputs["target_tokens"].data for datum in self.service.training.datums], [[65, 255], [66, 255]])
        self.assertEqual(self.service.training.optim_calls, 1)
        self.assertEqual(self.run.logs[0][1], 1)

    def test_forward_failure_does_not_optimize_or_advance_step(self) -> None:
        trainer = self.trainer()
        self.service.training.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            trainer.step([{"prompt": "P", "chosen": "A", "rejected": "B"}])
        self.assertEqual(self.service.training.optim_calls, 0)
        self.assertEqual(trainer.state.global_step, 0)

    def test_rejects_missing_fields_and_overlong_pairs(self) -> None:
        trainer = self.trainer(max_length=2)
        with self.assertRaises(KeyError):
            trainer.step([{"prompt": "P", "chosen": "A"}])
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            trainer.step([{"prompt": "P", "chosen": "A", "rejected": "B"}])
        self.assertEqual(self.service.training.optim_calls, 0)

    def test_train_saves_final_checkpoint_and_finishes_swanlab(self) -> None:
        trainer = self.trainer()
        trainer.train_dataset = [{"prompt": "P", "chosen": "A", "rejected": "B"}]
        state = trainer.train()
        self.assertEqual(state.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.service.training.save_calls, 1)
        self.assertEqual(self.run.finished, "success")
        self.assertEqual(self.run.texts, [("checkpoint/path", state.checkpoint_path)])

    def test_async_step_uses_async_clients(self) -> None:
        trainer = self.trainer(save_every=1)
        result = asyncio.run(trainer.step_async([{"prompt": "P", "chosen": "A", "rejected": "B"}]))
        self.assertEqual(result.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.service.training.optim_calls, 1)

    def test_async_train_saves_final_checkpoint(self) -> None:
        trainer = self.trainer()
        trainer.train_dataset = [{"prompt": "P", "chosen": "A", "rejected": "B"}]
        state = asyncio.run(trainer.train_async())
        self.assertEqual(state.checkpoint_path, "checkpoint://dpo-step-00000001")
        self.assertEqual(self.run.finished, "success")

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
        trainer = self.trainer()
        trainer._tokenizer = _ChatTokenizer()
        pairs = trainer._preferences([{
            "prompt": [{"role": "user", "content": "P"}],
            "chosen": [{"role": "assistant", "content": "A"}],
            "rejected": [{"role": "assistant", "content": "B"}],
        }])
        self.assertEqual(pairs[0].chosen, [ord("A"), ord(";"), 255])
        self.assertEqual(pairs[0].rejected, [ord("B"), ord(";"), 255])


if __name__ == "__main__":
    unittest.main()
