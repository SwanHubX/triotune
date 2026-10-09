from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trio_cookbook.recipes import sft_huanhuan_trio
from trio_cookbook.trainer import SFTConfig, SFTTrainer


class _Tokenizer:
    eos_token_id = 255

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
    ) -> object:
        segments = [f"{message['role']}:{message['content']};" for message in messages]
        text = "".join(segments) + ("assistant:" if add_generation_prompt else "")
        if not tokenize:
            return text
        tokens = self.encode(text)
        if not return_dict:
            return tokens
        mask = [bit for message, segment in zip(messages, segments, strict=True)
                for bit in [int(message["role"] == "assistant")] * len(segment)]
        return {"input_ids": tokens, "assistant_masks": mask}


class _NoMaskTokenizer(_Tokenizer):
    def apply_chat_template(self, *args: object, **kwargs: object) -> object:
        output = super().apply_chat_template(*args, **kwargs)
        if isinstance(output, dict):
            output.pop("assistant_masks")
        return output


class _Client:
    def __init__(self) -> None:
        self.datums: list[object] = []
        self.optim_calls = 0
        self.fail_forward = False
        self.saved: list[str] = []

    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()

    def forward_backward(self, datums: list[object], **kwargs: object) -> object:
        self.datums = datums
        assert kwargs == {"loss_fn": "cross_entropy", "auto_shift": False}
        if self.fail_forward:
            raise RuntimeError("forward failed")
        return SimpleNamespace(metrics={"loss_mean": 0.5, "token_count": 2})

    def optim_step(self, params: object) -> object:
        self.optim_calls += 1
        return SimpleNamespace(metrics={"grad_norm": 0.1})

    def save_state(self, *, name: str) -> object:
        self.saved.append(name)
        return SimpleNamespace(path=f"checkpoint://{name}")


class _Run:
    def __init__(self) -> None:
        self.logs: list[object] = []
        self.text: tuple[str, str, int] | None = None
        self.finished = False

    def log(self, values: object, *, step: int) -> None:
        self.logs.append((values, step))

    def log_text(self, *, key: str, data: str, step: int) -> None:
        self.text = (key, data, step)

    def finish(self, **kwargs: object) -> None:
        self.finished = True


class SFTTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _Client()
        self.run = _Run()
        service = patch("trio_cookbook.trainer.sft_trainer.trio.ServiceClient")
        swanlab = patch("trio_cookbook.trainer.sft_trainer.swanlab.init", return_value=self.run)
        self.service = service.start()
        swanlab.start()
        self.addCleanup(service.stop)
        self.addCleanup(swanlab.stop)
        self.service.return_value.create_lora_training_client.return_value = self.client

    def trainer(self, **kwargs: object) -> SFTTrainer:
        return SFTTrainer("test-model", args=SFTConfig(**kwargs))

    def test_text_and_completion_labels_are_shifted(self) -> None:
        trainer = self.trainer()
        metrics = trainer.step([{"text": "ab"}, {"prompt": "P", "completion": "A"}])
        self.assertEqual([datum.model_input.to_ints() for datum in self.client.datums],
                         [[97, 98], [80, 65]])
        self.assertEqual([datum.loss_fn_inputs["target_tokens"].data for datum in self.client.datums],
                         [[98, 255], [65, 255]])
        self.assertEqual(metrics["train/tokens"], 4)
        self.assertEqual(metrics["train/loss"], 0.5)
        self.assertEqual(self.client.optim_calls, 1)

    def test_assistant_mask_and_pretokenized_labels(self) -> None:
        trainer = self.trainer(assistant_only_loss=True)
        trainer.step([{"messages": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"},
        ]}])
        labels = self.client.datums[0].loss_fn_inputs["target_tokens"].data
        self.assertEqual(labels, [-100] * (len("user:Q;") - 1) + [ord(c) for c in "assistant:A;"])

        trainer = self.trainer()
        trainer.step([{"input_ids": [10, 11, 12], "labels": [-100, -100, 12]}])
        self.assertEqual(self.client.datums[0].loss_fn_inputs["target_tokens"].data, [-100, 12])

    def test_chat_completion_only_masks_prompt(self) -> None:
        trainer = self.trainer()
        trainer.step([{
            "prompt": [{"role": "user", "content": "Q"}],
            "completion": [{"role": "assistant", "content": "A"}],
        }])
        labels = self.client.datums[0].loss_fn_inputs["target_tokens"].data
        self.assertEqual(labels[-2:], [ord("A"), ord(";")])
        self.assertTrue(all(token == -100 for token in labels[:-2]))

    def test_invalid_masks_truncation_and_forward_failure(self) -> None:
        trainer = self.trainer(max_length=2)
        with self.assertRaisesRegex(ValueError, "no next-token labels"):
            trainer.step([{"prompt": "PP", "completion": "A"}])
        with self.assertRaisesRegex(ValueError, "require labels"):
            self.trainer(completion_only_loss=True).step([{"input_ids": [10, 11]}])
        with self.assertRaisesRegex(ValueError, "same length"):
            self.trainer().step([{"input_ids": [10, 11], "labels": [10]}])
        self.client.get_tokenizer = lambda: _NoMaskTokenizer()
        with self.assertRaisesRegex(ValueError, "assistant token mask"):
            self.trainer(assistant_only_loss=True).step([{"messages": [
                {"role": "assistant", "content": "A"},
            ]}])
        self.assertEqual(self.client.optim_calls, 0)

        self.client.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.trainer().step([{"text": "ab"}])
        self.assertEqual(self.client.optim_calls, 0)

    def test_train_saves_and_logs_checkpoint(self) -> None:
        trainer = self.trainer(report_to="swanlab", save_every=2, batch_size=1, max_steps=3)
        trainer.train_dataset = [{"text": "ab"}] * 3
        state = trainer.train()
        self.assertEqual(state.global_step, 3)
        self.assertEqual(self.client.saved, ["sft-step-00000002", "sft-step-00000003"])
        self.assertEqual(self.run.text, ("checkpoint/path", state.checkpoint_path, 3))
        self.assertEqual(len(self.run.logs), 3)
        self.assertTrue(self.run.finished)

        self.client.saved.clear()
        trainer = self.trainer(max_steps=1)
        trainer.train_dataset = [{"text": "ab"}]
        self.assertEqual(trainer.train().checkpoint_path, "checkpoint://sft-step-00000001")
        self.assertEqual(self.client.saved, ["sft-step-00000001"])


class SFTRecipeTests(unittest.TestCase):
    def test_config_defaults_and_cli_overrides(self) -> None:
        sample = '[{"instruction":"Q","input":"","output":"A"}]'
        with (
            patch("pathlib.Path.read_text", return_value=sample),
            patch.object(sft_huanhuan_trio, "SFTTrainer") as trainer_class,
            patch("builtins.print"),
        ):
            trainer_class.return_value.train.return_value = SimpleNamespace(
                global_step=0, checkpoint_path=None, metrics={}
            )
            with patch("sys.argv", ["sft_huanhuan_trio.py"]):
                sft_huanhuan_trio.main()
            defaults = trainer_class.call_args.kwargs
            self.assertEqual(defaults["model"], "Qwen/Qwen3.5-4B")
            self.assertEqual(defaults["lora_rank"], 8)
            self.assertEqual(defaults["args"].max_steps, 100)
            self.assertEqual(defaults["args"].report_to, "swanlab")
            self.assertEqual(list(defaults["train_dataset"])[0]["completion"][0]["content"], "A")

            with patch("sys.argv", [
                "sft_huanhuan_trio.py", "--steps", "2", "--batch-size", "2", "--no-swanlab"
            ]):
                sft_huanhuan_trio.main()
            overridden = trainer_class.call_args.kwargs["args"]
            self.assertEqual((overridden.max_steps, overridden.batch_size), (2, 2))
            self.assertIsNone(overridden.report_to)


if __name__ == "__main__":
    unittest.main()
