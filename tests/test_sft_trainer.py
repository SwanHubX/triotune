from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from trio_cookbook.recipes import sft_huanhuan, sft_huanhuan_trio
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
        enable_thinking: bool = False,
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


class _Future:
    def __init__(self, value: object) -> None:
        self.value = value

    async def _resolve(self) -> object:
        await asyncio.sleep(0)
        return self.value

    def result(self) -> object:
        raise AssertionError("The async trainer must not block on result().")

    def __await__(self):
        return self._resolve().__await__()


class _Client:
    def __init__(self) -> None:
        self.datums: list[object] = []
        self.optim_calls = 0
        self.fail_forward = False
        self.saved: list[str] = []
        self.events: list[str] = []

    def get_tokenizer(self) -> _Tokenizer:
        return _Tokenizer()

    def forward_backward(self, datums: list[object], **kwargs: object) -> object:
        self.events.append("forward")
        self.datums = datums
        assert kwargs == {"loss_fn": "cross_entropy", "auto_shift": False}
        if self.fail_forward:
            raise RuntimeError("forward failed")
        return SimpleNamespace(metrics={"loss_mean": 0.5, "token_count": 2})

    def optim_step(self, params: object) -> object:
        self.events.append("optim")
        self.optim_calls += 1
        return SimpleNamespace(metrics={"grad_norm": 0.1})

    def save_state(self, *, name: str) -> object:
        self.events.append(name)
        self.saved.append(name)
        return SimpleNamespace(path=f"checkpoint://{name}")

    async def forward_backward_async(self, datums: list[object], **kwargs: object) -> object:
        return _Future(self.forward_backward(datums, **kwargs))

    async def optim_step_async(self, params: object) -> object:
        return _Future(self.optim_step(params))

    async def save_state_async(self, *, name: str) -> object:
        return _Future(self.save_state(name=name))


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


class SFTTrainerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = _Client()
        self.run = _Run()
        service = patch("trio_cookbook.trainer.sft.trainer.trio.ServiceClient")
        swanlab = patch("trio_cookbook.trainer.sft.trainer.swanlab.init", return_value=self.run)
        self.service = service.start()
        swanlab.start()
        self.addCleanup(service.stop)
        self.addCleanup(swanlab.stop)
        self.service.return_value.create_lora_training_client.return_value = self.client

    def trainer(self, **kwargs: object) -> SFTTrainer:
        return SFTTrainer("test-model", args=SFTConfig(**kwargs))

    async def test_optimizer_submitted_before_forward_result(self) -> None:
        trainer = self.trainer()
        submitted, release = asyncio.Event(), asyncio.Event()

        class PendingFuture(_Future):
            async def _resolve(self) -> object:
                submitted.set()
                await release.wait()
                return self.value

        async def forward(datums: list[object], **kwargs: object) -> object:
            return PendingFuture(SimpleNamespace(metrics={"loss_mean": 0.5}))

        with patch.object(self.client, "forward_backward_async", side_effect=forward):
            task = asyncio.create_task(trainer.step([{"text": "ab"}]))
            try:
                await asyncio.wait_for(submitted.wait(), timeout=1)
                self.assertEqual(self.client.optim_calls, 1)
                self.assertEqual(trainer.state.global_step, 0)
                release.set()
                await asyncio.wait_for(task, timeout=1)
                self.assertEqual(self.client.optim_calls, 1)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_pipeline_submits_batches_and_checkpoints_in_order(self) -> None:
        trainer = self.trainer(report_to="swanlab", max_steps=3, save_every=2)
        trainer.train_dataset = [{"text": "ab"}] * 3
        release, all_submitted = asyncio.Event(), asyncio.Event()
        count = 0

        class PendingFuture(_Future):
            async def _resolve(self) -> object:
                await release.wait()
                return self.value

        async def forward(datums: list[object], **kwargs: object) -> object:
            nonlocal count
            result = self.client.forward_backward(datums, **kwargs)
            count += 1
            if count == 3:
                all_submitted.set()
            return PendingFuture(result) if count == 1 else _Future(result)

        with patch.object(self.client, "forward_backward_async", side_effect=forward):
            task = asyncio.create_task(trainer.train())
            try:
                await asyncio.wait_for(all_submitted.wait(), 1)
                self.assertEqual(self.client.events, [
                    "forward", "optim", "forward", "optim", "sft-step-00000002", "forward", "optim",
                ])
                self.assertEqual(self.run.logs, [])
                self.assertEqual(trainer.state.global_step, 0)
                release.set()
                state = await asyncio.wait_for(task, 1)
                self.assertEqual(state.global_step, 3)
                self.assertEqual([step for _, step in self.run.logs], [1, 2, 3])
                self.assertEqual(state.checkpoint_path, "checkpoint://sft-step-00000003")
                total = sum(values["time/step"] for values, _ in self.run.logs)
                self.assertAlmostEqual(total, state.metrics["time/train"])
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_pipeline_limits_outstanding_batches(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_in_flight"):
            SFTConfig(max_in_flight=0)
        trainer = self.trainer(max_steps=4, max_in_flight=2)
        trainer.train_dataset = [{"text": "ab"}] * 4
        submitted, release = asyncio.Event(), asyncio.Event()
        count = 0

        class PendingFuture(_Future):
            async def _resolve(self) -> object:
                await release.wait()
                return self.value

        async def forward(datums: list[object], **kwargs: object) -> object:
            nonlocal count
            count += 1
            if count == 2:
                submitted.set()
            return PendingFuture(self.client.forward_backward(datums, **kwargs))

        with patch.object(self.client, "forward_backward_async", side_effect=forward):
            task = asyncio.create_task(trainer.train())
            try:
                await asyncio.wait_for(submitted.wait(), 1)
                await asyncio.sleep(0)
                self.assertEqual(count, 2)
                release.set()
                self.assertEqual((await asyncio.wait_for(task, 1)).global_step, 4)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_remote_failure_reclaims_background_tasks(self) -> None:
        trainer = self.trainer(report_to="swanlab", max_steps=3)
        trainer.train_dataset = [{"text": "ab"}] * 3
        tasks = []
        create_task = asyncio.create_task

        class FailedFuture(_Future):
            async def _resolve(self) -> object:
                await asyncio.sleep(0)
                raise RuntimeError("remote forward failed")

        def tracked_task(coro):
            task = create_task(coro)
            tasks.append(task)
            return task

        with (
            patch.object(self.client, "forward_backward_async", return_value=FailedFuture(None)),
            patch("trio_cookbook.trainer.sft.trainer.asyncio.create_task", side_effect=tracked_task),
        ):
            with self.assertRaisesRegex(RuntimeError, "remote forward failed"):
                await trainer.train()
        self.assertTrue(tasks)
        self.assertTrue(all(task.done() for task in tasks))
        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(self.client.saved, [])
        self.assertTrue(self.run.finished)

    async def test_swanlab_options_and_disabled_reporting(self) -> None:
        with patch("trio_cookbook.trainer.sft.trainer.swanlab.init") as init:
            self.trainer()
            init.assert_not_called()
            self.trainer(report_to="swanlab")
            defaults = init.call_args.kwargs
            for key in ("workspace", "tags", "log_dir"):
                self.assertIsNone(defaults[key])
            self.assertEqual(defaults["name"], "test-model-sft")

            self.trainer(
                report_to="swanlab", workspace="team", tags=["sft", "huanhuan"],
                run_name="custom-run", log_dir="/tmp/sft-logs",
            )
            custom = init.call_args.kwargs
            self.assertEqual(custom["workspace"], "team")
            self.assertEqual(custom["tags"], ["sft", "huanhuan"])
            self.assertEqual(custom["name"], "custom-run")
            self.assertEqual(custom["log_dir"], "/tmp/sft-logs")

    async def test_text_and_completion_labels_are_shifted(self) -> None:
        trainer = self.trainer()
        metrics = await trainer.step([{"text": "ab"}, {"prompt": "P", "completion": "A"}])
        self.assertEqual([datum.model_input.to_ints() for datum in self.client.datums],
                         [[97, 98], [80, 65]])
        self.assertEqual([datum.loss_fn_inputs["target_tokens"].data for datum in self.client.datums],
                         [[98, 255], [65, 255]])
        self.assertEqual(metrics["train/tokens"], 4)
        self.assertEqual(metrics["train/loss"], 0.5)
        self.assertEqual(self.client.optim_calls, 1)

    async def test_assistant_mask_and_pretokenized_labels(self) -> None:
        trainer = self.trainer(assistant_only_loss=True)
        await trainer.step([{"messages": [
            {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"},
        ]}])
        labels = self.client.datums[0].loss_fn_inputs["target_tokens"].data
        self.assertEqual(labels, [-100] * (len("user:Q;") - 1) + [ord(c) for c in "assistant:A;"])

        trainer = self.trainer()
        await trainer.step([{"input_ids": [10, 11, 12], "labels": [-100, -100, 12]}])
        self.assertEqual(self.client.datums[0].loss_fn_inputs["target_tokens"].data, [-100, 12])

    async def test_chat_completion_only_masks_prompt(self) -> None:
        trainer = self.trainer()
        await trainer.step([{
            "prompt": [{"role": "user", "content": "Q"}],
            "completion": [{"role": "assistant", "content": "A"}],
        }])
        labels = self.client.datums[0].loss_fn_inputs["target_tokens"].data
        self.assertEqual(labels[-2:], [ord("A"), ord(";")])
        self.assertTrue(all(token == -100 for token in labels[:-2]))

    async def test_invalid_masks_truncation_and_forward_failure(self) -> None:
        trainer = self.trainer(max_length=2)
        with self.assertRaisesRegex(ValueError, "no next-token labels"):
            await trainer.step([{"prompt": "PP", "completion": "A"}])
        with self.assertRaisesRegex(ValueError, "require labels"):
            await self.trainer(completion_only_loss=True).step([{"input_ids": [10, 11]}])
        with self.assertRaisesRegex(ValueError, "same length"):
            await self.trainer().step([{"input_ids": [10, 11], "labels": [10]}])
        self.client.get_tokenizer = lambda: _NoMaskTokenizer()
        with self.assertRaisesRegex(ValueError, "assistant token mask"):
            await self.trainer(assistant_only_loss=True).step([{"messages": [
                {"role": "assistant", "content": "A"},
            ]}])
        self.assertEqual(self.client.optim_calls, 0)

        self.client.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            await self.trainer().step([{"text": "ab"}])
        self.assertEqual(self.client.optim_calls, 0)

    async def test_train_saves_and_logs_checkpoint(self) -> None:
        trainer = self.trainer(report_to="swanlab", save_every=2, batch_size=1, max_steps=3)
        trainer.train_dataset = [{"text": "ab"}] * 3
        state = await trainer.train()
        self.assertEqual(state.global_step, 3)
        self.assertEqual(self.client.saved, ["sft-step-00000002", "sft-step-00000003"])
        self.assertEqual(self.run.text, ("checkpoint/path", state.checkpoint_path, 3))
        self.assertEqual(len(self.run.logs), 3)
        self.assertTrue(self.run.finished)

        self.client.saved.clear()
        trainer = self.trainer(max_steps=1)
        trainer.train_dataset = [{"text": "ab"}]
        self.assertEqual((await trainer.train()).checkpoint_path, "checkpoint://sft-step-00000001")
        self.assertEqual(self.client.saved, ["sft-step-00000001"])


class SFTRecipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_recipe_awaits_sdk_futures(self) -> None:
        class SDKTokenizer(_Tokenizer):
            def apply_chat_template(self, *args: object, **kwargs: object) -> object:
                output = super().apply_chat_template(*args, **kwargs)
                return {"input_ids": output} if isinstance(output, list) else output

        release, submitted = asyncio.Event(), asyncio.Event()

        class PendingFuture(_Future):
            async def _resolve(self) -> object:
                await release.wait()
                return self.value

        class QueuedClient(_Client):
            async def forward_backward_async(self, datums: list[object], **kwargs: object) -> object:
                result = self.forward_backward(datums, **kwargs)
                return PendingFuture(result) if self.events.count("forward") == 1 else _Future(result)

            async def optim_step_async(self, params: object) -> object:
                result = self.optim_step(params)
                if self.optim_calls == 3:
                    submitted.set()
                return _Future(result)

        client = QueuedClient()
        client.get_tokenizer = lambda: SDKTokenizer()
        run = _Run()
        with (
            patch("pathlib.Path.read_text", return_value=json.dumps([
                {"instruction": "Q", "input": "", "output": "A"},
            ] * 3)),
            patch.object(sft_huanhuan.trio, "ServiceClient") as service,
            patch.object(sft_huanhuan.swanlab, "init", return_value=run),
            patch("sys.argv", ["sft_huanhuan.py", "--steps", "3", "--batch-size", "1", "--save-every", "2", "--swanlab"]),
            patch("builtins.print"),
        ):
            service.return_value.create_lora_training_client_async = AsyncMock(return_value=client)
            task = asyncio.create_task(sft_huanhuan.main())
            try:
                await asyncio.wait_for(submitted.wait(), 1)
                self.assertEqual(run.logs, [])
                self.assertEqual(client.events, [
                    "forward", "optim", "forward", "optim", "sft-step-00000002", "forward", "optim",
                ])
                release.set()
                await asyncio.wait_for(task, 1)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(client.optim_calls, 3)
        self.assertEqual(client.saved, ["sft-step-00000002", "sft-step-00000003"])
        self.assertEqual([step for _, step in run.logs], [1, 2, 3])
        self.assertTrue(run.finished)

    async def test_config_defaults_and_cli_overrides(self) -> None:
        sample = '[{"instruction":"Q","input":"","output":"A"}]'
        with (
            patch("pathlib.Path.read_text", return_value=sample),
            patch.object(sft_huanhuan_trio, "SFTTrainer") as trainer_class,
            patch("builtins.print"),
        ):
            trainer_class.return_value.train = AsyncMock(return_value=SimpleNamespace(
                global_step=0, checkpoint_path=None, metrics={}
            ))
            with patch("sys.argv", ["sft_huanhuan_trio.py"]):
                await sft_huanhuan_trio.main()
            defaults = trainer_class.call_args.kwargs
            self.assertEqual(defaults["model"], "Qwen/Qwen3.5-4B")
            self.assertEqual(defaults["lora_rank"], 8)
            self.assertEqual(defaults["args"].max_steps, 100)
            self.assertEqual(defaults["args"].max_in_flight, 32)
            self.assertEqual(defaults["args"].report_to, "swanlab")
            self.assertEqual(list(defaults["train_dataset"])[0]["completion"][0]["content"], "A")

            with patch("sys.argv", [
                "sft_huanhuan_trio.py", "--steps", "2", "--batch-size", "2", "--max-in-flight", "2", "--no-swanlab"
            ]):
                await sft_huanhuan_trio.main()
            overridden = trainer_class.call_args.kwargs["args"]
            self.assertEqual((overridden.max_steps, overridden.batch_size), (2, 2))
            self.assertEqual(overridden.max_in_flight, 2)
            self.assertIsNone(overridden.report_to)

            settings = {
                "model": {"name": "test-model", "lora_rank": 8},
                "data": {"file": "data/huanhuan.json", "epochs": 1},
                "sft": {"batch_size": 1, "max_steps": 1},
                "swanlab": {
                    "report_to": "swanlab", "workspace": "team", "tags": ["sft"],
                    "run_name": "toml-run", "log_dir": "/tmp/sft-logs",
                },
            }
            with (
                patch.object(sft_huanhuan_trio.tomllib, "load", return_value=settings),
                patch("sys.argv", ["sft_huanhuan_trio.py"]),
            ):
                await sft_huanhuan_trio.main()
            configured = trainer_class.call_args.kwargs["args"]
            self.assertEqual(configured.workspace, "team")
            self.assertEqual(configured.tags, ["sft"])
            self.assertEqual(configured.run_name, "toml-run")
            self.assertEqual(configured.log_dir, "/tmp/sft-logs")


if __name__ == "__main__":
    unittest.main()
