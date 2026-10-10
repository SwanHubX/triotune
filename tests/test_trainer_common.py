"""Shared support must preserve metric windows and SDK future handling."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from trio_cookbook.trainer.common.checkpoint import (
    checkpoint_path,
    save_checkpoint,
    save_checkpoint_async,
)
from trio_cookbook.trainer.common.logging import (
    accumulate_metrics,
    averaged_metrics,
    finish_run,
    start_run,
)
from trio_cookbook.trainer.grpo import GRPOConfig


class LoggingTests(unittest.TestCase):
    def test_sparse_metrics_have_independent_counts_and_windows(self):
        window = {}
        accumulate_metrics(window, {"loss": 2.0, "grad_norm": 4.0})
        accumulate_metrics(window, {"loss": 6.0})
        self.assertEqual(averaged_metrics(window), {"loss": 4.0, "grad_norm": 4.0})
        self.assertEqual(window["loss"], (8.0, 2))
        window.clear()
        self.assertEqual(averaged_metrics(window), {})
        accumulate_metrics(window, {"loss": 10.0})
        self.assertEqual(averaged_metrics(window), {"loss": 10.0})

    def test_run_initialization_retains_config_and_metadata(self):
        args = GRPOConfig(run_name="custom", tags=["grpo"], workspace="team")
        with patch("trio_cookbook.trainer.common.logging.swanlab.init") as init:
            self.assertIs(start_run(args, default_name="fallback", metadata={"model": "test"}), init.return_value)
        self.assertEqual(init.call_args.kwargs["name"], "custom")
        self.assertEqual(init.call_args.kwargs["workspace"], "team")
        self.assertEqual(init.call_args.kwargs["config"]["model"], "test")
        self.assertEqual(init.call_args.kwargs["config"]["num_generations"], args.num_generations)
        self.assertEqual(init.call_args.kwargs["tags"], ["grpo"])

    def test_finish_disabled_and_failed_runs(self):
        finish_run(None, state="success")
        run = Mock()
        finish_run(run, state="crashed", error="remote failure")
        run.finish.assert_called_once_with(state="crashed", error="remote failure")


class CheckpointTests(unittest.TestCase):
    def test_sync_save_resolves_sdk_future_and_preserves_name(self):
        future = SimpleNamespace(result=lambda: SimpleNamespace(path="trio://saved"))
        client = SimpleNamespace(save_state=Mock(return_value=future))
        self.assertEqual(save_checkpoint(client, "dpo", 12), "trio://saved")
        client.save_state.assert_called_once_with(name="dpo-step-00000012")

    def test_invalid_checkpoint_paths_are_rejected(self):
        for value in (None, "", 7, False):
            with self.subTest(path=value), self.assertRaisesRegex(TypeError, "save_state_async"):
                checkpoint_path(SimpleNamespace(path=value), method="save_state_async")
        with self.assertRaisesRegex(TypeError, "no checkpoint path"):
            checkpoint_path(SimpleNamespace())

    def test_sync_remote_error_propagates(self):
        client = SimpleNamespace(save_state=Mock(side_effect=RuntimeError("save failed")))
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            save_checkpoint(client, "grpo", 1)


class AsyncCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_save_resolves_both_sdk_stages(self):
        class Future:
            def result(self):
                raise AssertionError("Async saves must not call blocking result().")

            def __await__(self):
                async def resolve():
                    await asyncio.sleep(0)
                    return SimpleNamespace(path="trio://async")
                return resolve().__await__()

        client = SimpleNamespace(save_state_async=AsyncMock(return_value=Future()))
        self.assertEqual(await save_checkpoint_async(client, "opd", 2), "trio://async")
        client.save_state_async.assert_awaited_once_with(name="opd-step-00000002")

    async def test_missing_path_and_remote_error_propagate(self):
        client = SimpleNamespace(save_state_async=AsyncMock(return_value=SimpleNamespace(path="")))
        with self.assertRaisesRegex(TypeError, "save_state_async"):
            await save_checkpoint_async(client, "grpo", 1)
        client.save_state_async.side_effect = RuntimeError("remote failure")
        with self.assertRaisesRegex(RuntimeError, "remote failure"):
            await save_checkpoint_async(client, "grpo", 1)

    async def test_cancelled_save_drains_the_request(self):
        entered, released = asyncio.Event(), asyncio.Event()

        async def save_state_async(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        task = asyncio.create_task(save_checkpoint_async(
            SimpleNamespace(save_state_async=save_state_async), "opd", 1,
        ))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(released.is_set())
