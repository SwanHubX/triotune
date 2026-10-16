"""Cross-cutting trainer utilities preserve metrics and checkpoint behavior."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from trio_cookbook.utils.checkpoint import (
    checkpoint_path,
    save_checkpoint,
    save_checkpoint_async,
)
from trio_cookbook.utils.dataloader import iter_batches
from trio_cookbook.utils.datums import causal_datum, materialize_datum
from trio_cookbook.utils.metrics import (
    accumulate_metrics,
    averaged_metrics,
    finish_run,
    start_run,
)
from trio_cookbook.utils.metrics import MetricsAPI, normalize_metrics
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
        args = GRPOConfig(
            report_to="swanlab",
            run_name="custom",
            tags=["grpo"],
            workspace="team",
        )
        with patch("trio_cookbook.utils.metrics.swanlab.init") as init:
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

    def test_metrics_api_normalizes_and_averages_arbitrary_metrics(self):
        run = Mock()
        api = MetricsAPI(run, logging_step=2)

        api.record(
            {
                "loss_mean": 2.0,
                "optim/lr": 1e-5,
                "reward/score": 0.5,
                "ignored": float("nan"),
                "flag": True,
            },
            step=1,
        )
        api.record(
            {"loss_mean": 4.0, "reward/score": 1.5},
            step=2,
        )

        run.log.assert_called_once_with(
            {
                "train/loss_mean": 3.0,
                "train/learning_rate": 1e-5,
                "reward/score": 1.0,
                "train/step": 2.0,
            },
            step=2,
        )

    def test_metrics_api_publishes_eval_and_finishes_once(self):
        run = Mock()
        api = MetricsAPI(run)

        self.assertEqual(
            api.publish({"eval/loss": 0.25}, step=3),
            {"eval/loss": 0.25},
        )
        api.log_text(key="checkpoint/path", data="trio://saved", step=3)
        api.finish()
        api.finish()

        run.log.assert_called_once_with(
            {"eval/loss": 0.25, "train/step": 3.0},
            step=3,
        )
        run.log_text.assert_called_once_with(
            key="checkpoint/path",
            data="trio://saved",
            step=3,
        )
        run.finish.assert_called_once_with(state="success", error=None)

    def test_normalize_metrics_supports_algorithm_namespaces(self):
        self.assertEqual(
            normalize_metrics(
                {
                    "train/loss": 1,
                    "eval/loss": 2,
                    "policy/entropy": 3,
                    "distill/reverse_kl": 4,
                    "grad_norm": 5,
                    "text": "not-a-scalar",
                }
            ),
            {
                "train/loss": 1.0,
                "eval/loss": 2.0,
                "policy/entropy": 3.0,
                "distill/reverse_kl": 4.0,
                "train/grad_norm": 5.0,
            },
        )

    def test_iter_batches_is_lazy_and_limits_validation_batches(self):
        rows = (index for index in range(5))
        self.assertEqual(list(iter_batches(rows, 2)), [[0, 1], [2, 3], [4]])
        self.assertEqual(
            list(iter_batches(range(10), 3, max_batches=2)),
            [[0, 1, 2], [3, 4, 5]],
        )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            list(iter_batches([], 0))

    def test_generic_datum_helpers_preserve_algorithm_loss_inputs(self):
        calls = []

        def model_input(tokens):
            calls.append(("model", list(tokens)))
            return ("model", list(tokens))

        def datum_factory(*, model_input, loss_fn_inputs):
            calls.append(("datum", model_input, loss_fn_inputs))
            return {"model_input": model_input, "loss_fn_inputs": loss_fn_inputs}

        datum = causal_datum(
            [10, 11, 12],
            [-100, 12],
            model_input_factory=model_input,
            datum_factory=datum_factory,
        )
        self.assertEqual(datum["model_input"], ("model", [10, 11]))
        self.assertEqual(datum["loss_fn_inputs"], {"target_tokens": [-100, 12]})
        self.assertEqual(calls[0], ("model", [10, 11]))

        generic = materialize_datum(
            [1, 2],
            {"target_tokens": [2], "advantages": [0.5]},
            model_input_factory=model_input,
            datum_factory=datum_factory,
        )
        self.assertEqual(
            generic["loss_fn_inputs"],
            {"target_tokens": [2], "advantages": [0.5]},
        )


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
