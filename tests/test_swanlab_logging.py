from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trio_cookbook.trainer import GRPOConfig, GRPOResult, GRPOTrainer, TrainerState


class _Run:
    def __init__(self) -> None:
        self.payload: dict[str, float] | None = None
        self.step: int | None = None
        self.logs: list[tuple[dict[str, float], int]] = []
        self.texts: dict[str, tuple[object, int]] = {}
        self.events: list[str] = []

    def log(self, payload: dict[str, float], *, step: int) -> None:
        self.payload = payload
        self.step = step
        self.logs.append((payload, step))

    def log_text(self, *, key: str, data: object, step: int) -> None:
        self.texts[key] = (data, step)
        self.events.append(key)

    def finish(self, **_: str) -> None:
        self.events.append("finish")


class _Sampler:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    def sample(self, *, prompt: object, num_samples: int, sampling_params: object) -> object:
        self.prompts.append(prompt.to_ints())
        assert num_samples == 1
        return SimpleNamespace(sequences=[SimpleNamespace(text="final answer", tokens=[2])])

    async def sample_async(self, **kwargs: object) -> object:
        return self.sample(**kwargs)


def _final_trainer(run: _Run, sampler: _Sampler) -> GRPOTrainer:
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer._swanlab_run = run
    trainer.config = GRPOConfig(max_steps=1, batch_size=5, output_dir=True)
    trainer.state = TrainerState(global_step=1, checkpoint_path="trio://final-checkpoint")
    trainer._total_input_tokens = 0
    trainer._tokenizer = SimpleNamespace(encode=lambda text, **_: [1])
    trainer._decode = lambda tokens: "decoded answer"
    trainer._new_sampling_client = lambda: sampler

    async def new_sampling_client_async() -> _Sampler:
        return sampler

    trainer._new_sampling_client_async = new_sampling_client_async
    return trainer


class SwanLabLoggingTests(unittest.TestCase):
    def test_init_passes_workspace_and_tags_to_swanlab(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.model_name = "Qwen/Qwen3.5-9B"
        trainer.config = GRPOConfig(report_to="swanlab", workspace="my-organization", tags=["grpo", "gsm8k"])

        with patch("trio_cookbook.trainer.grpo_trainer.swanlab.init") as init:
            trainer._init_swanlab_run(lora_rank=32)

        self.assertEqual(init.call_args.kwargs["workspace"], "my-organization")
        self.assertEqual(init.call_args.kwargs["tags"], ["grpo", "gsm8k"])
        self.assertNotIn("mode", init.call_args.kwargs)

    def test_swanlab_workspace_and_tags_default_to_none(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.model_name = "Qwen/Qwen3.5-9B"
        trainer.config = GRPOConfig(report_to="swanlab")

        with patch("trio_cookbook.trainer.grpo_trainer.swanlab.init") as init:
            trainer._init_swanlab_run(lora_rank=32)

        self.assertIsNone(init.call_args.kwargs["workspace"])
        self.assertIsNone(init.call_args.kwargs["tags"])

    def test_none_report_to_uses_terminal_only(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = GRPOConfig()
        with patch("trio_cookbook.trainer.grpo_trainer.swanlab.init") as init:
            self.assertIsNone(trainer._init_swanlab_run(lora_rank=32))
        init.assert_not_called()

    def test_rejects_unsupported_report_targets(self) -> None:
        for target in ("disabled", "online", "local", "offline"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "report_to"):
                GRPOConfig(report_to=target)

    def test_logging_step_must_be_positive_integer(self) -> None:
        self.assertEqual(GRPOConfig().logging_step, 1)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "logging_step"):
                GRPOConfig(logging_step=value)

    def test_rejects_empty_workspace_and_tags(self) -> None:
        with self.assertRaisesRegex(ValueError, "workspace"):
            GRPOConfig(workspace=" ")
        with self.assertRaisesRegex(ValueError, "tags"):
            GRPOConfig(tags=["grpo", ""])
        with self.assertRaisesRegex(ValueError, "tags"):
            GRPOConfig(tags="grpo")
        with self.assertRaisesRegex(TypeError, "output_dir"):
            GRPOConfig(output_dir="checkpoint")

    def test_logs_batch_metrics_without_per_prompt_metrics(self) -> None:
        run = _Run()
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer._swanlab_run = run
        trainer.config = GRPOConfig()
        result = GRPOResult(
            step=3,
            metrics={
                "0/reward/mean": 1.0,
                "0/advantage/std": 0.0,
                "prompt-a/reward/std": 0.5,
                "reward/mean": 0.75,
                "reward/std": 0.25,
                "reward/boxed_answer_reward/mean": 0.75,
                "optim/lr": 4e-5,
                "optim/grad_norm": 1.25,
                "train/loss": -0.1,
                "trainer/backend_metric": 0.4,
                "loss_sum": 2.0,
                "loss_mean": 0.25,
                "token_count": 8.0,
                "scalar/other_metric": 0.6,
                "policy/entropy": 1.5,
                "policy/kl_sample_train": 0.02,
                "time/checkpoint": 0.5,
                "time/step": 3.0,
            },
            num_datums=8,
            skipped_groups=1,
            checkpoint_path="s3://example/checkpoint",
        )

        trainer._log_swanlab(result)

        assert run.payload is not None
        self.assertNotIn("0/reward/mean", run.payload)
        self.assertNotIn("0/advantage/std", run.payload)
        self.assertNotIn("prompt-a/reward/std", run.payload)
        self.assertEqual(run.payload["reward/mean"], 0.75)
        self.assertEqual(run.payload["reward/std"], 0.25)
        self.assertEqual(run.payload["reward/boxed_answer_reward/mean"], 0.75)
        self.assertEqual(run.payload["train/learning_rate"], 4e-5)
        self.assertNotIn("optim/lr", run.payload)
        self.assertEqual(run.payload["optim/grad_norm"], 1.25)
        self.assertEqual(run.payload["train/loss"], -0.1)
        self.assertEqual(run.payload["train/backend_metric"], 0.4)
        self.assertEqual(run.payload["train/other_metric"], 0.6)
        self.assertEqual(run.payload["train/loss_sum"], 2.0)
        self.assertEqual(run.payload["train/loss_mean"], 0.25)
        self.assertEqual(run.payload["train/token_count"], 8.0)
        self.assertFalse(any(key.startswith("trainer/") for key in run.payload))
        self.assertTrue(all("/" in key for key in run.payload))
        self.assertEqual(run.payload["policy/entropy"], 1.5)
        self.assertEqual(run.payload["policy/kl_sample_train"], 0.02)
        self.assertNotIn("time/checkpoint", run.payload)
        self.assertNotIn("checkpoint/saved", run.payload)
        self.assertEqual(run.payload["time/step"], 3.0)
        self.assertEqual(run.step, 3)

    def test_logging_step_averages_all_scalars_and_flushes_final_partial_window(self) -> None:
        run = _Run()
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer._swanlab_run = run
        trainer.config = GRPOConfig(logging_step=2)
        for step, loss in ((1, 0.2), (2, 0.4), (3, 0.6)):
            trainer._log_swanlab(GRPOResult(
                step=step,
                metrics={"train/loss": loss, "reward/mean": loss + 1, "time/step": 2.0},
                num_datums=step * 2,
            ))
            self.assertEqual(len(run.logs), step // 2)
        trainer._flush_swanlab(3)

        self.assertEqual([step for _, step in run.logs], [2, 3])
        self.assertAlmostEqual(run.logs[0][0]["train/loss"], 0.3)
        self.assertAlmostEqual(run.logs[0][0]["reward/mean"], 1.3)
        self.assertEqual(run.logs[0][0]["train/num_datums"], 3.0)
        self.assertEqual(run.logs[0][0]["train/step"], 2.0)
        self.assertAlmostEqual(run.logs[1][0]["train/loss"], 0.6)

    def test_final_checkpoint_is_logged_to_swanlab(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)

        with patch("trio_cookbook.trainer.grpo_trainer.tqdm.write") as terminal:
            trainer._log_final_checkpoint()

        terminal.assert_not_called()
        self.assertEqual(run.texts["checkpoint/path"], ("trio://final-checkpoint", 1))

    def test_final_checkpoint_respects_output_dir_and_missing_path(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)

        with patch("trio_cookbook.trainer.grpo_trainer.tqdm.write") as terminal:
            trainer.config = GRPOConfig(output_dir=False)
            trainer._log_final_checkpoint()
            trainer.config = GRPOConfig(output_dir=True)
            trainer.state.checkpoint_path = None
            trainer._log_final_checkpoint()

        terminal.assert_not_called()

    def test_final_inference_is_logged_as_text(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)
        rows = [{"messages": f"Prompt {i}", "question": f"Question {i}"} for i in range(5)]

        trainer._log_final_samples(rows)

        texts, step = run.texts["sample/text"]
        self.assertEqual(step, 1)
        self.assertEqual(len(texts), 4)
        self.assertEqual([text.caption for text in texts], [f"Question {i}" for i in range(4)])
        self.assertEqual([text.content for text in texts], ["final answer"] * 4)
        self.assertEqual(sampler.prompts, [[1]] * 4)

    def test_final_inference_decodes_missing_text(self) -> None:
        response = SimpleNamespace(sequences=[SimpleNamespace(text=None, tokens=[2])])
        self.assertEqual(GRPOTrainer._sample_text(response, lambda _: "decoded answer"), "decoded answer")
        with self.assertRaisesRegex(ValueError, "no sequences"):
            GRPOTrainer._sample_text(SimpleNamespace(sequences=[]), lambda _: "unused")

    def test_train_saves_and_logs_final_step_after_early_exhaustion(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)
        trainer.config = GRPOConfig(max_steps=3, batch_size=1, save_every=3, output_dir=True, logging_step=10)
        trainer.state = TrainerState()
        trainer.train_dataset = [{"messages": "Only question"}]
        trainer._training_client = SimpleNamespace(
            save_state=lambda *, name: SimpleNamespace(path=f"trio://{name}")
        )

        def step(rows: object) -> GRPOResult:
            trainer.state.global_step += 1
            trainer._total_input_tokens += 4
            trainer.state.metrics = {"train/loss": 0.25, "reward/mean": 0.75}
            trainer._save_state()
            result = GRPOResult(step=trainer.state.global_step, metrics=trainer.state.metrics)
            trainer._log_swanlab(result)
            return result

        trainer.step = step

        with patch("trio_cookbook.trainer.grpo_trainer.tqdm.write") as terminal:
            state = trainer.train()

        self.assertEqual(state.checkpoint_path, "trio://grpo-step-00000001")
        self.assertEqual(run.step, 1)
        self.assertEqual(run.logs[0][0]["train/loss"], 0.25)
        self.assertGreater(run.payload["time/train_runtime"], 0)
        self.assertGreater(run.payload["time/train_steps_per_second"], 0)
        self.assertGreater(run.payload["time/train_tokens_per_second"], 0)
        self.assertEqual(run.payload["train/total_input_tokens"], 4)
        self.assertEqual(run.texts["checkpoint/path"], (state.checkpoint_path, 1))
        texts, _ = run.texts["sample/text"]
        self.assertEqual(texts[0].caption, "Only question")
        summary, step = run.texts["train/final_summary"]
        self.assertEqual(step, 1)
        self.assertIn("Training finished at step 1", summary)
        self.assertIn('"train/loss": 0.25', summary)
        self.assertIn('"time/train_runtime":', summary)
        self.assertEqual([call.args[0] for call in terminal.call_args_list], summary.splitlines())
        self.assertLess(run.events.index("train/final_summary"), run.events.index("finish"))


class AsyncSwanLabLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_inference_uses_async_sampler(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)

        await trainer._log_final_samples_async([{"messages": "Prompt", "question": "Question"}])

        texts, step = run.texts["sample/text"]
        self.assertEqual(step, 1)
        self.assertEqual([(text.caption, text.content) for text in texts], [("Question", "final answer")])
        self.assertEqual(sampler.prompts, [[1]])

    async def test_async_train_saves_final_step_after_early_exhaustion(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)
        trainer.config = GRPOConfig(max_steps=3, batch_size=1, save_every=3, output_dir=True)
        trainer.state = TrainerState()
        trainer.train_dataset = [{"messages": "Only question"}]

        async def save_state_async(*, name: str) -> object:
            return SimpleNamespace(path=f"trio://{name}")

        trainer._training_client = SimpleNamespace(save_state_async=save_state_async)

        async def step_async(rows: object) -> GRPOResult:
            trainer.state.global_step += 1
            trainer._total_input_tokens += 2
            trainer.state.metrics = {"train/loss": 0.25}
            await trainer._save_state_async()
            return GRPOResult(step=trainer.state.global_step)

        trainer.step_async = step_async

        with patch("trio_cookbook.trainer.grpo_trainer.tqdm.write") as terminal:
            state = await trainer.train_async()

        self.assertEqual(state.checkpoint_path, "trio://grpo-step-00000001")
        self.assertGreater(state.metrics["time/train_runtime"], 0)
        self.assertEqual(state.metrics["train/total_input_tokens"], 2)
        self.assertEqual(run.texts["checkpoint/path"], (state.checkpoint_path, 1))
        self.assertIn("sample/text", run.texts)
        summary, step = run.texts["train/final_summary"]
        self.assertEqual(step, 1)
        self.assertIn('"train/loss": 0.25', summary)
        self.assertEqual([call.args[0] for call in terminal.call_args_list], summary.splitlines())
        self.assertLess(run.events.index("train/final_summary"), run.events.index("finish"))


if __name__ == "__main__":
    unittest.main()
