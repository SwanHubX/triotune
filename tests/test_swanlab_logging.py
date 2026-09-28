from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trio_cookbook.trainer import GRPOConfig, GRPOResult, GRPOTrainer, TrainerState


class _Run:
    def __init__(self) -> None:
        self.payload: dict[str, float] | None = None
        self.step: int | None = None
        self.texts: dict[str, tuple[object, int]] = {}
        self.events: list[str] = []

    def log(self, payload: dict[str, float], *, step: int) -> None:
        self.payload = payload
        self.step = step

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
    trainer.config = GRPOConfig(max_steps=1, batch_size=5)
    trainer.state = TrainerState(global_step=1, checkpoint_path="trio://final-checkpoint")
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
        trainer.config = GRPOConfig(mode="online", workspace="my-organization", tags=["grpo", "gsm8k"])

        with patch("trio_cookbook.trainer.grpo_trainer.swanlab.init") as init:
            trainer._init_swanlab_run(lora_rank=32)

        self.assertEqual(init.call_args.kwargs["workspace"], "my-organization")
        self.assertEqual(init.call_args.kwargs["tags"], ["grpo", "gsm8k"])

    def test_swanlab_workspace_and_tags_default_to_none(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.model_name = "Qwen/Qwen3.5-9B"
        trainer.config = GRPOConfig(mode="online")

        with patch("trio_cookbook.trainer.grpo_trainer.swanlab.init") as init:
            trainer._init_swanlab_run(lora_rank=32)

        self.assertIsNone(init.call_args.kwargs["workspace"])
        self.assertIsNone(init.call_args.kwargs["tags"])

    def test_rejects_empty_workspace_and_tags(self) -> None:
        with self.assertRaisesRegex(ValueError, "workspace"):
            GRPOConfig(workspace=" ")
        with self.assertRaisesRegex(ValueError, "tags"):
            GRPOConfig(tags=["grpo", ""])
        with self.assertRaisesRegex(ValueError, "tags"):
            GRPOConfig(tags="grpo")

    def test_logs_batch_metrics_without_per_prompt_metrics(self) -> None:
        run = _Run()
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer._swanlab_run = run
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
        self.assertEqual(run.payload["optim/lr"], 4e-5)
        self.assertEqual(run.payload["optim/grad_norm"], 1.25)
        self.assertNotIn("train/loss", run.payload)
        self.assertEqual(run.payload["trainer/loss"], -0.1)
        self.assertEqual(run.payload["policy/entropy"], 1.5)
        self.assertEqual(run.payload["policy/kl_sample_train"], 0.02)
        self.assertNotIn("time/checkpoint", run.payload)
        self.assertNotIn("checkpoint/saved", run.payload)
        self.assertEqual(run.payload["time/step"], 3.0)
        self.assertEqual(run.step, 3)

    def test_final_checkpoint_and_inference_are_logged_as_text(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)
        rows = [{"prompt": f"Prompt {i}", "question": f"Question {i}"} for i in range(5)]

        trainer._log_final_checkpoint()
        trainer._log_final_samples(rows)

        self.assertEqual(run.texts["checkpoint/path"], ("trio://final-checkpoint", 1))
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
        trainer.config = GRPOConfig(max_steps=3, batch_size=1, save_every=3)
        trainer.state = TrainerState()
        trainer.train_dataset = [{"prompt": "Only question"}]
        trainer._training_client = SimpleNamespace(
            save_state=lambda *, name: SimpleNamespace(path=f"trio://{name}")
        )

        def step(rows: object) -> GRPOResult:
            trainer.state.global_step += 1
            trainer.state.metrics = {"train/loss": 0.25, "reward/mean": 0.75}
            trainer._save_state()
            return GRPOResult(step=trainer.state.global_step)

        trainer.step = step

        with patch("trio_cookbook.trainer.grpo_trainer.tqdm.write") as terminal:
            state = trainer.train()

        self.assertEqual(state.checkpoint_path, "trio://grpo-step-00000001")
        self.assertEqual(run.texts["checkpoint/path"], (state.checkpoint_path, 1))
        texts, _ = run.texts["sample/text"]
        self.assertEqual(texts[0].caption, "Only question")
        summary, step = run.texts["trainer/final_summary"]
        self.assertEqual(step, 1)
        self.assertIn("Training finished at step 1", summary)
        self.assertIn('"train/loss": 0.25', summary)
        self.assertIn("Latest checkpoint: trio://grpo-step-00000001", summary)
        self.assertEqual([call.args[0] for call in terminal.call_args_list], summary.splitlines())
        self.assertLess(run.events.index("trainer/final_summary"), run.events.index("finish"))


class AsyncSwanLabLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_inference_uses_async_sampler(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)

        await trainer._log_final_samples_async([{"prompt": "Prompt", "question": "Question"}])

        texts, step = run.texts["sample/text"]
        self.assertEqual(step, 1)
        self.assertEqual([(text.caption, text.content) for text in texts], [("Question", "final answer")])
        self.assertEqual(sampler.prompts, [[1]])

    async def test_async_train_saves_final_step_after_early_exhaustion(self) -> None:
        run, sampler = _Run(), _Sampler()
        trainer = _final_trainer(run, sampler)
        trainer.config = GRPOConfig(max_steps=3, batch_size=1, save_every=3)
        trainer.state = TrainerState()
        trainer.train_dataset = [{"prompt": "Only question"}]

        async def save_state_async(*, name: str) -> object:
            return SimpleNamespace(path=f"trio://{name}")

        trainer._training_client = SimpleNamespace(save_state_async=save_state_async)

        async def step_async(rows: object) -> GRPOResult:
            trainer.state.global_step += 1
            trainer.state.metrics = {"train/loss": 0.25}
            await trainer._save_state_async()
            return GRPOResult(step=trainer.state.global_step)

        trainer.step_async = step_async

        with patch("trio_cookbook.trainer.grpo_trainer.tqdm.write") as terminal:
            state = await trainer.train_async()

        self.assertEqual(state.checkpoint_path, "trio://grpo-step-00000001")
        self.assertEqual(run.texts["checkpoint/path"], (state.checkpoint_path, 1))
        self.assertIn("sample/text", run.texts)
        summary, step = run.texts["trainer/final_summary"]
        self.assertEqual(step, 1)
        self.assertIn('"train/loss": 0.25', summary)
        self.assertEqual([call.args[0] for call in terminal.call_args_list], summary.splitlines())
        self.assertLess(run.events.index("trainer/final_summary"), run.events.index("finish"))


if __name__ == "__main__":
    unittest.main()
