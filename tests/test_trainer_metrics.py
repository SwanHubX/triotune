from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from trio_cookbook.trainer import GRPOBatch, GRPOConfig, GRPOResult, GRPOTrainer, SequenceSample, TrainerState
from trio_cookbook.utils.metrics import MetricsAPI
from trio_cookbook.trainer.grpo.metrics import _rollout_metrics


class TrainerMetricsTests(unittest.TestCase):
    def test_collects_lr_loss_grad_norm_and_sample_train_kl(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = GRPOConfig(learning_rate=4e-5)
        datum = SimpleNamespace(
            loss_fn_inputs={
                "logprobs": SimpleNamespace(data=[0.0, -1.0, -2.0]),
                "advantages": SimpleNamespace(data=[0.0, 1.0, 1.0]),
            }
        )
        forward_result = SimpleNamespace(
            metrics={"loss_mean": 0.25},
            loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-0.1, -1.2, -2.2])}],
        )
        optimizer_result = SimpleNamespace(metrics={"grad_norm": 1.5})

        metrics = trainer._training_metrics([datum], forward_result, optimizer_result)

        self.assertEqual(metrics["optim/lr"], 4e-5)
        self.assertEqual(metrics["train/loss"], 0.25)
        self.assertEqual(metrics["optim/grad_norm"], 1.5)
        self.assertAlmostEqual(metrics["policy/kl_sample_train"], 0.02)

    def test_collects_entropy_and_completion_lengths_from_rollouts(self) -> None:
        group = GRPOBatch(
            prompt_id="0",
            prompt="prompt",
            prompt_tokens=[1],
            samples=[
                SequenceSample(tokens=[2], logprobs=[-1.0], text="a"),
                SequenceSample(tokens=[3, 4], logprobs=[-2.0, -3.0], text="b"),
            ],
        )

        metrics = _rollout_metrics([group])

        self.assertEqual(metrics["policy/entropy"], 2.0)
        self.assertEqual(metrics["rollout/completion_length_mean"], 1.5)
        self.assertEqual(metrics["rollout/completion_length_min"], 1.0)
        self.assertEqual(metrics["rollout/completion_length_max"], 2.0)

    def test_progress_postfix_uses_core_metrics(self) -> None:
        result = GRPOResult(
            step=1,
            metrics={
                "train/loss": 0.25,
                "reward/mean": 0.75,
                "policy/entropy": 1.5,
                "policy/kl_sample_train": 0.02,
                "optim/grad_norm": 2.0,
            },
        )

        self.assertEqual(
            GRPOTrainer._progress_postfix(result),
            {
                "loss": "0.2500",
                "reward": "0.7500",
                "entropy": "1.5000",
                "kl": "0.0200",
                "grad_norm": "2.0000",
            },
        )

    def test_step_records_prepare_save_and_input_token_throughput(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = GRPOConfig(save_every=1)
        trainer.state = TrainerState()
        trainer._total_input_tokens = 0
        trainer._metrics_api = MetricsAPI()
        group = GRPOBatch(
            prompt_id="0",
            prompt="question",
            prompt_tokens=[1],
            samples=[SequenceSample(tokens=[2], logprobs=[-0.1], text="answer")],
        )
        trainer._groups = lambda rows: [group]
        trainer._new_sampling_client = lambda: object()
        trainer._sample_groups = lambda groups, client: groups
        trainer.reward_evaluator = SimpleNamespace(score=lambda groups: ([[1.0]], {}))
        trainer._build_datums = lambda groups, rewards: (
            [SimpleNamespace(model_input=SimpleNamespace(length=4))], 0, {}
        )
        trainer._train_datums = lambda datums: {"time/forward_backward": 0.1, "time/optim": 0.2}
        trainer._save_state = lambda: "checkpoint://step-1"

        metrics = trainer.step([{"messages": "question"}]).metrics

        self.assertEqual(metrics["train/input_tokens"], 4)
        self.assertEqual(trainer._total_input_tokens, 4)
        self.assertEqual(metrics["time/forward_backward"], 0.1)
        self.assertEqual(metrics["time/optim"], 0.2)
        self.assertGreaterEqual(metrics["time/prepare"], 0)
        self.assertNotIn("time/save", metrics)
        self.assertAlmostEqual(metrics["time/tokens_per_second"], 4 / metrics["time/step"])

        async def sampling_client_async() -> object:
            return object()

        async def sample_groups_async(groups: object, client: object) -> object:
            return groups

        async def score_async(groups: object) -> tuple[list[list[float]], dict[str, float]]:
            return [[1.0]], {}

        async def train_datums_async(datums: object) -> dict[str, float]:
            return {"time/forward_backward": 0.1, "time/optim": 0.2}

        async def save_state_async() -> None:
            return None

        trainer._new_sampling_client_async = sampling_client_async
        trainer._sample_groups_async = sample_groups_async
        trainer.reward_evaluator.score_async = score_async
        trainer._train_datums_async = train_datums_async
        trainer._save_state_async = save_state_async

        async_metrics = asyncio.run(trainer.step_async([{"messages": "question"}])).metrics
        self.assertEqual(trainer._total_input_tokens, 8)
        self.assertEqual(async_metrics["train/input_tokens"], 4)
        self.assertGreaterEqual(async_metrics["time/prepare"], 0)
        self.assertNotIn("time/save", async_metrics)
        self.assertAlmostEqual(async_metrics["time/tokens_per_second"], 4 / async_metrics["time/step"])

    def test_sync_and_async_optimizer_timing_only_when_datums_exist(self) -> None:
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.config = GRPOConfig()
        events: list[str] = []

        class Client:
            def forward_backward(self, datums: list[object], **kwargs: object) -> object:
                events.append("forward")
                return SimpleNamespace(metrics={"loss_mean": 0.5}, loss_fn_outputs=[])

            def optim_step(self, params: object) -> object:
                events.append("optim")
                return SimpleNamespace(metrics={})

            async def forward_backward_async(self, datums: list[object], **kwargs: object) -> object:
                return self.forward_backward(datums, **kwargs)

            async def optim_step_async(self, params: object) -> object:
                return self.optim_step(params)

        trainer._training_client = Client()
        for metrics in (trainer._train_datums([object()]), asyncio.run(trainer._train_datums_async([object()]))):
            self.assertEqual(metrics["train/loss"], 0.5)
            self.assertGreaterEqual(metrics["time/forward_backward"], 0)
            self.assertGreaterEqual(metrics["time/optim"], 0)
        self.assertEqual(events, ["forward", "optim", "forward", "optim"])
        self.assertNotIn("time/forward_backward", trainer._train_datums([]))
        self.assertNotIn("time/optim", asyncio.run(trainer._train_datums_async([])))


if __name__ == "__main__":
    unittest.main()
