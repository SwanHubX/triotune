from __future__ import annotations

from types import SimpleNamespace
import unittest

from trio_cookbook.trainer import GRPOBatch, GRPOConfig, GRPOResult, GRPOTrainer, SequenceSample
from trio_cookbook.trainer.grpo_trainer import _rollout_metrics


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


if __name__ == "__main__":
    unittest.main()
