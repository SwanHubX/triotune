from __future__ import annotations

import unittest

from trio_cookbook.trainer import GRPOResult, GRPOTrainer


class _Run:
    def __init__(self) -> None:
        self.payload: dict[str, float] | None = None
        self.step: int | None = None

    def log(self, payload: dict[str, float], *, step: int) -> None:
        self.payload = payload
        self.step = step


class SwanLabLoggingTests(unittest.TestCase):
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
                "time/step": 3.0,
            },
            num_datums=8,
            skipped_groups=1,
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
        self.assertEqual(run.payload["train/loss"], -0.1)
        self.assertEqual(run.payload["policy/entropy"], 1.5)
        self.assertEqual(run.payload["policy/kl_sample_train"], 0.02)
        self.assertEqual(run.payload["time/step"], 3.0)
        self.assertEqual(run.step, 3)


if __name__ == "__main__":
    unittest.main()
