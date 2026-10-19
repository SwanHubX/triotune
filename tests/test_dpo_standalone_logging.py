from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _load_recipe():
    path = Path(__file__).parents[1] / "triotune" / "recipes" / "dpo_rlhf.py"
    spec = importlib.util.spec_from_file_location("dpo_rlhf_standalone_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _TrainingClient:
    def get_tokenizer(self):
        return object()

    def forward_backward_custom(self, data, loss_fn):
        return _Future(SimpleNamespace(metrics={
            "dpo/loss": 0.4,
            "dpo/accuracy": 1.0,
            "dpo/margin": 0.3,
            "dpo/chosen_reward": 0.2,
            "dpo/rejected_reward": -0.1,
            "loss_sum": 9.0,
            "loss_mean": 3.0,
            "token_count": 2,
        }))

    def optim_step(self, params):
        return _Future(None)

    def save_weights_for_sampler(self, name):
        return _Future(SimpleNamespace(path="weights://test"))


class StandaloneDPOLoggingTests(unittest.TestCase):
    def test_logs_real_dpo_loss_and_timing_without_surrogate_loss(self):
        recipe = _load_recipe()
        training = _TrainingClient()
        service = SimpleNamespace(
            create_lora_training_client=lambda **kwargs: training,
            create_sampling_client=lambda **kwargs: object(),
        )
        args = SimpleNamespace(
            seed=1,
            dataset_subsets=["helpful-base"],
            base_model="test-model",
            lora_rank=4,
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.95,
            adam_eps=1e-8,
            steps=1,
            dpo_beta=0.1,
            save_weights_name="test",
        )
        datums = [SimpleNamespace(model_input=SimpleNamespace(length=3)) for _ in range(2)]
        with (
            patch.object(recipe, "load_hh_rlhf", return_value=[object()]),
            patch.object(recipe.trio, "ServiceClient", return_value=service),
            patch.object(recipe, "build_step_batch", return_value=datums),
            patch.object(recipe, "compute_reference_logprob_seqs", return_value=[-1.0, -1.0]),
            patch.object(recipe, "make_dpo_loss_fn", return_value=object()),
            patch.object(recipe, "start_swanlab", return_value=object()),
            patch.object(recipe.swanlab, "log") as log,
            patch.object(recipe.swanlab, "finish"),
            patch.object(recipe.swanlab, "Text", side_effect=lambda value: value),
            redirect_stdout(io.StringIO()),
        ):
            recipe.main(args)

        step_metrics = log.call_args_list[0].args[0]
        summary = log.call_args_list[1].args[0]
        self.assertEqual(step_metrics["train/loss"], 0.4)
        self.assertEqual(step_metrics["train/tokens"], 6)
        self.assertEqual(step_metrics["reward/reward_accuracy"], 1.0)
        self.assertEqual(step_metrics["reward/reward_margin"], 0.3)
        self.assertEqual(step_metrics["reward/reward_chosen"], 0.2)
        self.assertEqual(step_metrics["reward/reward_rejected"], -0.1)
        self.assertEqual(step_metrics["train/backend_token_count"], 2)
        self.assertEqual({name.split("/", 1)[0] for name in step_metrics}, {"train", "reward", "time"})
        self.assertNotIn("trainer/loss_sum", step_metrics)
        self.assertNotIn("trainer/loss_mean", step_metrics)
        self.assertGreater(step_metrics["time/step_elapsed_time"], 0)
        self.assertGreaterEqual(step_metrics["time/update"], 0)
        self.assertEqual(summary["train/total_tokens"], 6)
        self.assertEqual(summary["train/total_pairs"], 1)
        self.assertEqual(summary["train/weights_path"], "weights://test")
        self.assertEqual({name.split("/", 1)[0] for name in summary}, {"train", "time"})
        self.assertGreater(summary["time/train_runtime"], 0)


if __name__ == "__main__":
    unittest.main()
