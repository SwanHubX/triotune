from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


def _load_recipe() -> ModuleType:
    path = Path(__file__).parents[1] / "trio-cookbook" / "recipes" / "grpo_gsm8k.py"
    spec = importlib.util.spec_from_file_location("trio_cookbook_grpo_gsm8k_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recipe = _load_recipe()


class _Future:
    def __init__(
        self,
        events: list[str],
        event: str,
        value: object = None,
        error: Exception | None = None,
    ):
        self.events = events
        self.event = event
        self.value = value
        self.error = error

    def result(self) -> object:
        self.events.append(self.event)
        if self.error is not None:
            raise self.error
        return self.value


class _TrainingClient:
    def __init__(self, *, forward_error: Exception | None = None):
        self.events: list[str] = []
        self.forward_error = forward_error

    def forward_backward(self, datums: list[object], *, loss_fn: str) -> _Future:
        self.events.append("forward_submitted")
        return _Future(
            self.events,
            "forward_completed",
            SimpleNamespace(metrics={"loss_mean": 0.25}),
            self.forward_error,
        )

    def optim_step(self, adam_params: object) -> _Future:
        self.events.append("optimizer_submitted")
        return _Future(self.events, "optimizer_completed")


class OptimizationOrderTests(unittest.TestCase):
    def test_optimizer_is_submitted_after_forward_completes(self) -> None:
        client = _TrainingClient()

        metrics = recipe.run_optimization_step(
            client,
            [object()],
            loss_fn="importance_sampling",
            adam_params=object(),
        )

        self.assertEqual(metrics, {"loss_mean": 0.25})
        self.assertEqual(
            client.events,
            [
                "forward_submitted",
                "forward_completed",
                "optimizer_submitted",
                "optimizer_completed",
            ],
        )

    def test_forward_failure_does_not_submit_optimizer(self) -> None:
        client = _TrainingClient(forward_error=RuntimeError("forward failed"))

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            recipe.run_optimization_step(
                client,
                [object()],
                loss_fn="importance_sampling",
                adam_params=object(),
            )

        self.assertEqual(client.events, ["forward_submitted", "forward_completed"])


if __name__ == "__main__":
    unittest.main()
