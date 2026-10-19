"""PyTRIO loss contracts stay aligned across algorithm-facing adapters."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from triotune.api.loss_api import (
    BuiltinPyTrioObjective,
    LossSpec,
    PolicyLossInput,
    SupervisedLossInput,
)


class _Future:
    def __init__(self, value: object) -> None:
        self.value = value

    def result(self) -> object:
        return self.value

    def __await__(self):
        async def resolve():
            await asyncio.sleep(0)
            return self.value

        return resolve().__await__()


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], dict[str, object]]] = []

    def forward_backward(self, datums, **kwargs):
        self.calls.append((list(datums), dict(kwargs)))
        return _Future(SimpleNamespace(metrics={"loss_mean": 0.25}))

    async def forward_backward_async(self, datums, **kwargs):
        self.calls.append((list(datums), dict(kwargs)))
        return _Future(SimpleNamespace(metrics={"loss_mean": 0.25}))


class LossContractTests(unittest.IsolatedAsyncioTestCase):
    def test_supervised_and_policy_inputs_enforce_distinct_fields(self) -> None:
        supervised = SupervisedLossInput(
            model_tokens=[10, 11],
            target_tokens=[11, -100],
            weights=[1.0, 0.0],
        )
        policy = PolicyLossInput(
            model_tokens=[10, 11],
            target_tokens=[-100, 12],
            behavior_logprobs=[0.0, -0.5],
            advantages=[0.0, 1.0],
        )

        LossSpec("cross_entropy").validate_loss_inputs(supervised)
        LossSpec("ppo").validate_loss_inputs(policy)
        with self.assertRaisesRegex(TypeError, "requires policy inputs"):
            LossSpec("ppo").validate_loss_inputs(supervised)
        with self.assertRaisesRegex(TypeError, "requires supervised inputs"):
            LossSpec("cross_entropy").validate_loss_inputs(policy)

    def test_loss_options_are_validated_per_objective(self) -> None:
        self.assertEqual(
            LossSpec("ppo").effective_config,
            {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        )
        self.assertEqual(LossSpec("dro").effective_config, {"beta": 0.05})
        with self.assertRaisesRegex(ValueError, "does not support"):
            LossSpec("importance_sampling", {"beta": 0.1})
        with self.assertRaisesRegex(ValueError, "0 <= low <= high"):
            LossSpec("cispo", {"clip_low_threshold": 2.0, "clip_high_threshold": 1.0})

    async def test_builtin_objective_uses_one_sync_and_async_call_shape(self) -> None:
        datum = PolicyLossInput(
            model_tokens=[10, 11],
            target_tokens=[-100, 12],
            behavior_logprobs=[0.0, -0.5],
            advantages=[0.0, 1.0],
        ).to_datum()
        client = _Client()
        objective = BuiltinPyTrioObjective(
            LossSpec(
                "cispo",
                {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0},
            )
        )

        sync_result = objective.forward_backward(client, [datum])
        async_result = await objective.forward_backward_async(client, [datum])

        self.assertEqual(sync_result.metrics["loss_mean"], 0.25)
        self.assertEqual(async_result.metrics["loss_mean"], 0.25)
        expected = {
            "loss_fn": "cispo",
            "loss_fn_config": {
                "clip_low_threshold": 0.0,
                "clip_high_threshold": 4.0,
            },
            "auto_shift": False,
        }
        self.assertEqual(client.calls[0][1], expected)
        self.assertEqual(client.calls[1][1], expected)


if __name__ == "__main__":
    unittest.main()
