"""Two-run PyTRIO PPO: causal value readout, real SDK bridge and lifecycle."""

import asyncio
import math
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import swanlab
import torch
from pytrio.types.forward.tensor_data import TensorData
from test_ppo_family import Future, Tokenizer, TrainingClient, trajectory
from triotune.trainer.ppo import PPOConfig, PPOTrainer, PyTRIOCritic
from triotune.trainer.ppo.critic_loss import (
    critic_datums,
    make_value_loss_fn,
    values_from_logprobs,
)
from triotune.trainer.ppo.reporting import critic_metrics
from triotune.utils.metrics import MetricsAPI


class MarkerTokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        return {"1": [1], "0": [0]}.get(text, [10, 11])


class ActorClient(TrainingClient):
    def get_tokenizer(self):
        return MarkerTokenizer()


class CriticClient(TrainingClient):
    """A tiny differentiable causal LM; the two markers have real softmax logps."""

    def __init__(self):
        super().__init__()
        self.parameter = torch.tensor(0.2, requires_grad=True)
        self.grads = []

    def logprobs(self, data):
        rows = []
        for datum in data:
            tokens = datum.model_input.to_ints()
            labels = datum.loss_fn_inputs["target_tokens"].data
            row = []
            for i, label in enumerate(labels):
                feature = sum(tokens[: i + 1]) / 100
                logits = torch.stack(
                    [
                        self.parameter * 0,
                        self.parameter * feature,
                        self.parameter * 0 + 1,
                    ]
                )
                row.append(
                    logits.log_softmax(-1)[label]
                    if label != -100
                    else self.parameter * 0
                )
            rows.append(torch.stack(row))
        return rows

    async def forward_async(self, data, **kwargs):
        self.calls.append(("forward", data, kwargs))
        self.trace.append("submit:forward")
        return Future(
            SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": TensorData.from_torch(lp.detach())}
                    for lp in self.logprobs(data)
                ]
            ),
            self.trace,
            "forward",
        )

    async def forward_backward_custom_async(self, data, loss_fn):
        self.calls.append(("custom", data, loss_fn))
        self.trace.append("submit:custom")
        loss, metrics = loss_fn(data, self.logprobs(data))
        loss.backward()
        self.grads.append(self.parameter.grad.item())
        return Future(SimpleNamespace(metrics=metrics), self.trace, "custom")

    async def optim_step_async(self, params):
        self.trace.append("submit:optim")
        with torch.no_grad():
            self.parameter -= params.learning_rate * self.parameter.grad
            self.parameter.grad = None
        self.version += 1
        return Future(SimpleNamespace(metrics={}), self.trace, "optim")


class Service:
    def __init__(self):
        self.actor, self.critic = ActorClient(), CriticClient()
        self.calls = []

    async def create_lora_training_client_async(self, **kwargs):
        self.calls.append(kwargs)
        return self.actor if len(self.calls) == 1 else self.critic


class CriticLossTests(unittest.TestCase):
    def test_marker_targets_keep_context_and_include_bootstrap(self):
        t = trajectory(n=3)
        positive, negative = critic_datums([t], (1, 0))
        self.assertEqual(positive.model_input.to_ints(), [10, 11, 20, 21, 22])
        self.assertEqual(negative.model_input.to_ints(), positive.model_input.to_ints())
        self.assertEqual(
            positive.loss_fn_inputs["target_tokens"].data, [-100, 1, 1, 1, 1]
        )
        self.assertEqual(
            negative.loss_fn_inputs["target_tokens"].data, [-100, 0, 0, 0, 0]
        )

    def test_value_is_logit_difference_unbounded_and_signed(self):
        t = trajectory(n=1)
        c = PPOConfig(critic_value_scale=2, critic_value_bias=1)
        logits = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 4.0, 99.0], [0.0, -3.0, 99.0]], dtype=torch.float64
        )
        logp = logits.log_softmax(-1)
        values = values_from_logprobs([logp[:, 1], logp[:, 0]], [t], c)
        np.testing.assert_allclose(values[0], [9, -5])

    def test_mse_gradients_both_markers_mask_and_bootstrap(self):
        t = trajectory(n=3, values=[0, 0, 0, 100])
        c = PPOConfig()
        tensors = [
            torch.tensor([0.0, -1, -1, -1, -1], requires_grad=True) for _ in range(2)
        ]
        loss, metrics = make_value_loss_fn(
            [t], [[1, 99, 1]], [[True, False, True]], c, (1, 0)
        )(critic_datums([t], (1, 0)), tensors)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.5)
        self.assertAlmostEqual(metrics["critic/return_mean"], 1)
        np.testing.assert_allclose(tensors[0].grad, [0, -0.5, 0, -0.5, 0])
        np.testing.assert_allclose(tensors[1].grad, [0, 0.5, 0, 0.5, 0])

    def test_value_gradient_matches_finite_difference(self):
        t = trajectory(n=1, values=[0.1, 0.2])
        c = PPOConfig(critic_value_scale=1.5)
        data = critic_datums([t], (1, 0))
        fn = make_value_loss_fn([t], [[0.8]], [[True]], c, (1, 0))

        def evaluate(p, n):
            tensors = [
                torch.tensor([0, v, -1], dtype=torch.float64, requires_grad=True)
                for v in (p, n)
            ]
            loss, _ = fn(data, tensors)
            loss.backward()
            return loss.item(), [row.grad[1].item() for row in tensors]

        p, n, eps = -1.0, -1.1, 1e-6
        _, gradient = evaluate(p, n)
        numeric = [
            (evaluate(p + eps, n)[0] - evaluate(p - eps, n)[0]) / (2 * eps),
            (evaluate(p, n + eps)[0] - evaluate(p, n - eps)[0]) / (2 * eps),
        ]
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-8)

    def test_shapes_markers_and_nonfinite_inputs_rejected(self):
        for kwargs in (
            {"critic_token_ids": (1, 1)},
            {"critic_token_ids": (-1, 0)},
            {"critic_value_scale": 0},
            {"critic_eps_clip": -1},
            {"critic_n_minibatches": 0},
            {"critic_learning_rate": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PPOConfig(**kwargs)
        t = trajectory(values=[0, 0, 0])
        with self.assertRaisesRegex(ValueError, "frozen"):
            make_value_loss_fn([t], [[1]], [[True]], PPOConfig(), (1, 0))
        with self.assertRaisesRegex(ValueError, "finite"):
            make_value_loss_fn(
                [t], [[math.nan, 1]], [[True, True]], PPOConfig(), (1, 0)
            )
        fn = make_value_loss_fn([t], [[1, 1]], [[True, True]], PPOConfig(), (1, 0))
        with self.assertRaisesRegex(ValueError, "ordering"):
            fn(list(reversed(critic_datums([t], (1, 0)))), [])


class TwoRunPPOTests(unittest.IsolatedAsyncioTestCase):
    async def test_actor_reporter_finishes_on_main_thread(self):
        run = Mock()

        def check_thread(**kwargs):
            self.assertIs(threading.current_thread(), threading.main_thread())

        run.finish.side_effect = check_thread
        trainer = PPOTrainer(
            "test-model", PPOConfig.for_algorithm("grpo"), training_client=ActorClient()
        )
        trainer._metrics = MetricsAPI(run)
        await trainer.close()
        run.finish.assert_called_once()

    async def test_default_ppo_creates_two_independent_runs_and_saves_both(self):
        service = Service()
        config = PPOConfig(
            batch_size=1,
            max_steps=1,
            num_generations=2,
            seed=42,
            critic_lora_rank=4,
            critic_learning_rate=0.05,
            adv_norm=None,
        )
        trainer = PPOTrainer(
            "test-model",
            config,
            [{"messages": "hello"}],
            lambda **kw: 1.0,
            service_client=service,
            lora_rank=8,
        )
        state = await trainer.train()
        self.assertEqual(
            service.calls,
            [
                dict(base_model="test-model", rank=8, seed=42),
                dict(base_model="test-model", rank=4, seed=43),
            ],
        )
        self.assertIsInstance(trainer.critic, PyTRIOCritic)
        self.assertIsNot(trainer._client, trainer.critic.client)
        self.assertEqual(
            (state.global_step, state.optimizer_steps, state.critic_optimizer_steps),
            (1, 1, 1),
        )
        self.assertEqual(state.checkpoint_path, "trio://ppo-step-00000001")
        self.assertEqual(
            state.critic_checkpoint_path, "trio://ppo-critic-step-00000001"
        )
        self.assertNotEqual(service.critic.parameter.item(), 0.2)
        self.assertEqual(service.actor.version, 1)
        self.assertIn("critic/loss", state.metrics)
        self.assertNotEqual(state.metrics["train/loss"], 999)

    async def test_causal_values_cannot_see_future_actions(self):
        critic = PyTRIOCritic(CriticClient(), PPOConfig(), (1, 0))
        t = trajectory(n=3)
        changed = replace(t, completion_tokens=[20, 21, 99])
        before, after = await critic.compute_values([t, changed])
        self.assertEqual(len(before), 4)
        np.testing.assert_allclose(before[:-1], after[:-1])
        self.assertNotEqual(before[-1], after[-1])
        self.assertAlmostEqual(before[0], 0.2 * (10 + 11) / 100, places=6)

    async def test_critic_really_learns_return_and_keeps_actor_untouched(self):
        client, actor = CriticClient(), ActorClient()
        c = PPOConfig(critic_learning_rate=0.2)
        critic = PyTRIOCritic(client, c, (1, 0))
        t = trajectory(n=1)
        initial = (await critic.compute_values([t]))[0][0]
        for _ in range(6):
            t.values = (await critic.compute_values([t]))[0]
            await critic.update([t], [[1]], [[True]])
        final = (await critic.compute_values([t]))[0][0]
        self.assertLess((final - 1) ** 2, (initial - 1) ** 2)
        self.assertEqual(critic.optimizer_steps, 6)
        self.assertEqual(actor.version, 0)

    async def test_gae_uses_critic_values_and_freezes_returns_for_both_updates(self):
        service = Service()
        c = PPOConfig(
            adv_norm=None, discount=0.9, gae_lambda=0.8, critic_update_epochs=2
        )
        trainer = PPOTrainer("test-model", c, service_client=service)
        await trainer.initialize()
        observed = {}
        prepare = trainer._actor.prepare

        async def capture(ts):
            batch = await prepare(ts)
            observed["batch"] = batch
            return batch

        trainer._actor.prepare = capture
        t = trajectory(n=2, truncated=True)
        await trainer.step([t])
        v0, v1, bootstrap = t.values
        delta1 = 1 + 0.9 * bootstrap - v1
        expected = [0.9 * v1 - v0 + 0.9 * 0.8 * delta1, delta1]
        np.testing.assert_allclose(observed["batch"].advantages[0], expected)
        np.testing.assert_allclose(
            observed["batch"].returns[0], [expected[0] + v0, expected[1] + v1]
        )
        self.assertEqual(trainer.state.critic_optimizer_steps, 2)
        await trainer.close()

    async def test_actor_and_critic_update_concurrently_after_values_ready(self):
        actor, critic_client = ActorClient(), CriticClient()
        c = PPOConfig(critic_token_ids=(1, 0), adv_norm=None)
        trainer = PPOTrainer(
            "test-model", c, training_client=actor, critic_training_client=critic_client
        )
        await trainer.initialize()
        actor_started, critic_started = asyncio.Event(), asyncio.Event()
        actor_update, critic_update = trainer._actor.update, trainer.critic.update

        async def a(batch):
            self.assertIsNotNone(batch.trajectories[0].values)
            actor_started.set()
            await asyncio.wait_for(critic_started.wait(), timeout=1)
            return await actor_update(batch)

        async def b(ts, returns, masks):
            critic_started.set()
            await asyncio.wait_for(actor_started.wait(), timeout=1)
            return await critic_update(ts, returns, masks)

        trainer._actor.update, trainer.critic.update = a, b
        await trainer.step([trajectory()])
        await trainer.close()

    async def test_separate_reporters_and_both_finish_on_error(self):
        actor_run, critic_run = Mock(), Mock()
        service = Service()
        config = PPOConfig(report_to="swanlab", run_name="pair")
        trainer = PPOTrainer(
            "test-model",
            config,
            [{"messages": "hi"}],
            lambda **kw: 1,
            service_client=service,
        )

        async def failed(*args, **kwargs):
            raise RuntimeError("critic backend failed")

        service.critic.forward_backward_custom_async = failed
        with (
            patch(
                "triotune.trainer.ppo.trainer.MetricsAPI.from_config",
                return_value=MetricsAPI(actor_run),
            ) as actor_factory,
            patch(
                "triotune.trainer.ppo.trainer.critic_metrics",
                return_value=MetricsAPI(critic_run),
            ) as critic_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "critic backend"):
                await trainer.train()
        self.assertEqual(actor_factory.call_args.args[0].run_name, "pair-actor")
        self.assertEqual(critic_factory.call_args.kwargs["name"], "pair-critic")
        self.assertEqual(critic_factory.call_args.kwargs["metadata"]["role"], "critic")
        actor_run.finish.assert_called_once()
        critic_run.finish.assert_called_once()
        self.assertEqual(actor_run.finish.call_args.kwargs["state"], "crashed")
        self.assertEqual(critic_run.finish.call_args.kwargs["state"], "crashed")

    async def test_critic_failure_tracks_partial_updates_and_shared_client_rejected(
        self,
    ):
        client = ActorClient()
        with self.assertRaisesRegex(ValueError, "independent"):
            PPOTrainer(
                "test-model", training_client=client, critic_training_client=client
            )
        with self.assertRaisesRegex(ValueError, "use_critic"):
            PPOTrainer(
                "test-model",
                PPOConfig.for_algorithm("grpo"),
                critic_training_client=CriticClient(),
            )
        critic_client = CriticClient()
        c = PPOConfig(critic_token_ids=(1, 0), critic_update_epochs=2)
        trainer = PPOTrainer(
            "test-model",
            c,
            training_client=client,
            critic_training_client=critic_client,
        )
        backward = critic_client.forward_backward_custom_async

        async def fail_second(data, loss_fn):
            if critic_client.version == 1:
                raise RuntimeError("second critic update failed")
            return await backward(data, loss_fn)

        critic_client.forward_backward_custom_async = fail_second
        with self.assertRaisesRegex(RuntimeError, "second critic"):
            await trainer.step([trajectory()])
        self.assertEqual(trainer.state.critic_optimizer_steps, 1)
        self.assertEqual(trainer.state.global_step, 0)
        await trainer.close(state="crashed")

    async def test_real_sdk_bridge_transmits_both_value_gradients(self):
        from pytrio.client.training import TrainingClient as SDKTrainingClient

        client = object.__new__(SDKTrainingClient)
        t, config = trajectory(n=1, values=[0, 0]), PPOConfig()
        sent = []

        async def forward(**kwargs):
            return Future(
                SimpleNamespace(
                    loss_fn_outputs=[
                        {
                            "logprobs": TensorData(
                                data=[0.0, -1.0, -1.0], dtype="float32", shape=[3]
                            )
                        }
                        for _ in range(2)
                    ]
                )
            )

        async def backward(**kwargs):
            sent.extend(kwargs["data"])
            return Future(SimpleNamespace(metrics={}, loss_fn_outputs=[]))

        client._forward, client._forward_backward = forward, backward
        fn = make_value_loss_fn([t], [[1]], [[True]], config, (1, 0))
        future = await client._forward_backward_custom(
            critic_datums([t], (1, 0)), fn, None
        )
        output = await future
        self.assertAlmostEqual(output.metrics["critic/loss"], 0.5)
        self.assertEqual(sent[0].loss_fn_inputs["weights"].data, [0, 1, 0])
        self.assertEqual(sent[1].loss_fn_inputs["weights"].data, [0, -1, 0])


class ReporterProcessTests(unittest.TestCase):
    def test_two_swanlab_runs_without_closing_actor(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"SWANLAB_MODE": "disabled"}),
        ):
            actor = swanlab.init(
                project="ppo-test", name="actor", mode="disabled", log_dir=folder
            )
            metrics = None
            try:
                metrics = critic_metrics(
                    PPOConfig(report_to="swanlab", log_dir=folder),
                    name="critic",
                    metadata={"role": "critic"},
                )
                self.assertNotEqual(metrics._run.process.pid, os.getpid())
                self.assertIsNone(metrics._run.process.poll())
                metrics.record({"critic/loss": 1.0}, step=1)
                metrics.log_text(key="checkpoint/path", data="trio://critic", step=1)
                process = metrics._run.process
                metrics.finish()
                self.assertEqual(process.returncode, 0)
                # A critic run must not call reinit=True and finish this run.
                actor.log({"actor/loss": 0.5}, step=1)
            finally:
                if metrics is not None:
                    metrics.finish(state="crashed")
                actor.finish()


if __name__ == "__main__":
    unittest.main()
