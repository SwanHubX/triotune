"""PPO-family numerical, SDK-gradient and asynchronous integration contracts."""

import asyncio
import json
import math
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pytrio.types.forward.tensor_data import TensorData
from triotune.trainer.ppo import (
    NormConfig,
    PPOActor,
    PPOConfig,
    PPOTrainer,
    PreparedBatch,
    Trajectory,
    compute_gae,
    compute_token_level_gae,
    compute_turn_level_gae,
    prepare_advantages,
    value_loss,
)
from triotune.trainer.ppo.actor import minibatches
from triotune.trainer.ppo.datum import batch_datums, build_datum
from triotune.trainer.ppo.loss import make_loss_fn
from triotune.trainer.ppo.normalization import normalize


def trajectory(reward=1.0, group="g", n=2, **kwargs):
    return Trajectory(
        [10, 11], list(range(20, 20 + n)), [-1.0] * n, reward, group, **kwargs
    )


def raw_config(**kwargs):
    return PPOConfig(
        **{"use_critic": False, "reward_norm": None, "adv_norm": None, **kwargs}
    )


def objective(config, trajectories, values, advantages=None):
    batch = prepare_advantages(trajectories, config)
    if advantages is not None:
        batch.advantages = advantages
    tensors = [
        torch.tensor(
            [0.0] * (len(t.prompt_tokens) - 1) + lp,
            dtype=torch.float64,
            requires_grad=True,
        )
        for t, lp in zip(trajectories, values, strict=True)
    ]
    loss, metrics = make_loss_fn(batch, config)(batch_datums(batch), tensors)
    loss.backward()
    return loss.item(), [lp.grad.tolist() for lp in tensors], metrics


class AdvantageTests(unittest.TestCase):
    def test_singleton_group_normalization_uses_areal_fallbacks(self):
        self.assertEqual(
            normalize([[1, 3]], ["a"], NormConfig("group", None, True)), [[0, 0]]
        )
        np.testing.assert_allclose(
            normalize([[1, 3]], ["a"], NormConfig("group", "group")),
            [[-1 / 1.00001, 1 / 1.00001]],
        )
        self.assertEqual(normalize([], [], NormConfig()), [])

    def test_variable_groups_and_leave_one_out(self):
        rows, ids = [[1.0], [3.0], [2.0], [4.0], [6.0]], ["a", "a", "b", "b", "b"]
        self.assertEqual(
            normalize(rows, ids, NormConfig("group", None)),
            [[-1.0], [1.0], [-2.0], [0.0], [2.0]],
        )
        self.assertEqual(
            normalize(rows, ids, NormConfig("group", None, True)),
            [[-2.0], [2.0], [-3.0], [0.0], [3.0]],
        )
        scaled = normalize(rows, ids, NormConfig("group", "batch"))
        np.testing.assert_allclose(
            np.asarray(scaled).ravel(),
            np.array([-1, 1, -2, 0, 2]) / (math.sqrt(2.5) + 1e-5),
        )

    def test_reward_presets_do_not_cross_prompt_groups(self):
        ts = [trajectory(1), trajectory(3), trajectory(9, "other", n=3)]
        grpo = prepare_advantages(ts, PPOConfig.for_algorithm("grpo", adv_norm=None))
        rloo = prepare_advantages(ts, PPOConfig.for_algorithm("rloo", adv_norm=None))
        np.testing.assert_allclose(grpo.advantages[0], [-1 / (math.sqrt(2) + 1e-5)] * 2)
        self.assertEqual(rloo.advantages, [[-2, -2], [2, 2], [0, 0, 0]])

    def test_gae_terminal_bootstrap_and_tool_mask(self):
        args = ([0, 99, 1], [0.2, 99, 0.4, 0.8], [True, False, True])
        adv, ret = compute_gae(*args, discount=0.9, gae_lambda=0.8)
        np.testing.assert_allclose(adv, [0.592, 0, 0.6])
        np.testing.assert_allclose(ret, [0.792, 0, 1])
        adv, ret = compute_gae(*args, discount=0.9, gae_lambda=0.8, truncated=True)
        np.testing.assert_allclose(adv, [1.1104, 0, 1.32])
        np.testing.assert_allclose(ret, [1.3104, 0, 1.72])

    def test_token_and_turn_gae_advance_at_different_timestep_units(self):
        rewards = [0, 1, 0, 2]
        values = [0] * 5
        mask = [True] * 4

        token_adv, token_returns = compute_token_level_gae(
            rewards, values, mask, gae_lambda=0.5
        )
        turn_adv, turn_returns = compute_turn_level_gae(
            rewards, values, mask, [0, 0, 1, 1], gae_lambda=0.5
        )

        self.assertEqual(token_adv, [0.75, 1.5, 1.0, 2.0])
        self.assertEqual(token_returns, token_adv)
        self.assertEqual(turn_adv, [2.0, 2.0, 2.0, 2.0])
        self.assertEqual(turn_returns, turn_adv)

    def test_turn_gae_skips_masked_gaps_and_uses_first_turn_value(self):
        advantages, returns = compute_turn_level_gae(
            [0, 3, 0, 0, 4],
            [1, 5, 9, 2, 8, 8],
            [True, True, False, True, True],
            [0, 0, -1, 2, 2],
            discount=0.5,
            gae_lambda=0.25,
            truncated=True,
        )

        self.assertEqual(advantages, [3.75, 3.75, 0.0, 6.0, 6.0])
        self.assertEqual(returns, [4.75, 4.75, 0.0, 8.0, 8.0])

    def test_turn_advantages_keep_kl_token_local_and_returns_task_only(self):
        t = replace(
            trajectory(2, n=4, turn_ids=[0, 0, 1, 1]),
            logprobs=[-1.0, -0.8, -0.6, -1.0],
            reference_logprobs=[-1.0] * 4,
        )
        batch = prepare_advantages(
            [t], raw_config(gae_timestep_unit="turn", kl_ctl=1.0)
        )

        np.testing.assert_allclose(batch.advantages, [[2.0, 1.8, 1.6, 2.0]])
        self.assertEqual(batch.returns, [[2.0, 2.0, 2.0, 2.0]])

    def test_turn_gae_rejects_missing_or_malformed_turn_metadata(self):
        with self.assertRaisesRegex(ValueError, "requires.*turn_ids"):
            compute_gae(
                [1], [0, 0], [True], timestep_unit="turn"
            )
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            trajectory(n=3, turn_ids=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, "turn_id -1"):
            trajectory(n=3, loss_mask=[True, False, True], turn_ids=[0, 0, 1])

    def test_reference_kl_estimators_and_clamp(self):
        t = trajectory(0, reference_logprobs=[-2, -2])
        for estimator, expected in [("k1", 1), ("k2", 0.5), ("k3", math.exp(-1))]:
            batch = prepare_advantages(
                [t], raw_config(kl_ctl=0.1, kl_estimator=estimator)
            )
            np.testing.assert_allclose(
                batch.advantages[0], [-0.2 * expected, -0.1 * expected]
            )
        extreme = replace(t, reference_logprobs=[-100, -100])
        self.assertEqual(
            prepare_advantages([extreme], raw_config(kl_ctl=1)).advantages, [[-20, -10]]
        )

    def test_overlong_scaling_and_no_eos_zero_outcome_keep_kl(self):
        t = trajectory(4, n=4, truncated=True, reference_logprobs=[-2] * 4)
        c = raw_config(max_tokens=4, overlong_tokens=2, reward_scaling=2, reward_bias=1)
        self.assertEqual(prepare_advantages([t], c).advantages, [[8] * 4])
        batch = prepare_advantages(
            [t], replace(c, mask_no_eos_with_zero=True, kl_ctl=1)
        )
        self.assertEqual(batch.masks, [[True] * 4])
        self.assertEqual(batch.advantages, [[-4, -3, -2, -1]])

    def test_missing_values_reference_and_config_fail_early(self):
        with self.assertRaisesRegex(ValueError, "value predictions"):
            prepare_advantages([trajectory()], PPOConfig())
        with self.assertRaisesRegex(ValueError, "fixed-reference"):
            prepare_advantages([trajectory()], raw_config(kl_ctl=0.1))
        self.assertTrue(PPOTrainer("fake").args.use_critic)
        for kwargs in (
            {"temperature": 0.7},
            {"top_p": 0.9},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                raw_config(**kwargs)
        for kwargs in (
            {"loss_backend": "native", "loss_type": "sapo"},
            {"ppo_n_minibatches": 0},
            {"reward_norm": {}},
            {"rejection_upper": 2},
            {"c_clip": 1},
            {"gae_timestep_unit": "episode"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                raw_config(**kwargs)


class PolicyLossTests(unittest.TestCase):
    def test_policy_gradients_match_finite_difference(self):
        t = trajectory()
        values = [-0.9, -1.1]
        for config in (
            raw_config(),
            raw_config(loss_type="sapo"),
            raw_config(importance_sampling_level="sequence"),
        ):
            with self.subTest(
                loss=config.loss_type, ratio=config.importance_sampling_level
            ):
                _, grads, _ = objective(config, [t], [values], [[0.3, -0.2]])
                numeric = []
                for i in range(2):
                    plus, minus = list(values), list(values)
                    plus[i] += 1e-6
                    minus[i] -= 1e-6
                    a, _, _ = objective(config, [t], [plus], [[0.3, -0.2]])
                    b, _, _ = objective(config, [t], [minus], [[0.3, -0.2]])
                    numeric.append((a - b) / 2e-6)
                np.testing.assert_allclose(grads[0][1:], numeric, rtol=1e-6, atol=1e-8)

    def test_invalid_shapes_and_overflow_raise_before_backward(self):
        with self.assertRaisesRegex(FloatingPointError, "ratios"):
            objective(raw_config(), [trajectory()], [[1000, 1000]])
        batch = prepare_advantages([trajectory()], raw_config())
        with self.assertRaisesRegex(ValueError, "one logprob"):
            make_loss_fn(batch, raw_config())([], [])
        with self.assertRaisesRegex(ValueError, "causally"):
            make_loss_fn(batch, raw_config())(
                batch_datums(batch), [torch.tensor([-1.0], requires_grad=True)]
            )

    def test_ppo_clipping_asymmetric_and_dual_clipping_gradients(self):
        t = trajectory(n=4)
        lp = [[-1 + math.log(r) for r in (1.5, 0.5, 0.5, 4)]]
        loss, grad, _ = objective(raw_config(), [t], lp, [[1, -1, 1, -1]])
        self.assertAlmostEqual(loss, (-1.2 + 0.8 - 0.5 + 4) / 4)
        np.testing.assert_allclose(grad[0], [0, 0, 0, -0.125, 1])
        loss, grad, _ = objective(
            raw_config(c_clip=2, eps_clip_higher=0.4), [t], lp, [[1, -1, 1, -1]]
        )
        self.assertAlmostEqual(loss, (-1.4 + 0.8 - 0.5 + 2) / 4)
        self.assertEqual(grad[0][-1], 0)

    def test_policy_drift_metrics_cover_ratio_and_approximate_kl(self):
        ratios = (0.5, 1.0, 2.0)
        _, _, metrics = objective(
            raw_config(),
            [trajectory(n=3)],
            [[-1 + math.log(value) for value in ratios]],
            [[1, 1, 1]],
        )
        self.assertAlmostEqual(metrics["policy/ratio"], sum(ratios) / 3)
        self.assertEqual(metrics["policy/ratio_max"], 2.0)
        self.assertAlmostEqual(metrics["policy/clip_fraction_low"], 1 / 3)
        self.assertAlmostEqual(metrics["policy/clip_fraction_high"], 1 / 3)
        self.assertGreater(metrics["policy/update_approx_kl"], 0)

    def test_gspo_uses_geometric_ratio_and_mean_advantage(self):
        t = trajectory()
        loss, grad, _ = objective(
            raw_config(importance_sampling_level="sequence", eps_clip=0.8),
            [t],
            [[-1 + math.log(2), -1 + math.log(0.5)]],
            [[1, 3]],
        )
        self.assertAlmostEqual(loss, -2)
        np.testing.assert_allclose(grad[0], [0, -1, -1])

    def test_sapo_gradient_at_ratio_one_and_cispo_outside_clip(self):
        loss, grad, _ = objective(
            raw_config(loss_type="sapo"), [trajectory()], [[-1, -1]], [[1, -1]]
        )
        np.testing.assert_allclose(grad[0], [0, -0.5, 0.5])
        self.assertTrue(math.isfinite(loss))
        c = raw_config(loss_type="cispo", eps_clip=1, eps_clip_higher=4)
        _, grad, _ = objective(c, [trajectory(n=1)], [[-1 + math.log(8)]], [[2]])
        np.testing.assert_allclose(grad[0], [0, -10])

    def test_reductions_different_lengths_and_groups(self):
        ts = [trajectory(1, n=1), trajectory(3, n=3), trajectory(5, "b", n=2)]
        values = [[-1] * len(t.completion_tokens) for t in ts]
        for mode, expected in [
            ("token_mean", -20 / 6),
            ("seq_mean", -3),
            ("prompt_mean", -3.75),
            ("constant", -20 / 12),
        ]:
            c = raw_config(
                loss_aggregation=mode,
                loss_aggregation_divisor=4 if mode == "constant" else None,
            )
            loss, grad, _ = objective(c, ts, values)
            self.assertAlmostEqual(loss, expected)
            self.assertTrue(all(row[0] == 0 for row in grad))

    def test_rejection_keeps_original_denominator_and_masked_gradients_zero(self):
        ts = [trajectory(loss_mask=[True, False]), trajectory()]
        batch = prepare_advantages(ts, raw_config())
        batch.proximal_logprobs = [[-1, -1], [-1 + math.log(2)] * 2]
        c = raw_config(use_decoupled_loss=True, rejection_upper=1.5)
        tensors = [
            torch.tensor([0, -1, -1], dtype=torch.float64, requires_grad=True),
            torch.tensor(
                [0, -1 + math.log(2), -1 + math.log(2)],
                dtype=torch.float64,
                requires_grad=True,
            ),
        ]
        loss, m = make_loss_fn(batch, c)(batch_datums(batch), tensors)
        loss.backward()
        self.assertAlmostEqual(loss.item(), -1 / 3)
        np.testing.assert_allclose(tensors[0].grad, [0, -1 / 3, 0])
        np.testing.assert_allclose(tensors[1].grad, [0, 0, 0])
        self.assertAlmostEqual(m["policy/rejected_fraction"], 2 / 3, places=6)

    def test_zero_mask_rows_and_value_loss(self):
        t = trajectory()
        batch = PreparedBatch(
            [t, t],
            [[1, 1], [0, 0]],
            [[0, 0]] * 2,
            [[True, True], [False, False]],
            [[-1, -1]] * 2,
        )
        tensors = [torch.tensor([0.0, -1, -1], requires_grad=True) for _ in range(2)]
        loss, _ = make_loss_fn(batch, raw_config())(batch_datums(batch), tensors)
        loss.backward()
        np.testing.assert_allclose(tensors[1].grad, [0, 0, 0])
        values = torch.tensor([0.5, 10.0], requires_grad=True)
        loss = value_loss(
            values, torch.zeros(2), torch.ones(2), torch.tensor([True, False])
        )
        self.assertAlmostEqual(loss.item(), 0.32)
        loss.backward()
        self.assertEqual(values.grad.tolist(), [0, 0])

    def test_prompt_minibatch_scaling_with_uneven_group_count(self):
        ts = [trajectory(1, "a", n=1), trajectory(3, "b", n=3), trajectory(5, "c", n=2)]
        c = raw_config(loss_aggregation="prompt_mean")
        batch = prepare_advantages(ts, c)
        losses = []
        for chunk in minibatches(batch, 2):
            tensors = [
                torch.tensor(
                    [0.0] + [-1.0] * len(t.completion_tokens), requires_grad=True
                )
                for t in chunk.trajectories
            ]
            loss, _ = make_loss_fn(chunk, c)(batch_datums(chunk), tensors)
            losses.append(loss.item())
        self.assertAlmostEqual(sum(losses) / 2, -3, places=6)

    def test_shift_and_native_inputs_preserve_tool_context(self):
        t = trajectory(
            n=3,
            loss_mask=[True, False, True],
            turn_ids=[0, -1, 1],
        )
        d = build_datum(t, advantages=[1, 0, 2], old_logprobs=t.logprobs)
        self.assertEqual(d.model_input.to_ints(), [10, 11, 20, 21])
        self.assertEqual(d.loss_fn_inputs["target_tokens"].data, [-100, 20, -100, 22])
        self.assertEqual(d.loss_fn_inputs["advantages"].data, [0, 1, 0, 2])
        self.assertNotIn("turn_ids", d.loss_fn_inputs)


class Future:
    def __init__(self, value, trace=None, label=""):
        self.value, self.trace, self.label = value, trace, label

    def __await__(self):
        async def resolve():
            if self.trace is not None:
                self.trace.append(f"resolve:{self.label}")
            await asyncio.sleep(0)
            return self.value

        return resolve().__await__()

    def map(self, fn):
        return Future(fn(self.value))


class Tokenizer:
    def encode(self, text, **kwargs):
        return [10, 11]

    def decode(self, tokens, **kwargs):
        return str(tokens)


class Sampler:
    def __init__(self, trace, version=0):
        self.trace, self.version, self.seeds = trace, version, []

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.trace.append(f"sample:{self.version}")
        self.seeds.append(sampling_params.seed)
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=[20 + i, 30],
                    logprobs=[-1.0, -1.0],
                    text=str(i),
                    stop_reason="stop",
                )
                for i in range(num_samples)
            ]
        )

    async def compute_logprobs_async(self, *, prompt):
        return [None] + [-2.0] * (len(prompt.to_ints()) - 1)


class TrainingClient:
    def __init__(self):
        self.trace, self.calls, self.snapshots = [], [], []
        self.version = 0

    def get_tokenizer(self):
        return Tokenizer()

    async def forward_async(self, data, **kwargs):
        self.calls.append(("forward", data, kwargs))
        self.trace.append("submit:forward")
        result = SimpleNamespace(
            loss_fn_outputs=[
                {
                    "logprobs": TensorData(
                        data=[-1.2] * len(d.loss_fn_inputs["target_tokens"].data),
                        dtype="float32",
                        shape=[len(d.loss_fn_inputs["target_tokens"].data)],
                    )
                }
                for d in data
            ]
        )
        return Future(result, self.trace, "forward")

    async def forward_backward_custom_async(self, data, loss_fn):
        self.calls.append(("custom", data, loss_fn))
        self.trace.append("submit:custom")
        tensors = [
            torch.tensor(
                [-1.0] * len(d.loss_fn_inputs["target_tokens"].data), requires_grad=True
            )
            for d in data
        ]
        loss, metrics = loss_fn(data, tensors)
        loss.backward()
        metrics["loss"] = 999.0  # Remote CE proxy must not replace custom loss.
        return Future(SimpleNamespace(metrics=metrics), self.trace, "custom")

    async def forward_backward_async(self, data, **kwargs):
        self.calls.append(("native", data, kwargs))
        self.trace.append("submit:native")
        return Future(SimpleNamespace(metrics={"loss": 0.0}), self.trace, "native")

    async def optim_step_async(self, params):
        self.trace.append("submit:optim")
        self.version += 1
        return Future(SimpleNamespace(metrics={"grad_norm": 0.2}), self.trace, "optim")

    async def save_weights_and_get_sampling_client_async(self):
        sampler = Sampler(self.trace, self.version)
        self.snapshots.append(sampler)
        return sampler

    async def save_state_async(self, name):
        self.trace.append("submit:save")
        return Future(SimpleNamespace(path=f"trio://{name}"), self.trace, "save")


class AsyncTrainerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reporting_uses_configured_logging_interval(self):
        trainer = PPOTrainer(
            "fake",
            raw_config(logging_step=3),
            training_client=TrainingClient(),
        )
        await trainer.initialize()
        self.assertEqual(trainer._metrics._logging_step, 3)

    async def test_train_reports_progress_and_writes_local_metric_history(self):
        client = TrainingClient()
        with tempfile.TemporaryDirectory() as folder, patch(
            "triotune.trainer.ppo.trainer.tqdm"
        ) as progress:
            trainer = PPOTrainer(
                "fake",
                PPOConfig.for_algorithm(
                    "grpo",
                    batch_size=1,
                    num_generations=2,
                    max_steps=1,
                    log_dir=folder,
                ),
                [{"messages": "train"}],
                lambda **kwargs: {
                    "reward": float(kwargs["completion"]),
                    "correct": bool(float(kwargs["completion"])),
                },
                training_client=client,
            )

            state = await trainer.train()
            path = trainer._metrics.history_path
            with open(path, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]

        progress.return_value.update.assert_called_once_with(1)
        postfix = progress.return_value.set_postfix.call_args.args[0]
        self.assertEqual(set(postfix), {"loss", "reward", "grad_norm", "tokens"})
        self.assertEqual(rows[0]["event"], "run")
        self.assertEqual(rows[-1]["event"], "finish")
        self.assertEqual(rows[-1]["metrics"]["reward/mean"], 0.5)
        self.assertEqual(rows[-1]["metrics"]["reward/correct/mean"], 0.5)
        self.assertEqual(rows[-1]["metrics"]["reward/success_rate"], 0.5)
        self.assertIn("policy/entropy_sampled", rows[-1]["metrics"])
        self.assertIn("advantage/std", rows[-1]["metrics"])
        self.assertEqual(rows[-1]["metrics"]["rollout/completion_length_mean"], 2)
        self.assertIn("train/loss", rows[-1]["metrics"])
        self.assertEqual(rows[-1]["metrics"]["usage/train_tokens_total"], 12.0)
        self.assertEqual(rows[-1]["metrics"]["usage/actor_train_tokens"], 12.0)
        self.assertEqual(rows[-1]["metrics"]["usage/actor_train_tokens_total"], 12.0)
        self.assertEqual(rows[-1]["metrics"]["usage/critic_train_tokens"], 0.0)
        self.assertEqual(rows[-1]["metrics"]["usage/critic_train_tokens_total"], 0.0)

    async def test_evaluate_samples_rewards_without_optimizer_updates(self):
        client = TrainingClient()
        config = PPOConfig.for_algorithm(
            "cispo",
            batch_size=2,
            num_generations=2,
            eval_batch_size=1,
            eval_max_batches=2,
            seed=7,
        )
        trainer = PPOTrainer(
            "fake",
            config,
            reward_fn=lambda **kwargs: float(kwargs["completion"]),
            eval_dataset=[{"messages": "a"}, {"messages": "b"}, {"messages": "c"}],
            training_client=client,
        )

        with patch(
            "triotune.trainer.ppo.trainer.tqdm",
            side_effect=lambda batches, **kwargs: batches,
        ) as progress:
            metrics = await trainer.evaluate()

        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(trainer.state.optimizer_steps, 0)
        self.assertEqual(trainer._last_eval_step, 0)
        self.assertEqual(metrics["eval/reward_mean"], 0.5)
        self.assertEqual(metrics["eval/reward_std"], 0.5)
        self.assertEqual(metrics["eval/prompts"], 2.0)
        self.assertEqual(metrics["eval/trajectories"], 4.0)
        self.assertEqual(metrics["eval/batches"], 2.0)
        self.assertEqual(metrics["eval/completion_tokens"], 8.0)
        self.assertEqual(metrics["eval/completion_length_mean"], 2.0)
        self.assertEqual(metrics["eval/truncated_fraction"], 0.0)
        self.assertEqual(metrics["usage/eval_sample_input_tokens"], 4.0)
        self.assertEqual(metrics["usage/eval_sample_output_tokens"], 8.0)
        self.assertEqual(trainer._actor.train_tokens, 0)
        self.assertFalse(any("optim" in event for event in client.trace))
        self.assertEqual(
            progress.call_args.kwargs,
            {"total": 2, "desc": "PPO eval", "unit": "batch"},
        )

    async def test_train_schedules_before_periodic_and_final_evaluation(self):
        client = TrainingClient()
        config = PPOConfig.for_algorithm(
            "grpo",
            batch_size=1,
            num_generations=2,
            max_steps=3,
            eval_steps=2,
            eval_before_train=True,
            eval_after_train=True,
        )
        trainer = PPOTrainer(
            "fake",
            config,
            [{"messages": "train"}] * 3,
            lambda **kwargs: float(kwargs["completion"]),
            eval_dataset=[{"messages": "eval"}],
            training_client=client,
        )
        evaluated_at = []
        evaluate = trainer.evaluate

        async def track_evaluation(*args, **kwargs):
            evaluated_at.append(trainer.state.global_step)
            return await evaluate(*args, **kwargs)

        trainer.evaluate = track_evaluation
        state = await trainer.train()

        self.assertEqual(evaluated_at, [0, 2, 3])
        self.assertEqual(state.global_step, 3)
        self.assertEqual(state.optimizer_steps, 3)
        self.assertEqual(state.metrics["eval/prompts"], 1.0)

    async def test_evaluation_requires_reward_and_nonempty_data(self):
        trainer = PPOTrainer(
            "fake",
            raw_config(),
            eval_dataset=[],
            training_client=TrainingClient(),
        )
        with self.assertRaisesRegex(ValueError, "reward_fn"):
            await trainer.evaluate()
        trainer.reward_fn = lambda **kwargs: 1.0
        with self.assertRaisesRegex(ValueError, "at least one prompt"):
            await trainer.evaluate()

    async def test_end_to_end_rollouts_policy_refresh_seeds_and_checkpoints(self):
        client = TrainingClient()
        c = PPOConfig.for_algorithm(
            "grpo", batch_size=2, num_generations=2, max_steps=2, seed=42
        )
        trainer = PPOTrainer(
            "fake",
            c,
            [{"messages": "hello"}] * 4,
            lambda **kw: float(kw["completion"]),
            training_client=client,
        )
        self.assertIsNone(trainer._actor)
        result = await trainer.train()
        self.assertEqual((result.global_step, result.optimizer_steps), (2, 2))
        self.assertEqual([s.version for s in client.snapshots], [0, 1])
        self.assertEqual([s.seeds for s in client.snapshots], [[42, 43], [44, 45]])
        self.assertEqual(result.checkpoint_path, "trio://ppo-step-00000002")
        self.assertEqual(result.metrics["reward/mean"], 0.5)
        self.assertNotEqual(result.metrics["train/loss"], 999.0)
        self.assertEqual(result.metrics["train/action_tokens"], 8)
        self.assertEqual(result.metrics["usage/train_tokens"], 24)
        self.assertEqual(result.metrics["usage/train_tokens_total"], 48)
        self.assertEqual(result.metrics["usage/actor_train_tokens"], 24)
        self.assertEqual(result.metrics["usage/actor_train_tokens_total"], 48)
        self.assertEqual(result.metrics["usage/critic_train_tokens"], 0)
        self.assertEqual(result.metrics["usage/critic_train_tokens_total"], 0)
        self.assertEqual(result.metrics["usage/sample_input_tokens"], 4)
        self.assertEqual(result.metrics["usage/sample_output_tokens"], 8)
        self.assertEqual(result.metrics["usage/sample_input_tokens_total"], 8)
        self.assertEqual(result.metrics["usage/sample_output_tokens_total"], 16)
        self.assertEqual(result.metrics["train/grad_norm"], 0.2)
        sequence = [
            event for event in client.trace if "custom" in event or "optim" in event
        ]
        self.assertEqual(
            sequence,
            ["submit:custom", "resolve:custom", "submit:optim", "resolve:optim"] * 2,
        )

    async def test_all_presets_use_same_loop_and_native_threshold_contract(self):
        for name in (
            "grpo",
            "drgrpo",
            "liteppo",
            "rloo",
            "reinforce",
            "reinforce_baseline",
            "gspo",
            "sapo",
            "cispo",
            "dapo",
        ):
            with self.subTest(name=name):
                client = TrainingClient()
                c = PPOConfig.for_algorithm(name, ppo_n_minibatches=2, update_epochs=2)
                trainer = PPOTrainer("fake", c, training_client=client)
                ts = [
                    trajectory(0, "a"),
                    trajectory(1, "a"),
                    trajectory(0, "b"),
                    trajectory(1, "b"),
                ]
                metrics = await trainer.step(ts)
                self.assertEqual(trainer._actor.optimizer_steps, 4)
                self.assertEqual(metrics["usage/actor_train_tokens"], 48)
                self.assertEqual(metrics["usage/actor_train_tokens_total"], 48)
                self.assertEqual(metrics["usage/critic_train_tokens"], 0)
                self.assertEqual(metrics["usage/critic_train_tokens_total"], 0)
                self.assertEqual(metrics["usage/train_tokens_total"], 48)
                await trainer.close()
        client = TrainingClient()
        actor = PPOActor(
            client, raw_config(loss_backend="native", eps_clip=0.1, eps_clip_higher=0.3)
        )
        await actor.update(await actor.prepare([trajectory()]))
        _, data, kwargs = client.calls[0]
        self.assertEqual(kwargs["loss_fn"], "ppo")
        self.assertEqual(
            kwargs["loss_fn_config"],
            {"clip_low_threshold": 0.9, "clip_high_threshold": 1.3},
        )
        self.assertFalse(kwargs["auto_shift"])
        self.assertIn("logprobs", data[0].loss_fn_inputs)
        self.assertEqual(actor.train_tokens, 3)

    async def test_decoupled_vs_ordinary_recomputation_freezes_correct_policy(self):
        for decoupled in (False, True):
            c = raw_config(
                recompute_logprob=True, use_decoupled_loss=decoupled, kl_ctl=1
            )
            actor = PPOActor(TrainingClient(), c)
            batch = await actor.prepare([trajectory(0, reference_logprobs=[-2, -2])])
            self.assertEqual(actor.train_tokens, 3)
            self.assertEqual(batch.proximal_logprobs, [[-1.2, -1.2]])
            self.assertEqual(
                batch.trajectories[0].logprobs, [-1, -1] if decoupled else [-1.2, -1.2]
            )
            np.testing.assert_allclose(
                batch.advantages[0], [-2, -1] if decoupled else [-1.6, -0.8]
            )

    async def test_fixed_reference_and_external_critic_update(self):
        class ExternalCritic:
            async def compute_values(self, ts):
                return [[0.2, 0.4, 0.8] for _ in ts]

            async def update(self, ts, returns, masks):
                self.returns = returns
                return {"critic/loss": 0.25}

        critic, client = ExternalCritic(), TrainingClient()
        trainer = PPOTrainer(
            "fake",
            PPOConfig(adv_norm=None, kl_ctl=0.1),
            critic=critic,
            training_client=client,
            reference_client=Sampler([]),
        )
        metrics = await trainer.step([trajectory()])
        self.assertEqual(metrics["critic/loss"], 0.25)
        self.assertEqual(metrics["policy/reference_kl_sampled"], 1.0)
        self.assertEqual(metrics["reward/kl_penalty_mean"], -0.1)
        np.testing.assert_allclose(critic.returns, [[0.8, 0.9]])

    async def test_dynamic_sampling_refill_and_bounded_failure(self):
        async def reward(**kw):
            return 0.0 if kw["row"].get("boring") else float(kw["completion"])

        rows = [{"messages": "a", "boring": True}, {"messages": "b"}, {"messages": "c"}]
        c = PPOConfig.for_algorithm(
            "dapo", batch_size=2, num_generations=2, max_steps=1
        )
        trainer = PPOTrainer("fake", c, rows, reward, training_client=TrainingClient())
        state = await trainer.train()
        self.assertEqual(state.metrics["train/prompt_groups"], 2)
        self.assertEqual(state.metrics["rollout/rejected_groups"], 1)
        self.assertEqual(state.metrics["rollout/refill_count"], 1)
        self.assertAlmostEqual(state.metrics["rollout/degenerate_group_fraction"], 1 / 3)
        self.assertEqual(trainer._next_group, 3)
        c = replace(c, max_empty_batches=1)
        trainer = PPOTrainer(
            "fake", c, rows, lambda **kw: 0, training_client=TrainingClient()
        )
        with self.assertRaisesRegex(RuntimeError, "uninformative"):
            await trainer.train()

    async def test_inconsistent_groups_empty_data_and_update_failure(self):
        actor = PPOActor(TrainingClient(), raw_config())
        with self.assertRaisesRegex(ValueError, "identical"):
            await actor.prepare(
                [trajectory(), replace(trajectory(), prompt_tokens=[12])]
            )
        trainer = PPOTrainer(
            "fake", raw_config(), [], lambda **kw: 1, training_client=TrainingClient()
        )
        with self.assertRaisesRegex(ValueError, "No trainable"):
            await trainer.train()

        class FailingClient(TrainingClient):
            async def forward_backward_custom_async(self, data, loss_fn):
                if self.version == 1:
                    raise RuntimeError("backend failure")
                return await super().forward_backward_custom_async(data, loss_fn)

        trainer = PPOTrainer(
            "fake", raw_config(update_epochs=2), training_client=FailingClient()
        )
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            await trainer.step([trajectory()])
        self.assertEqual(trainer.state.optimizer_steps, 1)
        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(trainer._actor.train_tokens, 6)

    async def test_real_sdk_custom_bridge_sends_negative_local_gradient(self):
        # Exercise the installed SDK implementation itself, replacing only
        # remote transport. This catches sign/reduction mistakes in CE weights.
        from pytrio.client.training import TrainingClient as SDKTrainingClient

        client = object.__new__(SDKTrainingClient)
        ts = [trajectory(2, n=1)]
        batch = prepare_advantages(ts, raw_config())
        data = batch_datums(batch)
        sent = []

        async def forward(**kwargs):
            return Future(
                SimpleNamespace(
                    loss_fn_outputs=[
                        {
                            "logprobs": TensorData(
                                data=[0.0, -1.0], dtype="float32", shape=[2]
                            )
                        }
                    ]
                )
            )

        async def backward(**kwargs):
            sent.extend(kwargs["data"])
            return Future(SimpleNamespace(metrics={}, loss_fn_outputs=[]))

        client._forward, client._forward_backward = forward, backward
        future = await client._forward_backward_custom(
            data, make_loss_fn(batch, raw_config()), None
        )
        result = await future
        self.assertAlmostEqual(result.metrics["train/loss"], -2)
        self.assertEqual(sent[0].loss_fn_inputs["weights"].data, [0, 2])
        self.assertEqual(sent[0].loss_fn_inputs["target_tokens"].data, [-100, 20])


if __name__ == "__main__":
    unittest.main()
