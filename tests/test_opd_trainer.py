from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

from trio_cookbook import OPDTrainer
from trio_cookbook.trainer import OPDConfig, OPDTeacherGroup, OPDTeacherSpec
from trio_cookbook.trainer.opd.datum import build_opd_datum
from trio_cookbook.trainer.opd.loss import distillation_advantages
from trio_cookbook.trainer.opd.teacher_phase import teacher_completion_logprobs


class _Tokenizer:
    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(message["content"] for message in messages)
        text += "A" if add_generation_prompt else ""
        return self.encode(text) if tokenize else text


class _Future:
    def __init__(self, value=None, *, error=None):
        self.value, self.error = value, error

    def result(self):
        raise AssertionError("Async execution must not call blocking result().")

    async def _resolve(self):
        await asyncio.sleep(0)
        if self.error:
            raise self.error
        return self.value

    def __await__(self):
        return self._resolve().__await__()


class _Sampler:
    def __init__(self, service, version):
        self.service, self.version = service, version
        self.requests = []

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.requests.append((prompt.to_ints(), sampling_params, num_samples))
        self.service.sampling_active += 1
        self.service.sampling_peak = max(self.service.sampling_peak, self.service.sampling_active)
        try:
            await asyncio.sleep(0)
            if self.service.pipeline and prompt.to_ints()[0] == ord("B"):
                # A's teacher request must start before B's generation finishes.
                await self.service.teacher_started.wait()
            if self.service.block_sampling:
                self.service.sample_started.set()
                await asyncio.Event().wait()
            tokens = [65, 66][:sampling_params.max_tokens]
            if self.service.empty_sample:
                tokens = []
            return SimpleNamespace(sequences=[
                SimpleNamespace(tokens=tokens, logprobs=self.service.student_values[:len(tokens)])
                for _ in range(num_samples)
            ])
        finally:
            self.service.sampling_active -= 1


class _Teacher:
    def __init__(self, service, model="teacher"):
        self.service, self.model = service, model
        self.requests = []

    def get_tokenizer(self):
        tokenizer = _Tokenizer()
        if self.service.vocab_mismatch:
            tokenizer.get_vocab = lambda: {"A": 999}
        return tokenizer

    async def compute_logprobs_async(self, model_input):
        tokens = model_input.to_ints()
        self.requests.append(tokens)
        self.service.teacher_started.set()
        self.service.scoring_active += 1
        self.service.scoring_peak = max(self.service.scoring_peak, self.service.scoring_active)
        try:
            await asyncio.sleep(0)
            if self.service.teacher_error:
                raise RuntimeError("teacher failed")
            if self.service.bad_teacher:
                return [None] * len(tokens)
            # Prompt scores are irrelevant; each configured teacher may expose
            # a distinct completion score for ensemble tests.
            value = self.service.teacher_values.get(self.model, -1.0)
            return [None] + [value] * (len(tokens) - 1)
        finally:
            self.service.scoring_active -= 1


class _Training:
    def __init__(self, service):
        self.service = service
        self.datums = []
        self.events = []
        self.saved = []
        self.samplers = []
        self.optim_calls = 0
        self.fail_forward = False
        self.fail_optim = False
        self.block_forward = False
        self.forward_kwargs = None

    def get_tokenizer(self):
        return _Tokenizer()

    async def save_weights_and_get_sampling_client_async(self):
        self.events.append("snapshot")
        sampler = _Sampler(self.service, self.optim_calls)
        self.samplers.append(sampler)
        return _Future(sampler)

    async def forward_backward_async(self, datums, **kwargs):
        self.events.append("forward")
        self.datums = datums
        self.forward_kwargs = kwargs
        if self.block_forward:
            self.service.forward_started.set()
            await asyncio.Event().wait()
        return _Future(SimpleNamespace(metrics={"loss_mean": 0.3}),
                       error=RuntimeError("forward failed") if self.fail_forward else None)

    async def optim_step_async(self, params):
        self.events.append("optim")
        if self.fail_optim:
            raise RuntimeError("optim failed")
        self.optim_calls += 1
        return _Future(SimpleNamespace(metrics={"grad_norm": 0.2}))

    async def save_state_async(self, *, name):
        self.events.append("save")
        self.saved.append(name)
        return _Future(SimpleNamespace(path=f"checkpoint://{name}"))


class _Service:
    def __init__(self):
        self.training = _Training(self)
        self.teacher = _Teacher(self)
        self.teachers = {"teacher": self.teacher}
        self.teacher_values = {}
        self.student_values = [-2.0, -0.5]
        self.teacher_kwargs = None
        self.training_kwargs = None
        self.sampling_active = self.scoring_active = 0
        self.sampling_peak = self.scoring_peak = 0
        self.teacher_started = asyncio.Event()
        self.sample_started = asyncio.Event()
        self.forward_started = asyncio.Event()
        self.pipeline = self.block_sampling = self.empty_sample = False
        self.teacher_error = self.bad_teacher = self.vocab_mismatch = False

    def create_lora_training_client(self, **kwargs):
        self.training_kwargs = kwargs
        return self.training

    def create_sampling_client(self, **kwargs):
        self.teacher_kwargs = kwargs
        model = kwargs["base_model"]
        if model not in self.teachers:
            self.teachers[model] = _Teacher(self, model)
        return self.teachers[model]


class _Run:
    def __init__(self):
        self.logs, self.texts = [], []
        self.finished = None

    def log(self, payload, *, step):
        self.logs.append((payload, step))

    def log_text(self, **kwargs):
        self.texts.append(kwargs)

    def finish(self, *, state, error=None):
        self.finished = state


class OPDSignalTests(unittest.TestCase):
    def test_local_and_discounted_advantages(self):
        self.assertEqual(distillation_advantages([-2, -0.5], [-1, -1], coef=1, discount=0), [1, -0.5])
        self.assertEqual(distillation_advantages([-2, -0.5], [-1, -1], coef=2, discount=1), [1, -1])
        self.assertEqual(distillation_advantages([-2, -0.5], [-1, -1], coef=1, discount=0.5), [0.75, -0.5])

    def test_teacher_alignment_and_single_token_completion(self):
        self.assertEqual(teacher_completion_logprobs([None, None, -2], 2, 1), [-2])
        datum = build_opd_datum([10], [11], [-2], [1])
        self.assertEqual(datum.model_input.to_ints(), [10])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [11])

    def test_sdk_normalization_excludes_prompt_but_keeps_zero_kl_completion(self):
        from pytrio.client.training_batch import plan_training_batch
        from pytrio.model.limits import TrainingLimits

        datum = build_opd_datum([10, 11, 12], [13, 14], [-1, -1], [1, 0])
        plan = plan_training_batch([datum], loss_fn="importance_sampling", auto_shift=False,
                                   limits=TrainingLimits(None, None, None), tokenizer_vocab_size=256)
        self.assertEqual(plan.totals.total_active_tokens, 2)
        self.assertEqual(plan.totals.total_advantage_tokens, 1)

    def test_invalid_signals_fail(self):
        for values in ([None], [math.nan], [math.inf], [0.5], [True], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                distillation_advantages([-2], values, coef=1, discount=0)
        with self.assertRaises(ValueError):
            teacher_completion_logprobs([None, -1], 2, 1)
        with self.assertRaises(ValueError):
            build_opd_datum([True], [1], [-1], [1])
        with self.assertRaises(ValueError):
            build_opd_datum([1], [2, 3], [-1], [1])

    def test_surrogate_gradient_matches_exact_categorical_reverse_kl(self):
        # Enumerate a one-token categorical distribution. Its detached OPD
        # signal must give the exact KL gradient at the behavior policy.
        import numpy as np

        logits = np.array([0.4, -0.2, 0.1])
        teacher = np.array([0.2, 0.5, 0.3])
        student = np.exp(logits) / np.exp(logits).sum()
        advantage = np.array(distillation_advantages(np.log(student).tolist(),
                                                     np.log(teacher).tolist(), coef=1, discount=0))
        surrogate_gradient = -student * (advantage - np.dot(student, advantage))

        def kl(z):
            p = np.exp(z) / np.exp(z).sum()
            return np.dot(p, np.log(p) - np.log(teacher))

        numerical = []
        for index in range(3):
            shift = np.zeros(3)
            shift[index] = 1e-6
            numerical.append((kl(logits + shift) - kl(logits - shift)) / 2e-6)
        np.testing.assert_allclose(surrogate_gradient, numerical, atol=1e-8)


class OPDConfigTests(unittest.TestCase):
    def test_every_parameter_documents_default(self):
        for parameter in fields(OPDConfig):
            self.assertIn(f"{parameter.name} (`", OPDConfig.__doc__)

    def test_invalid_config(self):
        for kwargs in (
            {"batch_size": True}, {"max_steps": 0}, {"max_length": 1},
            {"num_generations": 0}, {"max_concurrent_sampling": 0},
            {"max_concurrent_scoring": 1.5}, {"max_completion_length": 0},
            {"kl_coef": math.nan}, {"kl_coef": 0}, {"discount_factor": 1.1},
            {"discount_factor": math.inf}, {"beta2": 1}, {"learning_rate": 0},
            {"importance_ratio_cap": 0}, {"importance_ratio_cap": math.nan},
            {"importance_ratio_cap": True},
            {"eps": -1}, {"save_every": -1}, {"logging_step": 0}, {"seed": -1},
            {"prompt_key": ""}, {"project": ""}, {"stop": []}, {"stop": "END"},
            {"report_to": "other"}, {"chat_template_kwargs": {"tokenize": True}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OPDConfig(**kwargs)


class OPDTrainerTests(unittest.TestCase):
    def setUp(self):
        self.service, self.run = _Service(), _Run()
        self.patch = patch("trio_cookbook.trainer.opd_trainer.trio.ServiceClient", return_value=self.service)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def trainer(self, **kwargs):
        return OPDTrainer("student", "teacher", args=OPDConfig(**kwargs))

    def test_eval_dataset_schedule_in_sync_and_async_training(self):
        rows = [{"messages": "P"}]
        for async_mode in (False, True):
            with self.subTest(async_mode=async_mode):
                trainer = OPDTrainer(
                    "student",
                    "teacher",
                    args=OPDConfig(
                        batch_size=1,
                        max_steps=4,
                        eval_steps=2,
                        eval_before_train=True,
                        eval_after_train=True,
                    ),
                    train_dataset=rows * 4,
                    eval_dataset=rows,
                )
                evaluated = []
                publish = trainer._publish_evaluation

                def track_evaluation(metrics):
                    evaluated.append(trainer.state.global_step)
                    return publish(metrics)

                trainer._publish_evaluation = track_evaluation
                if async_mode:
                    asyncio.run(trainer.train_async())
                else:
                    trainer.train()

                self.assertEqual(evaluated, [0, 2, 4])
                self.assertEqual(trainer.state.behavior_policy_version, 4)

    def test_sync_step_alignment_and_metrics(self):
        trainer = self.trainer()
        result = trainer.step([{"messages": "PQ"}])
        datum = self.service.training.datums[0]
        self.assertEqual(datum.model_input.to_ints(), [80, 81, 65])
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [-100, 65, 66])
        self.assertEqual(datum.loss_fn_inputs["logprobs"].data, [0, -2, -0.5])
        self.assertEqual(datum.loss_fn_inputs["advantages"].data, [0, 1, -0.5])
        self.assertEqual(self.service.teacher.requests, [[80, 81, 65, 66]])
        self.assertEqual(result.behavior_policy_version, 0)
        self.assertEqual(trainer.state.behavior_policy_version, 1)
        self.assertEqual(result.metrics["distill/reverse_kl_sampled"], -0.25)
        self.assertEqual(result.metrics["train/loss"], 0.3)
        self.assertEqual(result.metrics["train/loss"], result.metrics["loss_mean"])
        self.assertEqual(result.metrics["policy/importance_ratio_cap"], 5.0)
        self.assertEqual(
            self.service.training.forward_kwargs,
            {
                "loss_fn": "cispo",
                "auto_shift": False,
                "loss_fn_config": {
                    "clip_low_threshold": 0.0,
                    "clip_high_threshold": 5.0,
                },
            },
        )
        self.assertEqual(self.service.training.events, ["snapshot", "forward", "optim"])

    def test_none_ratio_cap_uses_unbounded_importance_sampling(self):
        result = self.trainer(importance_ratio_cap=None).step([{"messages": "P"}])

        self.assertEqual(
            self.service.training.forward_kwargs,
            {"loss_fn": "importance_sampling", "auto_shift": False},
        )
        self.assertNotIn("policy/importance_ratio_cap", result.metrics)

    def test_validation_scores_student_and_teacher_without_optimizer(self):
        trainer = self.trainer()

        metrics = trainer.evaluate([{"messages": "PQ"}])

        self.assertEqual(metrics["eval/prompts"], 1.0)
        self.assertEqual(metrics["eval/completions"], 1.0)
        self.assertAlmostEqual(metrics["eval/reverse_kl_sampled"], -0.25)
        self.assertEqual(self.service.training.optim_calls, 0)
        self.assertEqual(trainer.state.global_step, 0)

    def test_identical_teacher_student_keeps_zero_signal_without_group_filtering(self):
        self.service.student_values = [-1, -1]
        result = self.trainer().step([{"messages": "P"}])
        self.assertEqual(result.num_datums, 1)
        self.assertEqual(result.metrics["distill/reverse_kl_sampled"], 0)
        self.assertEqual(self.service.training.datums[0].loss_fn_inputs["advantages"].data, [0, 0])

    def test_invalid_and_misaligned_student_logprobs_fail_before_backward(self):
        for values in ([math.nan, -1], [0.2, -1], [-1]):
            with self.subTest(values=values):
                self.service.student_values = values
                with self.assertRaises(ExceptionGroup):
                    self.trainer().step([{"messages": "P"}])
                self.assertNotIn("forward", self.service.training.events)

    def test_prompt_formats_budget_and_seeds(self):
        trainer = self.trainer(max_length=3, seed=7)
        trainer.step([{"messages": [80, 81]}, {"messages": [{"role": "user", "content": "P"}]}])
        requests = self.service.training.samplers[0].requests
        self.assertEqual([request[1].max_tokens for request in requests], [1, 1])
        self.assertEqual([request[1].seed for request in requests], [7, 8])
        self.assertEqual([request[0] for request in requests], [[80, 81], [80, 65]])
        self.assertEqual(self.service.training.datums[0].loss_fn_inputs["advantages"].data, [0, 1])
        trainer.step([{"messages": "P"}])
        self.assertEqual(self.service.training.samplers[1].requests[0][1].seed, 9)

    def test_fixed_teacher_checkpoint_and_vocab_validation(self):
        OPDTrainer("student", "teacher", teacher_model_path="checkpoint://teacher")
        self.assertEqual(self.service.teacher_kwargs,
                         {"base_model": "teacher", "model_path": "checkpoint://teacher"})
        self.service.vocab_mismatch = True
        with self.assertRaisesRegex(ValueError, "vocabularies"):
            self.trainer()

    def test_multi_teacher_routing_and_weighted_geometric_ensemble(self):
        self.service.teacher_values = {"math-teacher": -1.0, "code-teacher": -3.0}
        trainer = OPDTrainer(
            "student",
            teachers=[
                OPDTeacherSpec("math", "math-teacher", "checkpoint://math"),
                OPDTeacherSpec("code", "code-teacher"),
            ],
            teacher_groups=[
                OPDTeacherGroup("blend", {"math": 2.0, "code": 1.0}),
                OPDTeacherGroup("code-only", {"code": 1.0}),
            ],
        )

        result = trainer.step([{"messages": "P", "teacher_group": "blend"}])

        # The normalized log score (-5/3) is the geometric ensemble used for
        # metrics, while the loss keeps raw group scale W=3: [-5 - 3*p_t].
        datum = self.service.training.datums[0]
        self.assertEqual(datum.loss_fn_inputs["advantages"].data, [1.0, -3.5])
        self.assertAlmostEqual(result.metrics["distill/reverse_kl_sampled"], 1.25)
        self.assertAlmostEqual(result.metrics["distill/teacher_logprob"], -5 / 3)
        self.assertEqual(result.metrics["distill/teacher_weight_sum"], 3.0)
        expected = [[80, 65, 66]]
        self.assertEqual(self.service.teachers["math-teacher"].requests, expected)
        self.assertEqual(self.service.teachers["code-teacher"].requests, expected)

        trainer.step([{"messages": "P", "teacher_group": "code-only"}])
        self.assertEqual(self.service.teachers["math-teacher"].requests, expected)
        self.assertEqual(
            self.service.teachers["code-teacher"].requests,
            expected + expected,
        )

        with self.assertRaisesRegex(ValueError, "must provide"):
            trainer.step([{"messages": "P"}])
        with self.assertRaisesRegex(ValueError, "unknown teacher group"):
            trainer.step([{"messages": "P", "teacher_group": "missing"}])

    def test_multi_teacher_configuration_rejects_ambiguous_or_unknown_routes(self):
        specs = [OPDTeacherSpec("math", "math-teacher")]
        with self.assertRaisesRegex(ValueError, "not both"):
            OPDTrainer("student", "teacher", teachers=specs)
        with self.assertRaisesRegex(ValueError, "unknown IDs"):
            OPDTrainer(
                "student",
                teachers=specs,
                teacher_groups=[OPDTeacherGroup("bad", {"missing": 1.0})],
            )

    def test_train_refreshes_sampler_and_saves_final_partial_batch(self):
        trainer = self.trainer(batch_size=2, seed=10)
        trainer.train_dataset = [{"messages": "P"}] * 3
        state = trainer.train()
        self.assertEqual(state.global_step, 2)
        self.assertEqual([sampler.version for sampler in self.service.training.samplers], [0, 1])
        self.assertEqual(self.service.training.samplers[1].requests[0][1].seed, 12)
        self.assertEqual(self.service.training.saved, ["opd-step-00000002"])
        self.assertEqual(state.checkpoint_path, "checkpoint://opd-step-00000002")

    def test_empty_dataset_and_step_limits(self):
        trainer = self.trainer()
        trainer.train_dataset = []
        self.assertEqual(trainer.train().global_step, 0)
        self.assertEqual(self.service.training.saved, [])
        trainer = self.trainer(max_steps=1, batch_size=1, save_every=1)
        trainer.train_dataset = [{"messages": "P"}] * 3
        self.assertEqual(trainer.train().global_step, 1)
        self.assertEqual(self.service.training.saved, ["opd-step-00000001"])

    def test_invalid_prompts_fail_before_snapshot(self):
        for rows in ([], [{}], [{"messages": []}], [{"messages": [-1]}], [{"messages": "LONG"}]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, TypeError)):
                self.trainer(max_length=3).step(rows)
        self.assertEqual(self.service.training.events, [])

    def test_backward_and_optimizer_failures_require_restart(self):
        for failure in ("fail_forward", "fail_optim"):
            with self.subTest(failure=failure):
                self.service = _Service()
                with patch("trio_cookbook.trainer.opd_trainer.trio.ServiceClient", return_value=self.service):
                    trainer = self.trainer()
                setattr(self.service.training, failure, True)
                with self.assertRaises(RuntimeError):
                    trainer.step([{"messages": "P"}])
                self.assertEqual(trainer.state.global_step, 0)
                self.assertEqual(self.service.training.optim_calls, 0)
                if failure == "fail_forward":
                    self.assertNotIn("optim", self.service.training.events)
                with self.assertRaisesRegex(RuntimeError, "recreate"):
                    trainer.step([{"messages": "P"}])

    def test_swanlab_flush_and_failure_finish(self):
        with patch(
            "trio_cookbook.utils.metrics.swanlab.init",
            return_value=self.run,
        ):
            trainer = self.trainer(report_to="swanlab", logging_step=10, batch_size=1)
        trainer.train_dataset = [{"messages": "P"}] * 2
        trainer.train()
        self.assertEqual(self.run.finished, "success")
        self.assertEqual(self.run.logs[0][1], 2)
        self.assertIn("distill/reverse_kl_sampled", self.run.logs[0][0])
        self.assertEqual(self.run.texts[0]["data"], trainer.state.checkpoint_path)
        self.run = _Run()
        with patch(
            "trio_cookbook.utils.metrics.swanlab.init",
            return_value=self.run,
        ):
            trainer = self.trainer(report_to="swanlab")
        self.service.training.fail_forward = True
        trainer.train_dataset = [{"messages": "P"}]
        with self.assertRaises(RuntimeError):
            trainer.train()
        self.assertEqual(self.run.finished, "crashed")

    def test_teacher_scoring_overlaps_unfinished_rollouts(self):
        self.service.pipeline = True
        trainer = self.trainer(max_concurrent_sampling=2, max_concurrent_scoring=1)

        async def run():
            return await asyncio.wait_for(trainer.step_async([{"messages": "A"}, {"messages": "B"}]), 2)

        self.assertEqual(asyncio.run(run()).num_datums, 2)

    def test_concurrency_limits_multiple_generations_and_order(self):
        trainer = self.trainer(max_concurrent_sampling=2, max_concurrent_scoring=3, num_generations=2)
        rows = [{"messages": [i]} for i in range(10, 20)]
        result = asyncio.run(trainer.step_async(rows))
        self.assertEqual(result.num_datums, 20)
        self.assertEqual(self.service.sampling_peak, 2)
        self.assertEqual(self.service.scoring_peak, 3)
        self.assertEqual([datum.model_input.to_ints()[0] for datum in self.service.training.datums],
                         [i for i in range(10, 20) for _ in range(2)])

    def test_rollout_failures_drain_siblings_and_never_train(self):
        for failure in ("teacher_error", "bad_teacher", "empty_sample"):
            with self.subTest(failure=failure):
                setattr(self.service, failure, True)
                trainer = self.trainer()
                with self.assertRaises(ExceptionGroup):
                    asyncio.run(trainer.step_async([{"messages": "P"}] * 4))
                self.assertNotIn("forward", self.service.training.events)
                self.assertEqual(self.service.sampling_active, 0)
                self.assertEqual(self.service.scoring_active, 0)
                self.assertFalse(trainer._requires_restart)
                setattr(self.service, failure, False)

    def test_cancellation_and_concurrent_call_guard(self):
        trainer = self.trainer()

        async def run():
            self.service.block_sampling = True
            task = asyncio.create_task(trainer.step_async([{"messages": "P"}] * 3))
            await self.service.sample_started.wait()
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                await trainer.step_async([{"messages": "Q"}])
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.service.sampling_active, 0)
            self.assertFalse(trainer._busy)
            self.assertFalse(trainer._requires_restart)
            self.service.block_sampling = False
            self.service.training.block_forward = True
            task = asyncio.create_task(trainer.step_async([{"messages": "P"}]))
            await self.service.forward_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(trainer._requires_restart)
            self.assertEqual(self.service.training.optim_calls, 0)

        asyncio.run(run())

    def test_sync_entry_in_running_loop_gives_actionable_error(self):
        trainer = self.trainer()

        async def run():
            with self.assertRaisesRegex(RuntimeError, "step_async"):
                trainer.step([{"messages": "P"}])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
