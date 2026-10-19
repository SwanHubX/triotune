from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from triotune.dataset import PromptExample
from triotune.trainer import OPDConfig
from triotune.trainer.opd import (
    OPDExecutionPlan,
    OPDMetricReducer,
    OPDRollout,
    OPDTargetBuilder,
    OPDTeacherGroup,
    OPDTeacherSpec,
    OPDTeacherTarget,
    SampledReverseKLLoss,
)
from triotune.trainer.opd.metrics import OPDMetricTotals


class _RolloutPhase:
    def __init__(self):
        self.calls = []

    async def sample_prompt(self, **kwargs):
        prompt = kwargs["prompt"]
        prompt_index = kwargs["prompt_index"]
        self.calls.append(prompt_index)
        return [
            OPDRollout(
                prompt_tokens=prompt.tokens,
                completion_tokens=[100 + prompt_index],
                behavior_logprobs=[-2.0],
                prompt_index=prompt_index,
                sample_index=0,
                behavior_policy_version=kwargs["behavior_policy_version"],
            )
        ]


class _TeacherPhase:
    def __init__(self):
        self.calls = []

    async def score(self, **kwargs):
        rollout = kwargs["rollout"]
        self.calls.append(rollout.prompt_index)
        return OPDTeacherTarget(rollout, [-1.0])


class OPDModuleTests(unittest.TestCase):
    def test_default_plan_binds_loss_configuration_once(self):
        config = OPDConfig(
            kl_coef=2.0,
            discount_factor=0.5,
            importance_ratio_cap=2.5,
        )
        plan = OPDExecutionPlan.from_config(config)

        self.assertEqual(plan.target_builder.loss.coefficient, 2.0)
        self.assertEqual(plan.target_builder.loss.discount_factor, 0.5)
        self.assertEqual(
            plan.objective.request_kwargs(),
            {
                "loss_fn": "cispo",
                "auto_shift": False,
                "loss_fn_config": {
                    "clip_low_threshold": 0.0,
                    "clip_high_threshold": 2.5,
                },
            },
        )
        config.kl_coef = 9.0
        config.importance_ratio_cap = None
        self.assertEqual(plan.target_builder.loss.coefficient, 2.0)
        self.assertEqual(plan.target_builder.loss.importance_ratio_cap, 2.5)

    def test_none_ratio_cap_restores_unbounded_importance_sampling(self):
        plan = OPDExecutionPlan.from_config(
            OPDConfig(importance_ratio_cap=None)
        )

        self.assertEqual(
            plan.objective.request_kwargs(),
            {"loss_fn": "importance_sampling", "auto_shift": False},
        )
        for cap in (0, -1, float("nan"), True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                SampledReverseKLLoss(importance_ratio_cap=cap)

    def test_custom_phases_run_in_stable_order_with_default_target(self):
        rollout = _RolloutPhase()
        teacher = _TeacherPhase()
        loss = SampledReverseKLLoss()
        plan = OPDExecutionPlan(
            rollout_phase=rollout,
            teacher_phase=teacher,
            target_builder=OPDTargetBuilder(loss),
            objective=loss.objective,
            metrics=OPDMetricReducer(),
        )
        prompts = [PromptExample(str(index), [index]) for index in (10, 11, 12)]
        config = SimpleNamespace(
            max_concurrent_sampling=2,
            max_concurrent_scoring=2,
        )

        samples = asyncio.run(
            plan.materialize(
                prompts,
                sampler=object(),
                teacher_clients={"teacher": object()},
                teacher_groups={"default": {"teacher": 1.0}},
                config=config,
                behavior_policy_version=7,
                seed_offset=0,
            )
        )

        self.assertEqual([sample.rollout.prompt_index for sample in samples], [0, 1, 2])
        self.assertEqual([sample.advantages for sample in samples], [[1.0]] * 3)
        self.assertEqual(
            [sample.rollout.behavior_policy_version for sample in samples],
            [7, 7, 7],
        )
        self.assertCountEqual(rollout.calls, [0, 1, 2])
        self.assertCountEqual(teacher.calls, [0, 1, 2])

    def test_streaming_metrics_match_batch_reduction(self):
        rollout = OPDRollout([1], [2], [-2.0], 0, 0, 0)
        target = OPDTeacherTarget(rollout, [-1.0])
        sample = OPDTargetBuilder(SampledReverseKLLoss()).build(target)
        reducer = OPDMetricReducer()
        direct = reducer.evaluation([sample, sample], prompt_count=2)
        totals = OPDMetricTotals()
        totals.add([sample], prompt_count=1)
        totals.add([sample], prompt_count=1)

        self.assertEqual(reducer.evaluation_totals(totals), direct)
        self.assertEqual(direct["eval/reverse_kl_sampled"], -1.0)
        self.assertEqual(direct["eval/advantage_mean"], 1.0)

    def test_contracts_reject_misaligned_scores(self):
        with self.assertRaisesRegex(ValueError, "align"):
            OPDRollout([1], [2, 3], [-1.0], 0, 0, 0)
        rollout = OPDRollout([1], [2], [-1.0], 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "align"):
            OPDTeacherTarget(rollout, [-1.0, -2.0])

    def test_teacher_specs_and_groups_validate_public_multi_teacher_input(self):
        spec = OPDTeacherSpec(" math ", "teacher-model", " checkpoint://math ")
        group = OPDTeacherGroup(" blend ", {"math": 2, "code": 0.5})

        self.assertEqual(spec.teacher_id, "math")
        self.assertEqual(spec.model_path, "checkpoint://math")
        self.assertEqual(group.name, "blend")
        self.assertEqual(dict(group.teachers), {"math": 2.0, "code": 0.5})
        for weights in ({}, {"math": 0}, {"math": float("nan")}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                OPDTeacherGroup("bad", weights)


if __name__ == "__main__":
    unittest.main()
