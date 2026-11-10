"""Contracts for WebShop TCOD-B2F expert replay and student takeover."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_test_support import AgentTokenizer
from test_opd_trainer import _Service
from train.collect_tcod_b2f_experts import episode_actions
from train.tcod_b2f import (
    DEFAULT_CONFIG,
    ExpertConfig,
    TCODB2FTrainingLoop,
    build_b2f_workflow,
    load_experiment,
    load_expert_actions,
    prepare_training_rows,
)
from train.train_tcod_b2f import prepare_and_train
from triotune.agentrl import Generation, SamplingSnapshot, current_execution_context
from triotune.trainer import OPDConfig, OPDTrainer


async def inline(function, /, *args, **kwargs):
    return function(*args, **kwargs)


class ShopPool:
    actions = ("search[shoes]", "click[product]", "click[buy now]")

    def __init__(self):
        self.sessions = {}
        self.executed = []
        self.closed = False

    def state(self, rollout_id):
        index = self.sessions[rollout_id]
        return {
            "observation": f"state {index}",
            "available_actions": {},
            "done": index == len(self.actions),
        }

    async def open(self, rollout_id, goal_index):
        self.sessions[rollout_id] = 0
        return self.state(rollout_id)

    async def step(self, rollout_id, action):
        index = self.sessions[rollout_id]
        if action != self.actions[index]:
            raise AssertionError(f"Expected {self.actions[index]}, got {action}")
        self.sessions[rollout_id] += 1
        self.executed.append(action)
        return self.state(rollout_id)

    async def outcome(self, rollout_id):
        done = self.sessions[rollout_id] == len(self.actions)
        return done, float(done)

    async def release(self, rollout_id):
        self.sessions.pop(rollout_id, None)

    async def close(self):
        self.sessions.clear()
        self.closed = True


class AdaptiveSampler:
    def __init__(self, pool):
        self.pool = pool
        self.tokenizer = AgentTokenizer()

    async def generate(self, prompt_tokens, *, max_tokens, seed=None):
        rollout_id = current_execution_context().rollout_id
        index = self.pool.sessions[rollout_id]
        value = (
            {
                "tool_calls": [
                    {
                        "name": "webshop",
                        "arguments": {"action": self.pool.actions[index]},
                    }
                ]
            }
            if index < len(self.pool.actions)
            else {"final": "done"}
        )
        response = json.dumps(value)
        tokens = tuple(self.tokenizer.encode(response) + [1])
        return Generation(tokens, (-1.0,) * len(tokens), response)


class B2FConfigurationTests(unittest.TestCase):
    def test_qwen38_b2f_config_and_expert_collection_settings(self):
        experiment = load_experiment(DEFAULT_CONFIG)
        self.assertEqual(experiment.trainer.model.name, "Qwen/Qwen3.5-4B")
        self.assertEqual(experiment.trainer.teacher.name, "Qwen/Qwen3.8-27B")
        self.assertEqual(experiment.curriculum.k_start, 1)
        self.assertEqual(experiment.curriculum.growth_interval, 4)
        self.assertEqual(experiment.experts.pass_k, 10)
        self.assertEqual(experiment.experts.batch_size, 4)
        self.assertEqual(experiment.agent.limits.max_turns, 15)
        self.assertEqual(experiment.trainer.batch_size, 16)
        self.assertEqual(experiment.trainer.max_steps, 200)

    def test_successful_experts_are_filtered_and_cycled_to_the_training_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experts.jsonl"
            records = [
                {
                    "goal_index": index,
                    "actions": ["search[item]", "click[product]", "click[buy now]"],
                    "success": True,
                }
                for index in range(16)
            ]
            path.write_text("".join(json.dumps(item) + "\n" for item in records))
            experiment = load_experiment(DEFAULT_CONFIG, steps=2)
            experiment = replace(
                experiment,
                experts=replace(experiment.experts, path=path),
            )
            train = [
                {"task_id": f"train-{index}", "goal_index": index}
                for index in range(20)
            ]
            eval_rows = [{"task_id": "eval", "goal_index": 100}]

            planned, evaluated = prepare_training_rows(train, eval_rows, experiment)

            self.assertEqual(len(planned), 32)
            self.assertEqual(planned[0]["goal_index"], 0)
            self.assertEqual(planned[16]["goal_index"], 0)
            self.assertEqual(planned[16]["expert_cycle"], 1)
            self.assertEqual(evaluated, eval_rows)
            self.assertEqual(len(load_expert_actions(path)), 16)


class B2FExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = _Service()
        self.service.teacher_values = {"teacher": -0.5}
        service_patch = patch(
            "triotune.trainer.opd_trainer.trio.ServiceClient",
            return_value=self.service,
        )
        service_patch.start()
        self.addCleanup(service_patch.stop)
        thread_patch = patch("asyncio.to_thread", side_effect=inline)
        thread_patch.start()
        self.addCleanup(thread_patch.stop)

    async def test_direct_entry_collects_experts_then_trains_and_resume_skips_collection(self):
        experiment = object()
        with (
            patch(
                "train.train_tcod_b2f.collect",
                AsyncMock(return_value={"cached_total": 20}),
            ) as collect_mock,
            patch(
                "train.train_tcod_b2f.run",
                AsyncMock(return_value={"status": "completed"}),
            ) as run_mock,
        ):
            report = await prepare_and_train(experiment)
            self.assertEqual(report["status"], "completed")
            collect_mock.assert_awaited_once_with(experiment, max_tasks=None)
            run_mock.assert_awaited_once_with(experiment, stop_after_step=None)

            collect_mock.reset_mock()
            run_mock.reset_mock()
            await prepare_and_train(experiment, resume=True)
            collect_mock.assert_not_awaited()
            run_mock.assert_awaited_once()

            collect_mock.reset_mock()
            run_mock.reset_mock()
            limited = await prepare_and_train(experiment, expert_max_tasks=4)
            self.assertEqual(limited["status"], "expert_collection_limited")
            collect_mock.assert_awaited_once_with(experiment, max_tasks=4)
            run_mock.assert_not_awaited()

    async def test_teacher_prefix_is_replayed_without_opd_gradients(self):
        row = {
            "task_id": "shop-0",
            "goal_index": 0,
            "expert_actions": list(ShopPool.actions),
        }
        trainer = OPDTrainer(
            "student",
            "teacher",
            args=OPDConfig(batch_size=1, num_generations=1, max_steps=1, seed=42),
            train_dataset=[row],
            eval_dataset=[{"task_id": "eval", "goal_index": 1}],
            defer_checkpoint_loop_binding=True,
        )
        pool = ShopPool()
        experiment = load_experiment(DEFAULT_CONFIG, steps=1)
        config = replace(
            experiment.agent,
            limits=replace(experiment.agent.limits, max_turns=15),
            retry=replace(
                experiment.agent.retry,
                max_attempts=2,
                initial_delay=0,
                max_delay=0,
            ),
        )
        workflow = build_b2f_workflow(config, AgentTokenizer(), pool, experiment.curriculum)
        loop = TCODB2FTrainingLoop(trainer, workflow, config=config)
        snapshots = []

        async def snapshot(*, stop=None, temperature=1.0, top_p=1.0):
            sampler = AdaptiveSampler(pool)
            snapshots.append((temperature, sampler))
            return SamplingSnapshot(
                f"student-{len(snapshots)}",
                sampler,
                sampler.tokenizer,
                policy_version=trainer.state.behavior_policy_version,
            )

        trainer.create_sampling_snapshot = snapshot
        update = await loop.collect_and_update([row])
        try:
            self.assertEqual(pool.executed, list(ShopPool.actions))
            self.assertEqual(len(update.rollouts), 2)
            self.assertEqual(update.result.num_datums, 2)
            self.assertEqual(len(self.service.teacher.requests), 2)
            prompt = AgentTokenizer().decode(
                update.rollouts[0].prompt_tokens,
                skip_special_tokens=True,
            )
            self.assertIn('"teacher_prefix_steps": 2', prompt)
            self.assertIn('"action": "click[product]"', prompt)
            metadata = update.batch.episodes[0].export_metadata
            self.assertEqual(metadata["teacher_prefix_length"], 2)
            self.assertEqual(metadata["expert_length"], 3)
            self.assertEqual(metadata["curriculum_k"], 1)
            self.assertEqual(
                loop.update_history[-1]["curriculum"]["teacher_prefix_mean"],
                2,
            )
            self.assertEqual(episode_actions(update.batch.episodes[0]), ["click[buy now]"])
            before_teacher = len(self.service.teacher.requests)
            before_optim = self.service.training.optim_calls
            metrics = await loop.evaluate()
            self.assertEqual(metrics["eval/agent/pass@1"], 1.0)
            self.assertEqual(len(loop.last_eval_batch.episodes[0].turns), 3)
            self.assertEqual(len(self.service.teacher.requests), before_teacher)
            self.assertEqual(self.service.training.optim_calls, before_optim)
            self.assertFalse(pool.sessions)
        finally:
            await workflow.close()
            await pool.close()


if __name__ == "__main__":
    unittest.main()
