"""Exercise the F2B curriculum through real collection and OPD boundaries."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml
from pytrio.errors.actor import ActorAPIError

from agent_test_support import AgentTokenizer
from test_opd_trainer import _Service
from train.tcod_f2b import (
    DEFAULT_CONFIG,
    CurriculumConfig,
    TCODAgentConfig,
    TCODF2BTrainingLoop,
    load_experiment,
    run,
)
from train.vanilla_opd import load_experiment as load_vanilla
from triotune.agentrl import (
    AgentRolloutBatch,
    EpisodeCollectionError,
    EpisodeResult,
    Generation,
    RetryEvent,
    SamplingSnapshot,
    current_execution_context,
)
from triotune.trainer import OPDConfig, OPDTrainer
from webshop_env.environment import build_workflow


async def inline(function, /, *args, **kwargs):
    return function(*args, **kwargs)


class ShopPool:
    """Require search, product selection, and purchase in separate actions."""

    def __init__(self):
        self.sessions = {}
        self.actions = []
        self.closed = False

    async def open(self, rollout_id, goal_index):
        self.sessions[rollout_id] = 0
        return {"observation": f"goal {goal_index}", "done": False}

    async def step(self, rollout_id, action):
        expected = ("search[shoes]", "click[product]", "click[buy now]")
        index = self.sessions[rollout_id]
        if action != expected[index]:
            raise AssertionError(f"Unexpected action: {action}")
        self.sessions[rollout_id] += 1
        self.actions.append((rollout_id, action))
        return {"observation": f"state {index + 1}", "done": index == 2}

    async def outcome(self, rollout_id):
        done = self.sessions[rollout_id] == 3
        return done, float(done)

    async def release(self, rollout_id):
        self.sessions.pop(rollout_id)

    async def close(self):
        self.closed = True


class ShopSampler:
    def __init__(self):
        self.tokenizer = AgentTokenizer()
        self.turns = {}
        self.seeds = []

    async def generate(self, prompt_tokens, *, max_tokens, seed=None):
        rollout_id = current_execution_context().rollout_id
        turn = self.turns.get(rollout_id, 0)
        self.turns[rollout_id] = turn + 1
        self.seeds.append(seed)
        actions = ("search[shoes]", "click[product]", "click[buy now]")
        value = (
            {"tool_calls": [{"name": "webshop", "arguments": {"action": actions[turn]}}]}
            if turn < 3
            else {"final": "done"}
        )
        response = json.dumps(value)
        tokens = tuple(self.tokenizer.encode(response) + [1])
        assert len(tokens) <= max_tokens
        return Generation(tokens, (-1.0,) * len(tokens), response)


class CurriculumConfigurationTests(unittest.TestCase):
    def test_defaults_keep_the_grpo_teacher_and_baseline_controls(self):
        experiment = load_experiment(DEFAULT_CONFIG)
        vanilla = load_vanilla(DEFAULT_CONFIG.with_name("vanilla-opd-grpo-teacher.yaml"))
        self.assertEqual(experiment.trainer.teacher, vanilla.trainer.teacher)
        for name in ("model", "training", "optimizer", "rollout", "objective", "evaluation"):
            self.assertEqual(getattr(experiment.trainer, name), getattr(vanilla.trainer, name))
        self.assertEqual(experiment.agent.curriculum, CurriculumConfig(1, 4))
        self.assertEqual(experiment.agent.limits.max_turns, 15)
        self.assertEqual(experiment.agent.evaluation.pass_ks, (1,))
        self.assertNotEqual(experiment.output_dir, vanilla.output_dir)
        self.assertNotEqual(experiment.trainer.checkpoint.state_dir, vanilla.trainer.checkpoint.state_dir)
        self.assertEqual(load_experiment(DEFAULT_CONFIG, steps=2).trainer.max_steps, 2)
        self.assertEqual(load_experiment(DEFAULT_CONFIG).trainer.max_steps, 200)

    def test_qwen38_teacher_config_matches_its_vanilla_baseline(self):
        config = DEFAULT_CONFIG.with_name("tcod-f2b-qwen3.8-27b-teacher.yaml")
        experiment = load_experiment(config)
        vanilla = load_vanilla(DEFAULT_CONFIG.with_name("vanilla-opd.yaml"))

        self.assertEqual(experiment.trainer.model.name, "Qwen/Qwen3.5-4B")
        self.assertEqual(experiment.trainer.teacher.name, "Qwen/Qwen3.8-27B")
        self.assertIsNone(experiment.trainer.teacher.model_path)
        for name in (
            "model",
            "training",
            "optimizer",
            "rollout",
            "objective",
            "evaluation",
        ):
            self.assertEqual(
                getattr(experiment.trainer, name),
                getattr(vanilla.trainer, name),
            )
        self.assertEqual(experiment.agent.curriculum, CurriculumConfig(1, 4))
        self.assertEqual(experiment.agent.limits.max_turns, 15)
        self.assertTrue(experiment.trainer.checkpoint.sampler.enabled)
        self.assertNotEqual(experiment.output_dir, vanilla.output_dir)
        self.assertNotEqual(
            experiment.trainer.checkpoint.state_dir,
            vanilla.trainer.checkpoint.state_dir,
        )

    def test_resume_override_uses_an_existing_shell_relative_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            manifest = Path(directory) / "checkpoint.json"
            manifest.write_text("{}\n")
            relative = manifest.relative_to(Path.cwd())

            experiment = load_experiment(DEFAULT_CONFIG, resume_from=relative)

            self.assertEqual(
                experiment.trainer.checkpoint.resume_from,
                str(manifest.resolve()),
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_experiment(
                    DEFAULT_CONFIG,
                    resume_from=Path(directory) / "missing.json",
                )

    def test_schedule_boundaries_and_cap_follow_completed_update_indices(self):
        schedule = CurriculumConfig()
        self.assertEqual(
            [schedule.horizon(step, 15) for step in (0, 3, 4, 7, 8, 55, 56, 200)],
            [1, 1, 2, 2, 3, 14, 15, 15],
        )
        self.assertEqual(CurriculumConfig(15, 1).horizon(0, 15), 15)
        for settings in ({"k_start": 0}, {"growth_interval": 0}, {"k_start": True}, {"growth_interval": 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                CurriculumConfig(**settings)
        for step, maximum in ((-1, 15), (True, 15), (0, 0), (0, True)):
            with self.subTest(step=step, maximum=maximum), self.assertRaises(ValueError):
                schedule.horizon(step, maximum)

    def test_invalid_or_missing_curriculum_is_rejected_before_remote_calls(self):
        values = yaml.safe_load(DEFAULT_CONFIG.read_text())
        for settings in (None, {"k_start": 16}, {"growth_interval": -1}, {"unknown": 1}):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                values["curriculum"] = settings
                path = Path(directory) / "invalid.yaml"
                path.write_text(yaml.safe_dump(values))
                with patch("triotune.trainer.opd_trainer.trio.ServiceClient") as service:
                    with self.assertRaises((ValueError, TypeError)):
                        load_experiment(path)
                    service.assert_not_called()


class F2BLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = _Service()
        self.service.teacher_values = {"teacher": -0.5}
        service_patch = patch("triotune.trainer.opd_trainer.trio.ServiceClient", return_value=self.service)
        service_patch.start()
        self.addCleanup(service_patch.stop)
        thread_patch = patch("asyncio.to_thread", side_effect=inline)
        thread_patch.start()
        self.addCleanup(thread_patch.stop)
        self.snapshots = []

    def attach_sampler(self, trainer):
        async def snapshot(*, stop=None, temperature=1.0, top_p=1.0):
            sampler = ShopSampler()
            self.snapshots.append((trainer.state.behavior_policy_version, temperature, sampler))
            return SamplingSnapshot(
                f"shop-{len(self.snapshots)}", sampler, sampler.tokenizer,
                policy_version=trainer.state.behavior_policy_version,
            )
        trainer.create_sampling_snapshot = snapshot

    def make_loop(self, *, checkpoint=None, curriculum=None):
        rows = [{"task_id": f"shop-{i}", "goal_index": i} for i in range(4)]
        args = OPDConfig(
            batch_size=1, num_generations=1, max_steps=4, seed=42,
            eval_steps=1, eval_before_train=True, eval_after_train=True,
            checkpoint=checkpoint or {},
        )
        trainer = OPDTrainer(
            "student", "teacher", args=args, train_dataset=rows,
            eval_dataset=[{"task_id": "eval-shop", "goal_index": 99}],
            defer_checkpoint_loop_binding=True,
        )
        self.attach_sampler(trainer)
        config = TCODAgentConfig(
            limits={"max_turns": 15},
            curriculum=curriculum or CurriculumConfig(1, 1),
            retry={
                "max_attempts": 2,
                "initial_delay": 0,
                "max_delay": 0,
            },
            evaluation={"num_generations": 1, "pass_ks": [1], "temperature": 0.4},
        )
        pool = ShopPool()
        workflow = build_workflow(config, AgentTokenizer(), pool)
        self.addAsyncCleanup(workflow.close)
        loop = TCODF2BTrainingLoop(trainer, workflow, config=config)
        return loop, pool, rows

    def assert_full_limits(self, loop):
        self.assertEqual(loop.config.limits.max_turns, 15)
        self.assertEqual(loop.workflow.runner.limits.max_turns, 15)
        self.assertEqual(loop.workflow.runner.runner.limits.max_turns, 15)

    async def test_short_zero_reward_prefixes_train_and_evaluation_runs_to_purchase(self):
        loop, pool, rows = self.make_loop()
        for index, horizon in enumerate((1, 2)):
            before = len(pool.actions)
            update = await loop.collect_and_update(rows[index:index + 1])
            self.assertEqual(len(pool.actions) - before, horizon)
            self.assertEqual(len(update.rollouts), horizon)
            episode = update.batch.episodes[0]
            self.assertEqual(episode.status, "max_turns")
            self.assertEqual(episode.score.reward, 0.0)
            self.assertTrue(episode.trainable)
            self.assertEqual(update.result.num_datums, horizon)
            self.assertEqual(loop.update_history[-1]["curriculum"]["horizon"], horizon)
            self.assert_full_limits(loop)
            # Every teacher request uses the original action and its context.
            for rollout, request in zip(update.rollouts, self.service.teacher.requests[-horizon:], strict=True):
                self.assertEqual(request, rollout.prompt_tokens + rollout.completion_tokens)
            for rollout, datum in zip(update.rollouts, self.service.training.datums, strict=True):
                targets = datum.loss_fn_inputs["target_tokens"].data
                self.assertEqual([t for t in targets if t != -100], rollout.completion_tokens)
            before_teacher = len(self.service.teacher.requests)
            before_updates = self.service.training.optim_calls
            metrics = await loop.evaluate()
            self.assertEqual(metrics["eval/agent/pass@1"], 1.0)
            self.assertEqual(len(loop.last_eval_batch.episodes[0].turns), 3)
            self.assertEqual(len(self.service.teacher.requests), before_teacher)
            self.assertEqual(self.service.training.optim_calls, before_updates)
            self.assertFalse(pool.sessions)
            self.assert_full_limits(loop)
        self.assertEqual([item[0] for item in self.snapshots], [0, 1, 1, 2])
        self.assertEqual([item[1] for item in self.snapshots], [1.0, 0.4, 1.0, 0.4])
        train_seeds = [seed for _, temp, sampler in self.snapshots if temp == 1.0 for seed in sampler.seeds]
        self.assertEqual(train_seeds, [42, 43, 44])

    async def test_teacher_failure_and_cancellation_restore_both_runner_limits(self):
        loop, pool, rows = self.make_loop()
        self.service.teacher_error = True
        with self.assertRaises(ExceptionGroup):
            await loop.collect_and_update(rows[:1])
        self.assert_full_limits(loop)
        self.assertFalse(pool.sessions)
        self.assertEqual(self.service.training.optim_calls, 0)
        self.assertEqual(loop.trainer.state.global_step, 0)
        with patch.object(
            loop.workflow,
            "collect",
            AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await loop.collect_and_update(rows[:1])
        self.assert_full_limits(loop)

    async def test_sampling_timeout_refreshes_snapshot_and_reuses_seed(self):
        loop, _, rows = self.make_loop()
        original_collect = loop.workflow.collect
        attempts = []

        async def flaky_collect(rows, snapshot, **kwargs):
            attempts.append((snapshot.snapshot_id, kwargs["seed"]))
            if len(attempts) == 1:
                timed_out = EpisodeResult(
                    "timeout-episode",
                    "timeout-group",
                    snapshot.snapshot_id,
                    status="timeout",
                    error="Student sampling timed out.",
                    retries=[
                        RetryEvent(
                            "sampling",
                            0,
                            2,
                            "TimeoutError",
                            "",
                            False,
                        )
                    ],
                )
                raise EpisodeCollectionError(AgentRolloutBatch([timed_out], []))
            return await original_collect(rows, snapshot, **kwargs)

        loop.workflow.collect = flaky_collect
        update = await loop.collect_and_update(rows[:1])

        self.assertEqual(loop.trainer.state.global_step, 1)
        self.assertEqual(update.result.num_datums, 1)
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0][0], attempts[1][0])
        self.assertEqual([seed for _, seed in attempts], [42, 42])
        self.assertEqual(loop._seed_offset, 1)
        self.assertEqual(
            loop.update_history[-1]["curriculum"]["collection_attempts"],
            2,
        )
        self.assertEqual(
            loop.trainer.state.metrics["curriculum/collection_attempts"],
            2.0,
        )
        self.assert_full_limits(loop)

    async def test_teacher_503_replays_before_optimizer_with_the_same_seed(self):
        loop, _, rows = self.make_loop()
        original_step = loop.trainer.step_rollouts_async
        scoring_attempts = 0

        async def flaky_step(rollouts, *, prompt_count):
            nonlocal scoring_attempts
            scoring_attempts += 1
            if scoring_attempts == 1:
                raise ExceptionGroup(
                    "teacher unavailable",
                    [
                        ActorAPIError(
                            "teacher actor unavailable",
                            status_code=503,
                            retryable=True,
                        )
                    ],
                )
            return await original_step(rollouts, prompt_count=prompt_count)

        loop.trainer.step_rollouts_async = flaky_step
        update = await loop.collect_and_update(rows[:1])

        self.assertEqual(scoring_attempts, 2)
        self.assertEqual(loop.trainer.state.global_step, 1)
        self.assertEqual(self.service.training.optim_calls, 1)
        self.assertEqual(update.result.num_datums, 1)
        self.assertEqual(len(self.snapshots), 2)
        self.assertIsNot(self.snapshots[0][2], self.snapshots[1][2])
        self.assertEqual(
            [seed for _, _, sampler in self.snapshots for seed in sampler.seeds],
            [42, 42],
        )
        self.assertEqual(loop._seed_offset, 1)
        self.assertEqual(
            loop.update_history[-1]["curriculum"]["collection_attempts"],
            2,
        )

    async def test_pause_resume_keeps_schedule_seed_cursor_and_rejects_schedule_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            loop, pool, _ = self.make_loop(checkpoint={"state_dir": directory, "save_every": 1})
            state = await loop.train(stop_after_step=2)
            path = state.checkpoint_manifest_path
            manifest = json.loads(Path(path).read_text())
            self.assertEqual(manifest["loop"]["identity"]["curriculum"], {"k_start": 1, "growth_interval": 1})
            self.assertEqual(manifest["loop"]["identity"]["limits"]["max_turns"], 15)
            self.assertEqual(manifest["loop"]["state"]["seed_offset"], 3)
            self.assertEqual([r["curriculum"]["horizon"] for r in manifest["loop"]["state"]["updates"]], [1, 2])
            self.assertFalse(pool.sessions)
            self.service.create_training_client_from_state_with_optimizer = lambda **kwargs: self.service.training
            checkpoint = {"state_dir": directory, "save_every": 1, "resume_from": path}
            resumed, _, _ = self.make_loop(checkpoint=checkpoint)
            self.assertEqual(resumed.trainer.state.global_step, 2)
            self.assertEqual(resumed._seed_offset, 3)
            state = await resumed.train()
            self.assertEqual(state.global_step, 4)
            self.assertEqual([r["curriculum"]["horizon"] for r in resumed.update_history], [1, 2, 3, 4])
            self.assertEqual(resumed._seed_offset, 10)
            # Purchase at k=3 needs no positive reward to receive OPD targets;
            # k=4 also includes the protocol's final model reply.
            self.assertEqual(resumed.update_history[-1]["turns"], 4)
            self.assert_full_limits(resumed)
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                self.make_loop(checkpoint=checkpoint, curriculum=CurriculumConfig(1, 2))

    async def test_entry_point_writes_tcod_report_using_the_shared_resource_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            original = load_experiment(DEFAULT_CONFIG)
            args = OPDConfig(
                model={"name": "student"}, teacher={"name": "teacher"},
                batch_size=1, max_steps=2, num_generations=1,
                checkpoint={"state_dir": str(Path(directory) / "state"), "save_every": 1},
                eval_steps=1, eval_before_train=True, eval_after_train=True,
            )
            experiment = replace(
                original, trainer=args, output_dir=Path(directory) / "runs",
                agent=replace(original.agent, curriculum=CurriculumConfig(1, 1)),
            )
            pool = ShopPool()
            rows = [{"task_id": str(i), "goal_index": i} for i in range(2)]
            original_init = OPDTrainer.__init__

            def init(trainer, *args, **kwargs):
                original_init(trainer, *args, **kwargs)
                self.attach_sampler(trainer)

            def workflow(config, tokenizer, pool):
                return build_workflow(config, AgentTokenizer(), pool)

            with (
                patch("train.vanilla_opd.cached_files", return_value={}),
                patch("train.vanilla_opd.WebShopBackend", return_value=SimpleNamespace(goals=[])),
                patch("train.vanilla_opd.SessionPool", return_value=pool),
                patch("train.vanilla_opd.task_rows", side_effect=lambda goals, data, split: rows if split == "train" else rows[:1]),
                patch("train.vanilla_opd.build_workflow", side_effect=workflow),
                patch.object(OPDTrainer, "__init__", init),
            ):
                report = await run(experiment)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["algorithm"], "agent_tcod_f2b")
            self.assertEqual([r["curriculum"]["horizon"] for r in report["updates"]], [1, 2])
            self.assertTrue(pool.closed)
            self.assertFalse(pool.sessions)
            written = json.loads((Path(report["output_dir"]) / "report.json").read_text())
            self.assertEqual(written["agent"]["curriculum"], {"k_start": 1, "growth_interval": 1})
            self.assertEqual(written["metrics"]["eval/agent/pass@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
