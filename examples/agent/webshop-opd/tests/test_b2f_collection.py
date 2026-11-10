"""Exercise expert collection with real agent sessions and local SDK transport."""

import asyncio
import fcntl
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_test_support import AgentTokenizer
from test_tcod_b2f import AdaptiveSampler, ShopPool
from train.collect_tcod_b2f_experts import atomic_write, collect
from train.tcod_b2f import DEFAULT_CONFIG, load_experiment, load_expert_actions
from train.tcod_b2f import prepare_training_rows
from train.tcod_f2b import CurriculumConfig
from train.train_tcod_b2f import prepare_and_train
from test_opd_trainer import _Service
from triotune.agentrl import SamplingSnapshot
from triotune.trainer import OPDTrainer


class TeacherClient:
    def __init__(self):
        self.requests = []
        self.failure = None
        self.pool = None

    def get_tokenizer(self):
        return AgentTokenizer()

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.requests.append(sampling_params)
        if self.failure:
            raise self.failure
        generation = await AdaptiveSampler(self.pool).generate(
            prompt.to_ints(), max_tokens=sampling_params.max_tokens, seed=sampling_params.seed,
        )
        return SimpleNamespace(sequences=[SimpleNamespace(
            tokens=list(generation.tokens), logprobs=list(generation.logprobs), stop_reason="stop",
        )])


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.client = TeacherClient()
        self.service = SimpleNamespace(create_sampling_client_async=AsyncMock(return_value=self.client))
        self.pools = []
        self.goals = [{"id": i} for i in range(1504)]
        self.files = {"products": self.root / "products.json"}
        self.files["products"].write_text("[]")
        original = load_experiment(DEFAULT_CONFIG, steps=1)
        original.trainer.training.batch_size = 2
        self.experiment = replace(
            original,
            experts=replace(original.experts, path=self.root / "experts.jsonl", pass_k=2),
            agent=replace(original.agent, retry=replace(original.agent.retry, max_attempts=1)),
        )

        def pool(backend):
            item = ShopPool()
            self.pools.append(item)
            self.client.pool = item
            return item

        for target, kwargs in (
            ("train.collect_tcod_b2f_experts.cached_files", {"return_value": self.files}),
            ("train.collect_tcod_b2f_experts.WebShopBackend", {"return_value": SimpleNamespace(goals=self.goals)}),
            ("train.collect_tcod_b2f_experts.SessionPool", {"side_effect": pool}),
            ("train.collect_tcod_b2f_experts.pytrio.ServiceClient", {"return_value": self.service}),
        ):
            guard = patch(target, **kwargs)
            guard.start()
            self.addCleanup(guard.stop)

    async def test_collects_actual_teacher_paths_resumes_and_preserves_goal_seeds(self):
        first = await collect(self.experiment, max_tasks=1)
        self.assertEqual(first["cached_total"], 1)
        stored = load_expert_actions(self.experiment.experts.path)
        self.assertEqual(list(stored.values()), [ShopPool.actions])
        self.assertEqual(len(self.client.requests), 6)
        self.service.create_sampling_client_async.assert_awaited_once_with(
            base_model="Qwen/Qwen3.5-4B",
            model_path=self.experiment.trainer.teacher.model_path,
        )
        goal = next(iter(stored))
        base_seed = self.experiment.experts.seed + goal * 2 * 15
        self.assertEqual(sorted(p.seed for p in self.client.requests),
                         sorted([base_seed + offset for offset in (0, 1, 2, 15, 16, 17)]))
        second = await collect(self.experiment)
        self.assertEqual(second["requested_tasks"], 1)
        self.assertEqual(second["cached_total"], 2)
        calls = len(self.client.requests)
        complete = await collect(self.experiment)
        self.assertEqual(complete["requested_tasks"], 0)
        self.assertEqual(len(self.client.requests), calls)
        for item in self.pools:
            self.assertFalse(item.sessions)
            self.assertTrue(item.closed)

    async def test_changed_teacher_or_assets_fail_before_creating_sampling_client(self):
        await collect(self.experiment, max_tasks=1)
        self.service.create_sampling_client_async.reset_mock()
        original = self.experiment.trainer.teacher.model_path
        self.experiment.trainer.teacher.model_path = "trio://another-teacher"
        with self.assertRaisesRegex(ValueError, "settings changed"):
            await collect(self.experiment)
        self.experiment.trainer.teacher.model_path = original
        self.files["products"].write_text("[1]")
        with self.assertRaisesRegex(ValueError, "settings changed"):
            await collect(self.experiment)
        self.service.create_sampling_client_async.assert_not_awaited()
        self.assertTrue(all(pool.closed for pool in self.pools))

    async def test_service_failure_and_cancellation_keep_prior_successes_and_close_sessions(self):
        await collect(self.experiment, max_tasks=1)
        previous = self.experiment.experts.path.read_bytes()
        for error in (ConnectionError("teacher unavailable"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.client.failure = error
                with self.assertRaises((RuntimeError, asyncio.CancelledError)):
                    await collect(self.experiment)
                self.assertEqual(self.experiment.experts.path.read_bytes(), previous)
                self.assertTrue(self.pools[-1].closed)
                self.assertFalse(self.pools[-1].sessions)

    async def test_failed_purchases_are_not_added_to_cache(self):
        async def partial(pool, rollout_id):
            done = pool.sessions[rollout_id] == 3
            return done, 0.5 if done else 0.0

        with patch.object(ShopPool, "outcome", partial):
            report = await collect(self.experiment, max_tasks=1)
        self.assertEqual(report["cached_total"], 0)
        self.assertFalse(self.experiment.experts.path.exists())

    async def test_invalid_limits_and_unverifiable_cache_fail_without_model_calls(self):
        for maximum in (0, -1, True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                await collect(self.experiment, max_tasks=maximum)
        self.experiment.experts.path.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "no provenance"):
            await collect(self.experiment)
        self.service.create_sampling_client_async.assert_not_awaited()

    async def test_concurrent_collector_is_rejected_without_model_calls(self):
        path = self.experiment.experts.path.with_suffix(".jsonl.lock")
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another collector"):
                await collect(self.experiment)
        self.service.create_sampling_client_async.assert_not_awaited()

    async def test_real_collection_training_checkpoint_and_resume_lifecycle(self):
        service = _Service()
        tokenizer = AgentTokenizer()
        tokenizer.get_vocab = lambda: {chr(i): i for i in range(256)}
        service.training.get_tokenizer = lambda: tokenizer
        service.create_sampling_client_async = self.service.create_sampling_client_async
        service.create_training_client_from_state_with_optimizer = lambda **kwargs: service.training
        args = self.experiment.trainer
        args.training.max_steps = 2
        args.training.batch_size = 1
        args.checkpoint.save_every = 1
        args.checkpoint.state_dir = str(self.root / "state")
        args.checkpoint.sampler.enabled = False
        args.reporting.report_to = None
        args.reporting.log_dir = str(self.root / "logs")
        args.evaluation.steps = 1
        curriculum = CurriculumConfig(1, 1)
        experiment = replace(
            self.experiment, curriculum=curriculum,
            agent=replace(self.experiment.agent, curriculum=curriculum),
            data=replace(self.experiment.data, eval_limit=1),
            output_dir=self.root / "runs",
        )

        async def snapshot(trainer, *, stop=None, temperature=1.0, top_p=1.0):
            sampler = AdaptiveSampler(self.pools[-1])
            return SamplingSnapshot(f"student-{len(self.pools)}-{trainer.state.global_step}",
                                    sampler, sampler.tokenizer,
                                    policy_version=trainer.state.behavior_policy_version)

        def pool(backend):
            item = ShopPool()
            self.pools.append(item)
            return item

        with (
            patch("train.vanilla_opd.cached_files", return_value=self.files),
            patch("train.vanilla_opd.WebShopBackend", return_value=SimpleNamespace(goals=self.goals)),
            patch("train.vanilla_opd.SessionPool", side_effect=pool),
            patch("pytrio.ServiceClient", return_value=service),
            patch.object(OPDTrainer, "create_sampling_snapshot", snapshot),
        ):
            first = await prepare_and_train(experiment, stop_after_step=1)
            self.assertEqual(first["global_step"], 1)
            manifest = self.root / "state/opd-step-00000001.json"
            self.assertTrue(manifest.exists())
            original_cache = experiment.experts.path.read_bytes()
            resumed_args = replace(args, checkpoint=replace(args.checkpoint, resume_from=str(manifest)))
            resumed = replace(experiment, trainer=resumed_args)
            second = await prepare_and_train(resumed, resume=True)
            self.assertEqual(second["global_step"], 2)
            self.assertNotEqual(first["output_dir"], second["output_dir"])
            self.assertEqual([u["curriculum"]["k"] for u in second["updates"]], [1, 2])
            self.assertEqual(service.training.optim_calls, 2)
            self.assertEqual(len(service.teachers[args.teacher.name].requests), 5)
            saved = json.loads((self.root / "state/opd-step-00000002.json").read_text())
            self.assertEqual(saved["loop"]["state"]["seed_offset"], 30)
            self.assertEqual(experiment.experts.path.read_bytes(), original_cache)
            self.assertEqual(second["metrics"]["eval/agent/pass@1"], 1.0)
            altered = original_cache.decode().replace("search[shoes]", "search[other]")
            experiment.experts.path.write_text(altered)
            with self.assertRaisesRegex(ValueError, "dataset"):
                await prepare_and_train(resumed, resume=True)
        self.assertTrue(all(p.closed and not p.sessions for p in self.pools))

    def test_atomic_replace_failure_keeps_the_previous_cache(self):
        path = self.experiment.experts.path
        path.write_text("previous\n")
        with patch("train.collect_tcod_b2f_experts.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                atomic_write(path, "new\n")
        self.assertEqual(path.read_text(), "previous\n")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["experts.jsonl", "products.json"])


class CacheValidationTests(unittest.TestCase):
    def test_duplicate_epoch_rows_do_not_count_as_distinct_expert_goals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experts.jsonl"
            path.write_text(json.dumps({"goal_index": 1, "success": True, "actions": ["click[buy now]"]}) + "\n")
            experiment = load_experiment(DEFAULT_CONFIG)
            experiment = replace(experiment, experts=replace(experiment.experts, path=path))
            with self.assertRaisesRegex(ValueError, "full batch"):
                prepare_training_rows([{"goal_index": 1}] * 16, [], experiment)

    def test_malformed_duplicate_unfinished_or_terminal_prefix_paths_are_rejected(self):
        valid = {"goal_index": 0, "success": True, "actions": ["click[buy now]"]}
        records = [
            None, [], "not JSON", {**valid, "goal_index": True},
            {**valid, "actions": ["search[x]"]},
            {**valid, "actions": ["click[buy now]", "click[buy now]"]},
            {**valid, "actions": ["click[buy now]", "search[x]"]},
            {**valid, "actions": [None]}, {**valid, "success": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experts.jsonl"
            for record in records:
                with self.subTest(record=record):
                    path.write_text(json.dumps(record) + "\n")
                    with self.assertRaises(ValueError):
                        load_expert_actions(path)
            path.write_text((json.dumps(valid) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_expert_actions(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "no successful"):
                load_expert_actions(path)
