"""LR contracts at real SDK optimizer boundaries, including pending requests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytrio
import test_checkpoint_resume as resume_support
from checkpoint_test_support import Service
from test_sft_trainer import _Client

from triotune import PPOTrainer
from triotune.api.client_api import optimizer_params
from triotune.trainer import (
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    SFTConfig,
    SFTTrainer,
)
from triotune.utils.checkpoint import load_training_checkpoint
from triotune.utils.scheduler import learning_rate_for_step, optimizer_horizons


class SchedulerPolicyTests(unittest.TestCase):
    def optimizer(self, kind="linear", **kwargs):
        return OptimizerConfig(learning_rate=1.0, scheduler={"type": kind, **kwargs})

    def assert_rates(self, optimizer, expected, *, total=None):
        for step, value in enumerate(expected, 1):
            with self.subTest(step=step):
                self.assertAlmostEqual(
                    learning_rate_for_step(optimizer, step, total_steps=total), value
                )

    def test_first_warmup_boundary_endpoint_and_floor(self):
        self.assert_rates(
            self.optimizer(total_steps=4, min_lr=0.1), [1.0, 0.7, 0.4, 0.1, 0.1]
        )
        self.assert_rates(
            self.optimizer(warmup_steps=2, total_steps=5, min_lr=0.1),
            [0.5, 1.0, 0.7, 0.4, 0.1, 0.1],
        )
        self.assert_rates(
            self.optimizer("cosine", warmup_steps=2, total_steps=5, min_lr=0.1),
            [0.5, 1.0, 0.775, 0.325, 0.1],
        )
        self.assert_rates(
            self.optimizer("constant", warmup_steps=4, min_lr=0.4),
            [0.4, 0.5, 0.75, 1.0, 1.0],
        )
        self.assert_rates(self.optimizer(total_steps=1, min_lr=0.2), [0.2, 0.2])

    def test_unknown_horizon_and_explicit_optimizer_budget(self):
        self.assert_rates(self.optimizer("constant", warmup_steps=2), [0.5, 1.0, 1.0])
        for kind in ("linear", "cosine"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "known total_steps"),
            ):
                learning_rate_for_step(self.optimizer(kind), 1)
        config = SFTConfig(
            optimizer={"scheduler": {"type": "linear"}}, training={"batch_size": 2}
        )
        self.assertEqual(optimizer_horizons(config, [0] * 5), {"actor": 3})
        config.optimizer.scheduler.total_steps = 7
        self.assertEqual(optimizer_horizons(config, iter([0])), {"actor": 7})

    def test_invalid_schedule_settings_fail_locally(self):
        for values in (
            {"type": "unknown"},
            {"warmup_steps": -1},
            {"warmup_steps": True},
            {"total_steps": 0},
            {"total_steps": True},
            {"min_lr": -1},
            {"type": "linear", "min_lr": 0},
            {"min_lr": float("nan")},
            {"min_lr": float("inf")},
            {"type": "cosine", "warmup_steps": 2, "total_steps": 2},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SchedulerConfig(**values)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.optimizer(min_lr=2)
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            learning_rate_for_step(self.optimizer(warmup_steps=2), 1, total_steps=2)
        for index in (0, -1, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                learning_rate_for_step(self.optimizer("constant"), index)

    def test_sdk_receives_only_adam_fields_and_uses_a_positive_default_floor(self):
        optimizer = self.optimizer(total_steps=2)
        with self.assertRaisesRegex(ValueError, "requires an optimizer step"):
            optimizer_params(optimizer)
        values = optimizer_params(optimizer, dict, step=2)
        self.assertEqual(values["learning_rate"], 0.01)
        self.assertNotIn("scheduler", values)
        self.assertEqual(pytrio.AdamParams(**values).learning_rate, 0.01)
        self.assertEqual(optimizer.learning_rate, 1.0)
        self.assertEqual(
            optimizer_params(self.optimizer("constant"), dict)["learning_rate"], 1.0
        )

    def test_ppo_inherits_independent_scheduler_configs_and_budgets(self):
        config = PPOConfig(
            training={"max_steps": 3},
            optimizer={"scheduler": {"type": "linear"}},
            actor={"update_epochs": 2, "ppo_n_minibatches": 2},
            critic={"update_epochs": 3, "n_minibatches": 2},
        )
        self.assertIsNot(config.optimizer.scheduler, config.critic.optimizer.scheduler)
        self.assertEqual(optimizer_horizons(config), {"actor": 12, "critic": 18})
        config.optimizer.scheduler.warmup_steps = 2
        self.assertEqual(config.critic.optimizer.scheduler.warmup_steps, 0)
        config.max_steps = None
        config.dynamic_sampling = True
        with self.assertRaisesRegex(ValueError, "known total_steps"):
            optimizer_horizons(config, [0] * 100)


class SchedulerTrainingTests(unittest.IsolatedAsyncioTestCase):
    def scheduled_config(self, helper, kind, path, policy, *, resume=None):
        config = helper.config(kind, path, resume=resume)
        config.optimizer.scheduler = SchedulerConfig(
            type=policy, warmup_steps=2 if policy == "constant" else 1, min_lr=0.0001
        )
        return config

    async def test_continuous_and_resumed_lrs_weights_and_moments(self):
        helper = resume_support.ResumeTests()
        for kind in ("sft", "dpo", "opd", "agent"):
            for policy in ("constant", "linear", "cosine"):
                with (
                    self.subTest(kind=kind, policy=policy),
                    tempfile.TemporaryDirectory() as root,
                ):
                    root = Path(root)
                    full_service = Service(agent=kind == "agent")
                    with patch("pytrio.ServiceClient", return_value=full_service):
                        _full, owner = await helper.make(
                            kind,
                            self.scheduled_config(helper, kind, root / "full", policy),
                        )
                        await helper.train(kind, owner)
                    split_service = Service(agent=kind == "agent")
                    with patch("pytrio.ServiceClient", return_value=split_service):
                        head, owner = await helper.make(
                            kind,
                            self.scheduled_config(helper, kind, root / "split", policy),
                        )
                        await helper.train(kind, owner, stop=1)
                        path = head.state.checkpoint_manifest_path
                        resumed, owner = await helper.make(
                            kind,
                            self.scheduled_config(
                                helper, kind, root / "split", policy, resume=path
                            ),
                        )
                        await helper.train(kind, owner)
                    complete = full_service.clients[0]
                    prefix, suffix = split_service.clients
                    self.assertEqual(complete.lrs, prefix.lrs + suffix.lrs)
                    self.assertEqual(complete.inputs, prefix.inputs + suffix.inputs)
                    helper.assert_client_equal(complete, suffix)
                    expected_final = 0.001 if policy == "constant" else 0.0001
                    self.assertEqual(complete.lrs[-1], expected_final)
                    saved = load_training_checkpoint(
                        resumed.state.checkpoint_manifest_path
                    )
                    self.assertEqual(
                        saved["schedulers"]["actor"]["last_lr"], expected_final
                    )
                    self.assertEqual(saved["schedulers"]["actor"]["optimizer_steps"], 3)
                    metric = (
                        "train/learning_rate"
                        if kind in ("sft", "agent")
                        else "optim/lr"
                    )
                    self.assertEqual(resumed.state.metrics[metric], expected_final)

    async def test_actor_and_critic_count_every_minibatch_and_epoch_independently(self):
        helper = resume_support.ResumeTests()
        data = [row for row in helper.rows("ppo") for _ in range(2)]

        def config(path, resume=None):
            args = self.scheduled_config(helper, "ppo", path, "linear", resume=resume)
            args.batch_size = 2
            args.ppo_n_minibatches = 2
            args.critic_n_minibatches = 2
            args.critic_update_epochs = 3
            args.critic.optimizer.scheduler = SchedulerConfig(
                type="cosine", warmup_steps=3, min_lr=0.0002
            )
            return args

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            full_service, split_service = Service(), Service()
            with patch("pytrio.ServiceClient", return_value=full_service):
                full = PPOTrainer(
                    args=config(root / "full"),
                    train_dataset=data,
                    reward_fn=lambda **kw: kw["row"]["reward"],
                )
                await full.train()
            with patch("pytrio.ServiceClient", return_value=split_service):
                head = PPOTrainer(
                    args=config(root / "split"),
                    train_dataset=data,
                    reward_fn=lambda **kw: kw["row"]["reward"],
                )
                await head.train(stop_after_step=1)
                resumed = PPOTrainer(
                    args=config(root / "split", head.state.checkpoint_manifest_path),
                    train_dataset=data,
                    reward_fn=lambda **kw: kw["row"]["reward"],
                )
                await resumed.train()
            self.assertEqual(
                (
                    resumed.state.global_step,
                    resumed.state.optimizer_steps,
                    resumed.state.critic_optimizer_steps,
                ),
                (3, 12, 18),
            )
            for index, count, floor in ((0, 12, 0.0001), (1, 18, 0.0002)):
                complete = full_service.clients[index]
                head_client, tail_client = (
                    split_service.clients[index],
                    split_service.clients[index + 2],
                )
                self.assertEqual(len(complete.lrs), count)
                self.assertEqual(complete.lrs, head_client.lrs + tail_client.lrs)
                self.assertEqual(complete.lrs[-1], floor)
                helper.assert_client_equal(complete, tail_client)
            self.assertEqual(resumed.state.metrics["train/learning_rate"], 0.0001)
            self.assertEqual(resumed.state.metrics["critic/learning_rate"], 0.0002)
            manifest = load_training_checkpoint(resumed.state.checkpoint_manifest_path)
            self.assertEqual(manifest["schedulers"]["actor"]["total_steps"], 12)
            self.assertEqual(manifest["schedulers"]["critic"]["total_steps"], 18)

    async def test_sft_pipeline_assigns_distinct_lrs_before_results_complete(self):
        observations, holder = [], {}

        class Client(_Client):
            async def optim_step_async(self, params):
                observations.append(
                    (params.learning_rate, holder["trainer"].state.optimizer_steps)
                )
                return await super().optim_step_async(params)

        client = Client()
        config = SFTConfig(
            model={"name": "policy"},
            training={"max_steps": 4, "batch_size": 1},
            optimizer={
                "learning_rate": 1.0,
                "scheduler": {"type": "linear", "warmup_steps": 1, "min_lr": 0.1},
            },
            execution={"max_in_flight": 4},
        )
        data = [{"input_ids": [10 + i, 20, 30]} for i in range(4)]
        with patch(
            "pytrio.ServiceClient",
            return_value=SimpleNamespace(
                create_lora_training_client=lambda **kw: client
            ),
        ):
            holder["trainer"] = SFTTrainer(args=config, train_dataset=data)
            await holder["trainer"].train()
        for actual, expected in zip(
            [value for value, _ in observations], [1.0, 0.7, 0.4, 0.1], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual([completed for _, completed in observations], [0, 0, 0, 0])
        self.assertEqual(holder["trainer"].state.optimizer_steps, 4)

    async def test_pipeline_and_serial_checkpoint_restore_share_the_same_lr_sequence(
        self,
    ):
        helper = resume_support.ResumeTests()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = self.scheduled_config(helper, "sft", root / "unused", "cosine")
            args.checkpoint.state_dir = None
            full_service, split_service = Service(), Service()
            with patch("pytrio.ServiceClient", return_value=full_service):
                _full, owner = await helper.make("sft", args)
                await helper.train("sft", owner)
            with patch("pytrio.ServiceClient", return_value=split_service):
                head, owner = await helper.make(
                    "sft",
                    self.scheduled_config(helper, "sft", root / "split", "cosine"),
                )
                await helper.train("sft", owner, stop=1)
                _resumed, owner = await helper.make(
                    "sft",
                    self.scheduled_config(
                        helper,
                        "sft",
                        root / "split",
                        "cosine",
                        resume=head.state.checkpoint_manifest_path,
                    ),
                )
                await helper.train("sft", owner)
            self.assertEqual(
                full_service.clients[0].lrs,
                split_service.clients[0].lrs + split_service.clients[1].lrs,
            )
            helper.assert_client_equal(
                full_service.clients[0], split_service.clients[1]
            )

    async def test_unknown_iterable_budget_and_failed_forward_do_not_advance_schedule(
        self,
    ):
        config = SFTConfig(
            model={"name": "policy"}, optimizer={"scheduler": {"type": "linear"}}
        )
        with patch("pytrio.ServiceClient") as remote:
            with self.assertRaisesRegex(ValueError, "known total_steps"):
                SFTTrainer(args=config, train_dataset=iter([{"text": "abc"}]))
            remote.assert_not_called()
        config.optimizer.scheduler = SchedulerConfig(type="constant", warmup_steps=2)
        service = Service()
        with patch("pytrio.ServiceClient", return_value=service):
            trainer = SFTTrainer(args=config, train_dataset=iter([{"text": "abc"}] * 3))
            await trainer.train()
        self.assertEqual(
            service.clients[0].lrs,
            [config.learning_rate / 2, config.learning_rate, config.learning_rate],
        )
        config.optimizer.scheduler = SchedulerConfig(type="linear", total_steps=2)
        with patch("pytrio.ServiceClient", return_value=Service()):
            trainer = SFTTrainer(args=config)
            with (
                patch.object(
                    trainer._client,
                    "forward_backward_async",
                    side_effect=RuntimeError("backward failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "backward failed"),
            ):
                await trainer.step([{"text": "abc"}])
            self.assertEqual(trainer.state.optimizer_steps, 0)
            self.assertEqual(trainer._submitted_optimizer_steps, 0)

    async def test_resume_rejects_changed_inferred_horizon_before_remote_allocation(
        self,
    ):
        helper = resume_support.ResumeTests()
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient", return_value=Service()),
        ):
            config = self.scheduled_config(helper, "sft", root, "linear")
            head, owner = await helper.make("sft", config)
            await helper.train("sft", owner, stop=1)
            resumed_config = self.scheduled_config(
                helper,
                "sft",
                root,
                "linear",
                resume=head.state.checkpoint_manifest_path,
            )
            resumed_config.max_steps = 4
            with patch("pytrio.ServiceClient") as remote:
                with self.assertRaisesRegex(ValueError, "scheduler horizon"):
                    await helper.make("sft", resumed_config)
                remote.assert_not_called()

    async def test_version_one_constant_lr_checkpoint_remains_resumable(self):
        helper = resume_support.ResumeTests()
        for kind in ("sft", "ppo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                service = Service()
                with patch("pytrio.ServiceClient", return_value=service):
                    head, owner = await helper.make(kind, helper.config(kind, root))
                    await helper.train(kind, owner, stop=1)
                    legacy = load_training_checkpoint(
                        head.state.checkpoint_manifest_path
                    )
                    legacy["format_version"] = 1
                    legacy.pop("schedulers")
                    values = legacy["identity"]["config"]
                    values["optimizer"].pop("scheduler")
                    if "critic" in values:
                        values["critic"]["optimizer"].pop("scheduler")
                    legacy["identity_sha256"] = hashlib.sha256(
                        json.dumps(
                            legacy["identity"],
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode()
                    ).hexdigest()
                    path = root / "v1.json"
                    path.write_text(json.dumps(legacy))
                    resumed, owner = await helper.make(
                        kind, helper.config(kind, root, resume=str(path))
                    )
                    await helper.train(kind, owner)
                saved = load_training_checkpoint(resumed.state.checkpoint_manifest_path)
                self.assertEqual(saved["format_version"], 2)
                self.assertEqual(resumed.state.global_step, 3)
                self.assertEqual(saved["schedulers"]["actor"]["last_lr"], 0.001)

    async def test_corrupt_scheduler_progress_is_rejected_before_remote_allocation(
        self,
    ):
        helper = resume_support.ResumeTests()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch("pytrio.ServiceClient", return_value=Service()):
                head, owner = await helper.make(
                    "sft", self.scheduled_config(helper, "sft", root, "linear")
                )
                await helper.train("sft", owner, stop=1)
            original = load_training_checkpoint(head.state.checkpoint_manifest_path)
            for field, value in (
                ("optimizer_steps", 0),
                ("total_steps", 4),
                ("last_lr", 0.5),
            ):
                changed = json.loads(json.dumps(original))
                changed["schedulers"]["actor"][field] = value
                path = root / f"bad-{field}.json"
                path.write_text(json.dumps(changed))
                config = self.scheduled_config(
                    helper, "sft", root, "linear", resume=str(path)
                )
                with self.subTest(field=field), patch("pytrio.ServiceClient") as remote:
                    with self.assertRaisesRegex(ValueError, "scheduler"):
                        await helper.make("sft", config)
                    remote.assert_not_called()

    async def test_explicit_horizon_allows_extending_total_target_without_resetting_lr(
        self,
    ):
        helper = resume_support.ResumeTests()
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient", return_value=Service()) as remote,
        ):
            config = self.scheduled_config(helper, "sft", root, "linear")
            config.max_steps = 2
            config.optimizer.scheduler.total_steps = 3
            head, owner = await helper.make("sft", config)
            await helper.train("sft", owner, stop=1)
            resumed_config = self.scheduled_config(
                helper,
                "sft",
                root,
                "linear",
                resume=head.state.checkpoint_manifest_path,
            )
            resumed_config.optimizer.scheduler.total_steps = 3
            _resumed, owner = await helper.make("sft", resumed_config)
            await helper.train("sft", owner)
            self.assertEqual(remote.return_value.clients[1].lrs, [0.00055, 0.0001])


if __name__ == "__main__":
    unittest.main()
