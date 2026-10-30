"""Continuous N-step runs must match independent K + (N-K) trainer instances."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from checkpoint_test_support import Service
from recipes.calculator_agent import calculator_rows, calculator_workflow

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import AgentConfig, AgentTrainingLoop
from triotune.trainer import (
    DPOConfig,
    DPOTrainer,
    OPDConfig,
    OPDTrainer,
    SFTConfig,
    SFTTrainer,
)
from triotune.utils.checkpoint import capture_rng, load_training_checkpoint, restore_rng
from triotune.utils.dataloader import dataset_identity
from triotune.utils.metrics import MetricsAPI


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    def test_default_agent_loop_keeps_pre_gigpo_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = PPOTrainer(
                args=self.config("agent", root),
                train_dataset=self.rows("agent"),
                eval_dataset=self.rows("agent")[:1],
            )
            AgentTrainingLoop(
                trainer,
                calculator_workflow(),
                config=AgentConfig(max_concurrency=1),
            )
            identity = trainer._checkpoint.loop["identity"]
            self.assertNotIn("batch_advantage_provider", identity)
            self.assertNotIn("anchor_observer", identity)
            self.assertNotIn(
                "reference_kl_coef",
                trainer._checkpoint.identity["config"]["actor"],
            )
            enabled_config = self.config("agent", Path(root) / "reference-kl")
            enabled_config.reference_kl_coef = 0.01
            enabled = PPOTrainer(
                args=enabled_config,
                train_dataset=self.rows("agent"),
                eval_dataset=self.rows("agent")[:1],
            )
            self.assertEqual(
                enabled._checkpoint.identity["config"]["actor"][
                    "reference_kl_coef"
                ],
                0.01,
            )

    async def test_ppo_initialization_failure_keeps_reporter_cleanup(self):
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as root:
            trainer = PPOTrainer(
                args=self.config("ppo", root),
                train_dataset=self.rows("ppo"),
                reward_fn=lambda **kwargs: 1.0,
            )
            with (
                patch.object(
                    trainer,
                    "initialize",
                    new=AsyncMock(side_effect=ConnectionError("load failed")),
                ),
                patch.object(trainer, "close", new=AsyncMock()) as close,
                self.assertRaisesRegex(ConnectionError, "load failed"),
            ):
                await trainer.train()
            close.assert_awaited_once_with(state="crashed", error="load failed")
            self.assertFalse(trainer._checkpoint.in_train_loop)

    async def test_filtered_rows_resume_at_raw_cursor_not_step_times_batch_size(self):
        data = [{"messages": [10 + i, 20], "drop": i % 2 == 0} for i in range(4)]

        def reward(*, row, completion_ids, **kwargs):
            return float(not row["drop"] and completion_ids[0] == 65)

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)

            def config(path, resume=None):
                value = self.config("ppo", path, resume=resume)
                value.max_steps = 2
                value.num_generations = 2
                value.dynamic_sampling = True
                return value

            with patch("pytrio.ServiceClient", return_value=Service()):
                full = PPOTrainer(
                    args=config(root / "full"), train_dataset=data, reward_fn=reward
                )
                await full.train()
            service = Service()
            with patch("pytrio.ServiceClient", return_value=service):
                head = PPOTrainer(
                    args=config(root / "split"), train_dataset=data, reward_fn=reward
                )
                await head.train(stop_after_step=1)
                self.assertEqual(head.state.data_cursor, 2)
                resumed = PPOTrainer(
                    args=config(root / "split", head.state.checkpoint_manifest_path),
                    train_dataset=data,
                    reward_fn=reward,
                )
                await resumed.train()
            self.assertEqual(resumed.state.data_cursor, 4)
            self.assertEqual(resumed._next_group, 4)
            self.assert_client_equal(full._client, resumed._client)

    def test_sync_dpo_and_opd_resume_keep_optimizer_state(self):
        for kind, trainer_type in (("dpo", DPOTrainer), ("opd", OPDTrainer)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                data = self.rows(kind)
                with patch("pytrio.ServiceClient", return_value=Service()):
                    full = trainer_type(
                        args=self.config(kind, root / "full"), train_dataset=data
                    )
                    full.train()
                service = Service()
                with patch("pytrio.ServiceClient", return_value=service):
                    head = trainer_type(
                        args=self.config(kind, root / "split"), train_dataset=data
                    )
                    head.train(stop_after_step=1)
                    resumed = trainer_type(
                        args=self.config(
                            kind,
                            root / "split",
                            resume=head.state.checkpoint_manifest_path,
                        ),
                        train_dataset=data,
                    )
                    resumed.train()
                self.assert_client_equal(
                    full._training_client, resumed._training_client
                )
                self.assertEqual(resumed.state.data_cursor, 3)

    def config(self, kind, root, *, resume=None, save_every=1):
        values = {
            "model": {"name": "policy", "lora_rank": 2},
            "training": {"max_steps": 3, "batch_size": 1, "seed": 0},
            "optimizer": {"learning_rate": 0.001},
            "checkpoint": {
                "state_dir": str(root),
                "resume_from": resume,
                "save_every": save_every,
            },
            "evaluation": {"before_train": True, "after_train": True, "steps": 2},
            "reporting": {"report_to": None},
        }
        if kind == "sft":
            return SFTConfig(**values, execution={"max_in_flight": 4})
        if kind == "dpo":
            return DPOConfig(**values)
        if kind == "opd":
            return OPDConfig(
                **values,
                teacher={"name": "teacher"},
                rollout={"num_generations": 1, "max_completion_length": 4},
            )
        return PPOConfig.for_algorithm(
            "reinforce" if kind == "agent" else "ppo",
            **values,
            rollout={"num_generations": 1, "max_tokens": 256},
            actor={
                "reward_norm": None,
                "adv_norm": None,
                "update_epochs": 2 if kind == "ppo" else 1,
            },
            critic={"update_epochs": 2} if kind == "ppo" else None,
        )

    def rows(self, kind):
        if kind == "agent":
            return [dict(calculator_rows()[0], task_id=f"task-{i}") for i in range(3)]
        if kind == "sft":
            return [
                {"input_ids": [10 + i, 20, 30 + i], "labels": [-100, 20, 30 + i]}
                for i in range(3)
            ]
        if kind == "dpo":
            return [
                {"messages": [10 + i, 20], "chosen": [30 + i], "rejected": [40 + i]}
                for i in range(3)
            ]
        return [{"messages": [10 + i, 20], "reward": float(i % 2)} for i in range(3)]

    async def make(self, kind, config):
        rows = self.rows(kind)
        valid = rows[:1]
        if kind in {"ppo", "agent"}:
            trainer = PPOTrainer(
                args=config,
                train_dataset=rows,
                eval_dataset=valid,
                reward_fn=lambda **kw: kw["row"]["reward"],
            )
            if kind == "agent":
                owner = AgentTrainingLoop(
                    trainer,
                    calculator_workflow(),
                    config=AgentConfig(max_concurrency=1),
                )
            else:
                owner = trainer
            await trainer.initialize()
        else:
            trainer = {"sft": SFTTrainer, "dpo": DPOTrainer, "opd": OPDTrainer}[kind](
                args=config, train_dataset=rows, eval_dataset=valid
            )
            owner = trainer
        return trainer, owner

    async def train(self, kind, owner, stop=None):
        method = owner.train_async if kind in {"dpo", "opd"} else owner.train
        return await method(stop_after_step=stop)

    def assert_client_equal(self, before, after):
        torch.testing.assert_close(before.weight, after.weight, rtol=0, atol=0)
        left, right = before.optimizer.state_dict(), after.optimizer.state_dict()
        self.assertEqual(left["param_groups"], right["param_groups"])
        for key in left["state"][0]:
            torch.testing.assert_close(
                left["state"][0][key], right["state"][0][key], rtol=0, atol=0
            )

    async def test_all_trainers_match_weights_adam_inputs_lr_and_accounting(self):
        for kind in ("sft", "dpo", "opd", "ppo", "agent"):
            for interval in (0, 1, 2):
                with (
                    self.subTest(kind=kind, save_every=interval),
                    tempfile.TemporaryDirectory() as root,
                ):
                    root = Path(root)
                    continuous = Service(agent=kind == "agent")
                    with patch("pytrio.ServiceClient", return_value=continuous):
                        full, owner = await self.make(
                            kind, self.config(kind, root / "full", save_every=interval)
                        )
                        await self.train(kind, owner)
                    split = Service(agent=kind == "agent")
                    with patch("pytrio.ServiceClient", return_value=split):
                        head, first_owner = await self.make(
                            kind, self.config(kind, root / "split", save_every=interval)
                        )
                        await self.train(kind, first_owner, stop=1)
                        manifest = head.state.checkpoint_manifest_path
                        self.assertEqual(head.state.data_cursor, 1)
                        self.assertEqual(
                            load_training_checkpoint(manifest)["state"]["global_step"],
                            1,
                        )
                        # Simulate unrelated process RNGs before reconstruction.
                        random.seed(999)
                        np.random.seed(999)
                        torch.manual_seed(999)
                        resumed, second_owner = await self.make(
                            kind,
                            self.config(
                                kind,
                                root / "split",
                                resume=manifest,
                                save_every=interval,
                            ),
                        )
                        self.assertEqual(resumed.state.global_step, 1)
                        await self.train(kind, second_owner)
                    component_count = 2 if kind == "ppo" else 1
                    self.assertEqual(len(split.loaded), component_count)
                    for i in range(component_count):
                        self.assert_client_equal(
                            continuous.clients[i], split.clients[i + component_count]
                        )
                        self.assertEqual(
                            continuous.clients[i].inputs,
                            split.clients[i].inputs
                            + split.clients[i + component_count].inputs,
                        )
                        self.assertEqual(
                            continuous.clients[i].lrs,
                            split.clients[i].lrs
                            + split.clients[i + component_count].lrs,
                        )
                        self.assertEqual(
                            continuous.clients[i].random_draws,
                            split.clients[i].random_draws
                            + split.clients[i + component_count].random_draws,
                        )
                    self.assertEqual(continuous.samples, split.samples)
                    self.assertEqual(full.state.global_step, resumed.state.global_step)
                    self.assertEqual(
                        full.state.optimizer_steps, resumed.state.optimizer_steps
                    )
                    self.assertEqual(full.state.data_cursor, resumed.state.data_cursor)
                    expected = load_training_checkpoint(
                        full.state.checkpoint_manifest_path
                    )
                    actual = load_training_checkpoint(
                        resumed.state.checkpoint_manifest_path
                    )
                    self.assertEqual(expected["runtime"], actual["runtime"])
                    for key in full.state.metrics:
                        if key.startswith(("usage/", "eval/")) and "elapsed" not in key:
                            self.assertEqual(
                                full.state.metrics[key], resumed.state.metrics[key], key
                            )
                    if kind == "agent":
                        self.assertEqual(owner._seed_offset, second_owner._seed_offset)
                        self.assertEqual(
                            owner._baseline_episodes, second_owner._baseline_episodes
                        )
                        self.assertEqual(
                            [u["policy_versions"] for u in owner.update_history],
                            [u["policy_versions"] for u in second_owner.update_history],
                        )

    async def test_bad_identity_is_rejected_before_remote_client_creation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            service = Service()
            with patch("pytrio.ServiceClient", return_value=service):
                first, owner = await self.make("sft", self.config("sft", root))
                await self.train("sft", owner, stop=1)
            path = first.state.checkpoint_manifest_path
            for change in ("lr", "data", "version", "rng"):
                config = self.config("sft", root, resume=path)
                rows = self.rows("sft")
                payload = json.loads(Path(path).read_text())
                if change == "lr":
                    config.learning_rate *= 2
                elif change == "data":
                    rows.reverse()
                else:
                    if change == "version":
                        payload["format_version"] = 999
                    else:
                        payload["rng"]["python"] = []
                    altered = root / f"{change}.json"
                    altered.write_text(json.dumps(payload))
                    config.checkpoint.resume_from = str(altered)
                with (
                    self.subTest(change=change),
                    patch("pytrio.ServiceClient") as remote,
                ):
                    with self.assertRaises((ValueError, TypeError, IndexError)):
                        SFTTrainer(
                            args=config,
                            train_dataset=rows,
                            eval_dataset=rows[:1]
                            if change != "data"
                            else self.rows("sft")[:1],
                        )
                    remote.assert_not_called()

    async def test_failed_remote_save_keeps_previous_manifest(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient", return_value=Service()) as service,
        ):
            trainer, owner = await self.make("sft", self.config("sft", root))
            await self.train("sft", owner, stop=1)
            original = Path(trainer.state.checkpoint_manifest_path).read_bytes()
            service.return_value.fail_save = True
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                await trainer._save_state()
            self.assertEqual(
                Path(trainer.state.checkpoint_manifest_path).read_bytes(), original
            )

    async def test_metric_failure_does_not_publish_a_partial_dpo_boundary(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient", return_value=Service()),
        ):
            trainer, owner = await self.make("dpo", self.config("dpo", root))
            with (
                patch.object(
                    trainer._metrics_api,
                    "record",
                    side_effect=OSError("metric write failed"),
                ),
                self.assertRaisesRegex(OSError, "metric write failed"),
            ):
                await self.train("dpo", owner)
            self.assertIsNone(trainer.state.checkpoint_manifest_path)
            self.assertFalse(list(Path(root).glob("*.json")))

    async def test_interrupted_boundary_replays_pending_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            full_service, split_service = Service(), Service()
            config = self.config("ppo", root / "full")
            config.eval_steps = 1
            with patch("pytrio.ServiceClient", return_value=full_service):
                full, owner = await self.make("ppo", config)
                await self.train("ppo", owner)
            config = self.config("ppo", root / "split")
            config.eval_steps = 1
            with patch("pytrio.ServiceClient", return_value=split_service):
                head, owner = await self.make("ppo", config)
                save = head._checkpoint.remote_saved

                def interrupted(trainer):
                    save(trainer)
                    raise ConnectionError("interrupted before scheduled evaluation")

                with (
                    patch.object(
                        head._checkpoint, "remote_saved", side_effect=interrupted
                    ),
                    self.assertRaises(ConnectionError),
                ):
                    await self.train("ppo", owner)
                path = head.state.checkpoint_manifest_path
                self.assertEqual(
                    load_training_checkpoint(path)["runtime"]["_last_eval_step"], 0
                )
                config = self.config("ppo", root / "split", resume=path)
                config.eval_steps = 1
                resumed, owner = await self.make("ppo", config)
                await self.train("ppo", owner)
            self.assertEqual(full_service.samples, split_service.samples)
            self.assert_client_equal(full_service.clients[0], split_service.clients[2])
            self.assertEqual(
                full.state.metrics["usage/train_tokens_total"],
                resumed.state.metrics["usage/train_tokens_total"],
            )

    async def test_atomic_local_publish_failure_retains_last_complete_record(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient", return_value=Service()),
        ):
            trainer, owner = await self.make("sft", self.config("sft", root))
            await self.train("sft", owner, stop=1)
            path = Path(trainer.state.checkpoint_manifest_path)
            original = path.read_bytes()
            with (
                patch(
                    "triotune.utils.checkpoint.os.replace",
                    side_effect=OSError("disk unavailable"),
                ),
                self.assertRaises(OSError),
            ):
                trainer._checkpoint.publish(trainer)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(Path(root).glob(".state-*")))

    def test_reporting_window_resumes_without_flushing_a_partial_window(self):
        from unittest.mock import Mock

        full_run, head_run, tail_run = Mock(), Mock(), Mock()
        full = MetricsAPI(full_run, logging_step=3)
        head = MetricsAPI(head_run, logging_step=3)
        for api in (full, head):
            api.record(
                {"train/loss": 1.0, "reward/mean": 1.0, "reward/std": 0.2}, step=1
            )
        state = json.loads(json.dumps(head.state_dict()))
        head.finish(flush=False)
        head_run.log.assert_not_called()
        tail = MetricsAPI(tail_run, logging_step=3)
        tail.load_state_dict(state)
        for api in (full, tail):
            api.record(
                {"train/loss": 3.0, "reward/mean": 2.0, "reward/std": 0.4}, step=2
            )
            api.record(
                {"train/loss": 5.0, "reward/mean": 3.0, "reward/std": 0.6}, step=3
            )
        self.assertEqual(full_run.log.call_args, tail_run.log.call_args)

    async def test_one_shot_source_rejected_before_remote_work(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient") as remote,
        ):
            with self.assertRaisesRegex(ValueError, "random-access"):
                SFTTrainer(
                    args=self.config("sft", root), train_dataset=iter(self.rows("sft"))
                )
            remote.assert_not_called()

    def test_rng_roundtrip_and_data_order_identity(self):
        state = capture_rng()
        expected = (random.random(), np.random.random(), torch.rand(2))
        restore_rng(json.loads(json.dumps(state)))
        actual = (random.random(), np.random.random(), torch.rand(2))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        self.assertNotEqual(dataset_identity([1, 2]), dataset_identity([2, 1]))

    def test_calculator_manifest_paths_follow_experiment_location(self):
        from examples.agent.calculator.train import load_experiment

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "experiment.yaml"
            path.write_text(
                "trainer:\n  model: {name: policy}\n  checkpoint:\n"
                "    state_dir: states\n    resume_from: states/previous.json\n"
            )
            config, *_ = load_experiment(path)
            self.assertEqual(config.checkpoint.state_dir, str(Path(root) / "states"))
            self.assertEqual(
                config.checkpoint.resume_from, str(Path(root) / "states/previous.json")
            )

    async def test_complete_mode_rejects_manual_steps_without_a_data_cursor(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pytrio.ServiceClient", return_value=Service()),
        ):
            trainer, _ = await self.make("sft", self.config("sft", root))
            with self.assertRaisesRegex(ValueError, "manual step"):
                await trainer.step(self.rows("sft")[:1])
            self.assertEqual(trainer._client.steps, 0)


if __name__ == "__main__":
    unittest.main()
