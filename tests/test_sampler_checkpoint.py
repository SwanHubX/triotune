"""Sampler retention, exact CPU weights and all trainer lifecycle paths."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from checkpoint_test_support import Client, Service
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_ppo_family import Future

import triotune
from triotune.agentrl import AgentOPDTrainingLoop, AgentTrainingLoop
from triotune.api import CheckpointConfig, SamplerCheckpointConfig
from triotune.trainer import (
    DPOConfig,
    DPOTrainer,
    OPDConfig,
    OPDTrainer,
    PPOConfig,
    PPOTrainer,
    SFTConfig,
    SFTTrainer,
)
from triotune.utils.checkpoint import load_training_checkpoint
from triotune.utils.config import load_yaml_config
from triotune.utils.sampler_checkpoint import SamplerCheckpoint

ROOT = Path(__file__).resolve().parents[1]
PRESETS = (
    "ppo",
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
)


class ExportClient(Client):
    def get_tokenizer(self):
        tokenizer = super().get_tokenizer()
        if not hasattr(tokenizer, "get_vocab"):
            tokenizer.get_vocab = lambda: {chr(i): i for i in range(256)}
        return tokenizer

    def save_weights_for_sampler(self, *, name, ttl_seconds):
        if self.service.export_error:
            raise RuntimeError("sampler export failed")
        self.service.exports.append(
            {
                "name": name,
                "ttl_seconds": ttl_seconds,
                "step": self.steps,
                "client": self,
                "weight": self.weight.detach().clone(),
            }
        )
        return SimpleNamespace(path=f"trio://sampler/{name}")

    async def save_weights_for_sampler_async(self, **kwargs):
        return Future(self.save_weights_for_sampler(**kwargs))


class ExportService(Service):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exports = []
        self.export_error = False

    def create_lora_training_client(self, **kwargs):
        client = ExportClient(self)
        self.clients.append(client)
        return client

    def create_training_client_from_state_with_optimizer(self, *, path):
        self.loaded.append(path)
        client = ExportClient(self, self.saved[path])
        self.clients.append(client)
        return client


def config(kind, *, enabled=True, every=0, train_every=2, steps=3, **checkpoint):
    values = {
        "model": {"name": "policy", "lora_rank": 2},
        "training": {"batch_size": 1, "max_steps": steps, "seed": 0},
        "checkpoint": {
            "save_every": train_every,
            "sampler": {"enabled": enabled, "save_every": every, "ttl_seconds": 86400},
            **checkpoint,
        },
        "evaluation": {"before_train": False, "after_train": False},
    }
    if kind == "sft":
        return SFTConfig(**values, execution={"max_in_flight": 4})
    if kind == "dpo":
        return DPOConfig(**values)
    if kind == "opd":
        return OPDConfig(
            **values,
            teacher={"name": "teacher"},
            rollout={
                "num_generations": 1,
                "max_completion_length": 4,
            },
        )
    extra = {"actor": {"overlong_tokens": 2}} if kind == "dapo" else {}
    return PPOConfig.for_algorithm(
        kind, **values, rollout={"num_generations": 2, "max_tokens": 4}, **extra
    )


def rows(kind, steps=3):
    if kind == "sft":
        return [
            {"input_ids": [10 + i, 20, 30], "labels": [-100, 20, 30]}
            for i in range(steps)
        ]
    if kind == "dpo":
        return [
            {"messages": [10 + i, 20], "chosen": [30], "rejected": [40]}
            for i in range(steps)
        ]
    return [{"messages": [10 + i, 20]} for i in range(steps)]


def trainer_for(kind, args, data):
    if kind == "sft":
        return SFTTrainer(args=args, train_dataset=data)
    if kind == "dpo":
        return DPOTrainer(args=args, train_dataset=data)
    if kind == "opd":
        return OPDTrainer(args=args, train_dataset=data)
    return PPOTrainer(
        args=args,
        train_dataset=data,
        reward_fn=lambda completion_ids, **_: float(completion_ids[0] == 65),
    )


async def train(trainer):
    if isinstance(trainer, (DPOTrainer, OPDTrainer)):
        return await trainer.train_async()
    return await trainer.train()


class ConfigurationTests(unittest.TestCase):
    def test_template_and_all_configs_roundtrip(self):
        self.assertEqual(Path(triotune.__file__).resolve().parent, ROOT / "triotune")
        values = load_yaml_config(ROOT / "tests/configs/checkpoint.yaml")
        for kind in ("sft", "dpo", "opd", *PRESETS):
            with self.subTest(kind=kind):
                args = config(kind, **values["checkpoint"])
                self.assertIsInstance(args.checkpoint.sampler, SamplerCheckpointConfig)
                self.assertFalse(args.checkpoint.sampler.enabled)
                self.assertEqual(args.checkpoint.sampler.save_every, 0)
                self.assertEqual(asdict(args.checkpoint), values["checkpoint"])
                loaded = type(args).from_modules({"algorithm": kind, **values})
                self.assertEqual(asdict(loaded.checkpoint), values["checkpoint"])

    def test_invalid_config_fails_locally(self):
        for fields in (
            {"enabled": "true"},
            {"enabled": 1},
            {"save_every": -1},
            {"save_every": True},
            {"save_every": 1.5},
            {"ttl_seconds": 0},
            {"ttl_seconds": -1},
            {"ttl_seconds": True},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                CheckpointConfig(sampler=fields)
        self.assertFalse(CheckpointConfig().sampler.enabled)
        self.assertIsNone(CheckpointConfig().sampler.ttl_seconds)
        self.assertEqual(
            CheckpointConfig(sampler={"enabled": True}).sampler.save_every, 0
        )


class TrainingSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_algorithms_disabled_final_and_independent_intervals(self):
        for kind in ("sft", "dpo", "opd", *PRESETS):
            for enabled, every, expected in (
                (False, 1, []),
                (True, 0, [3]),
                (True, 2, [2, 3]),
                (True, 3, [3]),
            ):
                with self.subTest(kind=kind, enabled=enabled, every=every):
                    service = ExportService()
                    with patch("pytrio.ServiceClient", return_value=service):
                        trainer = trainer_for(
                            kind, config(kind, enabled=enabled, every=every), rows(kind)
                        )
                        state = await train(trainer)
                    actor = (
                        trainer._client
                        if kind == "sft" or kind in PRESETS
                        else trainer._training_client
                    )
                    self.assertEqual(
                        [item["step"] for item in service.exports], expected
                    )
                    self.assertTrue(
                        all(item["client"] is actor for item in service.exports)
                    )
                    self.assertTrue(
                        all(item["ttl_seconds"] == 86400 for item in service.exports)
                    )
                    self.assertEqual(
                        [
                            saved["steps"]
                            for uri, saved in service.saved.items()
                            if "critic" not in uri
                        ],
                        [2, 3],
                    )
                    self.assertEqual(state.global_step, 3)
                    self.assertNotIn("/sampler/", state.checkpoint_path)
                    if enabled:
                        self.assertIn("/sampler/", state.sampler_checkpoint_path)
                        self.assertTrue(
                            torch.equal(service.exports[-1]["weight"], actor.weight)
                        )
                        # A manual export at the already saved final step is a no-op.
                        self.assertEqual(
                            await trainer.save_sampler_checkpoint_async(),
                            state.sampler_checkpoint_path,
                        )
                        self.assertEqual(len(service.exports), len(expected))
                    else:
                        self.assertIsNone(state.sampler_checkpoint_path)
                        self.assertIsNone(await trainer.save_sampler_checkpoint_async())

    async def test_sft_pipeline_exports_the_named_step_before_later_updates(self):
        service = ExportService()
        with patch("pytrio.ServiceClient", return_value=service):
            trainer = trainer_for(
                "sft", config("sft", every=2, train_every=3, steps=5), rows("sft", 5)
            )
            await trainer.train()
        self.assertEqual([e["step"] for e in service.exports], [2, 4, 5])
        self.assertEqual(
            [e["name"] for e in service.exports],
            [f"sft-sampler-step-{step:08d}" for step in (2, 4, 5)],
        )
        self.assertEqual([s["steps"] for s in service.saved.values()], [3, 5])

    def test_sync_dpo_and_opd_entrypoints(self):
        for kind in ("dpo", "opd"):
            with self.subTest(kind=kind):
                service = ExportService()
                with patch("pytrio.ServiceClient", return_value=service):
                    trainer = trainer_for(kind, config(kind, every=2), rows(kind))
                    state = trainer.train()
                self.assertEqual([e["step"] for e in service.exports], [2, 3])
                self.assertIn("/sampler/", state.sampler_checkpoint_path)
                if kind == "dpo":
                    self.assertEqual(
                        trainer.save_sampler_checkpoint(), state.sampler_checkpoint_path
                    )

    async def test_pause_resume_and_old_manifests_keep_train_separate(self):
        for kind in ("sft", "dpo", "opd", "grpo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                service = ExportService()
                with patch("pytrio.ServiceClient", return_value=service):
                    args = config(kind, state_dir=directory)
                    head = trainer_for(kind, args, rows(kind))
                    method = head.train_async if kind in ("dpo", "opd") else head.train
                    await method(stop_after_step=1)
                    self.assertEqual(service.exports, [])
                    # Simulate an existing manifest written before Sampler metadata.
                    path = Path(head.state.checkpoint_manifest_path)
                    payload = json.loads(path.read_text())
                    payload["state"].pop("sampler_checkpoint_path", None)
                    path.write_text(json.dumps(payload))
                    self.assertEqual(
                        load_training_checkpoint(path)["state"]["global_step"], 1
                    )
                    resumed = trainer_for(
                        kind,
                        config(kind, state_dir=directory, resume_from=str(path)),
                        rows(kind),
                    )
                    state = await train(resumed)
                self.assertEqual([e["step"] for e in service.exports], [3])
                self.assertEqual(
                    load_training_checkpoint(state.checkpoint_manifest_path)["state"][
                        "sampler_checkpoint_path"
                    ],
                    state.sampler_checkpoint_path,
                )

    async def test_export_failure_preserves_train_checkpoint(self):
        for kind in ("sft", "dpo", "opd", "grpo"):
            with self.subTest(kind=kind):
                service = ExportService()
                service.export_error = True
                with patch("pytrio.ServiceClient", return_value=service):
                    trainer = trainer_for(kind, config(kind, steps=1), rows(kind, 1))
                    with self.assertRaisesRegex(RuntimeError, "sampler export failed"):
                        await train(trainer)
                self.assertIsNotNone(trainer.state.checkpoint_path)
                self.assertIsNone(trainer.state.sampler_checkpoint_path)
                self.assertIsNone(trainer._sampler_checkpoint.saved_step)

    async def test_final_evaluation_failure_skips_final_only_export(self):
        for kind in ("sft", "dpo", "opd", "grpo"):
            with self.subTest(kind=kind):
                service = ExportService()
                with patch("pytrio.ServiceClient", return_value=service):
                    args = config(kind, steps=1)
                    args.evaluation.after_train = True
                    trainer = trainer_for(kind, args, rows(kind, 1))
                    trainer.eval_dataset = rows(kind, 1)
                    method = {
                        "sft": "evaluate",
                        "dpo": "evaluate_async",
                        "opd": "_evaluate_async",
                        "grpo": "evaluate",
                    }[kind]
                    with patch.object(
                        trainer,
                        method,
                        new=AsyncMock(side_effect=RuntimeError("eval failed")),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "eval failed"):
                            await train(trainer)
                self.assertEqual(service.exports, [])
                self.assertIsNone(trainer.state.sampler_checkpoint_path)

    async def test_agent_ppo_and_opd_use_same_policy(self):
        for kind in ("reinforce", "opd"):
            for every, expected in ((0, [3]), (2, [2, 3]), (3, [3])):
                with self.subTest(kind=kind, every=every):
                    service = ExportService(agent=True)
                    with patch("pytrio.ServiceClient", return_value=service):
                        trainer = trainer_for(
                            kind, config(kind, every=every), calculator_rows()[:3]
                        )
                        loop_type = (
                            AgentOPDTrainingLoop if kind == "opd" else AgentTrainingLoop
                        )
                        state = await loop_type(trainer, calculator_workflow()).train()
                    self.assertEqual([e["step"] for e in service.exports], expected)
                    self.assertIn("/sampler/", state.sampler_checkpoint_path)
                    self.assertNotIn("/sampler/", state.checkpoint_path)

    async def test_no_completed_updates_and_invalid_response_do_not_publish(self):
        policy = SamplerCheckpoint(SamplerCheckpointConfig(enabled=True), "sft")
        state = SimpleNamespace(global_step=0, sampler_checkpoint_path=None)
        client = Mock(spec=["save_weights_for_sampler"])
        reporter = Mock()
        self.assertIsNone(await policy.save_async(client, state, reporter))
        client.assert_not_called()
        self.assertEqual(client.mock_calls, [])
        state.global_step = 1
        client.save_weights_for_sampler.return_value = SimpleNamespace(path="")
        with self.assertRaisesRegex(TypeError, "no checkpoint path"):
            await policy.save_async(client, state, reporter)
        self.assertIsNone(state.sampler_checkpoint_path)
        reporter.log_text.assert_not_called()
        client.save_weights_for_sampler.assert_called_once_with(
            name="sft-sampler-step-00000001", ttl_seconds=None
        )

    async def test_future_failure_and_cancellation_leave_export_unpublished(self):
        for error in (RuntimeError("remote save failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                policy = SamplerCheckpoint(SamplerCheckpointConfig(enabled=True), "opd")
                state = SimpleNamespace(
                    global_step=1, sampler_checkpoint_path="trio://previous"
                )
                reporter = Mock()

                class FailedFuture:
                    def __await__(self):
                        async def resolve():
                            await asyncio.sleep(0)
                            raise error

                        return resolve().__await__()

                class AsyncClient:
                    async def save_weights_for_sampler_async(self, **kwargs):
                        return FailedFuture()

                with self.assertRaises(type(error)):
                    await policy.save_async(AsyncClient(), state, reporter)
                self.assertEqual(state.sampler_checkpoint_path, "trio://previous")
                self.assertIsNone(policy.saved_step)
                reporter.log_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
