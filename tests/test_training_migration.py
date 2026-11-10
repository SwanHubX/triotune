"""Compatibility, sampling safety, real reporting, and export restoration."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_checkpoint_resume as resume_support
import test_sampler_checkpoint as sampler_support
from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentContractError,
    AgentOPDTrainingLoop,
    AgentTrainingLoop,
)
from triotune.agentrl.integrations.openai import PyTRIOBackend
from triotune.agentrl.runtime.sampling import create_sampling_snapshot
from triotune.utils.checkpoint import load_training_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class SamplingCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        guard = patch(
            "pytrio.ServiceClient",
            side_effect=AssertionError("Tests must use local clients."),
        )
        guard.start()
        self.addCleanup(guard.stop)

    async def test_legacy_stop_only_trainer_runs_training_and_evaluation(self):
        class LegacyTrainer(PPOTrainer):
            async def create_sampling_snapshot(self, *, stop=None):
                return await super().create_sampling_snapshot(stop=stop)

        client = AgentTrainingClient()
        trainer = LegacyTrainer(
            "fake",
            args=PPOConfig.for_algorithm(
                "grpo",
                max_steps=1,
                num_generations=2,
                eval_before_train=True,
                eval_after_train=True,
            ),
            training_client=client,
            train_dataset=calculator_rows()[:1],
            eval_dataset=calculator_rows()[:1],
        )
        state = await AgentTrainingLoop(trainer, calculator_workflow()).train()
        self.assertEqual(state.global_step, 1)
        self.assertEqual(len(client.snapshots), 3)
        with self.assertRaisesRegex(AgentContractError, "Custom trainer must accept"):
            await create_sampling_snapshot(trainer, temperature=0.4)

    async def test_factory_internal_typeerror_is_not_retried(self):
        factory = AsyncMock(side_effect=TypeError("factory implementation failed"))
        with self.assertRaisesRegex(TypeError, "factory implementation failed"):
            await create_sampling_snapshot(
                SimpleNamespace(create_sampling_snapshot=factory)
            )
        factory.assert_awaited_once()

    async def test_evaluation_snapshot_is_rejected_before_training_collection(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake", args=PPOConfig.for_algorithm("grpo"), training_client=client
        )
        for options in ({"temperature": 0.4}, {"top_p": 0.9}):
            with self.subTest(options=options):
                snapshot = await trainer.create_sampling_snapshot(**options)
                workflow = calculator_workflow()
                with self.assertRaisesRegex(
                    AgentContractError, "Training snapshots require"
                ):
                    await workflow.collect(calculator_rows()[:1], snapshot)
                self.assertFalse(workflow._resource_started)
                with self.assertRaisesRegex(
                    AgentContractError, "Training snapshots require"
                ):
                    PyTRIOBackend(snapshot, model="fake")
                batch = await workflow.collect(
                    calculator_rows()[:1], snapshot, is_eval=True
                )
                self.assertEqual(len(batch.episodes), 1)
                await workflow.close()
        self.assertFalse(client.calls)
        await trainer.close()

    async def test_custom_snapshot_factory_cannot_bypass_training_check(self):
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm("grpo"),
            training_client=AgentTrainingClient(),
        )
        snapshot = await trainer.create_sampling_snapshot(temperature=0.4)
        loop = AgentTrainingLoop(
            trainer,
            calculator_workflow(),
            snapshot_factory=AsyncMock(return_value=snapshot),
        )
        with self.assertRaisesRegex(AgentContractError, "Training snapshots require"):
            await loop._collect(iter(calculator_rows()[:1]))
        self.assertEqual(trainer.state.optimizer_steps, 0)
        await trainer.close()


class CheckpointMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampler_failure_keeps_acknowledged_train_manifest_resumable(self):
        for case in ("sft", "dpo", "opd", "grpo", "agent_grpo", "agent_opd"):
            for every in (0, 1):
                with (
                    self.subTest(case=case, every=every),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    agent = case.startswith("agent_")
                    kind = case.removeprefix("agent_")
                    data = (
                        calculator_rows()[:2]
                        if agent
                        else sampler_support.rows(kind, 2)
                    )
                    service = sampler_support.ExportService(agent=agent)

                    def make(resume=None):
                        args = sampler_support.config(
                            kind,
                            steps=2,
                            every=every,
                            train_every=1,
                            state_dir=directory,
                            resume_from=resume,
                        )
                        if agent:
                            args.max_tokens = 256
                        trainer = sampler_support.trainer_for(kind, args, data)
                        owner = trainer
                        if agent:
                            loop = (
                                AgentOPDTrainingLoop
                                if kind == "opd"
                                else AgentTrainingLoop
                            )
                            owner = loop(trainer, calculator_workflow())
                        method = (
                            owner.train_async
                            if not agent and kind in ("dpo", "opd")
                            else owner.train
                        )
                        return trainer, method

                    with patch("pytrio.ServiceClient", return_value=service):
                        service.export_error = True
                        trainer, method = make()
                        with self.assertRaisesRegex(
                            RuntimeError, "sampler export failed"
                        ):
                            await method()
                        manifest = trainer.state.checkpoint_manifest_path
                        payload = load_training_checkpoint(manifest)
                        expected_step = 1 if every else 2
                        self.assertEqual(payload["state"]["global_step"], expected_step)
                        self.assertEqual(payload["state"]["data_cursor"], expected_step)
                        if agent:
                            self.assertEqual(
                                len(payload["loop"]["state"]["updates"]), expected_step
                            )
                        service.export_error = False
                        resumed, method = make(manifest)
                        await method()
                        self.assertEqual(resumed.state.global_step, 2)
                        self.assertEqual(
                            [export["step"] for export in service.exports], [2]
                        )

    async def test_agent_resume_accepts_legacy_and_vendored_default_identities(self):
        helper = resume_support.ResumeTests()
        for vendored in (False, True):
            with (
                self.subTest(vendored=vendored),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                service = resume_support.Service(agent=True)
                with patch("pytrio.ServiceClient", return_value=service):
                    trainer, owner = await helper.make(
                        "agent", helper.config("agent", root)
                    )
                    state = await helper.train("agent", owner, stop=1)
                    path = Path(state.checkpoint_manifest_path)
                    payload = json.loads(path.read_text())
                    evaluation = payload["loop"]["identity"]["evaluation"]
                    self.assertNotIn("temperature", evaluation)
                    if vendored:
                        evaluation.update(temperature=1.0, top_p=1.0)
                        payload["state"]["sampler_checkpoint_path"] = (
                            "trio://old-sampler"
                        )
                        path.write_text(json.dumps(payload))
                    resumed, loop = await helper.make(
                        "agent", helper.config("agent", root, resume=str(path))
                    )
                    self.assertEqual(loop._seed_offset, owner._seed_offset)
                    await helper.train("agent", loop)
                    self.assertEqual(resumed.state.global_step, 3)
                    self.assertEqual(len(loop.update_history), 3)
                    if vendored:
                        self.assertEqual(
                            load_training_checkpoint(
                                resumed.state.checkpoint_manifest_path
                            )["state"]["sampler_checkpoint_path"],
                            "trio://old-sampler",
                        )

    async def test_changed_nondefault_evaluation_is_still_rejected(self):
        helper = resume_support.ResumeTests()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "pytrio.ServiceClient", return_value=resume_support.Service(agent=True)
            ),
        ):
            trainer, owner = await helper.make(
                "agent", helper.config("agent", directory)
            )
            state = await helper.train("agent", owner, stop=1)
            restored = PPOTrainer(
                args=helper.config(
                    "agent", directory, resume=state.checkpoint_manifest_path
                ),
                train_dataset=helper.rows("agent"),
                eval_dataset=helper.rows("agent")[:1],
                reward_fn=lambda **kw: kw["row"]["reward"],
            )
            with self.assertRaisesRegex(
                ValueError, "execution loop or agent configuration"
            ):
                AgentTrainingLoop(
                    restored,
                    calculator_workflow(),
                    config=AgentConfig(
                        max_concurrency=1, evaluation={"temperature": 0.4}
                    ),
                )

    async def test_sampler_metadata_restores_step_ttl_and_deduplicates_after_restart(
        self,
    ):
        for kind in ("sft", "dpo", "opd", *sampler_support.PRESETS):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                service = sampler_support.ExportService()
                with patch("pytrio.ServiceClient", return_value=service):
                    args = sampler_support.config(
                        kind, steps=2, every=1, state_dir=directory
                    )
                    trainer = sampler_support.trainer_for(
                        kind, args, sampler_support.rows(kind, 2)
                    )
                    state = await sampler_support.train(trainer)
                    path = Path(state.checkpoint_manifest_path)
                    raw = json.loads(path.read_text())
                    self.assertNotIn("sampler_checkpoint_path", raw["state"])
                    self.assertEqual(
                        raw["components"]["actor"]["sampler_checkpoint"],
                        dict(
                            uri=state.sampler_checkpoint_path, step=2, ttl_seconds=86400
                        ),
                    )
                    resumed = sampler_support.trainer_for(
                        kind,
                        sampler_support.config(
                            kind,
                            steps=2,
                            every=1,
                            state_dir=directory,
                            resume_from=str(path),
                        ),
                        sampler_support.rows(kind, 2),
                    )
                    if kind in sampler_support.PRESETS:
                        await resumed.initialize()
                    self.assertEqual(resumed._sampler_checkpoint.saved_step, 2)
                    self.assertEqual(
                        await resumed.save_sampler_checkpoint_async(),
                        state.sampler_checkpoint_path,
                    )
                    self.assertEqual([item["step"] for item in service.exports], [1, 2])
                    if kind in sampler_support.PRESETS:
                        await resumed.close()
                    elif kind == "opd":
                        await resumed.close()
                    else:
                        resumed._metrics_api.finish()

    async def test_corrupt_sampler_metadata_fails_before_remote_client_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            service = sampler_support.ExportService()
            with patch("pytrio.ServiceClient", return_value=service):
                trainer = sampler_support.trainer_for(
                    "sft",
                    sampler_support.config("sft", steps=1, state_dir=directory),
                    sampler_support.rows("sft", 1),
                )
                state = await sampler_support.train(trainer)
            path = Path(state.checkpoint_manifest_path)
            payload = json.loads(path.read_text())
            for updates in (
                {"step": True},
                {"step": 2},
                {"uri": ""},
                {"ttl_seconds": 0},
            ):
                damaged = copy.deepcopy(payload)
                damaged["components"]["actor"]["sampler_checkpoint"].update(updates)
                path.write_text(json.dumps(damaged))
                with (
                    self.subTest(updates=updates),
                    self.assertRaisesRegex(ValueError, "sampler export metadata"),
                ):
                    load_training_checkpoint(path)


class RealReportingTests(unittest.TestCase):
    def test_webshop_entries_select_root_and_reject_a_preloaded_copy(self):
        example = ROOT / "examples/agent/webshop-opd"
        if not example.exists():
            self.skipTest("WebShop application is not present in this checkout.")
        scripts = [
            *example.joinpath("train").glob("*.py"),
            example / "eval/beforetraining_eval.py",
        ]
        scripts = [path for path in scripts if "sys.path.insert" in path.read_text()]
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts:
                code = (
                    "import runpy, pathlib\n"
                    f"runpy.run_path({str(script)!r})\n"
                    "import triotune\n"
                    f"assert pathlib.Path(triotune.__file__).resolve().parent == pathlib.Path({str(ROOT / 'triotune')!r})\n"
                )
                result = subprocess.run(
                    [sys.executable, "-P", "-c", code],
                    cwd=directory,
                    env={**os.environ, "PYTHONPATH": str(example)},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, str(script) + result.stderr)
            code = (
                "import triotune, runpy\n"
                f"runpy.run_path({str(example / 'train/vanilla_opd.py')!r})\n"
            )
            result = subprocess.run(
                [sys.executable, "-P", "-c", code],
                cwd=directory,
                env={**os.environ, "PYTHONPATH": str(example)},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Start a fresh Python process", result.stderr)

    def test_swanlab_sdk_records_training_eval_sampler_and_terminal_after_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tests/audit_training_reporting.py"),
                    "--output",
                    directory,
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "SWANLAB_MODE": "offline",
                    "PYTHONPATH": f"{ROOT}:{ROOT}/tests",
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = next(
                line.removeprefix("REPORTING_AUDIT=")
                for line in result.stdout.splitlines()
                if line.startswith("REPORTING_AUDIT=")
            )
            self.assertEqual(len(json.loads(summary)), 8)
            terminal = result.stdout + result.stderr
            for text in (
                "SFT train",
                "DPO train",
                "OPD train",
                "PPO train",
                "Agent PPO train",
                "Agent OPD train",
                "loss=",
            ):
                self.assertIn(text, terminal)
