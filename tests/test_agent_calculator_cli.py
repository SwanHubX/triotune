"""Public/legacy entry points fail clearly before unnecessary remote calls."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from recipes import train_calculator_agent as legacy

from examples.agent.calculator import train
from triotune import PPOConfig
from triotune.agentrl import AgentConfig, AgentContractError, CostConfig

ROOT = Path(__file__).resolve().parents[1]


def arguments(**overrides):
    return SimpleNamespace(
        **{
            "config": ROOT / "examples/agent/calculator/config.yaml",
            "model": None,
            "steps": None,
            "mode": "train",
            "checkpoint": None,
            **overrides,
        }
    )


class CalculatorCLITests(unittest.IsolatedAsyncioTestCase):
    async def test_insufficient_rows_fail_before_creating_service_client(self):
        with (
            patch.object(train.pytrio, "ServiceClient") as service,
            self.assertRaisesRegex(ValueError, "train_tasks"),
        ):
            await train.run(arguments(steps=1000))
        service.assert_not_called()

    async def test_failure_report_and_original_error_survive_reporter_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = PPOConfig.for_algorithm("grpo", model={"name": "fake"})
            failure = AgentContractError("snapshot unavailable")
            trainer = SimpleNamespace(
                create_sampling_snapshot=AsyncMock(side_effect=failure),
                close=AsyncMock(side_effect=RuntimeError("cleanup error")),
            )
            settings = (config, AgentConfig(), CostConfig(), train.DataConfig(), output)
            with (
                patch.object(train, "load_experiment", return_value=settings),
                patch.object(train.pytrio, "ServiceClient", return_value=object()),
                patch.object(train, "PPOTrainer", return_value=trainer),
                self.assertRaisesRegex(AgentContractError, "snapshot unavailable"),
            ):
                await train.run(arguments(mode="probe"))
            report = json.loads((output / "probe-report.json").read_text())
            self.assertEqual(report["error"], "snapshot unavailable")
            self.assertTrue(any("cleanup error" in note for note in failure.__notes__))
            trainer.close.assert_awaited_once()

    def test_checkpoint_rejected_outside_evaluation(self):
        message = io.StringIO()
        with (
            patch.object(train.asyncio, "run") as run,
            self.assertRaises(SystemExit) as error,
            redirect_stderr(message),
        ):
            train.main(["--mode", "train", "--checkpoint", "trio://example"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("only in evaluate mode", message.getvalue())
        run.assert_not_called()

    def test_legacy_probe_delegates_to_public_entry_point(self):
        with (
            patch("sys.argv", ["legacy", "--model", "MODEL", "--probe-only"]),
            patch.object(legacy, "calculator_main") as main,
        ):
            legacy.main()
        main.assert_called_once_with(["--model", "MODEL", "--mode", "probe"])

    def test_legacy_script_is_directly_executable_from_repository_root(self):
        result = subprocess.run(
            [sys.executable, "tests/recipes/train_calculator_agent.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        self.assertIn("--mode", result.stdout)

    def test_unknown_experiment_sections_fail_before_training(self):
        with (
            patch.object(train, "load_yaml_config", return_value={"typo": {}}),
            self.assertRaisesRegex(ValueError, "Unknown experiment"),
        ):
            train.load_experiment(Path("unused.yaml"))
