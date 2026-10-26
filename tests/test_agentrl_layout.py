"""Guard agentrl package boundaries, public exports and wheel declarations."""

import ast
import importlib
import tomllib
import unittest
from pathlib import Path

from triotune import agentrl

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "triotune" / "agentrl"


class AgentRLLayoutTests(unittest.TestCase):
    def test_root_contains_only_shared_configuration_contracts_and_exports(self):
        self.assertEqual(
            {path.name for path in PACKAGE.glob("*.py")},
            {"__init__.py", "config.py", "types.py"},
        )
        self.assertEqual(
            {path.parent.name for path in PACKAGE.glob("*/__init__.py")},
            {"runtime", "rollout", "training", "diagnostics"},
        )
        self.assertFalse((ROOT / "triotune" / "agent").exists())

    def test_public_exports_resolve_to_functional_modules(self):
        modules = {
            "config": ("AgentConfig", "AgentEvaluationConfig", "RetryPolicy"),
            "types": (
                "Generation",
                "AgentTurn",
                "EpisodeResult",
                "EpisodeScore",
                "EpisodeLimits",
            ),
            "runtime.conversation": (
                "ChatTemplateRenderer",
                "TokenAppendRenderer",
                "JsonActionParser",
                "JsonObservationFormatter",
            ),
            "runtime.runner": ("EpisodeRunner",),
            "runtime.sampling": ("PyTRIOAgentSampler", "SamplingSnapshot"),
            "runtime.tools": ("FunctionTool", "FunctionToolExecutor"),
            "rollout.trajectory": ("EpisodeTrajectoryBuilder",),
            "rollout.workflow": (
                "AgentRolloutBatch",
                "AgentRolloutWorkflow",
                "EpisodeCollectionError",
            ),
            "training.loop": ("AgentTrainingLoop",),
            "training.rewards": (
                "CompositeScorer",
                "CostConfig",
                "RewardTerm",
                "ToolCostScorer",
                "TurnRewardScorer",
            ),
            "training.evaluation": ("evaluation_metrics", "paired_task_metrics"),
            "diagnostics.logging": ("JsonlEpisodeLogger",),
            "diagnostics.validation": ("probe_tokenizer",),
        }
        self.assertEqual(len(agentrl.__all__), len(set(agentrl.__all__)))
        for suffix, names in modules.items():
            module = importlib.import_module(f"triotune.agentrl.{suffix}")
            for name in names:
                with self.subTest(module=suffix, name=name):
                    self.assertIn(name, agentrl.__all__)
                    self.assertIs(getattr(agentrl, name), getattr(module, name))

    def test_setuptools_includes_every_functional_subpackage(self):
        settings = tomllib.loads((ROOT / "pyproject.toml").read_text())
        declared = set(settings["tool"]["setuptools"]["packages"])
        actual = {
            ".".join(path.parent.relative_to(ROOT).parts)
            for path in PACKAGE.rglob("__init__.py")
        }
        self.assertEqual(
            {
                name
                for name in declared
                if name == "triotune.agentrl" or name.startswith("triotune.agentrl.")
            },
            actual,
        )
        self.assertNotIn("triotune.agent", declared)

    def test_runtime_and_examples_use_the_new_namespace(self):
        for directory in (ROOT / "triotune", ROOT / "examples"):
            for path in directory.rglob("*.py"):
                tree = ast.parse(path.read_text())
                modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        modules.append(node.module)
                    elif isinstance(node, ast.Import):
                        modules.extend(alias.name for alias in node.names)
                for name in modules:
                    with self.subTest(path=path, module=name):
                        self.assertFalse(
                            name == "triotune.agent"
                            or name.startswith("triotune.agent.")
                        )
