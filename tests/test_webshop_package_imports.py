"""Keep WebShop imports independent of removed copies and installed versions."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/agent/webshop-opd"


@unittest.skipUnless(EXAMPLE.is_dir(), "WebShop application is not present.")
class WebShopPackageImportTests(unittest.TestCase):
    def run_python(self, code, directory):
        return subprocess.run(
            [sys.executable, "-P", "-c", code],
            cwd=directory,
            env={**os.environ, "PYTHONPATH": f"{directory}:{EXAMPLE}"},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_direct_application_imports_prefer_repository_over_an_old_install(self):
        modules = (
            "webshop_env.environment",
            "webshop_env.data_prepare",
            "webshop_env.metrics",
            "train",
            "train.vanilla_opd",
            "train.grpo_teacher",
            "train.resume_grpo_teacher",
            "train.resume_grpo_teacher_native",
            "train.tcod_f2b",
            "train.tcod_b2f",
            "train.collect_tcod_b2f_experts",
            "train.train_tcod_b2f",
        )
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "triotune"
            old.mkdir()
            (old / "__init__.py").write_text(
                "raise AssertionError('Old Triotune installation selected.')\n"
            )
            # Run train and environment first in separate processes: neither
            # may rely on another entry point having already selected the root.
            for first in ("webshop_env.environment", "train"):
                code = (
                    "import importlib, pathlib, sys\n"
                    f"importlib.import_module({first!r})\n"
                    f"for name in {modules!r}: importlib.import_module(name)\n"
                    "import triotune\n"
                    f"expected = pathlib.Path({str(ROOT / 'triotune')!r}).resolve()\n"
                    "assert pathlib.Path(triotune.__file__).resolve().parent == expected\n"
                    "for name, module in tuple(sys.modules.items()):\n"
                    "    if name.startswith('triotune.'):\n"
                    "        assert pathlib.Path(module.__file__).resolve().is_relative_to(expected), name\n"
                )
                result = self.run_python(code, directory)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_preloaded_copy_missing_origin_and_mixed_submodules_require_restart(self):
        scenarios = (
            "old = types.ModuleType('triotune'); old.__file__ = '/removed-copy/triotune/__init__.py'; sys.modules['triotune'] = old",
            "old = types.ModuleType('triotune'); sys.modules['triotune'] = old",
            f"sys.path.insert(0, {str(ROOT)!r}); import triotune; old = types.ModuleType('triotune.agentrl'); old.__file__ = '/removed-copy/triotune/agentrl/__init__.py'; sys.modules['triotune.agentrl'] = old",
        )
        with tempfile.TemporaryDirectory() as directory:
            for scenario in scenarios:
                with self.subTest(scenario=scenario):
                    code = (
                        "import sys, types\n"
                        + scenario + "\n"
                        + "import webshop_env.environment\n"
                    )
                    result = self.run_python(code, directory)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("RuntimeError: WebShop requires the repository", result.stderr)
                    self.assertIn("Start a fresh Python process", result.stderr)
                    self.assertNotIn("ImportError:", result.stderr)


if __name__ == "__main__":
    unittest.main()
