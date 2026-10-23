"""Build in a fresh temporary tree and verify agentrl imports from the wheel.

    .venv/bin/python tests/verify_agentrl_wheel.py --output output/agentrl/dist

This avoids stale build/lib files and makes no model/service calls.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(output: Path) -> Path:
    output = output.resolve()
    with tempfile.TemporaryDirectory(prefix="triotune-agentrl-wheel-") as directory:
        source = Path(directory) / "source"
        source.mkdir()
        for name in ("pyproject.toml", "MANIFEST.in", "README.md"):
            shutil.copy2(ROOT / name, source / name)
        shutil.copytree(
            ROOT / "triotune",
            source / "triotune",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        subprocess.run(
            ["uv", "build", str(source), "--wheel", "--out-dir", str(output)],
            check=True,
            timeout=120,
        )
    wheels = sorted(
        output.glob("triotune-*.whl"), key=lambda path: path.stat().st_mtime
    )
    if not wheels:
        raise RuntimeError("Wheel builder returned no triotune artifact.")
    wheel = wheels[-1]
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "triotune" / "agentrl").rglob("*.py")
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    missing = expected - names
    old = {name for name in names if name.startswith("triotune/agent/")}
    if missing or old:
        raise RuntimeError(
            f"Invalid wheel contents: missing={sorted(missing)}, obsolete={sorted(old)}"
        )
    code = """
import importlib
import sys
sys.path.insert(0, sys.argv[1])
for name in sys.argv[2:]:
    module = importlib.import_module(name)
    assert module.__file__.startswith(sys.argv[1]), (name, module.__file__)
print('All agentrl modules imported from the wheel.')
"""
    modules = sorted(
        path[:-3].replace("/", ".")
        for path in expected
        if not path.endswith("/__init__.py")
    )
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheel), "triotune.agentrl", *modules],
        check=True,
        timeout=30,
    )
    print(f"Verified {len(expected)} agentrl files in {wheel}")
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/agentrl/dist")
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
