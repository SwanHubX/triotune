"""Compatibility CLI for the public calculator example.

Prefer examples/agent/calculator/train.py for new experiments. Legacy
--probe-only, --model, --steps, --trace and --template-kwargs remain accepted.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.agent.calculator.train import main as calculator_main


def main():
    argv = sys.argv[1:]
    probe = "--probe-only" in argv
    argv = [value for value in argv if value != "--probe-only"]
    if not any(value == "--mode" or value.startswith("--mode=") for value in argv):
        argv.extend(["--mode", "probe" if probe else "train"])
    calculator_main(argv)


if __name__ == "__main__":
    main()
