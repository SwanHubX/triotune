"""Validate the prepared GSM8K train/test JSONL datasets.

These tests intentionally live outside the trainer tests: they verify the
local tutorial assets and their split isolation, not PPO numerical behavior.
Run them after preparing both files under ``data/``.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "gsm8k_train.jsonl"
TEST_FILE = ROOT / "data" / "gsm8k_test.jsonl"
QUESTION_SUFFIX = "Provide a numerical answer without units, written inside \\boxed{}."
NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read one nonempty JSON object per line with useful source errors."""

    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise AssertionError(f"Blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AssertionError(f"Non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


class GSM8KDataTests(unittest.TestCase):
    """Check official split sizes, trainer fields and prompt consistency."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = [path for path in (TRAIN_FILE, TEST_FILE) if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "Prepare local GSM8K data first: "
                + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            )
        cls.train_rows = load_rows(TRAIN_FILE)
        cls.test_rows = load_rows(TEST_FILE)

    def test_official_split_sizes_and_disjoint_questions(self) -> None:
        self.assertEqual(len(self.train_rows), 7473)
        self.assertEqual(len(self.test_rows), 1319)
        train_questions = {row["question"] for row in self.train_rows}
        test_questions = {row["question"] for row in self.test_rows}
        self.assertEqual(len(train_questions), len(self.train_rows))
        self.assertEqual(len(test_questions), len(self.test_rows))
        self.assertTrue(train_questions.isdisjoint(test_questions))

    def test_rows_follow_the_ppo_reward_contract(self) -> None:
        for split, rows in (("train", self.train_rows), ("test", self.test_rows)):
            for index, row in enumerate(rows):
                with self.subTest(split=split, index=index):
                    self.assertEqual(set(row), {"messages", "answer", "question"})
                    self.assertIsInstance(row["question"], str)
                    self.assertTrue(row["question"].strip())
                    self.assertRegex(row["answer"], NUMBER)
                    messages = row["messages"]
                    self.assertEqual(
                        [message.get("role") for message in messages],
                        ["user", "assistant", "user"],
                    )
                    self.assertIn(r"\boxed{3}", messages[1]["content"])
                    self.assertEqual(
                        messages[-1]["content"],
                        row["question"] + QUESTION_SUFFIX,
                    )


if __name__ == "__main__":
    unittest.main()
