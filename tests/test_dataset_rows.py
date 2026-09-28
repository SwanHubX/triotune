from __future__ import annotations

import unittest

from trio_cookbook.trainer import GRPOConfig, GRPOTrainer, TrainerState


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [len(text)]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        return "decoded tokens"

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str | list[int]:
        return [3, 4] if tokenize else "chat template"


class DatasetRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = GRPOTrainer.__new__(GRPOTrainer)
        self.trainer.config = GRPOConfig()
        self.trainer.state = TrainerState()
        self.trainer._tokenizer = _Tokenizer()

    def test_prompt_column_accepts_text_tokens_and_chat(self) -> None:
        rows = [
            {"prompt": "question"},
            {"prompt": [1, 2]},
            {"prompt": [{"role": "user", "content": "question"}]},
        ]

        groups = self.trainer._groups(rows)

        self.assertEqual([group.prompt_tokens for group in groups], [[8], [1, 2], [3, 4]])
        self.assertEqual(
            [group.prompt for group in groups],
            ["question", "decoded tokens", "chat template"],
        )

    def test_missing_prompt_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "prompt"):
            self.trainer._groups([{"messages": "question"}])


if __name__ == "__main__":
    unittest.main()
