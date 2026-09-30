from __future__ import annotations

import unittest

from trio_cookbook.trainer import GRPOConfig, GRPOTrainer, TrainerState


class _Tokenizer:
    eos_token = "<|endoftext|>"

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

    def test_default_messages_accepts_text_tokens_and_chat(self) -> None:
        rows = [
            {"messages": "question"},
            {"messages": [1, 2]},
            {"messages": [{"role": "user", "content": "question"}]},
        ]

        groups = self.trainer._groups(rows)

        self.assertEqual([group.prompt_tokens for group in groups], [[8], [1, 2], [3, 4]])
        self.assertEqual(
            [group.prompt for group in groups],
            ["question", "decoded tokens", "chat template"],
        )

    def test_default_rejects_old_prompt_column(self) -> None:
        with self.assertRaisesRegex(KeyError, "messages"):
            self.trainer._groups([{"prompt": "question"}])

    def test_custom_prompt_key_reads_old_column(self) -> None:
        self.trainer.config = GRPOConfig(prompt_key="prompt")

        groups = self.trainer._groups([{"prompt": "question"}])

        self.assertEqual(groups[0].prompt, "question")

    def test_stop_sequences_follow_config(self) -> None:
        self.assertEqual(self.trainer._stop_sequences(), ["<|endoftext|>", "<|im_end|>"])

        self.trainer.config = GRPOConfig(stop=False)
        self.assertIsNone(self.trainer._stop_sequences())


if __name__ == "__main__":
    unittest.main()
