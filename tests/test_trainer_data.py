"""Algorithm data preparation can run independently of remote trainers."""

import unittest

from trio_cookbook.trainer.dpo.config import DPOConfig
from trio_cookbook.trainer.dpo.data import prepare_preferences
from trio_cookbook.trainer.sft.config import SFTConfig
from trio_cookbook.trainer.sft.data import tokens_and_labels


class Tokenizer:
    eos_token_id = 255

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class DPODataTests(unittest.TestCase):
    def test_preference_order_eos_and_exact_length_boundary(self):
        pairs = prepare_preferences(
            [{"messages": "PQ", "chosen": "A", "rejected": [66, 255]}],
            Tokenizer(), DPOConfig(max_length=4),
        )
        self.assertEqual(pairs[0].prompt, [80, 81])
        self.assertEqual(pairs[0].chosen, [65, 255])
        self.assertEqual(pairs[0].rejected, [66, 255])

    def test_empty_missing_and_overlength_preferences_fail(self):
        for rows, error in (
            ([], ValueError),
            ([{"messages": "P", "chosen": "A"}], KeyError),
            ([{"messages": "P", "chosen": "", "rejected": "B"}], ValueError),
            ([{"messages": "PP", "chosen": "AA", "rejected": "B"}], ValueError),
        ):
            with self.subTest(rows=rows), self.assertRaises(error):
                prepare_preferences(rows, Tokenizer(), DPOConfig(max_length=3))


class SFTDataTests(unittest.TestCase):
    def test_completion_and_pretokenized_masks(self):
        tokens, labels = tokens_and_labels(
            {"prompt": "PQ", "completion": "A"}, Tokenizer(), SFTConfig(),
        )
        self.assertEqual(tokens, [80, 81, 65, 255])
        self.assertEqual(labels, [-100, -100, 65, 255])
        tokens, labels = tokens_and_labels(
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            Tokenizer(), SFTConfig(max_length=2),
        )
        self.assertEqual((tokens, labels), ([1, 2], [-100, 2]))

    def test_invalid_or_fully_masked_next_token_labels_fail(self):
        for row in (
            {"input_ids": [1, 2], "labels": [-100, -100]},
            {"input_ids": [1, 2], "labels": [1]},
            {"input_ids": [1, -1]},
            {"input_ids": [1, 2], "labels": [1, -1]},
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                tokens_and_labels(row, Tokenizer(), SFTConfig())

    def test_truncation_cannot_remove_all_completion_labels(self):
        with self.assertRaisesRegex(ValueError, "no next-token labels"):
            tokens_and_labels(
                {"prompt": "PQ", "completion": "A"}, Tokenizer(), SFTConfig(max_length=2),
            )
