"""Unit coverage for the algorithm-neutral dataset preprocessing contract."""

from __future__ import annotations

import unittest

from trio_cookbook.trainer.dataset import (
    PreferenceExample,
    PromptExample,
    SupervisedExample,
    prepare_preference,
    prepare_prompt,
    prepare_rows,
    prepare_sft,
)


class Tokenizer:
    eos_token_id = 255

    def encode(self, text: str, **_: object) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], **_: object) -> str:
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **_: object,
    ) -> object:
        segments = [f"{message['role']}:{message['content']};" for message in messages]
        text = "".join(segments) + ("assistant:" if add_generation_prompt else "")
        if not tokenize:
            return text
        input_ids = self.encode(text)
        if not return_dict:
            return input_ids
        mask = [
            bit
            for message, segment in zip(messages, segments, strict=True)
            for bit in [int(message["role"] == "assistant")] * len(segment)
        ]
        if add_generation_prompt:
            mask.extend([0] * len("assistant:"))
        return {"input_ids": input_ids, "assistant_masks": mask}


class PromptProcessingTests(unittest.TestCase):
    def test_text_tokens_and_chat_share_one_result_type(self) -> None:
        tokenizer = Tokenizer()
        examples = [
            prepare_prompt({"messages": "Q"}, tokenizer),
            prepare_prompt({"messages": [81]}, tokenizer),
            prepare_prompt(
                {"messages": [{"role": "user", "content": "Q"}]}, tokenizer
            ),
        ]

        self.assertTrue(all(isinstance(example, PromptExample) for example in examples))
        self.assertEqual([example.input_ids for example in examples[:2]], [[81], [81]])
        self.assertEqual(examples[1].text, "Q")
        self.assertTrue(examples[2].text.endswith("assistant:"))

    def test_context_budget_reserves_completion_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_length"):
            prepare_prompt(
                {"messages": "AB"},
                Tokenizer(),
                max_length=2,
                min_completion_tokens=1,
            )


class SFTProcessingTests(unittest.TestCase):
    def test_prompt_completion_builds_mask_and_pytrio_shift(self) -> None:
        example = prepare_sft(
            {"prompt": "PQ", "completion": "A"}, Tokenizer()
        )

        self.assertIsInstance(example, SupervisedExample)
        self.assertEqual(example.input_ids, [80, 81, 65, 255])
        self.assertEqual(example.labels, [-100, -100, 65, 255])
        self.assertEqual(example.loss_mask, [0, 0, 1, 1])
        self.assertEqual(example.model_input, [80, 81, 65])
        self.assertEqual(example.target_tokens, [-100, 65, 255])

    def test_pretokenized_loss_mask_is_normalized_to_labels(self) -> None:
        example = prepare_sft(
            {"input_ids": [10, 11, 12], "loss_mask": [0, 1, 1]}, Tokenizer()
        )

        self.assertEqual(example.labels, [-100, 11, 12])
        self.assertEqual(
            example.as_dict(),
            {
                "input_ids": [10, 11, 12],
                "labels": [-100, 11, 12],
                "loss_mask": [0, 1, 1],
            },
        )
        self.assertEqual(prepare_sft(example.as_dict(), Tokenizer()), example)

    def test_assistant_only_chat_uses_template_mask(self) -> None:
        example = prepare_sft(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            },
            Tokenizer(),
            assistant_only_loss=True,
        )

        self.assertTrue(any(example.loss_mask))
        supervised = "".join(
            chr(token)
            for token, bit in zip(example.input_ids, example.loss_mask, strict=True)
            if bit
        )
        self.assertEqual(supervised, "assistant:A;")

    def test_invalid_or_fully_masked_rows_fail_early(self) -> None:
        rows = (
            {"input_ids": [1, 2], "loss_mask": [0]},
            {"input_ids": [1, 2], "loss_mask": [0, 2]},
            {"input_ids": [1, 2], "loss_mask": [0, 0]},
            {"input_ids": [1, 2], "labels": [-100, 2], "loss_mask": [1, 1]},
            {"input_ids": [1, -1], "labels": [-100, 1]},
            {"input_ids": [1, 2], "labels": [1, -2]},
        )
        for row in rows:
            with self.subTest(row=row), self.assertRaises((TypeError, ValueError)):
                prepare_sft(row, Tokenizer())

    def test_truncation_cannot_remove_all_completion_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "no next-token labels"):
            prepare_sft(
                {"prompt": "PQ", "completion": "A"},
                Tokenizer(),
                max_length=2,
            )


class PreferenceProcessingTests(unittest.TestCase):
    def test_preference_sequences_mask_the_shared_prompt(self) -> None:
        pair = prepare_preference(
            {"messages": "P", "chosen": "A", "rejected": [66, 255]},
            Tokenizer(),
            max_length=3,
        )

        self.assertIsInstance(pair, PreferenceExample)
        self.assertEqual((pair.prompt, pair.chosen, pair.rejected), ([80], [65, 255], [66, 255]))
        self.assertEqual(pair.chosen_loss_mask, [0, 1, 1])
        self.assertEqual(pair.sequence("chosen").target_tokens, [65, 255])

    def test_missing_empty_and_overlong_preferences_fail(self) -> None:
        rows = (
            {"messages": "P", "chosen": "A"},
            {"messages": "P", "chosen": "", "rejected": "B"},
            {"messages": "PP", "chosen": "AA", "rejected": "B"},
        )
        for row in rows:
            with self.subTest(row=row), self.assertRaises((KeyError, ValueError)):
                prepare_preference(row, Tokenizer(), max_length=3)

    def test_batch_dispatch_is_explicit_and_reports_row_index(self) -> None:
        rows = [{"messages": "P", "chosen": "A", "rejected": "B"}]
        prepared = prepare_rows(rows, Tokenizer(), format="preference")
        self.assertEqual(len(prepared), 1)
        with self.assertRaisesRegex(ValueError, "format"):
            prepare_rows(rows, Tokenizer(), format="auto")
        with self.assertRaisesRegex(KeyError, "Dataset row 0"):
            prepare_rows([{"messages": "P", "chosen": "A"}], Tokenizer(), format="preference")


if __name__ == "__main__":
    unittest.main()
