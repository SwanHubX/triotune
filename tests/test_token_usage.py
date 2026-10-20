import unittest
from types import SimpleNamespace

from triotune.utils.token_usage import (
    SampleTokens,
    compute_sample_tokens,
    compute_train_tokens,
)


class TokenUsageTests(unittest.TestCase):
    def test_compute_train_tokens_counts_model_inputs(self) -> None:
        datums = [
            SimpleNamespace(model_input=SimpleNamespace(length=3)),
            SimpleNamespace(model_input=SimpleNamespace(length=5)),
        ]

        self.assertEqual(compute_train_tokens(datums), 8)

    def test_compute_train_tokens_handles_empty_and_rejects_invalid_lengths(self) -> None:
        self.assertEqual(compute_train_tokens([]), 0)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            compute_train_tokens(
                [SimpleNamespace(model_input=SimpleNamespace(length=-1))]
            )

    def test_compute_sample_tokens_prefers_sdk_usage(self) -> None:
        response = SimpleNamespace(
            input_tokens=7,
            output_tokens=9,
            sequences=[SimpleNamespace(tokens=[1])],
        )

        self.assertEqual(
            compute_sample_tokens(response, prompt_tokens=3),
            SampleTokens(input_tokens=7, output_tokens=9),
        )

    def test_compute_sample_tokens_supports_compatible_response(self) -> None:
        response = {
            "sequences": [
                {"tokens": [1, 2]},
                {"completion_tokens": [3, 4, 5]},
            ]
        }

        self.assertEqual(
            compute_sample_tokens(response, prompt_tokens=4),
            SampleTokens(input_tokens=4, output_tokens=5),
        )

    def test_compute_sample_tokens_requires_input_fallback(self) -> None:
        response = SimpleNamespace(sequences=[])

        with self.assertRaisesRegex(ValueError, "prompt_tokens is required"):
            compute_sample_tokens(response)

    def test_compute_sample_tokens_accepts_empty_generation(self) -> None:
        response = SimpleNamespace(input_tokens=3, output_tokens=0, sequences=[])

        self.assertEqual(
            compute_sample_tokens(response),
            SampleTokens(input_tokens=3, output_tokens=0),
        )


if __name__ == "__main__":
    unittest.main()
