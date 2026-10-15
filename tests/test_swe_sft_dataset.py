"""SWE agent trajectories become masked rows consumable by SFTTrainer."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from trio_cookbook.dataset import prepare_sft
from trio_cookbook.dataset.swe_sft import (
    SWESFTConfig,
    prepare_swe_sft_dataset,
)


class Tokenizer:
    """Small chat tokenizer whose characters make mask assertions readable."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, object]], object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        return_assistant_tokens_mask: bool,
        tools: object = None,
        **_: object,
    ) -> dict[str, list[int]]:
        self.calls.append((messages, tools))
        text = ""
        mask: list[int] = []
        if tools:
            prefix = "[tools]|"
            text += prefix
            mask.extend([0] * len(prefix))
        for message in messages:
            content = message.get("content") or ""
            calls = ""
            for call in message.get("tool_calls", []):
                function = call["function"]
                calls += f"<call:{function['name']}:{function['arguments']}>"
            segment = f"[{message['role']}]{content}{calls}|"
            text += segment
            enabled = int(message["role"] == "assistant")
            mask.extend([enabled] * len(segment))
        return {"input_ids": [ord(char) for char in text], "assistant_masks": mask}


def trajectory(*, error: bool = True) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": "Fix the test"},
            {
                "role": "assistant",
                "content": "<thinking>inspect first</thinking>",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"command": "pytest"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "failed output",
                "tool_call_id": "call-1",
                "is_error": error,
            },
            {"role": "assistant", "content": "Apply the fix"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }


def supervised_text(row: dict[str, list[int]]) -> str:
    return "".join(
        chr(token)
        for token, label in zip(row["input_ids"], row["labels"], strict=True)
        if label != -100
    )


class SWESFTPairTests(unittest.TestCase):
    def test_pair_mode_filters_failed_action_and_trains_next_decision(self) -> None:
        tokenizer = Tokenizer()
        dataset = prepare_swe_sft_dataset([trajectory()], tokenizer)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.report.candidates, 2)
        self.assertEqual(dataset.report.error_segments, 1)
        self.assertEqual(supervised_text(dict(dataset[0])), "[assistant]Apply the fix|")
        rendered = "".join(chr(token) for token in dataset[0]["input_ids"])
        self.assertNotIn("inspect first", rendered)
        self.assertIsNotNone(tokenizer.calls[0][1])
        # The generic SFT path accepts the cached row without re-tokenizing it.
        self.assertEqual(prepare_sft(dataset[0], tokenizer).as_dict(), dataset[0])

    def test_disabling_error_filter_yields_each_progressive_pair(self) -> None:
        dataset = prepare_swe_sft_dataset(
            [trajectory()],
            Tokenizer(),
            SWESFTConfig(filter_error_segments=False),
        )

        self.assertEqual(len(dataset), 2)
        self.assertIn("inspect first", supervised_text(dict(dataset[0])))
        self.assertEqual(supervised_text(dict(dataset[1])), "[assistant]Apply the fix|")


class SWESFTTrajectoryTests(unittest.TestCase):
    def test_trajectory_mode_masks_only_the_failed_assistant_segment(self) -> None:
        dataset = prepare_swe_sft_dataset(
            [trajectory()],
            Tokenizer(),
            SWESFTConfig(sample_mode="trajectory"),
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.report.error_segments, 1)
        self.assertEqual(supervised_text(dict(dataset[0])), "[assistant]Apply the fix|")
        rendered = "".join(chr(token) for token in dataset[0]["input_ids"])
        self.assertIn("inspect first", rendered)
        self.assertIn("failed output", rendered)

    def test_nested_jsonl_processes_all_conversations_and_parses_arguments(self) -> None:
        first = trajectory(error=False)
        second = {
            "messages": [
                {"role": "user", "content": "Review"},
                {"role": "assistant", "content": "Looks good"},
            ]
        }
        record = {"conversations": [first, second]}
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = prepare_swe_sft_dataset(
                path,
                tokenizer,
                SWESFTConfig(
                    pre_split=True,
                    parse_tool_call_arguments=True,
                ),
            )

        self.assertEqual(dataset.report.records, 1)
        self.assertEqual(dataset.report.conversations, 2)
        self.assertEqual(len(dataset), 2)
        tool_call = tokenizer.calls[0][0][1]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["arguments"], {"command": "pytest"})


class SWESFTValidationTests(unittest.TestCase):
    def test_template_must_track_each_assistant_turn(self) -> None:
        class BrokenTokenizer(Tokenizer):
            def apply_chat_template(self, *args: object, **kwargs: object) -> object:
                output = super().apply_chat_template(*args, **kwargs)
                output["assistant_masks"] = [0] * len(output["input_ids"])
                return output

        with self.assertRaisesRegex(ValueError, "assistant spans"):
            prepare_swe_sft_dataset([trajectory(error=False)], BrokenTokenizer())

    def test_invalid_config_and_fully_filtered_data_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "pre_split"):
            SWESFTConfig(sample_mode="trajectory", pre_split=True)
        with self.assertRaisesRegex(ValueError, "overlong=1"):
            prepare_swe_sft_dataset(
                [{"messages": [{"role": "assistant", "content": "answer"}]}],
                Tokenizer(),
                SWESFTConfig(max_length=2),
            )


if __name__ == "__main__":
    unittest.main()
