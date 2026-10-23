"""Opt-in integration checks use a cached real tokenizer, never model weights.

TRIOTUNE_TEST_TOKENIZER=Qwen/Qwen3.5-4B PYTHONPATH=tests \
  .venv/bin/python -m unittest -v test_agent_tokenizer_integration
"""

import os
import unittest

from triotune.agentrl import (
    AgentContractError,
    AgentTurn,
    ChatTemplateRenderer,
    EpisodeResult,
    EpisodeTrajectoryBuilder,
    Generation,
    TokenAppendRenderer,
    probe_tokenizer,
)


@unittest.skipUnless(
    os.environ.get("TRIOTUNE_TEST_TOKENIZER"),
    "Set TRIOTUNE_TEST_TOKENIZER to a cached tokenizer path or ID",
)
class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["TRIOTUNE_TEST_TOKENIZER"], local_files_only=True
        )

    def test_eos_utf8_json_process_rewards_and_causal_mask(self):
        report = probe_tokenizer(self.tokenizer)
        self.assertEqual(len(report["cases"]), 2)
        self.assertTrue(
            all(
                case["prefix_preserved"] and case["causal_mask_aligned"]
                for case in report["cases"]
            )
        )

    def test_noncanonical_sample_ids_survive_history_rewriting(self):
        tokenizer = self.tokenizer
        messages = [{"role": "user", "content": "Calculate 2+3"}]
        renderer = TokenAppendRenderer(
            tokenizer, advertise_tools=False, enable_thinking=False
        )
        prompt = renderer.render(messages, [])
        text = '{"tool_calls":[{"name":"calculate","arguments":{"operation":"add","a":2,"b":3}}]}'
        # Sampling can produce a different valid segmentation than re-encoding
        # the same decoded string. This used to lose behavior-policy alignment.
        ids = [
            token
            for character in text
            for token in tokenizer.encode(character, add_special_tokens=False)
        ] + [tokenizer.eos_token_id]
        self.assertNotEqual(ids[:-1], tokenizer.encode(text, add_special_tokens=False))
        episode = EpisodeResult(
            "e",
            "g",
            "s",
            turns=[
                AgentTurn(
                    tuple(prompt),
                    Generation(tuple(ids), tuple([-1.0] * len(ids)), text),
                )
            ],
        )
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "5"},
        ]
        strict = ChatTemplateRenderer(
            tokenizer, advertise_tools=False, enable_thinking=False
        )
        with self.assertRaises(AgentContractError):
            EpisodeTrajectoryBuilder().validate_prefix(
                episode, strict.render(messages, [])
            )
        next_prompt = renderer.render_continuation(episode, messages, [])
        self.assertEqual(next_prompt[: len(prompt) + len(ids)], prompt + ids)
        EpisodeTrajectoryBuilder().validate_prefix(episode, next_prompt)
        suffix = tokenizer.decode(next_prompt[len(prompt) + len(ids) :])
        self.assertFalse(suffix.startswith(tokenizer.eos_token))

    def test_returned_eos_plus_newline_is_not_duplicated(self):
        tokenizer = self.tokenizer
        renderer = TokenAppendRenderer(
            tokenizer, advertise_tools=False, enable_thinking=False
        )
        messages = [{"role": "user", "content": "hello"}]
        prompt = renderer.render(messages, [])
        output = tokenizer.encode(
            '{"final":"5"}' + tokenizer.eos_token + "\n", add_special_tokens=False
        )
        episode = EpisodeResult(
            "e",
            "g",
            "s",
            turns=[
                AgentTurn(
                    tuple(prompt),
                    Generation(
                        tuple(output), tuple([-1.0] * len(output)), '{"final":"5"}'
                    ),
                )
            ],
        )
        next_prompt = renderer.render_continuation(
            episode,
            messages
            + [
                {"role": "assistant", "content": '{"final":"5"}'},
                {"role": "user", "content": "again"},
            ],
            [],
        )
        suffix = tokenizer.decode(next_prompt[len(prompt) + len(output) :])
        self.assertTrue(suffix.startswith("<|im_start|>user"))
