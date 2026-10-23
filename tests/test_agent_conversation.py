"""Exercise append-only serialization in ordinary CI without downloading files."""

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


class TemplateTokenizer:
    """Small template with explicit special IDs and a generated thinking prefix."""

    eos_token_id = 1
    eos_token = "<end>"

    def encode(self, text, **kwargs):
        result = []
        while text:
            if text.startswith(self.eos_token):
                result.append(1)
                text = text[len(self.eos_token) :]
            else:
                result.append(ord(text[0]) + 10)
                text = text[1:]
        return result

    def decode(self, tokens, **kwargs):
        return "".join(self.eos_token if t == 1 else chr(t - 10) for t in tokens)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        text = "".join(f"<{m['role']}>" + m["content"] + "<end>\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant><think></think>"
        if tokenize:
            return {
                "input_ids": self.encode(text),
                "attention_mask": [1] * len(self.encode(text)),
            }
        return text


class ConversationTests(unittest.TestCase):
    def test_mapping_result_and_full_tokenizer_probe(self):
        tokenizer = TemplateTokenizer()
        self.assertTrue(
            ChatTemplateRenderer(tokenizer).render(
                [{"role": "user", "content": "hello"}], []
            )
        )
        report = probe_tokenizer(tokenizer)
        self.assertEqual(len(report["cases"]), 2)
        self.assertTrue(all(case["context_tokens"] > 0 for case in report["cases"]))
        with self.assertRaises(AgentContractError):
            probe_tokenizer(tokenizer, renderer="chat_template")

    def test_append_deduplicates_only_returned_boundary_tokens(self):
        tokenizer = TemplateTokenizer()
        renderer = TokenAppendRenderer(tokenizer)
        messages = [{"role": "user", "content": "question"}]
        prompt = renderer.render(messages, [])
        for ending, expected in (
            ("", "<end>\n<user>"),
            ("<end>", "\n<user>"),
            ("<end>\n", "<user>"),
        ):
            text = '{"final":"5"}'
            ids = tokenizer.encode(text + ending)
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
            following = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "observation"},
            ]
            actual = renderer.render_continuation(episode, following, [])
            self.assertEqual(actual[: len(prompt) + len(ids)], prompt + ids)
            self.assertTrue(
                tokenizer.decode(actual[len(prompt) + len(ids) :]).startswith(expected)
            )
            EpisodeTrajectoryBuilder().validate_prefix(episode, actual)

    def test_unsupported_template_fails_before_sampling_another_turn(self):
        class BadTemplate(TemplateTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                return (
                    text.replace("TRIOTUNE_ASSISTANT_BOUNDARY_7e39d2", "lost")
                    if isinstance(text, str)
                    else text
                )

        tokenizer = BadTemplate()
        renderer = TokenAppendRenderer(tokenizer)
        messages = [{"role": "user", "content": "question"}]
        prompt = renderer.render(messages, [])
        episode = EpisodeResult(
            "e",
            "g",
            "s",
            turns=[AgentTurn(tuple(prompt), Generation((99,), (-1,), "x"))],
        )
        with self.assertRaisesRegex(AgentContractError, "placeholder"):
            renderer.render_continuation(
                episode,
                messages
                + [
                    {"role": "assistant", "content": "x"},
                    {"role": "user", "content": "next"},
                ],
                [],
            )
