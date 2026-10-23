"""Deterministic token-preserving calculator sampling for agent integration tests."""

import json
from types import SimpleNamespace

from test_ppo_family import TrainingClient


class AgentTokenizer:
    """Character tokens plus an EOS ID make template-prefix assertions exact."""

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text]

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(
            ""
            if token == 1 and skip_special_tokens
            else "<eos>"
            if token == 1
            else chr(token - 10)
            for token in tokens
        )

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        tokens = []
        for message in messages:
            tokens.extend(self.encode(f"<{message['role']}>" + message["content"]))
            tokens.append(1)
        if add_generation_prompt:
            tokens.extend(self.encode("<assistant>"))
        return tokens if tokenize else self.decode(tokens)


class CalculatorSampler:
    def __init__(self, tokenizer, trace, version, *, output=None, failure=None):
        self.tokenizer, self.trace, self.version = tokenizer, trace, version
        self.output, self.failure = output, failure
        self.requests = []

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.trace.append(f"agent_sample:{self.version}")
        self.requests.append((prompt.to_ints(), sampling_params))
        if self.failure:
            raise self.failure
        text = self.tokenizer.decode(prompt.to_ints(), skip_special_tokens=True)
        action = self.output
        if action is None:
            if '"tool_results"' in text:
                action = '{"final":"5"}'
            elif sampling_params.seed is not None and (sampling_params.seed // 8) % 2:
                action = '{"final":"0"}'
            else:
                action = json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "calculate",
                                "arguments": {"operation": "add", "a": 2, "b": 3},
                            }
                        ]
                    }
                )
        tokens = self.tokenizer.encode(action) + [1]
        reason = "stop"
        if len(tokens) > sampling_params.max_tokens:
            tokens, reason = tokens[: sampling_params.max_tokens], "length"
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=tokens,
                    logprobs=[-1.0] * len(tokens),
                    text="untrusted response text",
                    stop_reason=reason,
                )
            ],
            input_tokens=len(prompt.to_ints()),
            output_tokens=len(tokens),
        )


class AgentTrainingClient(TrainingClient):
    def get_tokenizer(self):
        return AgentTokenizer()

    async def save_weights_and_get_sampling_client_async(self):
        sampler = CalculatorSampler(self.get_tokenizer(), self.trace, self.version)
        self.snapshots.append(sampler)
        return sampler
