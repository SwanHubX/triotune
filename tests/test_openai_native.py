"""Native Qwen3.5 tools, exact-token continuation, and protocol rejection.

Unit tests use an explicit tokenizer double and a fake PyTRIO sampling client.
Real-tokenizer tests use the locally cached project model without downloading
weights or making model API calls. Set TRIOTUNE_TEST_TOKENIZER to select a
different local copy of that same tokenizer.
"""

import asyncio
import copy
import importlib.util
import json
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

from triotune.agentrl import ExecutionContext, PyTRIOAgentSampler, SamplingSnapshot
from triotune.agentrl.runtime.conversation import NativeToolContinuationRenderer
from triotune.agentrl.sdk.openai import (
    Agent,
    OpenAIClient,
    OpenAIProxyServer,
    OpenAISession,
    PyTRIOBackend,
    Qwen35ToolParser,
    function_tool,
)
from triotune.agentrl.training.rewards import RewardAssignment
from triotune.agentrl.types import (
    ActionParseError,
    AgentContractError,
    EpisodeScore,
    Generation,
)


class NativeTokenizer:
    """Minimal tokenizer with explicit markers and alternative ASCII token IDs."""

    chat_template = "<tool_call><function=name><parameter=arg>"
    markers = ("<|im_start|>", "<|im_end|>")
    eos_token_id = 2

    def encode(self, text, **kwargs):
        result = []
        while text:
            for index, marker in enumerate(self.markers, 1):
                if text.startswith(marker):
                    result.append(index)
                    text = text[len(marker) :]
                    break
            else:
                result.append(ord(text[0]) + 1000)
                text = text[1:]
        return result

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(
            ("" if skip_special_tokens else self.markers[token - 1])
            if token in (1, 2)
            else chr(token - (2000 if token >= 2000 else 1000))
            for token in tokens
        )

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, tools=(), **kwargs
    ):
        text = json.dumps(tools, sort_keys=True) if tools else ""
        for item in messages:
            if item["role"] == "tool":
                text += (
                    "<|im_start|>user\n<tool_response>\n"
                    + item["content"]
                    + "\n</tool_response><|im_end|>\n"
                )
            else:
                text += "<|im_start|>" + item["role"] + "\n"
                if item["role"] == "assistant":
                    text += (
                        "<think>\n"
                        + item.get("reasoning_content", "")
                        + "\n</think>\n\n"
                    )
                text += item["content"] + "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return self.encode(text) if tokenize else text


def xml_call(name="add", a="2", b="3"):
    """Render the Qwen3.5 grammar rather than a JSON response imitation."""
    return (
        f"<tool_call>\n<function={name}>\n<parameter=a>\n{a}\n</parameter>\n"
        f"<parameter=b>\n{b}\n</parameter>\n</function>\n</tool_call>"
    )


@function_tool
def add(a: int, b: int) -> int:
    """Add two integer values."""
    return a + b


class FakeSamplingClient:
    """Return raw sampled IDs and scores through the real call_async path."""

    def __init__(self, tokenizer, outputs, *, noncanonical=False):
        self.tokenizer, self.outputs = tokenizer, iter(outputs)
        self.calls, self.generated = [], []
        self.noncanonical = noncanonical

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.calls.append(list(prompt.to_ints()))
        await asyncio.sleep(0)
        text = next(self.outputs)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.noncanonical:
            if isinstance(self.tokenizer, NativeTokenizer):
                ids = [token + 1000 if token >= 1000 else token for token in ids]
            else:
                # Character-by-character encoding is a valid sampled BPE path
                # that differs from canonical encoding of the same display.
                eos = "<|im_end|>"
                ids = [
                    token
                    for char in text.removesuffix(eos)
                    for token in self.tokenizer.encode(char, add_special_tokens=False)
                ]
                if text.endswith(eos):
                    ids += self.tokenizer.encode(eos, add_special_tokens=False)
        self.generated.append(ids)
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=ids, logprobs=[-0.25] * len(ids), stop_reason="stop"
                )
            ],
            input_tokens=len(prompt.to_ints()),
            output_tokens=len(ids),
        )


class NativeToolsTests(unittest.IsolatedAsyncioTestCase):
    """Exercise both OpenAI protocols without a real model or network."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = NativeTokenizer()

    def setup_backend(self, outputs, **options):
        sampler = FakeSamplingClient(self.tokenizer, outputs, noncanonical=True)
        snapshot = SamplingSnapshot(
            "snapshot",
            PyTRIOAgentSampler(sampler, self.tokenizer),
            self.tokenizer,
            policy_version=3,
        )
        backend = PyTRIOBackend(
            snapshot, model="Qwen/Qwen3.5-4B", max_tokens=4096, **options
        )
        context = ExecutionContext(
            "rollout", "episode", "group", 0, "snapshot", policy_version=3
        )
        return (
            OpenAIClient(backend=backend),
            sampler,
            OpenAISession(context=context, training=True),
        )

    async def test_both_apis_multiple_tools_reasoning_and_noncanonical_prefix(self):
        raw = (
            "Need both sums.\n</think>\n\n"
            + xml_call()
            + "\n"
            + xml_call(a="4", b="5")
            + "<|im_end|>"
        )
        for api in ("chat_completions", "responses"):
            with self.subTest(api=api):
                client, sampler, session = self.setup_backend(
                    [raw, "Done.\n</think>\n\n14<|im_end|>"]
                )
                agent = Agent(
                    name="native",
                    instructions="Use add.",
                    tools=[add],
                    client=client,
                    api=api,
                )
                result = await agent.run(
                    "Compute two sums.", session=session, training=True
                )
                self.assertEqual(result.text, "14")
                first, second = session.interactions
                self.assertEqual(first.tool_results[0]["content"], "5")
                self.assertEqual(first.tool_results[1]["content"], "9")
                self.assertEqual(second.parent_interaction_id, first.interaction_id)
                self.assertEqual(tuple(sampler.generated[0]), first.output_ids)
                self.assertEqual(first.logprobs, (-0.25,) * len(first.output_ids))
                prefix = sampler.calls[0] + sampler.generated[0]
                self.assertEqual(sampler.calls[1][: len(prefix)], prefix)
                self.assertNotEqual(
                    self.tokenizer.encode(raw, add_special_tokens=False),
                    sampler.generated[0],
                )
                evidence = first.response["_triotune_sampling"]
                self.assertEqual(evidence["reasoning"], "Need both sums.")
                self.assertEqual(evidence["raw_text"], raw)
                self.assertIn("14", result.text)
                self.assertEqual(session.status, "completed")

    async def test_illegal_parameters_preserve_tokens_before_raising(self):
        for invalid in (
            xml_call(a="true"),
            xml_call(a="NaN"),
            xml_call(a='{"x":1,"x":2}'),
            xml_call(name="unknown"),
            xml_call() + "suffix",
            xml_call().replace("</parameter>", "", 1),
        ):
            with self.subTest(output=invalid):
                client, sampler, session = self.setup_backend(
                    ["</think>\n" + invalid + "<|im_end|>"]
                )
                with self.assertRaises(ActionParseError):
                    await client.create_chat_completion(
                        messages=[{"role": "user", "content": "go"}],
                        tools=[
                            {
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "parameters": add.parameters,
                                },
                            }
                        ],
                        session=session,
                    )
                self.assertEqual(session.status, "failed")
                self.assertEqual(
                    session.interactions[0].output_ids, tuple(sampler.generated[0])
                )
                self.assertTrue(session.export_trajectory().interactions[0]["logprobs"])

    async def test_changed_history_tool_results_and_schemas_fail_before_sampling(self):
        for alteration in ("history", "call", "result", "tools"):
            client, sampler, session = self.setup_backend(
                ["</think>\n" + xml_call() + "<|im_end|>"]
            )
            tools = [
                {
                    "type": "function",
                    "function": {"name": "add", "parameters": add.parameters},
                }
            ]
            initial = [{"role": "user", "content": "go"}]
            response = await client.create_chat_completion(
                messages=initial, tools=tools, session=session
            )
            first = session.interactions[0]
            session.record_tool_results(
                first,
                [
                    {
                        "call_id": response["choices"][0]["message"]["tool_calls"][0][
                            "id"
                        ],
                        "name": "add",
                        "content": "5",
                    }
                ],
            )
            assistant = copy.deepcopy(response["choices"][0]["message"])
            messages = initial + [
                assistant,
                {
                    "role": "tool",
                    "content": "5",
                    "tool_call_id": assistant["tool_calls"][0]["id"],
                },
            ]
            if alteration == "history":
                messages[0] = {"role": "user", "content": "changed"}
            elif alteration == "call":
                messages[-1]["tool_call_id"] = "forged"
            elif alteration == "result":
                messages[-1]["content"] = "forged"
            else:
                tools = copy.deepcopy(tools)
                tools[0]["function"]["description"] = "changed"
            with (
                self.subTest(alteration=alteration),
                self.assertRaises(AgentContractError),
            ):
                await client.create_chat_completion(
                    messages=messages,
                    tools=tools,
                    session=session,
                    parent_interaction_id=first.interaction_id,
                )
            self.assertEqual(len(sampler.calls), 1)

    async def test_unsupported_choice_messages_format_and_override_are_rejected(self):
        for options in (
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "function": {"name": "add"}}},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {}}],
                    }
                ]
            },
            {"_triotune_input_ids": [4]},
            {"logit_bias": {"1": 1}},
        ):
            client, sampler, session = self.setup_backend([])
            with self.subTest(options=options), self.assertRaises(AgentContractError):
                await client.create_chat_completion(
                    session=session,
                    **{"messages": [{"role": "user", "content": "go"}], **options},
                )
            self.assertEqual(sampler.calls, [])
        client, _, _ = self.setup_backend([])
        with self.assertRaises(AgentContractError):
            PyTRIOBackend(
                client.backend.snapshot, model="other", model_format="unsupported"
            )

    def test_parser_end_markers_and_open_thinking_prefix(self):
        parser = Qwen35ToolParser(self.tokenizer)
        prefix = self.tokenizer.encode("<think>\n")
        for text, valid in (
            ("reason</think>\nanswer<|im_end|>\n", True),
            ("reason</think>\nanswer<|im_end|>bad", False),
            ("reason</think>\nanswer<|im_end|><|im_end|>", False),
            ("unclosed", False),
            ("reason</think>\n<|im_start|>user", False),
        ):
            ids = tuple(self.tokenizer.encode(text))
            generation = Generation(ids, (-0.2,) * len(ids), "untrusted display")
            if valid:
                parsed = parser.parse_generation(
                    generation, tools=[], prompt_tokens=prefix
                )
                self.assertEqual(parsed.action.final_text, "answer")
                self.assertEqual(parsed.reasoning, "reason")
                self.assertIs(parsed.generation, generation)
            else:
                with self.assertRaises(ActionParseError):
                    parser.parse_generation(generation, tools=[], prompt_tokens=prefix)

    def test_schema_strings_unions_enums_and_nested_values(self):
        """Exercise schema-dependent XML values rather than string coercion."""
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "choice": {"enum": ["first", "second"]},
                "data": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["text", "choice", "data"],
            "additionalProperties": False,
        }
        raw = (
            "</think>\n<tool_call>\n<function=typed>\n"
            + "".join(
                f"<parameter={key}>\n{value}\n</parameter>\n"
                for key, value in {
                    "text": "literal text",
                    "optional": "not quoted",
                    "choice": "first",
                    "data": "[1,2]",
                }.items()
            )
            + "</function>\n</tool_call><|im_end|>"
        )
        tokens = tuple(self.tokenizer.encode(raw, add_special_tokens=False))
        parsed = Qwen35ToolParser(self.tokenizer).parse_generation(
            Generation(tokens, (-0.2,) * len(tokens), "untrusted"),
            tools=[
                {
                    "type": "function",
                    "function": {"name": "typed", "parameters": schema},
                }
            ],
            prompt_tokens=self.tokenizer.encode("<think>\n"),
        )
        self.assertEqual(
            parsed.action.tool_calls[0].arguments,
            {
                "text": "literal text",
                "optional": "not quoted",
                "choice": "first",
                "data": [1, 2],
            },
        )
        for bad in (
            {"type": ["string", "null"]},
            {"type": "object", "properties": []},
            {"type": "string", "pattern": ".*"},
            {"anyOf": []},
            {"type": "object", "required": ["missing"]},
        ):
            with self.subTest(schema=bad), self.assertRaises(AgentContractError):
                Qwen35ToolParser.check_schema(bad)

    async def test_none_choice_and_unknown_model_tools_are_explicit(self):
        client, _, session = self.setup_backend(["</think>\nanswer<|im_end|>"])
        response = await client.create_chat_completion(
            messages=[{"role": "user", "content": "go"}],
            tool_choice="none",
            session=session,
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "answer")
        client, sampler, session = self.setup_backend(
            ["</think>\n" + xml_call() + "<|im_end|>"]
        )
        with self.assertRaises(ActionParseError):
            await client.create_chat_completion(
                messages=[{"role": "user", "content": "go"}],
                tool_choice="none",
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "add", "parameters": add.parameters},
                    }
                ],
                session=session,
            )
        self.assertEqual(len(sampler.calls), 1)
        client, sampler, session = self.setup_backend([])
        adapter = OpenAIClient(
            backend=PyTRIOBackend(client.backend.snapshot, model="unknown-format")
        )
        with self.assertRaises(AgentContractError):
            await adapter.create_chat_completion(
                messages=[{"role": "user", "content": "go"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "add", "parameters": add.parameters},
                    }
                ],
                session=session,
            )
        self.assertEqual(sampler.calls, [])

    async def test_repeated_tool_turns_keep_full_prefix_and_unique_call_ids(self):
        client, sampler, session = self.setup_backend(
            [
                "</think>\n" + xml_call() + "<|im_end|>",
                "</think>\n" + xml_call(a="8", b="9") + "<|im_end|>",
                "</think>\n17<|im_end|>",
            ]
        )
        result = await Agent(
            name="repeat", instructions="Use add", tools=[add], client=client
        ).run("go", session=session, training=True)
        self.assertEqual(result.text, "17")
        self.assertNotEqual(
            session.interactions[0].tool_calls[0]["id"],
            session.interactions[1].tool_calls[0]["id"],
        )
        for index in (1, 2):
            expected = sampler.calls[index - 1] + sampler.generated[index - 1]
            self.assertEqual(sampler.calls[index][: len(expected)], expected)

    @unittest.skipUnless(
        importlib.util.find_spec("openai") and importlib.util.find_spec("fastapi"),
        "OpenAI proxy extras are unavailable.",
    )
    async def test_official_sdk_models_roundtrip_native_calls_over_proxy(self):
        """Use real SDK response models and ASGI transport, not only dictionaries."""
        import httpx
        from openai import AsyncOpenAI

        for responses in (False, True):
            adapter, sampler, session = self.setup_backend(
                ["</think>\n" + xml_call() + "<|im_end|>", "</think>\n5<|im_end|>"]
            )
            proxy = OpenAIProxyServer(
                adapter, session_factory=lambda session_id, bound=session: bound
            )
            async with AsyncOpenAI(
                api_key="offline-test",
                base_url="http://proxy/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=proxy.create_app())
                ),
            ) as sdk:
                tools = (
                    [{"type": "function", "name": "add", "parameters": add.parameters}]
                    if responses
                    else [
                        {
                            "type": "function",
                            "function": {"name": "add", "parameters": add.parameters},
                        }
                    ]
                )
                if responses:
                    initial = [{"role": "user", "content": "go"}]
                    first = await sdk.responses.create(
                        model="Qwen/Qwen3.5-4B", input=initial, tools=tools
                    )
                    call = next(
                        item for item in first.output if item.type == "function_call"
                    )
                    result = await sdk.responses.create(
                        model="Qwen/Qwen3.5-4B",
                        input=initial
                        + [item.model_dump(exclude_none=True) for item in first.output]
                        + [
                            {
                                "type": "function_call_output",
                                "call_id": call.call_id,
                                "output": "5",
                            }
                        ],
                        tools=tools,
                        extra_headers={"X-Triotune-Session-Id": session.session_id},
                    )
                    self.assertEqual(result.output_text, "5")
                else:
                    initial = [{"role": "user", "content": "go"}]
                    first = await sdk.chat.completions.create(
                        model="Qwen/Qwen3.5-4B", messages=initial, tools=tools
                    )
                    call = first.choices[0].message.tool_calls[0]
                    result = await sdk.chat.completions.create(
                        model="Qwen/Qwen3.5-4B",
                        messages=initial
                        + [first.choices[0].message.model_dump()]
                        + [{"role": "tool", "tool_call_id": call.id, "content": "5"}],
                        tools=tools,
                        extra_headers={"X-Triotune-Session-Id": session.session_id},
                    )
                    self.assertEqual(result.choices[0].message.content, "5")
            self.assertEqual(len(sampler.calls), 2)
            prefix = sampler.calls[0] + sampler.generated[0]
            self.assertEqual(sampler.calls[1][: len(prefix)], prefix)

    def test_incompatible_template_boundary_is_rejected(self):
        tokenizer = NativeTokenizer()
        original = tokenizer.apply_chat_template

        def broken(messages, **kwargs):
            value = original(messages, **kwargs)
            return (
                value.replace("TRIOTUNE_NATIVE_ACTION_BOUNDARY_12ac69", "removed")
                if isinstance(value, str)
                else value
            )

        tokenizer.apply_chat_template = broken
        with self.assertRaises(AgentContractError):
            NativeToolContinuationRenderer(tokenizer).render_after_tool(
                [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "content": "5"},
                ],
                [],
                assistant_index=1,
                prompt_tokens=[5],
                output_tokens=[6],
            )

    async def test_same_label_sampler_switches_cannot_change_a_session_policy(self):
        """Opaque snapshot labels alone cannot prove the same sampling source."""
        for swap_client in (False, True):
            client, original, session = self.setup_backend(
                ["</think>\n" + xml_call() + "<|im_end|>"]
            )
            tools = [
                {
                    "type": "function",
                    "function": {"name": "add", "parameters": add.parameters},
                }
            ]
            initial = [{"role": "user", "content": "go"}]
            first = await client.create_chat_completion(
                messages=initial, tools=tools, session=session
            )
            message = first["choices"][0]["message"]
            replacement = FakeSamplingClient(self.tokenizer, ["</think>\n5<|im_end|>"])
            if swap_client:
                client.backend.snapshot.sampler.client = replacement
            else:
                snapshot = replace(
                    client.backend.snapshot,
                    sampler=PyTRIOAgentSampler(replacement, self.tokenizer),
                )
                client.backend = PyTRIOBackend(snapshot, model="Qwen/Qwen3.5-4B")
            with (
                self.subTest(swap_client=swap_client),
                self.assertRaises(AgentContractError),
            ):
                await client.create_chat_completion(
                    messages=initial
                    + [
                        message,
                        {
                            "role": "tool",
                            "tool_call_id": message["tool_calls"][0]["id"],
                            "content": "5",
                        },
                    ],
                    tools=tools,
                    session=session,
                )
            self.assertEqual(len(original.calls), 1)
            self.assertEqual(replacement.calls, [])

    async def test_exact_ids_bound_delayed_token_rewards_and_process_vectors(self):
        """P1 reward positions must refer to the P2 backend's original output."""
        client, _, session = self.setup_backend(["</think>\n5<|im_end|>"])
        await client.create_chat_completion(
            messages=[{"role": "user", "content": "go"}], session=session
        )
        session.finish()
        count = len(session.interactions[0].output_ids)
        for reward in (
            RewardAssignment("token", 1, position=(0, count)),
            EpisodeScore(1, True, process_rewards={0: (1,)}),
        ):
            with self.subTest(reward=reward), self.assertRaises(AgentContractError):
                session.submit_reward(reward, submission_id="invalid")
            self.assertEqual(session.rewards, [])
            self.assertEqual(session.interactions[0].reward_submissions, [])
        session.submit_reward(
            RewardAssignment("token", 1, position=(0, count - 1)), submission_id="eos"
        )
        session.submit_reward(
            RewardAssignment("token", 1, position=(0, count - 1)), submission_id="eos"
        )
        self.assertEqual(len(session.rewards), 1)

    def test_template_boolean_and_null_spellings_match_declared_types(self):
        """Qwen's Jinja scalar rendering uses Python True/False/None spellings."""
        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}, "empty": {"type": "null"}},
            "required": ["flag", "empty"],
            "additionalProperties": False,
        }
        for flag, expected in (("True", True), ("False", False)):
            raw = f"</think>\n<tool_call>\n<function=typed>\n<parameter=flag>\n{flag}\n</parameter>\n<parameter=empty>\nNone\n</parameter>\n</function>\n</tool_call><|im_end|>"
            tokens = tuple(self.tokenizer.encode(raw, add_special_tokens=False))
            parsed = Qwen35ToolParser(self.tokenizer).parse_generation(
                Generation(tokens, (-0.2,) * len(tokens), "untrusted"),
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "typed", "parameters": schema},
                    }
                ],
                prompt_tokens=self.tokenizer.encode("<think>\n"),
            )
            self.assertEqual(
                parsed.action.tool_calls[0].arguments, {"flag": expected, "empty": None}
            )

    async def test_parser_contract_failures_keep_sampled_evidence(self):
        """Plugin failures remain diagnosable without inventing model actions."""
        for inconsistent in (False, True):

            def parse(generation, *, tools, prompt_tokens, conflict=inconsistent):
                if not conflict:
                    return None
                parsed = Qwen35ToolParser(self.tokenizer).parse_generation(
                    generation, tools=tools, prompt_tokens=prompt_tokens
                )
                return replace(
                    parsed,
                    assistant_message={"role": "assistant", "content": "rewritten"},
                )

            client, sampler, session = self.setup_backend(
                ["</think>\n5<|im_end|>"],
                output_parser=SimpleNamespace(parse_generation=parse),
            )
            with (
                self.subTest(inconsistent=inconsistent),
                self.assertRaises(AgentContractError),
            ):
                await client.create_chat_completion(
                    messages=[{"role": "user", "content": "go"}], session=session
                )
            self.assertEqual(session.status, "failed")
            self.assertEqual(
                session.interactions[0].output_ids, tuple(sampler.generated[0])
            )
            self.assertTrue(session.export_trajectory().interactions[0]["logprobs"])

    async def test_inference_and_fixed_policy_sources_cannot_mix(self):
        """The external OpenAI path stays separate from one fixed-policy attempt."""
        for pytrio_first in (False, True):
            calls = []

            async def create(_calls=calls, **kwargs):
                _calls.append(kwargs)
                return {
                    "id": "external",
                    "choices": [
                        {"message": {"role": "assistant", "content": "external"}}
                    ],
                }

            native, sampler, session = self.setup_backend(["</think>\n5<|im_end|>"])
            external = OpenAIClient(
                SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace(create=create))
                ),
                model="external",
            )
            initial = {
                "messages": [{"role": "user", "content": "go"}],
                "session": session,
            }
            first, second = (native, external) if pytrio_first else (external, native)
            await first.create_chat_completion(**initial)
            with (
                self.subTest(pytrio_first=pytrio_first),
                self.assertRaises(AgentContractError),
            ):
                await second.create_chat_completion(**initial)
            self.assertEqual(len(calls), 0 if pytrio_first else 1)
            self.assertEqual(len(sampler.calls), 1 if pytrio_first else 0)

    async def test_optional_seed_and_endpoint_token_budgets_match_sdk_contract(self):
        client, sampler, session = self.setup_backend(["</think>\n5<|im_end|>"])
        await client.create_chat_completion(
            messages=[{"role": "user", "content": "go"}], session=session, seed=None
        )
        self.assertEqual(len(sampler.calls), 1)
        for responses in (False, True):
            client, sampler, session = self.setup_backend([])
            with (
                self.subTest(responses=responses),
                self.assertRaises(AgentContractError),
            ):
                if responses:
                    await client.create_response(
                        input="go", session=session, max_tokens=1
                    )
                else:
                    await client.create_chat_completion(
                        messages=[{"role": "user", "content": "go"}],
                        session=session,
                        max_output_tokens=1,
                    )
            self.assertEqual(sampler.calls, [])

    async def test_native_request_preflight_does_not_consume_proxy_capacity(self):
        """Pure request validation must keep the earlier empty-session cleanup."""
        adapter, sampler, template = self.setup_backend(["</think>\n5<|im_end|>"])
        proxy = OpenAIProxyServer(
            adapter,
            max_sessions=1,
            session_factory=lambda identity: OpenAISession(
                context=template.context, training=True, session_id=identity
            ),
        )
        for invalid in (
            {"tool_choice": "required"},
            {"max_output_tokens": 2},
            {"temperature": 0.2},
        ):
            with self.subTest(options=invalid), self.assertRaises(AgentContractError):
                await proxy.chat_completions(
                    {"messages": [{"role": "user", "content": "go"}], **invalid},
                    session_id="invalid",
                )
            self.assertEqual(proxy.sessions, {})
            self.assertEqual(sampler.calls, [])
        await proxy.chat_completions(
            {"messages": [{"role": "user", "content": "go"}]}, session_id="valid"
        )
        self.assertEqual(len(sampler.calls), 1)


class RealQwen35ToolsTests(NativeToolsTests):
    """Run the same native contracts against the actual cached model tokenizer."""

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            cls.tokenizer = AutoTokenizer.from_pretrained(
                os.environ.get("TRIOTUNE_TEST_TOKENIZER", "Qwen/Qwen3.5-4B"),
                local_files_only=True,
            )
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest("Cached Qwen3.5 tokenizer is unavailable.") from exc

    async def test_project_non_thinking_configuration_preserves_native_prefix(self):
        """Use the calculator's real enable_thinking=False template setting."""
        client, sampler, session = self.setup_backend(
            [xml_call() + "<|im_end|>", "5<|im_end|>"],
            template_kwargs={"enable_thinking": False},
        )
        result = await Agent(
            name="nonthinking", instructions="Use add.", tools=[add], client=client
        ).run("go", session=session, training=True)
        self.assertEqual(result.text, "5")
        expected = sampler.calls[0] + sampler.generated[0]
        self.assertEqual(sampler.calls[1][: len(expected)], expected)
        self.assertEqual(
            session.interactions[0].response["_triotune_sampling"]["reasoning"], ""
        )
        suffix = self.tokenizer.decode(
            sampler.calls[1][len(expected) :], skip_special_tokens=False
        )
        self.assertFalse(suffix.startswith("<|im_end|>"))

    def test_actual_jinja_tool_scalars_roundtrip_through_model_parser(self):
        """Parse parameter spellings emitted by the real project's template."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "typed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flag": {"type": "boolean"},
                            "empty": {"type": "null"},
                        },
                        "required": ["flag", "empty"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        initial = [{"role": "user", "content": "go"}]
        assistant = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "typed",
                        "arguments": {"flag": True, "empty": None},
                    },
                }
            ],
        }
        prompt = self.tokenizer.apply_chat_template(
            initial, tools=tools, tokenize=False, add_generation_prompt=True
        )
        rendered = self.tokenizer.apply_chat_template(
            initial + [assistant],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        self.assertTrue(rendered.startswith(prompt))
        raw = rendered[len(prompt) :]
        self.assertIn("\nTrue\n", raw)
        self.assertIn("\nNone\n", raw)
        ids = tuple(self.tokenizer.encode(raw, add_special_tokens=False))
        parsed = Qwen35ToolParser(self.tokenizer).parse_generation(
            Generation(ids, (-0.2,) * len(ids), "untrusted"),
            tools=tools,
            prompt_tokens=self.tokenizer.encode(prompt, add_special_tokens=False),
        )
        self.assertEqual(
            parsed.action.tool_calls[0].arguments, {"flag": True, "empty": None}
        )
