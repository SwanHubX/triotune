"""Model-name-independent discovery and installable native tool adapters."""

import copy
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_openai_native import FakeSamplingClient, NativeTokenizer, add, xml_call
from triotune.agentrl import (
    AgentAction,
    AgentContractError,
    ExecutionContext,
    NativeToolFormat,
    PyTRIOAgentSampler,
    SamplingSnapshot,
    ToolCall,
    register_tool_format,
)
from triotune.agentrl.integrations.openai import (
    Agent,
    ModelOutput,
    OpenAIClient,
    OpenAISession,
    PyTRIOBackend,
    Qwen35ToolParser,
)
from triotune.agentrl.integrations.openai import tool_formats
from triotune.agentrl.runtime.conversation import NativeToolContinuationRenderer


class IsolateFormats:
    def setUp(self):
        super().setUp()
        registry = patch.dict(
            tool_formats._FORMATS,
            {"qwen3.5": tool_formats._FORMATS["qwen3.5"]},
            clear=True,
        )
        registry.start()
        self.addCleanup(registry.stop)
        points = patch.object(tool_formats, "entry_points", return_value=[])
        self.points = points.start()
        self.addCleanup(points.stop)
        tool_formats._plugin_formats.cache_clear()
        self.addCleanup(tool_formats._plugin_formats.cache_clear)

    @staticmethod
    def snapshot(tokenizer=None, outputs=()):
        tokenizer = tokenizer or NativeTokenizer()
        sampler = FakeSamplingClient(tokenizer, outputs, noncanonical=True)
        return SamplingSnapshot(
            "fixed", PyTRIOAgentSampler(sampler, tokenizer), tokenizer, 0,
        ), sampler


class ToolFormatDiscoveryTests(IsolateFormats, unittest.TestCase):
    def test_shared_grammar_is_detected_independently_of_catalog_model_id(self):
        # These are catalog labels, not claims about real model tokenizer files.
        for model in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B", "vendor/future-model"):
            with self.subTest(model=model):
                snapshot, _ = self.snapshot()
                backend = PyTRIOBackend(snapshot, model=model)
                self.assertEqual(backend.model_format, "qwen3.5")
                self.assertEqual(backend.tool_format_source, "template")
                self.assertIsInstance(backend.output_parser, Qwen35ToolParser)

    def test_metadata_and_explicit_selection_have_precedence(self):
        tokenizer = NativeTokenizer()
        tokenizer.init_kwargs = {"model_format": "qwen3.5"}
        snapshot, _ = self.snapshot(tokenizer)
        backend = PyTRIOBackend(snapshot, model="alias")
        self.assertEqual(backend.tool_format_source, "metadata")
        tokenizer.model_format = "uninstalled-grammar"
        with self.assertRaisesRegex(AgentContractError, "Conflicting"):
            PyTRIOBackend(snapshot, model="alias")
        backend = PyTRIOBackend(snapshot, model="alias", model_format="qwen3.5")
        self.assertEqual(backend.tool_format_source, "explicit")
        tokenizer.init_kwargs = {}
        with self.assertRaisesRegex(AgentContractError, "register an adapter"):
            PyTRIOBackend(snapshot, model="alias", model_format="auto")

    def test_explicit_names_and_metadata_do_not_bypass_template_validation(self):
        tokenizer = NativeTokenizer()
        tokenizer.chat_template = "different grammar"
        snapshot, _ = self.snapshot(tokenizer)
        for kwargs in ({"model_format": "qwen3.5"}, {}):
            tokenizer.init_kwargs = {"model_format": "qwen3.5"}
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(AgentContractError, "does not match"),
            ):
                PyTRIOBackend(snapshot, model="alias", **kwargs)

    def test_unknown_template_keeps_text_only_even_for_a_known_model_name(self):
        tokenizer = NativeTokenizer()
        tokenizer.chat_template = "plain text template"
        snapshot, _ = self.snapshot(tokenizer)
        backend = PyTRIOBackend(snapshot, model="Qwen/Qwen3.5-4B")
        self.assertFalse(backend.native_tools)
        self.assertIsNone(backend.model_format)

    def test_tool_use_template_dictionary_and_transformers_selection(self):
        tokenizer = NativeTokenizer()
        grammar = tokenizer.chat_template
        tokenizer.chat_template = {"default": "text only", "tool_use": grammar}
        snapshot, _ = self.snapshot(tokenizer)
        self.assertTrue(PyTRIOBackend(snapshot, model="alias").native_tools)

        class TemplateGetter(NativeTokenizer):
            chat_template = {"default": "text only", "custom": grammar}

            def get_chat_template(self, *, tools):
                self.saw_tools = tools
                return self.chat_template["custom"]

        tokenizer = TemplateGetter()
        snapshot, _ = self.snapshot(tokenizer)
        self.assertTrue(PyTRIOBackend(snapshot, model="alias").native_tools)
        self.assertTrue(tokenizer.saw_tools)

    def test_invalid_metadata_and_explicit_values_are_rejected(self):
        for value in (True, [], {}, "", "auto"):
            tokenizer = NativeTokenizer()
            tokenizer.init_kwargs = {"model_format": value}
            snapshot, _ = self.snapshot(tokenizer)
            with self.subTest(metadata=value), self.assertRaises(AgentContractError):
                PyTRIOBackend(snapshot, model="alias")
        snapshot, _ = self.snapshot()
        for value in (True, [], {}, "", "unsupported"):
            with self.subTest(explicit=value), self.assertRaises(AgentContractError):
                PyTRIOBackend(snapshot, model="alias", model_format=value)

    def test_ambiguous_detection_requires_explicit_selection(self):
        adapter = copy.copy(tool_formats._FORMATS["qwen3.5"])
        register_tool_format(replace(adapter, name="other-xml"))
        snapshot, _ = self.snapshot()
        with self.assertRaisesRegex(AgentContractError, "Ambiguous"):
            PyTRIOBackend(snapshot, model="alias")
        self.assertEqual(
            PyTRIOBackend(snapshot, model="alias", model_format="other-xml").model_format,
            "other-xml",
        )
        with self.assertRaisesRegex(AgentContractError, "already registered"):
            register_tool_format(adapter)

    def test_installed_entry_point_is_loaded_once_without_mutating_registry(self):
        adapter = NativeToolFormat(
            name="plugin-xml",
            template_matcher=lambda template: "PLUGIN_XML" in template,
            parser_factory=Qwen35ToolParser,
            schema_validator=Qwen35ToolParser.check_schema,
        )
        point = SimpleNamespace(name="plugin", load=lambda: adapter)
        self.points.return_value = [point]
        tokenizer = NativeTokenizer()
        tokenizer.chat_template = "PLUGIN_XML"
        snapshot, _ = self.snapshot(tokenizer)
        for _ in range(2):
            self.assertEqual(PyTRIOBackend(snapshot, model="new-model").model_format, "plugin-xml")
        self.points.assert_called_once_with(group="triotune.agentrl.tool_formats")
        self.assertNotIn("plugin-xml", tool_formats._FORMATS)

    def test_invalid_plugins_and_duplicate_plugin_names_are_rejected(self):
        for value in (object(), tool_formats._FORMATS["qwen3.5"]):
            tool_formats._plugin_formats.cache_clear()
            self.points.return_value = [SimpleNamespace(name="broken", load=lambda: value)]
            snapshot, _ = self.snapshot()
            with self.subTest(plugin=value), self.assertRaises(AgentContractError):
                PyTRIOBackend(snapshot, model="alias")

    def test_malformed_adapter_factories_fail_before_sampling(self):
        for factory in (lambda tokenizer: None, lambda tokenizer: object()):
            self.points.return_value = [SimpleNamespace(
                name="broken-parser",
                load=lambda: replace(
                    tool_formats._FORMATS["qwen3.5"], name="broken-parser",
                    parser_factory=factory,
                ),
            )]
            tool_formats._plugin_formats.cache_clear()
            snapshot, sampler = self.snapshot()
            with self.assertRaisesRegex(AgentContractError, "parse_generation"):
                PyTRIOBackend(snapshot, model="alias", model_format="broken-parser")
            self.assertEqual(sampler.calls, [])

    def test_non_boolean_matcher_is_rejected(self):
        register_tool_format(replace(
            tool_formats._FORMATS["qwen3.5"], name="bad-matcher",
            template_matcher=lambda template: "yes",
        ))
        snapshot, _ = self.snapshot()
        with self.assertRaisesRegex(AgentContractError, "must return bool"):
            PyTRIOBackend(snapshot, model="alias")


class CustomToolFormatTests(IsolateFormats, unittest.IsolatedAsyncioTestCase):
    async def test_new_grammar_uses_its_parser_schema_and_renderer_with_exact_tokens(self):
        observed = []

        class JsonParser:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def parse_generation(self, generation, *, tools, prompt_tokens):
                raw = self.tokenizer.decode(generation.tokens)
                text = raw.removesuffix("<|im_end|>")
                message = {"role": "assistant", "content": "", "reasoning_content": ""}
                if text.startswith("{"):
                    value = json.loads(text)
                    call = ToolCall("local-call", value["name"], value["arguments"])
                    message["tool_calls"] = [{
                        "id": call.call_id, "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }]
                    action = AgentAction(message, (call,))
                else:
                    message["content"] = text
                    action = AgentAction(message, final_text=text)
                return ModelOutput(generation, action, message, "", raw)

        class Renderer(NativeToolContinuationRenderer):
            def render_after_tool(self, *args, **kwargs):
                observed.append("continuation")
                return super().render_after_tool(*args, **kwargs)

        def check_schema(schema):
            observed.append("schema")
            self.assertEqual(schema["type"], "object")

        register_tool_format(NativeToolFormat(
            name="local-json",
            template_matcher=lambda template: "LOCAL_JSON" in template,
            parser_factory=JsonParser,
            schema_validator=check_schema,
            renderer_factory=Renderer,
        ))
        for api in ("chat_completions", "responses"):
            tokenizer = NativeTokenizer()
            tokenizer.chat_template = "LOCAL_JSON"
            snapshot, sampler = self.snapshot(tokenizer, [
                '{"name":"add","arguments":{"a":2,"b":3}}<|im_end|>',
                '5<|im_end|>',
            ])
            backend = PyTRIOBackend(snapshot, model="vendor/new-family")
            session = OpenAISession(context=ExecutionContext(
                "rollout", "episode", "group", 0, "fixed", policy_version=0,
            ), training=True)
            async with Agent(
                name="custom", instructions="Use add.", tools=[add],
                client=OpenAIClient(backend=backend), api=api,
            ) as agent:
                result = await agent.run("2+3", session=session, training=True)
            self.assertEqual(result.text, "5")
            prefix = sampler.calls[0] + sampler.generated[0]
            self.assertEqual(sampler.calls[1][:len(prefix)], prefix)
            self.assertEqual(list(session.interactions[0].output_ids), sampler.generated[0])
            self.assertEqual(backend.model_format, "local-json")
        self.assertIn("schema", observed)
        self.assertIn("continuation", observed)

    async def test_plugin_renderer_cannot_replace_sampled_prefix_tokens(self):
        class BrokenRenderer(NativeToolContinuationRenderer):
            def render_after_tool(self, *args, **kwargs):
                return super().render_after_tool(*args, **kwargs)[1:]

        register_tool_format(replace(
            tool_formats._FORMATS["qwen3.5"],
            name="broken-renderer", renderer_factory=BrokenRenderer,
        ))
        snapshot, sampler = self.snapshot(outputs=["</think>\n" + xml_call() + "<|im_end|>"])
        client = OpenAIClient(backend=PyTRIOBackend(
            snapshot, model="alias", model_format="broken-renderer",
        ))
        session = OpenAISession(context=ExecutionContext(
            "rollout", "episode", "group", 0, "fixed", policy_version=0,
        ), training=True)
        async with Agent(
            name="broken", instructions="Use add.", tools=[add], client=client,
        ) as agent:
            with self.assertRaisesRegex(AgentContractError, "sampled prefix"):
                await agent.run("2+3", session=session, training=True)
        self.assertEqual(len(sampler.calls), 1)

    async def test_detected_format_continues_for_new_model_ids_and_rejects_template_mutation(self):
        snapshot, sampler = self.snapshot(outputs=["</think>\n" + xml_call() + "<|im_end|>"])
        client = OpenAIClient(backend=PyTRIOBackend(snapshot, model="Qwen/Qwen3.8-27B"))
        session = OpenAISession(context=ExecutionContext(
            "rollout", "episode", "group", 0, "fixed", policy_version=0,
        ), training=True)
        await client.create_chat_completion(
            messages=[{"role": "user", "content": "2+3"}],
            tools=[{"type": "function", "function": {"name": "add", "parameters": add.parameters}}],
            session=session,
        )
        snapshot.tokenizer.chat_template = "changed"
        with self.assertRaisesRegex(AgentContractError, "template was changed"):
            await client.create_chat_completion(
                messages=[{"role": "user", "content": "other"}], session=session,
            )
        self.assertEqual(len(sampler.calls), 1)

    async def test_live_session_cannot_switch_registered_grammars(self):
        snapshot, sampler = self.snapshot(outputs=["</think>\n" + xml_call() + "<|im_end|>"])
        client = OpenAIClient(backend=PyTRIOBackend(snapshot, model="alias"))
        session = OpenAISession(context=ExecutionContext(
            "rollout", "episode", "group", 0, "fixed", policy_version=0,
        ), training=True)
        await client.create_chat_completion(
            messages=[{"role": "user", "content": "2+3"}],
            tools=[{"type": "function", "function": {"name": "add", "parameters": add.parameters}}],
            session=session,
        )
        register_tool_format(replace(tool_formats._FORMATS["qwen3.5"], name="other-xml"))
        client.backend = PyTRIOBackend(snapshot, model="alias", model_format="other-xml")
        with self.assertRaisesRegex(AgentContractError, "tool format changed"):
            await client.create_chat_completion(
                messages=[{"role": "user", "content": "other"}], session=session,
            )
        self.assertEqual(len(sampler.calls), 1)
