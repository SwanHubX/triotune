"""Offline tests for the optional OpenAI tool and session integration."""

import asyncio
import importlib.util
import json
import unittest
from types import SimpleNamespace

from triotune.agentrl import FunctionToolExecutor
from triotune.agentrl.sdk.openai import (
    Agent,
    OpenAIChatObservationFormatter,
    OpenAIClient,
    OpenAINativeActionParser,
    OpenAIProxyServer,
    OpenAISession,
    function_tool,
)
from triotune.agentrl.types import (
    ActionParseError,
    ToolCall,
    ToolResult,
    current_execution_context,
)


@function_tool(retry_safe=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


class _FakeChat:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                id="chat-1",
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    function=SimpleNamespace(
                                        name="add", arguments='{"a": 2, "b": 3}'
                                    ),
                                )
                            ],
                        ),
                        logprobs=None,
                    )
                ],
            )
        return SimpleNamespace(
            id="chat-2",
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="5", tool_calls=[]), logprobs=None
                )
            ],
        )


class OpenAISDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_typed_tool_schema_and_validation(self):
        self.assertEqual(add.parameters["properties"]["a"]["type"], "number")
        self.assertEqual(add.description, "Add two numbers.")
        value = await FunctionToolExecutor([add]).execute(
            ToolCall("call", "add", {"a": 2, "b": 3})
        )
        self.assertEqual(value.content, "5")

    def test_native_parser_and_formatter_preserve_call_id(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-7",
                                function=SimpleNamespace(
                                    name="add", arguments='{"a": 1, "b": 2}'
                                ),
                            )
                        ],
                    )
                )
            ]
        )
        action = OpenAINativeActionParser().parse_chat_completion(response)
        messages = OpenAIChatObservationFormatter().format(
            [ToolResult("call-7", "add", "3")]
        )
        self.assertEqual(action.tool_calls[0].call_id, "call-7")
        self.assertEqual(messages[0]["tool_call_id"], "call-7")

    async def test_agent_loop_records_session_request_and_delayed_reward(self):
        fake = _FakeChat()
        session = OpenAISession()
        agent = Agent(
            name="calculator",
            instructions="Use tools.",
            tools=[add],
            model="fake-model",
            client=OpenAIClient(
                SimpleNamespace(chat=SimpleNamespace(completions=fake)),
                session=session,
            ),
        )
        result = await agent.run("2 + 3", session=session)
        result.session.submit_reward(1.0)
        exported = result.export_trajectory()
        self.assertEqual(result.text, "5")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(session.interactions), 2)
        self.assertEqual(
            session.interactions[1].parent_interaction_id,
            session.interactions[0].interaction_id,
        )
        self.assertEqual(exported.session_id, session.session_id)
        self.assertEqual(exported.rewards, (1.0,))

    async def test_recursive_types_defaults_and_invalid_arguments(self):
        @function_tool
        async def total(values: list[float], scale: float | None = None):
            return sum(values) * (1 if scale is None else scale)

        executor = FunctionToolExecutor([total])
        result = await executor.execute(ToolCall("a", "total", {"values": [1, 2.5]}))
        self.assertEqual(result.content, "3.5")
        for arguments in (
            {"values": [True]},
            {"values": ["1"]},
            {"values": None},
            {"values": [1], "extra": 3},
            {"values": [float("nan")]},
            {},
        ):
            with self.subTest(arguments=arguments):
                self.assertTrue(
                    (await executor.execute(ToolCall("b", "total", arguments))).is_error
                )
        schema = total.parameters["properties"]["scale"]
        self.assertIn({"type": "null"}, schema["anyOf"])
        self.assertEqual(total.parameters["required"], ["values"])

    def test_invalid_native_calls_and_truncation(self):
        parser = OpenAINativeActionParser()
        for arguments in ('{"a":NaN}', '{"a":1,"a":2}', "[1]", "{"):
            with self.subTest(arguments=arguments), self.assertRaises(ActionParseError):
                parser.parse_response(
                    {
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "c",
                                "name": "add",
                                "arguments": arguments,
                            }
                        ],
                    }
                )
        with self.assertRaises(ActionParseError):
            parser.parse_chat_completion(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "unfinished"},
                        }
                    ]
                }
            )

    def test_unsupported_signatures_fail_at_registration(self):
        def positional(value: int, /):
            return value

        def variadic(*values: int):
            return values

        def unsupported(value: complex):
            return value

        for function in (positional, variadic, unsupported):
            with self.subTest(function=function), self.assertRaises(TypeError):
                function_tool(function)

    def test_delayed_reward_is_finite_idempotent_and_export_is_json(self):
        from dataclasses import asdict

        session = OpenAISession()
        session.submit_reward(1, submission_id="outcome")
        session.submit_reward(1, submission_id="outcome")
        self.assertEqual(len(session.rewards), 1)
        for value in (True, float("nan"), "1"):
            with self.assertRaises(ValueError):
                session.submit_reward(value)
        exported = session.export_trajectory()
        self.assertFalse(exported.metadata["trainable"])
        json.dumps(asdict(exported), allow_nan=False)


@unittest.skipUnless(
    importlib.util.find_spec("openai"),
    "Install triotune[openai] for SDK transport tests.",
)
class OfficialSDKTests(unittest.IsolatedAsyncioTestCase):
    """Use official SDK response models and mocked HTTP without credentials."""

    async def asyncSetUp(self):
        import httpx
        from openai import AsyncOpenAI

        self.requests = []
        self.contexts = []
        self.mode = "normal"

        async def handler(request):
            self.contexts.append(current_execution_context())
            payload = json.loads(request.content)
            self.requests.append((request, payload))
            if self.mode == "error":
                return httpx.Response(
                    503, json={"error": {"message": "offline", "type": "server_error"}}
                )
            if self.mode == "cancel":
                raise asyncio.CancelledError()
            if request.url.path.endswith("/responses"):
                final = any(
                    item.get("type") == "function_call_output"
                    for item in payload["input"]
                )
                output = (
                    [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "5",
                                    "annotations": [],
                                    "logprobs": [
                                        {
                                            "token": "5",
                                            "logprob": -0.1,
                                            "bytes": [53],
                                            "top_logprobs": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                    if final
                    else [
                        {"id": "rs_1", "type": "reasoning", "summary": []},
                        {
                            "id": "fc_1",
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "add",
                            "arguments": '{"a":2,"b":3}',
                            "status": "completed",
                        },
                    ]
                )
                body = {
                    "id": "resp_final" if final else "resp_tool",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "fake",
                    "output": output,
                    "usage": {"input_tokens": 8, "output_tokens": 0, "total_tokens": 8},
                }
            else:
                final = any(item.get("role") == "tool" for item in payload["messages"])
                message = (
                    {"role": "assistant", "content": "5"}
                    if final
                    else {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": '{"a":2,"b":3}',
                                },
                            }
                        ],
                    }
                )
                body = {
                    "id": "chat_final" if final else "chat_tool",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fake",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop" if final else "tool_calls",
                            "message": message,
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "5",
                                        "logprob": -0.1,
                                        "bytes": [53],
                                        "top_logprobs": [],
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 0,
                        "total_tokens": 4,
                    },
                }
            return httpx.Response(
                200, json=body, headers={"x-request-id": "server-request"}
            )

        self.sdk = AsyncOpenAI(
            api_key="offline-test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        self.adapter = OpenAIClient(self.sdk, model="fake")

    async def asyncTearDown(self):
        await self.sdk.close()

    async def test_both_apis_tool_loop_and_request_context(self):
        from dataclasses import asdict

        observed = []

        @function_tool
        def add(a: float, b: float):
            observed.append(current_execution_context())
            return a + b

        for api in ("chat_completions", "responses"):
            agent = Agent(
                name="calc",
                instructions="Use add.",
                tools=[add],
                client=self.adapter,
                api=api,
            )
            result = (
                await agent.run("2+3", logprobs=True)
                if api == "chat_completions"
                else await agent.run("2+3")
            )
            self.assertEqual(result.text, "5")
            first, second = result.session.interactions
            self.assertEqual(first.server_request_id, "server-request")
            self.assertEqual(first.output_tokens, 0)
            self.assertEqual(len(first.tool_calls), 1)
            self.assertEqual(first.tool_results[0]["call_id"], "call-1")
            self.assertEqual(observed[-1], first.context)
            self.assertEqual(second.parent_interaction_id, first.interaction_id)
            self.assertIsNotNone(second.logprobs)
            wire_request, wire = self.requests[-1]
            self.assertEqual(
                wire_request.headers["x-client-request-id"], second.request_id
            )
            if api == "responses":
                self.assertEqual(wire["tools"][0]["name"], "add")
                self.assertEqual(wire["input"][1]["type"], "reasoning")
                self.assertEqual(wire["input"][-1]["call_id"], "call-1")
            else:
                self.assertEqual(wire["messages"][-1]["tool_call_id"], "call-1")
            json.dumps(asdict(result.export_trajectory()), allow_nan=False)
        self.assertIsNone(current_execution_context())

    async def test_concurrent_runs_have_independent_sessions(self):
        agent = Agent(
            name="calc", instructions="Use add.", tools=[add], client=self.adapter
        )
        left, right = await asyncio.gather(agent.run("left"), agent.run("right"))
        self.assertNotEqual(left.session_id, right.session_id)
        self.assertEqual(len(left.session.interactions), 2)
        self.assertEqual(len(right.session.interactions), 2)
        self.assertEqual(
            left.session.interactions[0].request["messages"][-1]["content"], "left"
        )

    async def test_api_failure_and_cancellation_preserve_trace(self):
        from openai import APIStatusError

        for mode, exception in (
            ("error", APIStatusError),
            ("cancel", asyncio.CancelledError),
        ):
            self.mode = mode
            session = OpenAISession()
            with self.assertRaises(exception):
                await self.adapter.create_chat_completion(
                    messages=[{"role": "user", "content": "x"}], session=session
                )
            self.assertEqual(
                session.interactions[0].status,
                "cancelled" if mode == "cancel" else "failed",
            )
            self.assertIsNone(current_execution_context())

    async def test_streaming_and_multiple_choices_fail_before_http(self):
        for kwargs in ({"stream": True}, {"n": 2}):
            with self.assertRaises(ValueError):
                await self.adapter.create_chat_completion(messages=[], **kwargs)
        self.assertEqual(self.requests, [])

    async def test_turn_limit_and_injected_client_ownership(self):
        agent = Agent(
            name="calc",
            instructions="Use add.",
            tools=[add],
            client=self.adapter,
            max_turns=1,
        )
        session = OpenAISession()
        with self.assertRaisesRegex(RuntimeError, "max_turns"):
            await agent.run("2+3", session=session)
        self.assertEqual(len(session.interactions[0].tool_results), 1)
        self.assertFalse(session.running)
        await agent.close()
        self.assertFalse(self.sdk.is_closed())

    async def test_tool_timeout_and_cancellation_preserve_identity(self):
        def make_tool(cancelled):
            @function_tool(name="add")
            async def slow(a: float, b: float):
                if cancelled:
                    raise asyncio.CancelledError()
                await asyncio.sleep(1)

            return slow

        for cancelled in (False, True):
            agent = Agent(
                name="calc",
                instructions="Use add.",
                tools=[make_tool(cancelled)],
                client=self.adapter,
                tool_timeout=0.01,
            )
            session = OpenAISession()
            with self.assertRaises(
                asyncio.CancelledError if cancelled else TimeoutError
            ):
                await agent.run("2+3", session=session)
            self.assertFalse(session.running)
            self.assertEqual(
                session.interactions[-1].status,
                "cancelled" if cancelled else "tool_failure",
            )
            self.assertIsNone(current_execution_context())

    async def test_official_synchronous_client_runs_off_event_loop(self):
        import httpx
        from openai import OpenAI

        observed = []

        def handler(request):
            observed.append(current_execution_context())
            return httpx.Response(
                200,
                json={
                    "id": "sync",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fake",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        with OpenAI(
            api_key="offline-test",
            base_url="http://test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ) as sdk:
            agent = Agent(name="sync", instructions="Answer.", model="fake", client=sdk)
            result = await agent.run("hello")
            self.assertEqual(result.text, "done")
            self.assertEqual(observed[0], result.session.interactions[0].context)
            self.assertIsNone(current_execution_context())

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"),
        "Install triotune[openai-proxy] for HTTP routes.",
    )
    async def test_proxy_routes_with_official_sdk_and_session_header(self):
        import httpx
        from openai import AsyncOpenAI

        proxy = OpenAIProxyServer(self.adapter)
        async with AsyncOpenAI(
            api_key="offline-test",
            base_url="http://proxy/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy.create_app())
            ),
        ) as client:
            response = await client.chat.completions.create(
                model="fake",
                messages=[{"role": "user", "content": "x"}],
                extra_headers={"X-Triotune-Session-Id": "session-a"},
            )
            self.assertEqual(response.id, "chat_tool")
            await client.responses.create(
                model="fake",
                input=[{"role": "user", "content": "x"}],
                extra_headers={"X-Triotune-Session-Id": "session-b"},
            )
        self.assertEqual(set(proxy.sessions), {"session-a", "session-b"})
        self.assertEqual(
            proxy.sessions["session-a"].interactions[0].endpoint, "/v1/chat/completions"
        )
        proxy.release_session("session-a")
        self.assertNotIn("session-a", proxy.sessions)

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"),
        "Install triotune[openai-proxy] for HTTP routes.",
    )
    async def test_proxy_rejects_streams_and_preserves_upstream_failures(self):
        import httpx

        proxy = OpenAIProxyServer(self.adapter, max_sessions=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()),
            base_url="http://proxy",
        ) as client:
            payload = {"model": "fake", "messages": [{"role": "user", "content": "x"}]}
            headers = {"X-Triotune-Session-Id": "a"}
            response = await client.post(
                "/v1/chat/completions",
                json={**payload, "stream": True},
                headers=headers,
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(self.requests, [])
            self.mode = "error"
            response = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(proxy.sessions["a"].interactions[-1].status, "failed")
            with self.assertRaisesRegex(ValueError, "capacity"):
                proxy._session("b")
