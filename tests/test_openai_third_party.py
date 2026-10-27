"""Standard AsyncOpenAI integration without Triotune's built-in Agent loop.

The fake upstream returns official protocol objects through the real OpenAI
SDK and an in-process ASGI transport. User callbacks execute ordinary Python
tools and return observations over the wire, exercising identity injection,
trace capture, replay validation, and lifecycle ownership independently of
Triotune's function-tool executor.
"""

import asyncio
import copy
import importlib.util
import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from triotune.agentrl import ExecutionContext
from triotune.agentrl.config import RetryPolicy
from triotune.agentrl.integrations.openai import (
    OpenAIClient,
    OpenAIProxyServer,
    OpenAISession,
    OpenAIWorkflow,
    SDKRunResult,
)
from triotune.agentrl.types import (
    AgentContractError,
    _execution_context,
    current_execution_context,
)


@unittest.skipUnless(
    importlib.util.find_spec("openai") and importlib.util.find_spec("fastapi"),
    "Install the openai-proxy extra for standard SDK integration tests.",
)
class ThirdPartySDKTests(unittest.IsolatedAsyncioTestCase):
    """Run third-party code with a real AsyncOpenAI object and local evidence."""

    @staticmethod
    def context(index=0):
        return ExecutionContext(
            f"rollout-{index}",
            f"episode-{index}",
            "group",
            index,
            "snapshot",
            policy_version=3,
            group_size=4,
            resource_owner="collector",
        )

    def setUp(self):
        self.observed = []
        self.mode = "normal"
        self.entered = asyncio.Event()

        async def create(**payload):
            context = current_execution_context()
            self.observed.append((context, copy.deepcopy(payload)))
            await asyncio.sleep(0)
            if self.mode == "fail":
                raise ConnectionError("offline")
            if self.mode == "block":
                self.entered.set()
                await asyncio.Event().wait()
            responses = "input" in payload
            items = payload["input"] if responses else payload["messages"]
            items = [] if isinstance(items, str) else items
            final = any(
                item.get("role") == "tool" or item.get("type") == "function_call_output"
                for item in items
            )
            identity = context.interaction_id
            call_ids = [f"call-{identity}-{index}" for index in range(2)]
            if responses:
                return {
                    "id": f"response-{identity}",
                    "object": "response",
                    "created_at": 0,
                    "model": "fake",
                    "status": "completed",
                    "output": (
                        [
                            {
                                "id": "msg",
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "14",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ]
                        if final
                        else [
                            {
                                "id": f"fc-{index}",
                                "type": "function_call",
                                "call_id": call_id,
                                "name": "add",
                                "arguments": json.dumps(
                                    {"a": 2 + index * 2, "b": 3 + index * 2}
                                ),
                                "status": "completed",
                            }
                            for index, call_id in enumerate(call_ids)
                        ]
                    ),
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            return {
                "id": f"chat-{identity}",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop" if final else "tool_calls",
                        "message": (
                            {"role": "assistant", "content": "14"}
                            if final
                            else {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": call_id,
                                        "function": {
                                            "name": "add",
                                            "arguments": json.dumps(
                                                {"a": 2 + index * 2, "b": 3 + index * 2}
                                            ),
                                        },
                                    }
                                    for index, call_id in enumerate(call_ids)
                                ],
                            }
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

        self.upstream = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            responses=SimpleNamespace(create=create),
        )
        self.upstream.closed = False

        async def close():
            self.upstream.closed = True

        self.upstream.close = close
        self.adapter = OpenAIClient(self.upstream, model="fake")

    @staticmethod
    async def callback(sdk, question, *, responses=False):
        """A complete third-party loop using only the official SDK protocol."""
        from openai import AsyncOpenAI

        if not isinstance(sdk, AsyncOpenAI):
            raise TypeError("Third-party execution requires the official client.")
        initial = [{"role": "user", "content": question}]
        if responses:
            first = await sdk.responses.create(model="fake", input=initial)
            observations = []
            for call in first.output:
                arguments = json.loads(call.arguments)
                await asyncio.sleep(0)
                observations.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": str(arguments["a"] + arguments["b"]),
                    }
                )
            final = await sdk.responses.create(
                model="fake",
                input=initial
                + [item.model_dump(exclude_none=True) for item in first.output]
                + observations,
            )
            return {"answer": final.output_text}
        first = await sdk.chat.completions.create(model="fake", messages=initial)
        observations = []
        for call in first.choices[0].message.tool_calls:
            arguments = json.loads(call.function.arguments)
            await asyncio.sleep(0)
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(arguments["a"] + arguments["b"]),
                }
            )
        final = await sdk.chat.completions.create(
            model="fake",
            messages=initial
            + [first.choices[0].message.model_dump(exclude_none=True)]
            + observations,
        )
        return {"answer": final.choices[0].message.content}

    async def test_user_callbacks_both_protocols_capture_results_and_rewards(self):
        for responses in (False, True):
            context = self.context()
            workflow = OpenAIWorkflow(client=self.adapter)

            async def execute(sdk, question, selected=responses):
                return await self.callback(sdk, question, responses=selected)

            result = await workflow.run(
                "compute", execute=execute, context=context, training=True
            )
            self.assertIsInstance(result, SDKRunResult)
            self.assertEqual(result.output, {"answer": "14"})
            first, second = result.session.interactions
            self.assertEqual(
                [item["content"] for item in first.tool_results], ["5", "9"]
            )
            self.assertEqual(second.parent_interaction_id, first.interaction_id)
            events = [
                event for event in result.session.events if event.kind == "tool_result"
            ]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event.context == first.context for event in events))
            self.assertTrue(
                all(
                    item.context.rollout_id == context.rollout_id
                    for item in result.session.interactions
                )
            )
            self.assertEqual(result.session.status, "completed")
            await workflow.submit_reward(result, 1, submission_id="outcome")
            await workflow.submit_reward(result, 1, submission_id="outcome")
            trace = workflow.export_trajectory(result)
            self.assertEqual(len(trace.rewards), 1)
            self.assertEqual(trace.metadata["context"], asdict(context))
            self.assertFalse(trace.metadata["trainable"])
            result.session.release()
        self.assertIsNone(current_execution_context())

    async def test_bound_official_client_reuses_session_and_rejects_header_override(
        self,
    ):
        from openai import APIConnectionError, AsyncOpenAI

        session = OpenAISession()
        proxy = OpenAIProxyServer(self.adapter, max_sessions=1)
        async with proxy.create_client(session=session) as sdk:
            self.assertIsInstance(sdk, AsyncOpenAI)
            await self.callback(sdk, "compute")
            self.assertEqual(set(proxy.sessions), {session.session_id})
            self.assertEqual(len(session.interactions), 2)
            with self.assertRaises(APIConnectionError):
                await sdk.with_options(
                    default_headers={"X-Triotune-Session-Id": "other"}
                ).chat.completions.create(model="fake", messages=[])
            self.assertEqual(set(proxy.sessions), {session.session_id})
            self.assertEqual(len(session.interactions), 2)
        self.assertEqual(session.status, "open")
        self.assertTrue(sdk.is_closed())
        self.assertFalse(self.upstream.closed)
        proxy.finish_session(session.session_id)
        proxy.release_session(session.session_id)

    async def test_full_history_replays_are_idempotent_and_conflicts_are_atomic(self):
        from openai import BadRequestError

        session = OpenAISession()
        async with self.adapter.as_async_openai(session=session) as sdk:
            initial = [{"role": "user", "content": "compute"}]
            first = await sdk.chat.completions.create(model="fake", messages=initial)
            assistant = first.choices[0].message.model_dump(exclude_none=True)
            calls = first.choices[0].message.tool_calls
            observations = [
                {"role": "tool", "tool_call_id": call.id, "content": str(index)}
                for index, call in enumerate(calls)
            ]
            invalid = copy.deepcopy(observations)
            invalid[-1]["tool_call_id"] = "foreign"
            with self.assertRaises(BadRequestError):
                await sdk.chat.completions.create(
                    model="fake", messages=initial + [assistant] + invalid
                )
            with self.assertRaises(BadRequestError):
                await sdk.chat.completions.create(
                    model="fake",
                    messages=initial + [assistant] + [observations[0], observations[0]],
                )
            self.assertEqual(session.interactions[0].tool_results, [])
            self.assertEqual(len(session.interactions), 1)
            messages = initial + [assistant] + observations
            second = await sdk.chat.completions.create(model="fake", messages=messages)
            self.assertEqual(len(session.interactions[0].tool_results), 2)
            # Echoing the complete conversation must not re-add old tool
            # results or re-emit reward-bearing execution events.
            await sdk.chat.completions.create(
                model="fake",
                messages=messages
                + [
                    second.choices[0].message.model_dump(exclude_none=True),
                    {"role": "user", "content": "again"},
                ],
            )
            self.assertEqual(len(session.interactions[0].tool_results), 2)
            conflicting = copy.deepcopy(observations)
            conflicting[0]["content"] = "changed"
            with self.assertRaises(BadRequestError):
                await sdk.chat.completions.create(
                    model="fake", messages=initial + [assistant] + conflicting
                )
            self.assertEqual(len(self.observed), 3)

    async def test_responses_delta_and_foreign_session_call_ids(self):
        from openai import BadRequestError

        left, right = OpenAISession(), OpenAISession()
        async with (
            self.adapter.as_async_openai(session=left) as one,
            self.adapter.as_async_openai(session=right) as two,
        ):
            first = await one.responses.create(model="fake", input="left")
            foreign = await two.responses.create(model="fake", input="right")
            invalid = [
                {
                    "type": "function_call_output",
                    "call_id": foreign.output[0].call_id,
                    "output": "bad",
                }
            ]
            with self.assertRaises(BadRequestError):
                await one.responses.create(
                    model="fake", previous_response_id=first.id, input=invalid
                )
            self.assertEqual(left.interactions[0].tool_results, [])
            observations = [
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "done",
                }
                for call in first.output
            ]
            await one.responses.create(
                model="fake", previous_response_id=first.id, input=observations
            )
            self.assertEqual(len(left.interactions[0].tool_results), 2)
            self.assertEqual(
                left.interactions[1].parent_interaction_id,
                left.interactions[0].interaction_id,
            )

    async def test_parallel_candidate_identity_and_context_restoration(self):
        enclosing = self.context(3)
        token = _execution_context.set(enclosing)
        try:
            workflow = OpenAIWorkflow(client=self.adapter)
            results = await asyncio.gather(
                *(
                    workflow.run(
                        str(index),
                        execute=self.callback,
                        context=self.context(index),
                        training=True,
                    )
                    for index in range(3)
                )
            )
            self.assertEqual(len({result.session_id for result in results}), 3)
            self.assertEqual(
                sorted(result.session.context.sample_index for result in results),
                [0, 1, 2],
            )
            self.assertTrue(
                all(
                    record.context == result.session.context
                    or record.context.interaction_id is not None
                    for result in results
                    for record in result.session.events
                )
            )
            self.assertEqual(current_execution_context(), enclosing)
        finally:
            _execution_context.reset(token)

    async def test_failure_cancel_and_nonawaitable_callbacks_finalize(self):
        for mode in ("fail", "cancel", "sync", "empty"):
            self.mode = "fail" if mode == "fail" else "normal"
            session = OpenAISession()

            async def execute(sdk, question, selected=mode):
                if selected == "empty":
                    return "no requests"
                await sdk.chat.completions.create(
                    model="fake", messages=[{"role": "user", "content": question}]
                )
                if selected == "cancel":
                    raise asyncio.CancelledError()

            callback = (
                (lambda sdk, question: "not awaitable") if mode == "sync" else execute
            )
            with (
                self.subTest(mode=mode),
                self.assertRaises(
                    asyncio.CancelledError if mode == "cancel" else Exception
                ),
            ):
                await OpenAIWorkflow(client=self.adapter).run(
                    "compute", execute=callback, session=session
                )
            self.assertEqual(
                session.status, "cancelled" if mode == "cancel" else "failed"
            )
            self.assertFalse(session.running)
            session.export_trajectory()
            with self.assertRaises(AgentContractError):
                session.submit_reward(1)
            session.release()
            self.assertIsNone(current_execution_context())

    async def test_completed_tool_results_survive_provider_failure(self):
        """A failed next model request cannot erase already executed tools."""
        from openai import InternalServerError

        session = OpenAISession()

        async def execute(sdk, question):
            initial = [{"role": "user", "content": question}]
            first = await sdk.chat.completions.create(model="fake", messages=initial)
            calls = first.choices[0].message.tool_calls
            observations = [
                {"role": "tool", "tool_call_id": call.id, "content": "done"}
                for call in calls
            ]
            self.mode = "fail"
            await sdk.chat.completions.create(
                model="fake",
                messages=initial
                + [first.choices[0].message.model_dump(exclude_none=True)]
                + observations,
            )

        with self.assertRaises(InternalServerError):
            await OpenAIWorkflow(client=self.adapter).run(
                "compute", execute=execute, session=session
            )
        self.assertEqual(
            [item["content"] for item in session.interactions[0].tool_results],
            ["done", "done"],
        )
        self.assertEqual(session.interactions[-1].status, "failed")
        self.assertEqual(session.status, "failed")
        session.export_trajectory()
        session.release()

    async def test_callback_failure_cancels_only_its_inflight_request_tasks(self):
        """SDK transport teardown joins requests before session finalization."""
        self.mode = "block"
        session, tasks = OpenAISession(), []

        async def execute(sdk, question):
            tasks.append(
                asyncio.create_task(
                    sdk.chat.completions.create(
                        model="fake", messages=[{"role": "user", "content": question}]
                    )
                )
            )
            await self.entered.wait()
            raise RuntimeError("framework failed")

        unrelated = asyncio.create_task(asyncio.sleep(0.1, result="unrelated"))
        with self.assertRaisesRegex(RuntimeError, "framework failed"):
            await OpenAIWorkflow(client=self.adapter).run(
                "compute", execute=execute, session=session
            )
        self.assertTrue(tasks[0].done())
        self.assertTrue(tasks[0].cancelled())
        self.assertEqual(await unrelated, "unrelated")
        self.assertEqual(session.interactions[0].status, "cancelled")
        self.assertEqual(session.status, "failed")
        session.export_trajectory()
        session.release()

    async def test_factory_validation_does_not_allocate_or_steal_sessions(self):
        session = OpenAISession()
        proxy = OpenAIProxyServer(self.adapter, max_sessions=1)
        for options in (
            {"default_headers": []},
            {"default_headers": {"key": None}},
            {"default_headers": {"X-Triotune-Session-Id": "other"}},
            {"max_retries": 2},
            {"max_retries": False},
            {"base_url": "http://other/v1"},
            {"http_client": object()},
        ):
            with self.subTest(options=options), self.assertRaises(AgentContractError):
                proxy.create_client(session=session, **options)
            self.assertEqual(proxy.sessions, {})
        async with proxy.create_client(
            session=session,
            default_headers={"x-triotune-session-id": session.session_id},
        ) as sdk:
            await sdk.chat.completions.create(
                model="fake", messages=[{"role": "user", "content": "go"}]
            )
            with self.assertRaises(AgentContractError):
                proxy.create_client(
                    session=OpenAISession(session_id=session.session_id)
                )
        self.assertIs(proxy.sessions[session.session_id], session)

    async def test_callback_identity_and_snapshot_preflight_precede_user_execution(
        self,
    ):
        """Collector provenance is not replaceable through the SDK callback path."""
        from triotune.agentrl import SamplingSnapshot

        executed = []

        async def execute(sdk, question):
            executed.append(question)

        workflow = OpenAIWorkflow(client=self.adapter)
        context = self.context()
        for options in (
            {
                "context": self.context(1),
                "session": OpenAISession(context=context, training=True),
                "training": True,
            },
            {
                "context": context,
                "session": OpenAISession(context=context),
                "training": True,
            },
            {"training": True},
            {
                "context": context,
                "snapshot": SamplingSnapshot(
                    "other", object(), object(), policy_version=3
                ),
            },
            {
                "context": context,
                "snapshot": SamplingSnapshot(
                    "snapshot", object(), object(), policy_version=4
                ),
            },
        ):
            with self.subTest(options=options), self.assertRaises(AgentContractError):
                await workflow.run("compute", execute=execute, **options)
        self.assertEqual(executed, [])
        self.assertEqual(self.observed, [])

    async def test_explicit_request_id_replay_and_total_budgets(self):
        session = OpenAISession()
        payload = {"messages": [{"role": "user", "content": "once"}]}
        first = await self.adapter.create_chat_completion(
            **payload, session=session, request_id="request-fixed"
        )
        replay = await self.adapter.create_chat_completion(
            **payload, session=session, request_id="request-fixed"
        )
        self.assertEqual(first, replay)
        self.assertEqual(len(session.interactions), 1)
        with self.assertRaises(AgentContractError):
            await self.adapter.create_chat_completion(
                messages=[{"role": "user", "content": "changed"}],
                session=session,
                request_id="request-fixed",
            )
        limited = OpenAISession(max_generated_tokens=1, episode_timeout=1.0)
        await self.adapter.create_chat_completion(
            messages=[{"role": "user", "content": "first"}],
            session=limited,
            request_id="limited-request",
        )
        cached = await self.adapter.create_chat_completion(
            messages=[{"role": "user", "content": "first"}],
            session=limited,
            request_id="limited-request",
        )
        self.assertEqual(cached["id"], limited.interactions[0].response_id)
        with self.assertRaises(AgentContractError):
            await self.adapter.create_chat_completion(
                messages=[{"role": "user", "content": "second"}], session=limited
            )
        self.assertEqual(len(limited.interactions), 1)

    async def test_proxy_explicit_request_id_replays_cached_completion(self):
        """HTTP retries reuse explicit identity, never prompt equality alone."""
        import httpx

        session = OpenAISession()
        proxy = OpenAIProxyServer(
            self.adapter,
            session_factory=lambda identity: session,
        )
        headers = {
            "X-Triotune-Session-Id": session.session_id,
            "X-Triotune-Request-Id": "retry-id",
        }
        payload = {"model": "fake", "messages": [{"role": "user", "content": "same"}]}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
            second = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
            self.assertEqual(first.json(), second.json())
            self.assertEqual(first.headers["X-Client-Request-Id"], "retry-id")
            self.assertEqual(second.headers["X-Client-Request-Id"], "retry-id")
            conflict = await client.post(
                "/v1/chat/completions",
                json={**payload, "messages": [{"role": "user", "content": "changed"}]},
                headers=headers,
            )
            self.assertEqual(conflict.status_code, 400)
        self.assertEqual(len(session.interactions), 1)
        self.assertEqual(len(self.observed), 1)

    async def test_total_episode_timeout_finishes_failed_session(self):
        self.mode = "block"
        session = OpenAISession(episode_timeout=0.01)
        with self.assertRaises(TimeoutError):
            await self.adapter.create_chat_completion(
                messages=[{"role": "user", "content": "timeout"}], session=session
            )
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.interactions[0].status, "failed")
        session.export_trajectory()

    async def test_native_pytrio_callback_uses_exact_arrays_and_out_of_order_results(
        self,
    ):
        """Third-party code can use the fixed native backend without Agent."""
        from test_openai_native import FakeSamplingClient, NativeTokenizer, xml_call

        from triotune.agentrl import PyTRIOAgentSampler, SamplingSnapshot
        from triotune.agentrl.integrations.openai import PyTRIOBackend

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        for responses in (False, True):
            tokenizer = NativeTokenizer()
            sampling = FakeSamplingClient(
                tokenizer,
                [
                    "</think>\n"
                    + xml_call()
                    + "\n"
                    + xml_call(a="4", b="5")
                    + "<|im_end|>",
                    "</think>\n14<|im_end|>",
                ],
                noncanonical=True,
            )
            snapshot = SamplingSnapshot(
                "snapshot",
                PyTRIOAgentSampler(sampling, tokenizer),
                tokenizer,
                policy_version=3,
            )
            adapter = OpenAIClient(
                backend=PyTRIOBackend(
                    snapshot, model="Qwen/Qwen3.5-4B", max_tokens=4096
                )
            )

            async def execute(sdk, question, selected=responses):
                initial = [{"role": "user", "content": question}]
                if selected:
                    schemas = [
                        {"type": "function", **item["function"]} for item in tools
                    ]
                    first = await sdk.responses.create(
                        model="Qwen/Qwen3.5-4B", input=initial, tools=schemas
                    )
                    calls = [
                        call for call in first.output if call.type == "function_call"
                    ]
                    observations = [
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": str(sum(json.loads(call.arguments).values())),
                        }
                        for call in reversed(calls)
                    ]
                    final = await sdk.responses.create(
                        model="Qwen/Qwen3.5-4B",
                        input=initial
                        + [item.model_dump(exclude_none=True) for item in first.output]
                        + observations,
                        tools=schemas,
                    )
                    return final.output_text
                first = await sdk.chat.completions.create(
                    model="Qwen/Qwen3.5-4B", messages=initial, tools=tools
                )
                calls = first.choices[0].message.tool_calls
                observations = [
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(
                            sum(json.loads(call.function.arguments).values())
                        ),
                    }
                    for call in reversed(calls)
                ]
                final = await sdk.chat.completions.create(
                    model="Qwen/Qwen3.5-4B",
                    messages=initial
                    + [first.choices[0].message.model_dump(exclude_none=True)]
                    + observations,
                    tools=tools,
                )
                return final.choices[0].message.content

            result = await OpenAIWorkflow(client=adapter).run(
                "compute", execute=execute, context=self.context(), snapshot=snapshot
            )
            self.assertEqual(result.output, "14")
            first, second = result.session.interactions
            self.assertEqual(
                [item["content"] for item in first.tool_results], ["9", "5"]
            )
            prefix = sampling.calls[0] + sampling.generated[0]
            self.assertEqual(sampling.calls[1][: len(prefix)], prefix)
            self.assertEqual(first.output_ids, tuple(sampling.generated[0]))
            self.assertEqual(second.logprobs, (-0.25,) * len(second.output_ids))
            self.assertFalse(result.export_trajectory().metadata["trainable"])

    async def test_workflow_scoring_retry_acceptance_and_resource_lifecycle(self):
        """SDK postprocessing reuses runner semantics without entering PPO."""
        calls = []

        class Scorer:
            resource_owner = "workflow"

            async def start(self):
                calls.append("start")

            async def close(self):
                calls.append("close")

            async def score(self, row, result):
                calls.append("score")
                if calls.count("score") == 1:
                    raise ConnectionError("retry scorer")
                return 0.75

        async def accept(result, trace):
            calls.append("accept")
            self.assertIsNone(trace)
            self.assertFalse(result.trainable)
            self.assertEqual(result.status, "completed")
            return False

        workflow = OpenAIWorkflow(
            client=self.adapter,
            scorer=Scorer(),
            accept_sample=accept,
            scorer_retry=RetryPolicy(max_attempts=2, initial_delay=0, max_delay=0),
            scorer_timeout=1,
            resource_owner="workflow",
        )
        async with workflow:
            result = await workflow.run("compute", execute=self.callback)
            self.assertFalse(result.accepted)
            self.assertFalse(result.session.accepted)
            self.assertEqual(result.session.rewards, [0.75])
            self.assertEqual(len(result.session.retries), 1)
            self.assertEqual(calls, ["start", "score", "score", "accept"])
        self.assertEqual(calls[-1], "close")

    async def test_interaction_scorer_targets_original_request(self):
        from triotune.agentrl import EpisodeScore, RewardAssignment

        async def score(row, result, interaction):
            index = result.session.interactions.index(interaction)
            return EpisodeScore(
                0,
                False,
                assignments=(
                    RewardAssignment("interaction", index + 1, source="sdk-prm"),
                ),
            )

        result = await OpenAIWorkflow(
            client=self.adapter, interaction_scorer=score
        ).run("compute", execute=self.callback)
        first, second = result.session.interactions
        self.assertEqual(first.reward["position"], 0)
        self.assertEqual(second.reward["position"], 1)
        self.assertEqual(len(first.reward_submissions), 1)
        self.assertEqual(len(second.reward_submissions), 1)

    async def test_failure_classifier_covers_callback_and_scoring_failures(self):
        def classify(stage, error, record):
            return "sdk_failure", f"{stage}:{type(error).__name__}"

        async def broken_score(row, result):
            raise RuntimeError("bad scorer")

        for execution_failure in (False, True):
            session = OpenAISession()
            scorer = None if execution_failure else broken_score
            workflow = OpenAIWorkflow(
                client=self.adapter, scorer=scorer, failure_classifier=classify
            )

            async def execute(sdk, input, fail=execution_failure):
                if fail:
                    raise RuntimeError("bad callback")
                return await self.callback(sdk, input)

            with self.assertRaises(RuntimeError):
                await workflow.run("compute", execute=execute, session=session)
            self.assertEqual(session.failure_classification[0], "sdk_failure")
            self.assertIs(session.accepted, None if execution_failure else False)
