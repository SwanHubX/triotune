"""Offline tests for the optional OpenAI tool and session integration."""

import asyncio
import importlib.util
import json
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

from triotune.agentrl import FunctionToolExecutor
from triotune.agentrl.runtime.sampling import PyTRIOAgentSampler, SamplingSnapshot
from triotune.agentrl.integrations.openai import (
    Agent,
    OpenAIChatObservationFormatter,
    OpenAIClient,
    OpenAINativeActionParser,
    OpenAIProxyServer,
    OpenAISession,
    OpenAIWorkflow,
    PyTRIOBackend,
    function_tool,
)
from triotune.agentrl.types import (
    ActionParseError,
    AgentContractError,
    ExecutionContext,
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
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    type="function",
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
                    message=SimpleNamespace(
                        role="assistant", content="5", tool_calls=[]
                    ),
                    logprobs=None,
                )
            ],
        )


class OpenAISDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_headers_leave_no_active_request_or_caller_mutation(self):
        fake = _FakeChat()
        adapter = OpenAIClient(
            SimpleNamespace(chat=SimpleNamespace(completions=fake)), model="fake"
        )
        session = OpenAISession()
        for headers in ([], "invalid", 3, {3: "value"}, {"key": None}, {"": "value"}):
            with self.subTest(headers=headers), self.assertRaises(ValueError):
                await adapter.create_chat_completion(
                    messages=[], session=session, extra_headers=headers
                )
            self.assertEqual(session.interactions, [])
            self.assertIsNone(current_execution_context())
        self.assertEqual(fake.calls, [])
        # No request was registered, so the owner can abort and release the
        # empty session rather than inheriting an unfinishable running record.
        session.finish("failed", error="invalid configuration")
        session.export_trajectory()
        session.release()

        headers = {"x-client-request-id": "caller", "custom": "value"}
        session = OpenAISession()
        await adapter.create_chat_completion(
            messages=[], session=session, extra_headers=headers
        )
        self.assertEqual(headers, {"x-client-request-id": "caller", "custom": "value"})
        self.assertEqual(
            fake.calls[0]["extra_headers"]["X-Client-Request-Id"],
            session.interactions[0].request_id,
        )

    async def test_preparation_failure_after_registration_finalizes_and_preserves_error(
        self,
    ):
        class BrokenPayload(dict):
            def __setitem__(self, key, value):
                if key == "extra_headers":
                    raise RuntimeError("header preparation failed")
                return super().__setitem__(key, value)

        fake = _FakeChat()
        adapter = OpenAIClient(
            SimpleNamespace(chat=SimpleNamespace(completions=fake)), model="fake"
        )
        session = OpenAISession()
        with self.assertRaisesRegex(RuntimeError, "header preparation failed"):
            await adapter._create(
                "/v1/chat/completions",
                fake.create,
                session,
                BrokenPayload(model="fake", messages=[]),
            )
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.interactions[0].status, "failed")
        self.assertEqual(len(session.export_trajectory().interactions), 1)
        session.release()
        self.assertEqual(fake.calls, [])
        self.assertIsNone(current_execution_context())

        session = OpenAISession()
        with (
            patch.object(session, "finish", side_effect=ValueError("cleanup failed")),
            self.assertRaisesRegex(
                RuntimeError, "header preparation failed"
            ) as failure,
        ):
            await adapter._create(
                "/v1/chat/completions",
                fake.create,
                session,
                BrokenPayload(model="fake", messages=[]),
            )
        self.assertIn("cleanup failed", failure.exception.__notes__[0])
        self.assertEqual(session.interactions[0].status, "failed")
        self.assertIsNone(current_execution_context())
        session.finish("failed")
        session.release()

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
        self.assertEqual(session.status, "completed")
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
        interaction = session.start_request("/v1/chat/completions", {"messages": []})
        session.complete_request(interaction, {"id": "completed", "choices": []})
        session.finish()
        session.submit_reward(1, submission_id="outcome")
        session.submit_reward(1, submission_id="outcome")
        self.assertEqual(len(session.rewards), 1)
        for value in (True, float("nan"), "1"):
            with self.assertRaises(ValueError):
                session.submit_reward(value)
        exported = session.export_trajectory()
        self.assertFalse(exported.metadata["trainable"])
        json.dumps(asdict(exported), allow_nan=False)


class SessionContractTests(unittest.TestCase):
    """Reward targeting and lifecycle rules work without optional SDK packages."""

    @staticmethod
    def record(
        session,
        *,
        payload=None,
        endpoint="/v1/chat/completions",
        output=None,
        parent=None,
    ):
        payload = (
            payload
            if payload is not None
            else {"messages": [{"role": "user", "content": "question"}]}
        )
        interaction = session.start_request(
            endpoint, payload, parent_interaction_id=parent
        )
        response = {"id": f"response-{len(session.interactions)}"}
        if endpoint == "/v1/chat/completions":
            response["choices"] = [
                {"message": {"role": "assistant", "content": output or "answer"}}
            ]
        else:
            response.update(
                status="completed",
                output=output
                or [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answer"}],
                    }
                ],
            )
        session.complete_request(interaction, response)
        return interaction

    def test_reward_assignments_bind_positions_and_trajectory_scope(self):
        from triotune.agentrl import RewardAssignment

        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        session.finish()
        for target, position in (("interaction", 0), ("turn", 0), ("token", (0, 2))):
            assignment = RewardAssignment(
                target, 3, position=position, mode="advantage", source="prm"
            )
            session.submit_reward(assignment)
            self.assertEqual(
                first.reward["position"],
                list(position) if isinstance(position, tuple) else position,
            )
            self.assertEqual(first.reward["mode"], "advantage")
            self.assertIsNone(second.reward)
            self.assertEqual(session.events[-1].context, first.context)
        trajectory = RewardAssignment("trajectory", 1, source="outcome")
        session.submit_reward(trajectory)
        self.assertIsNone(second.reward)
        self.assertEqual(len(first.reward_submissions), 3)
        self.assertEqual(session.events[-1].context, session.context)
        self.assertEqual(len(session.rewards), 4)

    def test_episode_score_dispatches_nested_targets_and_preserves_original_once(self):
        from triotune.agentrl import EpisodeScore, RewardAssignment

        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        session.finish()
        score = EpisodeScore(
            4,
            True,
            {"accuracy": 1},
            {0: (0.25, -0.5)},
            (
                RewardAssignment("interaction", 5, position=0, source="judge"),
                RewardAssignment(
                    "turn",
                    2,
                    position=1,
                    source="prm",
                    mode="advantage",
                    aggregation="mean",
                    discount=0.9,
                ),
                RewardAssignment("trajectory", 3, source="bonus"),
            ),
        )
        session.submit_reward(score, submission_id="score")
        session.submit_reward(score, submission_id="score")
        self.assertEqual(session.rewards, [score])
        self.assertEqual(len(first.reward_submissions), 3)
        self.assertEqual(first.reward["value"], 5)
        self.assertEqual(first.reward["position"], 0)
        self.assertEqual(first.reward_submissions[0]["position"], [0, 0])
        self.assertEqual(first.reward_submissions[1]["value"], -0.5)
        self.assertEqual(
            second.reward_submissions, [session._jsonable(score.assignments[1])]
        )
        event = session.events[-1]
        self.assertEqual(event.context, session.context)
        self.assertEqual(
            event.details["bindings"][0],
            {
                "interaction_id": None,
                "reward": {"reward": 4, "success": True, "components": {"accuracy": 1}},
            },
        )
        self.assertIsNone(event.details["bindings"][-1]["interaction_id"])
        exported = session.export_trajectory()
        self.assertFalse(exported.metadata["trainable"])
        self.assertEqual(len(exported.rewards), 1)
        # Mutating caller-owned score containers cannot change the recorded
        # score, interaction views or the idempotency comparison snapshot.
        score.components["accuracy"] = 0
        self.assertEqual(session.rewards[0].components["accuracy"], 1)
        with self.assertRaisesRegex(AgentContractError, "submission_id"):
            session.submit_reward(score, submission_id="score")

    def test_episode_score_nested_validation_is_atomic(self):
        from triotune.agentrl import EpisodeScore, RewardAssignment

        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        session.finish()
        valid = RewardAssignment("interaction", 1, position=0)
        for score in (
            EpisodeScore(2, False, process_rewards={0: (1,), 8: ()}),
            EpisodeScore(
                2, False, assignments=(valid, RewardAssignment("turn", 2, position=8))
            ),
            EpisodeScore(
                2, False, assignments=(valid, RewardAssignment("interaction", 2))
            ),
            EpisodeScore(2, False, assignments=(valid, RewardAssignment("token", 2))),
        ):
            before = len(session.events)
            with self.assertRaises(AgentContractError):
                session.submit_reward(score, submission_id="atomic")
            self.assertEqual(session.rewards, [])
            self.assertEqual(first.reward_submissions, [])
            self.assertEqual(second.reward_submissions, [])
            self.assertEqual(len(session.events), before)
        score = EpisodeScore(2, True, assignments=(valid,))
        with self.assertRaises(AgentContractError):
            session.submit_reward(score, response_id=second.response_id)
        session.submit_reward(score, submission_id="atomic")
        self.assertEqual(len(session.rewards), 1)
        self.assertEqual(first.reward["value"], 1)
        self.assertIsNone(second.reward)

    def test_episode_score_outcome_stays_session_scoped(self):
        from triotune.agentrl import EpisodeScore

        session = OpenAISession()
        first = self.record(session)
        session.finish()
        session.submit_reward(EpisodeScore(2, True, {"accuracy": 1}))
        self.assertIsNone(first.reward)
        self.assertEqual(first.reward_submissions, [])
        self.assertEqual(session.rewards[0].reward, 2)

    def test_id_maps_resolve_and_rewards_accept_all_reference_types(self):
        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        session.finish()
        self.assertEqual(
            session.request_to_interaction[first.request_id], first.interaction_id
        )
        self.assertEqual(
            session.response_to_interaction[first.response_id], first.interaction_id
        )
        for reference in (
            {"interaction_id": first.interaction_id},
            {"request_id": first.request_id},
            {"response_id": first.response_id},
        ):
            self.assertIs(session.get_interaction(**reference), first)
            session.submit_reward(2, **reference)
            self.assertEqual(first.reward, 2)
            self.assertIsNone(second.reward)
        session.submit_reward(4)
        self.assertEqual(second.reward, 4)

    def test_invalid_reward_targets_are_atomic(self):
        from triotune.agentrl import RewardAssignment

        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        session.finish()
        cases = (
            (RewardAssignment("interaction", 1, position=9), {}),
            (RewardAssignment("token", 1, position=(9, 0)), {}),
            (RewardAssignment("turn", 1), {}),
            (
                RewardAssignment("interaction", 1, position=0),
                {"response_id": second.response_id},
            ),
            (
                RewardAssignment("trajectory", 1),
                {"interaction_id": first.interaction_id},
            ),
            (1, {"response_id": "foreign-response"}),
            (1, {"request_id": "foreign-request"}),
            (1, {"request_id": first.request_id, "response_id": first.response_id}),
        )
        for reward, reference in cases:
            with (
                self.subTest(reward=reward, reference=reference),
                self.assertRaises(AgentContractError),
            ):
                session.submit_reward(reward, submission_id="invalid", **reference)
        self.assertEqual(session.rewards, [])
        self.assertIsNone(first.reward)
        self.assertIsNone(second.reward)

    def test_submission_id_compares_payload_and_resolved_target(self):
        from triotune.agentrl import RewardAssignment

        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        session.finish()
        assignment = RewardAssignment("interaction", 1, source="score")
        session.submit_reward(
            assignment, submission_id="score", request_id=first.request_id
        )
        event_count = len(session.events)
        session.submit_reward(
            assignment, submission_id="score", response_id=first.response_id
        )
        self.assertEqual(len(session.events), event_count)
        for changed, target in (
            (RewardAssignment("interaction", 2, source="score"), first),
            (RewardAssignment("interaction", 1, source="other"), first),
            (
                RewardAssignment("interaction", 1, source="score", mode="advantage"),
                first,
            ),
            (assignment, second),
        ):
            with (
                self.subTest(changed=changed, target=target.interaction_id),
                self.assertRaisesRegex(AgentContractError, "submission_id"),
            ):
                session.submit_reward(
                    changed, submission_id="score", interaction_id=target.interaction_id
                )
        self.assertEqual(len(session.rewards), 1)
        self.assertEqual(len(first.reward_submissions), 1)
        self.assertIsNone(second.reward)

    def test_prefix_lineage_distinguishes_unrelated_and_repeated_requests(self):
        session = OpenAISession()
        first = self.record(session)
        repeated = self.record(session)
        unrelated = self.record(
            session, payload={"messages": [{"role": "user", "content": "other"}]}
        )
        self.assertIsNone(repeated.parent_interaction_id)
        self.assertIsNone(unrelated.parent_interaction_id)
        # The other question has a unique matching prefix even though it is
        # not required to be the most recently recorded interaction.
        self.record(session)
        child_messages = unrelated.request["messages"] + [
            unrelated.response["choices"][0]["message"],
            {"role": "user", "content": "continue"},
        ]
        child = self.record(session, payload={"messages": child_messages})
        self.assertEqual(child.parent_interaction_id, unrelated.interaction_id)
        self.assertNotEqual(child.parent_interaction_id, first.interaction_id)
        grandchild = self.record(
            session,
            payload={
                "messages": child_messages + [child.response["choices"][0]["message"]]
            },
        )
        self.assertEqual(grandchild.parent_interaction_id, child.interaction_id)

    def test_ambiguous_prefix_requires_explicit_parent(self):
        session = OpenAISession()
        first, second = self.record(session), self.record(session)
        child_payload = {
            "messages": first.request["messages"]
            + [first.response["choices"][0]["message"]]
        }
        with self.assertRaisesRegex(AgentContractError, "Ambiguous"):
            session.start_request("/v1/chat/completions", child_payload)
        child = self.record(
            session, payload=child_payload, parent=second.interaction_id
        )
        self.assertEqual(child.parent_interaction_id, second.interaction_id)
        with self.assertRaises(AgentContractError):
            session.start_request(
                "/v1/chat/completions", child_payload, parent_interaction_id="foreign"
            )

    def test_responses_lineage_accepts_full_prefix_or_previous_response_id(self):
        session = OpenAISession()
        first = self.record(
            session,
            endpoint="/v1/responses",
            payload={"input": "question"},
        )
        child = self.record(
            session,
            endpoint="/v1/responses",
            payload={
                "input": [{"role": "user", "content": "question"}]
                + first.response["output"]
            },
        )
        self.assertEqual(child.parent_interaction_id, first.interaction_id)
        explicit = self.record(
            session,
            endpoint="/v1/responses",
            payload={
                "previous_response_id": first.response_id,
                "input": [{"role": "user", "content": "branch"}],
            },
        )
        self.assertEqual(explicit.parent_interaction_id, first.interaction_id)
        for payload, parent in (
            ({"previous_response_id": "foreign", "input": []}, None),
            (
                {"previous_response_id": first.response_id, "input": []},
                child.interaction_id,
            ),
        ):
            with self.assertRaises(AgentContractError):
                session.start_request(
                    "/v1/responses", payload, parent_interaction_id=parent
                )

    def test_responses_delta_history_requires_explicit_parent(self):
        session = OpenAISession()
        root = self.record(
            session,
            endpoint="/v1/responses",
            payload={"input": [{"role": "user", "content": "root"}]},
        )
        delta = self.record(
            session,
            endpoint="/v1/responses",
            payload={
                "previous_response_id": root.response_id,
                "input": [{"role": "user", "content": "follow up"}],
            },
        )
        self.assertEqual(delta.parent_interaction_id, root.interaction_id)
        partial = {"input": delta.request["input"] + delta.response["output"]}
        unrelated = self.record(session, endpoint="/v1/responses", payload=partial)
        self.assertIsNone(unrelated.parent_interaction_id)
        # A valid root prefix may still resolve to that root, but the delta
        # candidate is never selected without an explicit reference.
        full = {
            "input": root.request["input"] + root.response["output"] + partial["input"]
        }
        inferred = self.record(session, endpoint="/v1/responses", payload=full)
        self.assertEqual(inferred.parent_interaction_id, root.interaction_id)
        explicit = self.record(
            session,
            endpoint="/v1/responses",
            payload=partial,
            parent=delta.interaction_id,
        )
        self.assertEqual(explicit.parent_interaction_id, delta.interaction_id)
        by_response = self.record(
            session,
            endpoint="/v1/responses",
            payload={"previous_response_id": delta.response_id, "input": "next"},
        )
        self.assertEqual(by_response.parent_interaction_id, delta.interaction_id)

    def test_response_mapping_rejects_foreign_and_duplicate_completion(self):
        session, foreign = OpenAISession(), OpenAISession()
        first = self.record(session)
        with self.assertRaises(AgentContractError):
            foreign.complete_request(first, {"id": "foreign"})
        second = session.start_request("/v1/chat/completions", {"messages": []})
        with self.assertRaisesRegex(AgentContractError, "unique"):
            session.complete_request(second, {"id": first.response_id})
        self.assertEqual(
            session.response_to_interaction[first.response_id], first.interaction_id
        )
        self.assertIsNone(second.response_id)
        session.complete_request(second, {"id": "distinct", "choices": []})

    def test_identity_maps_reject_unallocated_aliases_and_cross_endpoint_parents(self):
        session = OpenAISession()
        chat = self.record(session, endpoint="/v1/chat/completions")
        session.request_to_interaction["forged-request"] = chat.interaction_id
        with self.assertRaisesRegex(AgentContractError, "mappings"):
            session.get_interaction(request_id="forged-request")

        session.request_to_interaction.pop("forged-request")
        session.response_to_interaction["forged-response"] = chat.interaction_id
        with self.assertRaisesRegex(AgentContractError, "mappings"):
            session.get_interaction(response_id="forged-response")

        session.response_to_interaction.pop("forged-response")
        with self.assertRaisesRegex(AgentContractError, "same API endpoint"):
            session.start_request(
                "/v1/responses",
                {"input": []},
                parent_interaction_id=chat.interaction_id,
            )

    def test_terminal_sessions_reject_late_tool_results(self):
        session, foreign = OpenAISession(), OpenAISession()
        interaction = self.record(session)
        result = ToolResult("call", "add", "3")
        with self.assertRaises(AgentContractError):
            foreign.record_tool_results(interaction, [result])
        session.record_tool_results(interaction, [result])
        session.finish()
        session.release()
        with self.assertRaises(AgentContractError):
            session.record_tool_results(interaction, [result])
        self.assertEqual(len(interaction.tool_results), 1)

    def test_lifecycle_blocks_inflight_work_and_open_exports(self):
        session = OpenAISession()
        interaction = session.start_request("/v1/chat/completions", {"messages": []})
        for operation in (
            session.finish,
            session.export_trajectory,
            session.release,
            lambda: session.submit_reward(1),
        ):
            with self.assertRaisesRegex(AgentContractError, "active"):
                operation()
        session.complete_request(interaction, {"id": "done", "choices": []})
        with self.assertRaises(AgentContractError):
            session.export_trajectory()
        session.running = True
        with self.assertRaises(AgentContractError):
            session.finish()
        session.running = False
        session.finish()
        event_count = len(session.events)
        session.finish()
        self.assertEqual(len(session.events), event_count)
        session.submit_reward(1)
        session.release()
        self.assertTrue(session.export_trajectory().metadata["released"])
        for operation in (
            lambda: session.finish("failed"),
            lambda: session.submit_reward(2),
            lambda: session.start_request("/v1/chat/completions", {}),
        ):
            with self.assertRaises(AgentContractError):
                operation()

    def test_failed_and_cancelled_sessions_keep_partial_evidence_without_rewards(self):
        for status in ("failed", "cancelled"):
            session = OpenAISession()
            self.record(session)
            session.finish(status, error="stopped")
            exported = session.export_trajectory()
            self.assertEqual(exported.metadata["status"], status)
            self.assertEqual(exported.metadata["error"], "stopped")
            self.assertEqual(len(exported.interactions), 1)
            with self.assertRaises(AgentContractError):
                session.submit_reward(1)
            with self.assertRaises(AgentContractError):
                session.finish()
            session.release()
            self.assertTrue(session.released)


class SDKIdentityTests(unittest.IsolatedAsyncioTestCase):
    """Collector provenance survives API calls, local tools and trace export."""

    @staticmethod
    def context(index=0):
        return ExecutionContext(
            rollout_id=f"rollout-{index}",
            episode_id=f"episode-{index}",
            group_id="collector-group",
            sample_index=index,
            group_size=4,
            snapshot_id="snapshot-3",
            policy_version=3,
            task_id="task",
            resource_owner="collector",
        )

    @staticmethod
    def adapter(observed):
        async def create(**kwargs):
            before = current_execution_context()
            await asyncio.sleep(0)
            observed.append((before, current_execution_context()))
            responses = "input" in kwargs
            items = kwargs["input"] if responses else kwargs["messages"]
            final = any(
                item.get("type") == "function_call_output" or item.get("role") == "tool"
                for item in items
            )
            identity = f"response-{before.interaction_id}"
            if responses:
                return {
                    "id": identity,
                    "status": "completed",
                    "output": (
                        [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "5"}],
                            }
                        ]
                        if final
                        else [
                            {
                                "type": "function_call",
                                "call_id": "call",
                                "name": "add",
                                "arguments": '{"a":2,"b":3}',
                            }
                        ]
                    ),
                }
            return {
                "id": identity,
                "choices": [
                    {
                        "message": (
                            {"role": "assistant", "content": "5"}
                            if final
                            else {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": "call",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"a":2,"b":3}',
                                        },
                                    }
                                ],
                            }
                        )
                    }
                ],
            }

        return OpenAIClient(
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                responses=SimpleNamespace(create=create),
            ),
            model="fake",
        )

    def test_session_preserves_complete_context_and_separates_ids(self):
        context = self.context(2)
        session = OpenAISession(
            context=context, training=True, session_id="sdk-session"
        )
        self.assertIs(session.context, context)
        self.assertTrue(session.training)
        self.assertEqual(session.rollout_id, context.rollout_id)
        self.assertEqual(session.group_id, context.group_id)
        self.assertNotIn(
            session.session_id,
            (context.rollout_id, context.episode_id, context.snapshot_id),
        )
        for field, value in (
            ("context", replace(context, group_id="other")),
            ("session_id", "other"),
            ("rollout_id", "other"),
            ("group_id", "other"),
            ("training", False),
        ):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(session, field, value)
        with self.assertRaises(FrozenInstanceError):
            session.context.sample_index = 99
        inference = OpenAISession()
        ids = (
            inference.session_id,
            inference.context.episode_id,
            inference.rollout_id,
            inference.context.snapshot_id,
        )
        self.assertEqual(len(set(ids)), 4)
        legacy = OpenAISession(
            rollout_id="legacy",
            group_id="g",
            sample_index=2,
            snapshot_id="s",
            policy_version="label",
        )
        self.assertEqual(
            (
                legacy.rollout_id,
                legacy.group_id,
                legacy.context.sample_index,
                legacy.context.snapshot_id,
                legacy.context.policy_version,
            ),
            ("legacy", "g", 2, "s", "label"),
        )

    def test_training_context_rejects_missing_identity_and_conflicting_overrides(self):
        context = self.context(1)
        with self.assertRaises(AgentContractError):
            OpenAISession(
                training=True, rollout_id="manual", snapshot_id="s", policy_version=3
            )
        for kwargs in (
            {"context": replace(context, policy_version=None), "training": True},
            {"context": replace(context, interaction_id="child")},
            {"context": {}},
            {"training": 1},
            {"session_id": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(AgentContractError):
                OpenAISession(**kwargs)
        for field, value in (
            ("rollout_id", "changed"),
            ("group_id", "changed"),
            ("sample_index", 0),
            ("sample_index", True),
            ("snapshot_id", "changed"),
            ("policy_version", 4),
            ("resource_owner", "changed"),
        ):
            with self.subTest(field=field), self.assertRaises(AgentContractError):
                OpenAISession(context=context, **{field: value})
        matched = OpenAISession(
            context=context, training=True, sample_index=1, policy_version=3
        )
        self.assertIs(matched.context, context)

    async def test_concurrent_candidate_contexts_reach_sync_and_async_tools(self):
        def make_tool(tool_contexts, asynchronous):
            def sync_add(a: float, b: float):
                tool_contexts.append(current_execution_context())
                return a + b

            async def async_add(a: float, b: float):
                before = current_execution_context()
                await asyncio.sleep(0)
                self.assertEqual(current_execution_context(), before)
                tool_contexts.append(before)
                return a + b

            return function_tool(async_add if asynchronous else sync_add, name="add")

        for api in ("chat_completions", "responses"):
            for asynchronous in (False, True):
                api_contexts, tool_contexts = [], []
                tool = make_tool(tool_contexts, asynchronous)
                agent = Agent(
                    name="calc",
                    instructions="Use add.",
                    tools=[tool],
                    client=self.adapter(api_contexts),
                    api=api,
                )
                contexts = [self.context(i) for i in range(3)]
                workflow = OpenAIWorkflow(agent)
                results = await asyncio.gather(
                    *(
                        workflow.run("2+3", context=context, training=True)
                        for context in contexts
                    )
                )
                self.assertEqual(len({result.session_id for result in results}), 3)
                self.assertEqual(len(api_contexts), 6)
                self.assertTrue(all(left == right for left, right in api_contexts))
                for result, context in zip(results, contexts, strict=True):
                    self.assertIs(result.session.context, context)
                    first, second = result.session.interactions
                    self.assertIsNone(first.parent_interaction_id)
                    self.assertEqual(second.parent_interaction_id, first.interaction_id)
                    for interaction in (first, second):
                        self.assertEqual(
                            replace(
                                interaction.context,
                                interaction_id=None,
                                parent_interaction_id=None,
                            ),
                            context,
                        )
                    self.assertIn(first.context, tool_contexts)
                    self.assertTrue(
                        all(
                            event.context == context
                            or event.context in (first.context, second.context)
                            for event in result.session.events
                        )
                    )
                    await workflow.submit_reward(result, 1, submission_id="outcome")
                    trace = workflow.export_trajectory(result)
                    self.assertEqual(trace.metadata["context"], asdict(context))
                    self.assertEqual(trace.metadata["episode_id"], context.episode_id)
                    self.assertEqual(trace.metadata["group_size"], 4)
                    self.assertEqual(trace.logical_rollout_id, context.rollout_id)
                    self.assertTrue(trace.metadata["training_context"])
                    self.assertFalse(trace.metadata["trainable"])
                self.assertIsNone(current_execution_context())

    async def test_conflicting_context_and_identity_options_fail_before_sdk_call(self):
        observed = []
        agent = Agent(
            name="calc",
            instructions="Use add.",
            tools=[add],
            client=self.adapter(observed),
        )
        context = self.context()
        for field, value in (
            ("rollout_id", "other"),
            ("episode_id", "other"),
            ("group_id", "other"),
            ("sample_index", 1),
            ("snapshot_id", "other"),
            ("policy_version", 4),
            ("group_size", 9),
            ("task_id", "other"),
            ("resource_owner", "other"),
        ):
            altered = replace(context, **{field: value})
            session = OpenAISession(context=context, training=True)
            with self.subTest(field=field), self.assertRaises(AgentContractError):
                await agent.run("2+3", context=altered, session=session, training=True)
            self.assertEqual(session.interactions, [])
        for field in ("rollout_id", "snapshot_id", "policy_version", "sample_index"):
            with self.subTest(option=field), self.assertRaises(ValueError):
                await agent.run(
                    "2+3", context=context, training=True, **{field: "override"}
                )
        with self.assertRaises(AgentContractError):
            await agent.run("2+3", training=True)
        with self.assertRaises(AgentContractError):
            await agent.run(
                "2+3",
                context=context,
                session=OpenAISession(context=context),
                training=True,
            )
        self.assertEqual(observed, [])

    async def test_workflow_snapshot_checks_provenance_without_sampling(self):
        from unittest.mock import AsyncMock

        from triotune.agentrl.runtime.sampling import SamplingSnapshot

        observed = []
        sampler = SimpleNamespace(
            generate=AsyncMock(
                side_effect=AssertionError(
                    "External API traces must not sample PyTRIO."
                )
            )
        )
        snapshot = SamplingSnapshot("snapshot-3", sampler, object(), policy_version=3)
        workflow = OpenAIWorkflow(
            Agent(
                name="calc",
                instructions="Use add.",
                tools=[add],
                client=self.adapter(observed),
            )
        )
        context = self.context()
        for altered in (
            replace(context, snapshot_id="other"),
            replace(context, policy_version=4),
            None,
        ):
            with self.subTest(altered=altered), self.assertRaises(AgentContractError):
                await workflow.run("2+3", context=altered, snapshot=snapshot)
        self.assertEqual(observed, [])
        result = await workflow.run("2+3", context=context, snapshot=snapshot)
        self.assertTrue(result.session.training)
        self.assertFalse(result.export_trajectory().metadata["trainable"])
        sampler.generate.assert_not_awaited()
        session = OpenAISession(context=context, training=True)
        result = await workflow.run("2+3", session=session, snapshot=snapshot)
        self.assertIs(result.session, session)

    async def test_existing_collector_identity_reaches_sdk_and_ppo_datums(self):
        from recipes.calculator_agent import calculator_rows, calculator_workflow
        from test_agent_episode_runner import snapshot

        from triotune import PPOConfig
        from triotune.trainer.ppo import prepare_advantages, validate_policy_versions
        from triotune.trainer.ppo.datum import batch_datums

        policy, _ = snapshot()
        policy = replace(policy, policy_version=3)
        api_contexts, sdk_results, log_contexts = [], [], []
        sdk = OpenAIWorkflow(
            Agent(
                name="judge",
                instructions="Use add.",
                tools=[add],
                client=self.adapter(api_contexts),
            )
        )

        class Logger:
            async def write(self, episode):
                log_contexts.append((current_execution_context(), episode.context))

        collector = calculator_workflow(logger=Logger())
        original = collector.runner.scorer

        class Scorer:
            async def score(self, row, episode):
                context = current_execution_context()
                self_context = episode.context
                result = await sdk.run("2+3", context=self_context, snapshot=policy)
                # The nested SDK call must restore the enclosing scorer's
                # collector context before the original reward is computed.
                if current_execution_context() != context:
                    raise AssertionError("SDK leaked interaction context into scoring.")
                score = await original.score(row, episode)
                await sdk.submit_reward(result, score, submission_id="outcome")
                sdk_results.append(result)
                return score

        collector.runner.scorer = Scorer()
        batch = await collector.collect(calculator_rows()[:1], policy, n_samples=3)
        self.assertEqual(len(batch.trajectories), 3)
        self.assertEqual(len(sdk_results), 3)
        episodes = {episode.rollout_id: episode for episode in batch.episodes}
        self.assertEqual(
            sorted(result.session.context.sample_index for result in sdk_results),
            [0, 1, 2],
        )
        for result in sdk_results:
            episode = episodes[result.session.rollout_id]
            self.assertIs(result.session.context, episode.context)
            self.assertEqual(result.session.context.group_size, 3)
            self.assertEqual(result.session.context.snapshot_id, policy.snapshot_id)
            self.assertEqual(result.session.context.policy_version, 3)
            self.assertNotEqual(result.session_id, episode.episode_id)
            self.assertFalse(result.export_trajectory().metadata["trainable"])
        self.assertTrue(all(actual == expected for actual, expected in log_contexts))
        validate_policy_versions(batch.trajectories, current_version=3)
        prepared = prepare_advantages(
            batch.trajectories, PPOConfig.for_algorithm("reinforce", adv_norm=None)
        )
        datums = batch_datums(prepared)
        self.assertEqual(len(datums), 3)
        self.assertEqual(
            {item["rollout_id"] for item in datums.provenance}, set(episodes)
        )
        self.assertEqual({item["policy_version"] for item in datums.provenance}, {3})
        self.assertIsNone(current_execution_context())
        await collector.close()

    def test_interaction_context_tampering_is_rejected_at_each_boundary(self):
        context = self.context()
        session = OpenAISession(context=context, training=True)
        interaction = SessionContractTests.record(session)
        original = interaction.context
        for field, value in (
            ("rollout_id", "other"),
            ("episode_id", "other"),
            ("group_id", "other"),
            ("sample_index", 1),
            ("snapshot_id", "other"),
            ("policy_version", 4),
            ("task_id", "other"),
            ("resource_owner", "other"),
        ):
            interaction.context = replace(original, **{field: value})
            for operation in (
                lambda: session.start_request("/v1/chat/completions", {}),
                lambda: session.record_tool_results(interaction, []),
                session.finish,
            ):
                with self.subTest(field=field), self.assertRaises(AgentContractError):
                    operation()
        interaction.context = original
        session.finish()
        interaction.context = replace(original, policy_version=4)
        with self.assertRaises(AgentContractError):
            session.export_trajectory()
        interaction.context = original
        self.assertEqual(session.export_trajectory().metadata["policy_version"], 3)

    def test_event_context_tampering_is_rejected_before_export(self):
        context = self.context()
        session = OpenAISession(context=context, training=True)
        SessionContractTests.record(session)
        event = session.events[0]
        object.__setattr__(event, "context", replace(context, rollout_id="forged"))
        with self.assertRaisesRegex(AgentContractError, "event provenance"):
            session.export_trajectory()
        object.__setattr__(event, "context", context)
        session.finish()
        self.assertEqual(
            session.export_trajectory().metadata["context"]["rollout_id"], "rollout-0"
        )

    async def test_api_and_tool_failures_restore_enclosing_context(self):
        from triotune.agentrl.types import _execution_context

        enclosing = replace(self.context(3), rollout_id="enclosing")
        token = _execution_context.set(enclosing)
        try:
            for cancel in (False, True):

                async def fail(**kwargs):
                    self.assertEqual(
                        current_execution_context().rollout_id, "rollout-0"
                    )
                    raise asyncio.CancelledError()

                # API cancellation has the same provenance cleanup as transport
                # failure; use a separate function for a normal exception.
                async def error(**kwargs):
                    raise ConnectionError("offline")

                raw = SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=fail if cancel else error)
                    )
                )
                agent = Agent(
                    name="fail",
                    instructions="Answer.",
                    client=OpenAIClient(raw, model="fake"),
                )
                session = OpenAISession(context=self.context(), training=True)
                with self.assertRaises(
                    asyncio.CancelledError if cancel else ConnectionError
                ):
                    await agent.run("x", session=session, training=True)
                self.assertEqual(current_execution_context(), enclosing)
                self.assertEqual(session.status, "cancelled" if cancel else "failed")
                self.assertEqual(
                    session.export_trajectory().metadata["context"],
                    asdict(self.context()),
                )
        finally:
            _execution_context.reset(token)

    async def test_tool_failure_and_cancellation_keep_candidate_identity(self):
        from triotune.agentrl import ToolSystemError

        def make_tool(cancel, observed):
            @function_tool(name="add")
            async def fail(a: float, b: float):
                observed.append(current_execution_context())
                if cancel:
                    raise asyncio.CancelledError()
                raise RuntimeError("tool failed")

            return fail

        for cancel in (False, True):
            tool_contexts, api_contexts = [], []
            context = self.context(1)
            session = OpenAISession(context=context, training=True)
            agent = Agent(
                name="calc",
                instructions="Use add.",
                tools=[make_tool(cancel, tool_contexts)],
                client=self.adapter(api_contexts),
            )
            with self.assertRaises(
                asyncio.CancelledError if cancel else ToolSystemError
            ):
                await agent.run("2+3", session=session, training=True)
            self.assertEqual(tool_contexts, [session.interactions[0].context])
            self.assertEqual(
                replace(
                    tool_contexts[0], interaction_id=None, parent_interaction_id=None
                ),
                context,
            )
            self.assertEqual(session.status, "cancelled" if cancel else "failed")
            trace = session.export_trajectory(
                metadata={"context": {}, "sample_index": 99, "trainable": True}
            )
            self.assertEqual(trace.metadata["context"], asdict(context))
            self.assertEqual(trace.metadata["sample_index"], 1)
            self.assertFalse(trace.metadata["trainable"])
            self.assertIsNone(current_execution_context())


class ProxyAllocationTests(unittest.IsolatedAsyncioTestCase):
    """Direct handlers share the same capacity cleanup as the HTTP routes."""

    async def test_invalid_requests_reclaim_only_new_empty_sessions(self):
        async def create(**kwargs):
            return {
                "id": "done",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }

        adapter = OpenAIClient(
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                responses=SimpleNamespace(create=create),
            ),
            model="fake",
        )
        proxy = OpenAIProxyServer(adapter, max_sessions=1)
        for handler, payload in (
            (proxy.chat_completions, {"messages": [], "stream": True}),
            (proxy.responses, {"input": [], "background": True}),
            (proxy.chat_completions, {"messages": [], "n": 2}),
            (proxy.responses, {"input": [], "extra_headers": []}),
            (proxy.chat_completions, {}),
            (proxy.responses, {"input": [], "session": "invalid"}),
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaises((ValueError, TypeError)),
            ):
                await handler(payload, session_id="correctable")
            self.assertEqual(proxy.sessions, {})
            self.assertEqual(len(proxy._released_session_ids), 0)
        await proxy.chat_completions({"messages": []}, session_id="valid")
        existing = proxy.sessions["valid"]
        with self.assertRaises(ValueError):
            await proxy.responses({"input": [], "stream": True}, session_id="valid")
        self.assertIs(proxy.sessions["valid"], existing)
        self.assertEqual(len(existing.interactions), 1)
        self.assertEqual(existing.status, "open")
        proxy.finish_session("valid")
        proxy.release_session("valid")

    async def test_released_id_retention_is_bounded_fifo(self):
        proxy = OpenAIProxyServer(None, max_sessions=1, max_released_session_ids=2)
        old_trace = None
        for name in ("one", "two", "three"):
            session = proxy._session(name)
            SessionContractTests.record(session)
            proxy.finish_session(name)
            released = proxy.release_session(name)
            if name == "one":
                old_trace = released
            self.assertLessEqual(len(proxy._released_session_ids), 2)
            self.assertEqual(proxy.sessions, {})
        self.assertEqual(list(proxy._released_session_ids), ["two", "three"])
        for name in ("two", "three"):
            with self.assertRaisesRegex(ValueError, "released"):
                proxy._session(name)
        fresh = proxy._session("one")
        self.assertIsNot(fresh, old_trace)
        self.assertEqual(fresh.status, "open")
        self.assertTrue(old_trace.released)
        self.assertTrue(old_trace.export_trajectory().metadata["released"])
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                OpenAIProxyServer(None, max_released_session_ids=invalid)


class PyTRIOBackendTests(unittest.IsolatedAsyncioTestCase):
    """The OpenAI façade preserves fixed PyTRIO sampling evidence."""

    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            del kwargs
            values = []
            for message in messages:
                values.extend(ord(char) for char in message["content"])
                values.append(1)
            if add_generation_prompt:
                values.extend((2, 3))
            return values if tokenize else ""

        def decode(self, tokens, *, skip_special_tokens=False):
            return "".join(
                "" if skip_special_tokens and token in {1, 2, 3} else chr(token)
                for token in tokens
            )

    class SamplingClient:
        def __init__(self, *, logprobs=True, output="answer"):
            self.calls = []
            self.logprobs = logprobs
            self.output = output

        async def sample_async(self, *, prompt, num_samples, sampling_params):
            self.calls.append((prompt.to_ints(), num_samples, sampling_params))
            tokens = [ord(char) for char in self.output] + [1]
            reason = "stop"
            if (
                sampling_params.max_tokens is not None
                and len(tokens) > sampling_params.max_tokens
            ):
                tokens = tokens[: sampling_params.max_tokens]
                reason = "length"
            return SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        tokens=tokens,
                        logprobs=([-0.1] * len(tokens) if self.logprobs else None),
                        stop_reason=reason,
                    )
                ],
                input_tokens=len(prompt.to_ints()),
                output_tokens=len(tokens),
            )

    @classmethod
    def snapshot(cls, *, logprobs=True, output="answer"):
        tokenizer = cls.Tokenizer()
        client = cls.SamplingClient(logprobs=logprobs, output=output)
        sampler = PyTRIOAgentSampler(client, tokenizer)
        snapshot = SamplingSnapshot(
            "pytrio-snapshot", sampler, tokenizer, policy_version=7
        )
        return snapshot, client

    @staticmethod
    def context(snapshot_id="pytrio-snapshot", policy_version=7):
        return ExecutionContext(
            rollout_id="pytrio-rollout",
            episode_id="pytrio-episode",
            group_id="pytrio-group",
            sample_index=0,
            group_size=1,
            snapshot_id=snapshot_id,
            policy_version=policy_version,
            resource_owner="test",
        )

    async def test_snapshot_backend_forwards_sampling_and_preserves_raw_arrays(self):
        snapshot, sampling_client = self.snapshot()
        backend = PyTRIOBackend(snapshot, model="fake-model", max_tokens=12)
        adapter = OpenAIClient(backend=backend)
        session = OpenAISession(context=self.context(), training=True)
        response = await adapter.create_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            session=session,
            max_tokens=8,
            seed=13,
            temperature=1,
            top_p=1.0,
        )
        prompt, count, params = sampling_client.calls[0]
        self.assertEqual(prompt, [*map(ord, "hello"), 1, 2, 3])
        self.assertEqual(count, 1)
        self.assertEqual(params.max_tokens, 8)
        self.assertEqual(params.seed, 13)
        self.assertEqual(params.temperature, 1)
        self.assertEqual(params.top_p, 1)
        interaction = session.interactions[0]
        self.assertEqual(interaction.input_ids, tuple(prompt))
        self.assertEqual(interaction.output_ids, tuple(ord(c) for c in "answer") + (1,))
        self.assertEqual(interaction.logprobs, (-0.1,) * len(interaction.output_ids))
        self.assertEqual(interaction.stop_reason, "stop")
        self.assertEqual(interaction.sampling_source, "pytrio")
        self.assertEqual(response["choices"][0]["message"]["content"], "answer")
        session.finish()
        exported = session.export_trajectory()
        self.assertEqual(exported.interactions[0]["sampling_source"], "pytrio")
        self.assertEqual(exported.interactions[0]["input_ids"], prompt)

    async def test_responses_input_uses_same_snapshot_backend(self):
        snapshot, sampling_client = self.snapshot(output="response")
        adapter = OpenAIClient(backend=PyTRIOBackend(snapshot, model="fake-model"))
        session = OpenAISession(context=self.context(), training=True)
        response = await adapter.create_response(
            input="question", session=session, max_output_tokens=16
        )
        self.assertEqual(response["status"], "completed")
        self.assertEqual(sampling_client.calls[0][0], [*map(ord, "question"), 1, 2, 3])

    async def test_agent_uses_bound_backend_without_changing_session_provenance(self):
        snapshot, _ = self.snapshot(output="final")
        context = self.context()
        agent = Agent(
            name="pytrio-agent",
            instructions="Answer briefly.",
            model="fake-model",
            client=OpenAIClient(backend=PyTRIOBackend(snapshot, model="fake-model")),
        )
        result = await agent.run("question", context=context, training=True)
        self.assertEqual(result.text, "final")
        self.assertEqual(result.session.status, "completed")
        self.assertEqual(result.session.context, context)
        self.assertEqual(
            result.session.interactions[0].sampling_source,
            "pytrio",
        )

    async def test_proxy_factory_can_supply_collector_session_for_pytrio_backend(self):
        snapshot, sampling_client = self.snapshot(output="proxy")
        context = self.context()
        proxy = OpenAIProxyServer(
            backend=PyTRIOBackend(snapshot, model="fake-model"),
            session_factory=lambda session_id: OpenAISession(
                context=context, training=True, session_id=session_id
            ),
        )
        response = await proxy.chat_completions(
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "question"}],
            },
            session_id="proxy-session",
        )
        self.assertEqual(response["choices"][0]["message"]["content"], "proxy")
        self.assertEqual(len(sampling_client.calls), 1)
        proxy.finish_session("proxy-session")
        self.assertEqual(
            proxy.sessions["proxy-session"].interactions[0].sampling_source,
            "pytrio",
        )

    async def test_training_parameters_model_and_snapshot_version_are_strict(self):
        snapshot, sampling_client = self.snapshot()
        backend = PyTRIOBackend(snapshot, model="bound-model")
        cases = (
            {"model": "other"},
            {"temperature": 0.7},
            {"top_p": 0.9},
            {"stream": True},
            {"n": 2},
            {"stop": "different"},
        )
        for options in cases:
            session = OpenAISession(context=self.context(), training=True)
            adapter = OpenAIClient(backend=backend)
            prevalidated = (
                options.get("stream") or options.get("n", 1) != 1 or "stop" in options
            )
            expected_error = ValueError if prevalidated else AgentContractError
            with self.subTest(options=options), self.assertRaises(expected_error):
                await adapter.create_chat_completion(
                    messages=[{"role": "user", "content": "x"}],
                    session=session,
                    **options,
                )
            self.assertEqual(session.status, "open")
        mismatch = OpenAISession(
            context=self.context(snapshot_id="other"), training=True
        )
        with self.assertRaises(AgentContractError):
            await OpenAIClient(backend=backend).create_chat_completion(
                messages=[{"role": "user", "content": "x"}], session=mismatch
            )
        self.assertEqual(sampling_client.calls, [])

    async def test_missing_logprobs_and_truncation_remain_explicit_failures(self):
        snapshot, _ = self.snapshot(logprobs=False)
        session = OpenAISession(context=self.context(), training=True)
        with self.assertRaises(AgentContractError):
            await OpenAIClient(
                backend=PyTRIOBackend(snapshot, model="fake-model")
            ).create_chat_completion(
                messages=[{"role": "user", "content": "x"}], session=session
            )
        self.assertEqual(session.status, "failed")
        self.assertIsNone(session.interactions[0].output_ids)

        snapshot, _ = self.snapshot(output="long-output")
        session = OpenAISession(context=self.context(), training=True)
        response = await OpenAIClient(
            backend=PyTRIOBackend(snapshot, model="fake-model")
        ).create_chat_completion(
            messages=[{"role": "user", "content": "x"}],
            session=session,
            max_tokens=2,
        )
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.interactions[0].stop_reason, "length")
        self.assertEqual(len(session.interactions[0].output_ids), 2)


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
            if self.mode in {"connection_error", "timeout"}:
                from openai import APIConnectionError, APITimeoutError

                if self.mode == "timeout":
                    raise APITimeoutError(request=request)
                raise APIConnectionError(request=request)
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
            context = SDKIdentityTests.context(1)
            result = (
                await agent.run("2+3", logprobs=True, context=context, training=True)
                if api == "chat_completions"
                else await agent.run("2+3", context=context, training=True)
            )
            self.assertEqual(result.text, "5")
            first, second = result.session.interactions
            self.assertEqual(first.server_request_id, "server-request")
            self.assertEqual(first.output_tokens, 0)
            self.assertEqual(len(first.tool_calls), 1)
            self.assertEqual(first.tool_results[0]["call_id"], "call-1")
            self.assertEqual(observed[-1], first.context)
            self.assertEqual(second.parent_interaction_id, first.interaction_id)
            self.assertIs(result.session.context, context)
            self.assertEqual(self.contexts[-1], second.context)
            self.assertEqual(
                replace(
                    second.context, interaction_id=None, parent_interaction_id=None
                ),
                context,
            )
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
            self.assertEqual(
                session.status, "cancelled" if mode == "cancel" else "failed"
            )
            self.assertEqual(len(session.export_trajectory().interactions), 1)
            with self.assertRaises(AgentContractError):
                session.submit_reward(1)
            session.release()
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
        self.assertEqual(session.status, "failed")
        self.assertEqual(len(session.export_trajectory().interactions), 1)
        await agent.close()
        self.assertFalse(self.sdk.is_closed())

    async def test_incomplete_response_is_exportable_but_not_successful(self):
        async def create(**kwargs):
            return {
                "id": "incomplete",
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
            }

        agent = Agent(
            name="incomplete",
            instructions="Answer.",
            api="responses",
            client=OpenAIClient(
                SimpleNamespace(responses=SimpleNamespace(create=create)),
                model="fake",
            ),
        )
        session = OpenAISession()
        with self.assertRaises(ActionParseError):
            await agent.run("question", session=session)
        self.assertEqual(session.status, "failed")
        self.assertIn("ActionParseError", session.error)
        self.assertFalse(session.export_trajectory().metadata["trainable"])

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
            self.assertEqual(session.status, "cancelled" if cancelled else "failed")
            self.assertEqual(
                session.export_trajectory().metadata["status"], session.status
            )
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
        proxy.finish_session("session-a")
        session = proxy.sessions["session-a"]
        session.submit_reward(1, response_id="chat_tool")
        self.assertEqual(len(session.export_trajectory().interactions), 1)
        proxy.release_session("session-a")
        self.assertNotIn("session-a", proxy.sessions)
        self.assertTrue(session.released)
        with self.assertRaisesRegex(ValueError, "released"):
            proxy._session("session-a")

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

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"),
        "Install triotune[openai-proxy] for HTTP routes.",
    )
    async def test_proxy_failure_responses_expose_releasable_session_identity(self):
        import httpx

        proxy = OpenAIProxyServer(self.adapter, max_sessions=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()),
            base_url="http://proxy",
        ) as client:
            for mode, status in (
                ("error", 503),
                ("connection_error", 502),
                ("timeout", 504),
            ):
                self.mode = mode
                for endpoint, payload in (
                    (
                        "/v1/chat/completions",
                        {
                            "model": "fake",
                            "messages": [{"role": "user", "content": "question"}],
                        },
                    ),
                    ("/v1/responses", {"model": "fake", "input": "question"}),
                ):
                    with self.subTest(mode=mode, endpoint=endpoint):
                        failed = await client.post(endpoint, json=payload)
                        self.assertEqual(failed.status_code, status)
                        session_id = failed.headers["X-Triotune-Session-Id"]
                        session = proxy.sessions[session_id]
                        self.assertEqual(session.status, "failed")
                        interaction = session.get_interaction(
                            request_id=failed.headers["X-Client-Request-Id"]
                        )
                        self.assertEqual(
                            interaction.interaction_id,
                            failed.headers["X-Triotune-Interaction-Id"],
                        )
                        self.assertEqual(
                            failed.headers["X-Request-Id"], interaction.request_id
                        )
                        self.assertEqual(
                            len(session.export_trajectory().interactions), 1
                        )
                        proxy.release_session(session_id)
                        self.assertEqual(proxy.sessions, {})

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"),
        "Install triotune[openai-proxy] for HTTP routes.",
    )
    async def test_proxy_http_validation_reclaims_capacity_without_stale_request_ids(
        self,
    ):
        import httpx

        proxy = OpenAIProxyServer(self.adapter, max_sessions=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()),
            base_url="http://proxy",
        ) as client:
            for endpoint, payload in (
                (
                    "/v1/chat/completions",
                    {"model": "fake", "messages": [], "stream": True},
                ),
                ("/v1/responses", {"model": "fake", "input": [], "background": True}),
                ("/v1/chat/completions", {"model": "fake"}),
                (
                    "/v1/responses",
                    {"model": "fake", "input": [], "previous_response_id": "unknown"},
                ),
            ):
                invalid = await client.post(endpoint, json=payload)
                self.assertEqual(invalid.status_code, 400)
                self.assertEqual(proxy.sessions, {})
                self.assertNotIn("X-Client-Request-Id", invalid.headers)
            headers = {"X-Triotune-Session-Id": "retained"}
            payload = {
                "model": "fake",
                "messages": [{"role": "user", "content": "question"}],
            }
            valid = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
            self.assertEqual(valid.status_code, 200)
            invalid = await client.post(
                "/v1/chat/completions",
                json={**payload, "stream": True},
                headers=headers,
            )
            self.assertEqual(invalid.status_code, 400)
            self.assertEqual(invalid.headers["X-Triotune-Session-Id"], "retained")
            self.assertNotIn("X-Client-Request-Id", invalid.headers)
            self.assertNotIn("X-Triotune-Interaction-Id", invalid.headers)
            self.assertEqual(len(proxy.sessions["retained"].interactions), 1)
            self.assertEqual(proxy.sessions["retained"].status, "open")

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"),
        "Install triotune[openai-proxy] for HTTP routes.",
    )
    async def test_proxy_parent_header_and_terminal_session_are_enforced(self):
        import httpx

        proxy = OpenAIProxyServer(self.adapter)
        headers = {"X-Triotune-Session-Id": "branch"}
        payload = {
            "model": "fake",
            "messages": [{"role": "user", "content": "question"}],
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
            self.assertEqual(first.status_code, 200)
            parent = first.headers["X-Triotune-Interaction-Id"]
            # Use a different completion ID and unrelated input to exercise
            # explicit association independently from inferred message prefixes.
            child_payload = {
                "model": "fake",
                "messages": [
                    {"role": "user", "content": "other"},
                    {"role": "tool", "tool_call_id": "call-1", "content": "3"},
                ],
            }
            child = await client.post(
                "/v1/chat/completions",
                json=child_payload,
                headers={**headers, "X-Triotune-Parent-Interaction-Id": parent},
            )
            self.assertEqual(child.status_code, 200)
            session = proxy.sessions["branch"]
            self.assertEqual(session.interactions[1].parent_interaction_id, parent)
            proxy.finish_session("branch")
            before = len(self.requests)
            late = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
            self.assertEqual(late.status_code, 400)
            self.assertEqual(len(self.requests), before)

    @unittest.skipUnless(
        importlib.util.find_spec("fastapi"),
        "Install triotune[openai-proxy] for HTTP routes.",
    )
    async def test_proxy_unexpected_sdk_failure_retains_identity_and_diagnostics(self):
        import httpx

        async def create(**kwargs):
            raise RuntimeError("internal diagnostic")

        adapter = OpenAIClient(
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            ),
            model="fake",
        )
        proxy = OpenAIProxyServer(adapter)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.create_app()),
            base_url="http://proxy",
        ) as client:
            failed = await client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(failed.status_code, 500)
        self.assertNotIn("internal diagnostic", failed.text)
        session_id = failed.headers["X-Triotune-Session-Id"]
        session = proxy.sessions[session_id]
        interaction = session.get_interaction(
            request_id=failed.headers["X-Client-Request-Id"]
        )
        self.assertIn("internal diagnostic", interaction.error)
        self.assertEqual(session.status, "failed")
        self.assertEqual(
            session.export_trajectory().interactions[0]["request_id"],
            interaction.request_id,
        )
        proxy.release_session(session_id)
        self.assertEqual(proxy.sessions, {})

    async def test_proxy_cannot_finish_export_or_release_inflight_request(self):
        started, unblock = asyncio.Event(), asyncio.Event()

        async def create(**kwargs):
            started.set()
            await unblock.wait()
            return {
                "id": "done",
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
            }

        adapter = OpenAIClient(
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            ),
            model="fake",
        )
        proxy = OpenAIProxyServer(adapter)
        task = asyncio.create_task(
            proxy.chat_completions(
                {"messages": [{"role": "user", "content": "x"}]}, session_id="busy"
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            for operation in (
                lambda: proxy.finish_session("busy"),
                lambda: proxy.release_session("busy"),
                proxy.sessions["busy"].export_trajectory,
            ):
                with self.assertRaises(AgentContractError):
                    operation()
            self.assertIn("busy", proxy.sessions)
        finally:
            unblock.set()
            await task
        proxy.finish_session("busy")
        proxy.release_session("busy")
        self.assertNotIn("busy", proxy.sessions)
