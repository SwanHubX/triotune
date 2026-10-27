"""Shared runtime boundary contracts used by episode and SDK execution."""

import unittest
from types import SimpleNamespace

from triotune.agentrl import ExecutionContext, RuntimeBoundary
from triotune.agentrl.config import RetryPolicy
from triotune.agentrl.integrations.openai import Agent, OpenAIClient, function_tool
from triotune.agentrl.types import ToolSystemError


class RuntimeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Retry, classifier, and resource ownership behavior stays explicit."""

    def record(self):
        return SimpleNamespace(
            context=ExecutionContext("r", "e", "g", 0, "s"),
            events=[],
            retries=[],
            export_metadata={},
        )

    async def test_retry_records_attempts_but_does_not_retry_non_idempotent_work(self):
        boundary = RuntimeBoundary(retry=RetryPolicy(max_attempts=2, retry_tools=True))
        record = self.record()
        attempts = 0

        async def flaky():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ToolSystemError("temporary", retryable=True)
            return "ok"

        self.assertEqual(await boundary.operation(record, "tool", 0, flaky), "ok")
        self.assertEqual(attempts, 2)
        self.assertEqual(len(record.retries), 1)

        attempts = 0
        with self.assertRaises(ToolSystemError):
            await boundary.operation(record, "tool", 0, flaky, retry_safe=False)
        self.assertEqual(attempts, 1)

    async def test_failure_classifier_and_owned_resources_are_bounded(self):
        calls = []

        class Resource:
            async def start(self):
                calls.append("start")

            async def close(self):
                calls.append("close")

        def classify(stage, error, record):
            return "classified_failure", f"{stage}:{error}"

        boundary = RuntimeBoundary(failure_classifier=classify, resource_owner="owner")
        resource = Resource()
        await boundary.start(resource)
        await boundary.start(resource)
        record = self.record()
        self.assertEqual(
            boundary.classify("api", RuntimeError("offline"), record),
            ("classified_failure", "api:offline"),
        )
        self.assertTrue(record.export_metadata["system_failure"])
        await boundary.close()
        await boundary.close()
        self.assertEqual(calls, ["start", "close"])

    async def test_partial_resource_startup_is_rolled_back_and_can_retry(self):
        calls = []

        class Resource:
            def __init__(self, name, fail=False):
                self.name, self.fail = name, fail

            async def start(self):
                calls.append(f"start:{self.name}")
                if self.fail:
                    raise RuntimeError("startup failed")

            async def close(self):
                calls.append(f"close:{self.name}")

        first, second = Resource("first"), Resource("second", fail=True)
        boundary = RuntimeBoundary()
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            await boundary.start(first, second)
        self.assertFalse(boundary.started)
        self.assertEqual(calls, ["start:first", "start:second", "close:first"])
        second.fail = False
        await boundary.start(first, second)
        await boundary.close()
        self.assertEqual(
            calls[-4:],
            ["start:first", "start:second", "close:second", "close:first"],
        )

    async def test_sdk_agent_retries_only_explicitly_safe_tools(self):
        calls = []

        @function_tool(retry_safe=True, name="unstable")
        def unstable() -> str:
            calls.append("tool")
            if len(calls) == 1:
                raise ConnectionError("temporary")
            return "done"

        async def create(**payload):
            final = any(item.get("role") == "tool" for item in payload["messages"])
            return {
                "id": f"response-{len(calls)}-{final}",
                "choices": [
                    {
                        "message": (
                            {"role": "assistant", "content": "complete"}
                            if final
                            else {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": "call",
                                        "function": {
                                            "name": "unstable",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        )
                    }
                ],
            }

        client = OpenAIClient(
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            ),
            model="fake",
        )
        agent = Agent(
            name="retry",
            instructions="Use tool",
            tools=[unstable],
            client=client,
            tool_retry=RetryPolicy(
                max_attempts=2,
                initial_delay=0,
                max_delay=0,
                retry_tools=True,
            ),
        )
        result = await agent.run("go")
        self.assertEqual(result.text, "complete")
        self.assertEqual(calls, ["tool", "tool"])
        self.assertEqual(len(result.session.retries), 1)
        self.assertEqual(len(result.session.interactions[0].tool_results), 1)
