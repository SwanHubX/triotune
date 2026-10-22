"""Transient sampling retries preserve prompts and never retry invalid input."""

import asyncio
import unittest

from test_swe_runtime import Sampler, Tokenizer

from triotune.swe import PyTRIOSWEModel, RetryConfig, SWESamplingConfig
from triotune.swe.retry import is_transient, retry_call


class SWERetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_transport_retries_keep_the_same_request_and_scores(self):
        class FlakySampler(Sampler):
            calls = 0

            async def sample_async(self, **kwargs):
                self.calls += 1
                if self.calls < 3:
                    raise ConnectionError("temporary connection loss")
                return await super().sample_async(**kwargs)

        client = FlakySampler()
        model = PyTRIOSWEModel(
            client,
            Tokenizer(),
            policy_version="fixed",
            config=SWESamplingConfig(retry={"max_retries": 2, "initial_delay": 0}),
        )
        sample = await model.sample([], [])
        self.assertEqual(client.calls, 3)
        self.assertEqual(sample.request_retries, 2)
        self.assertEqual(sample.logprobs, (-0.5, -0.7))
        self.assertEqual(sample.prompt_tokens, (10, 11))

    async def test_invalid_inputs_and_authentication_are_not_retried(self):
        class AuthError(Exception):
            status_code = 401
            retryable = True

        for error in (ValueError("context"), AuthError("auth")):
            calls = 0

            async def fail(error=error):
                nonlocal calls
                calls += 1
                raise error

            with self.assertRaises(type(error)):
                await retry_call(fail, RetryConfig(initial_delay=0))
            self.assertEqual(calls, 1)
            self.assertFalse(is_transient(error))

    async def test_retry_limit_and_cancellation_are_bounded(self):
        calls = 0
        failed = asyncio.Event()

        async def fail():
            nonlocal calls
            calls += 1
            failed.set()
            raise ConnectionError("offline")

        with self.assertRaises(ConnectionError):
            await retry_call(fail, RetryConfig(max_retries=2, initial_delay=0))
        self.assertEqual(calls, 3)
        failed.clear()
        pending = asyncio.create_task(retry_call(fail, RetryConfig(initial_delay=5)))
        await failed.wait()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(calls, 4)

    def test_retry_configuration_rejects_invalid_budgets(self):
        for values in (
            {"max_retries": -1},
            {"max_retries": True},
            {"initial_delay": float("nan")},
            {"max_delay": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RetryConfig(**values)
