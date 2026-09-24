from __future__ import annotations

import asyncio
from concurrent.futures import Future
import threading
import time
import unittest

from pytrio.types.future.api_future import APIFuture

from trio_cookbook.utils.client import call_async, wait_async


class _BlockingFuture:
    def result(self) -> str:
        time.sleep(0.05)
        return "blocking-result"


class WaitAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_checkpoint_api_future_does_not_block_event_loop(self) -> None:
        future: Future[str] = Future()
        threading.Timer(0.05, future.set_result, args=("checkpoint",)).start()

        class Client:
            async def save_state_async(self, *, name: str) -> APIFuture[str]:
                self.name = name
                return APIFuture(future)

        loop_progressed = asyncio.Event()

        async def tick() -> None:
            await asyncio.sleep(0.01)
            loop_progressed.set()

        client = Client()
        tick_task = asyncio.create_task(tick())
        result = await call_async(client, "save_state", name="step-1")

        self.assertEqual(result, "checkpoint")
        self.assertEqual(client.name, "step-1")
        self.assertTrue(loop_progressed.is_set())
        await tick_task

    async def test_resolves_coroutine_returning_native_future(self) -> None:
        async def submit() -> asyncio.Future[str]:
            future = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(future.set_result, "native-result")
            return future

        self.assertEqual(await wait_async(submit()), "native-result")

    async def test_blocking_result_does_not_block_event_loop(self) -> None:
        loop_progressed = asyncio.Event()

        async def tick() -> None:
            await asyncio.sleep(0.01)
            loop_progressed.set()

        tick_task = asyncio.create_task(tick())
        result = await wait_async(_BlockingFuture())

        self.assertEqual(result, "blocking-result")
        self.assertTrue(loop_progressed.is_set())
        await tick_task


if __name__ == "__main__":
    unittest.main()
