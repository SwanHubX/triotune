"""Small adapters around public PyTRIO client methods.

This module is algorithm-independent: GRPO, PPO, DPO and supervised loops all
need the same handling for synchronous API futures and ``*_async`` methods.
No trainer-specific loss or Datum field is defined here.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any


async def wait_async(value: Any) -> Any:
    """Await a coroutine/future, or resolve a synchronous API future."""

    # PyTRIO training methods have two async stages: awaiting the method
    # coroutine submits the operation and returns an awaitable APIFuture, which
    # must then be awaited for the remote result. Keep unwrapping awaitables so
    # the second stage is never resolved through its blocking ``result()``.
    while inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if callable(result_async):
        value = result_async()
        while inspect.isawaitable(value):
            value = await value
        return value
    result = getattr(value, "result", None)
    if callable(result):
        # Compatibility futures without ``__await__`` may block in result().
        # Resolve those in a worker thread to keep the caller's loop responsive.
        value = await asyncio.to_thread(result)
        while inspect.isawaitable(value):
            value = await value
    return value


def wait(value: Any) -> Any:
    """Synchronous counterpart to :func:`wait_async`."""

    result = getattr(value, "result", None)
    if callable(result):
        # PyTRIO's APIFuture implements ``__await__`` as well as the blocking
        # ``result()`` API.  Check for ``result`` first: treating every
        # awaitable as an async-only value would reject all synchronous
        # TrainingClient calls (forward_backward, optim_step, save_state).
        value = result()
        if inspect.isawaitable(value):
            raise RuntimeError(
                "A synchronous PyTRIO call returned an awaitable result; "
                "use the async trainer API."
            )
        return value
    if inspect.isawaitable(value):
        raise RuntimeError("An async PyTRIO method was returned; use the async trainer API.")
    return value


def call(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call a public client method and resolve a returned API future."""

    function = getattr(client, method, None)
    if function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}().")
    return wait(function(*args, **kwargs))


async def call_async(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Prefer a public ``*_async`` method and resolve its future."""

    async_function = getattr(client, f"{method}_async", None)
    if async_function is not None:
        return await wait_async(async_function(*args, **kwargs))

    sync_function = getattr(client, method, None)
    if sync_function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}() or {method}_async().")

    # Compatible clients may omit a public async variant. Run a synchronous
    # fallback in a worker thread so it cannot block the caller's event loop.
    return await asyncio.to_thread(call, client, method, *args, **kwargs)


async def submit_async(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Submit without waiting for the returned remote result."""

    function = getattr(client, f"{method}_async", None) or getattr(client, method, None)
    if function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}().")
    result = function(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def sdk_object(name: str, values: dict[str, Any], factory: Any = None) -> Any:
    """Build an SDK object, falling back to a plain mapping for test clients."""

    if factory is not None:
        return factory(**values)
    for module_name in ("pytrio", "tinker"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        cls = getattr(module, name, None)
        if cls is not None:
            return cls(**values)
    return values


def sampling_params(config: Any, factory: Any = None) -> Any:
    return sdk_object("SamplingParams", config.sampling_params, factory)


def optimizer_params(config: Any, factory: Any = None) -> Any:
    return sdk_object("AdamParams", config.optimizer_params(), factory)
