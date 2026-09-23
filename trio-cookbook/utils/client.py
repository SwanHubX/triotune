"""Small adapters around public PyTRIO client methods.

This module is algorithm-independent: GRPO, PPO, DPO and supervised loops all
need the same handling for synchronous API futures and ``*_async`` methods.
No trainer-specific loss or Datum field is defined here.
"""

from __future__ import annotations

import inspect
from typing import Any


async def wait_async(value: Any) -> Any:
    """Await a coroutine/future, or resolve a synchronous API future."""

    if inspect.isawaitable(value):
        value = await value
    result_async = getattr(value, "result_async", None)
    if result_async is not None:
        result = result_async()
        return await result if inspect.isawaitable(result) else result
    result = getattr(value, "result", None)
    if callable(result):
        value = result()
        if inspect.isawaitable(value):
            value = await value
    return value


def wait(value: Any) -> Any:
    """Synchronous counterpart to :func:`wait_async`."""

    if inspect.isawaitable(value):
        raise RuntimeError("An async PyTRIO method was returned; use the async trainer API.")
    result = getattr(value, "result", None)
    return result() if callable(result) else value


def call(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call a public client method and resolve a returned API future."""

    function = getattr(client, method, None)
    if function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}().")
    return wait(function(*args, **kwargs))


async def call_async(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Prefer a public ``*_async`` method and resolve its future."""

    function = getattr(client, f"{method}_async", None) or getattr(client, method, None)
    if function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}() or {method}_async().")
    return await wait_async(function(*args, **kwargs))


async def submit_async(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Submit without waiting for the returned remote result.

    This preserves PyTRIO's clock cycle: submit forward/backward, submit the
    optimizer, and only then wait for both operations.
    """

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
