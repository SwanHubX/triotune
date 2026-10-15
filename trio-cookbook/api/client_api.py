"""Public adapters around PyTRIO training and sampling client methods.

Every algorithm uses a training client, while rollout algorithms also use a
sampling client. Both expose the same future and async conventions, so those
mechanics belong to the API contract layer. Sampling-client lifetime does
not: each algorithm decides when a new checkpoint must be exposed through
``save_weights_and_get_sampling_client`` and passes that client to these helpers.

No trainer-specific loss, Datum field, checkpoint cadence, or client-refresh
policy is defined here.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any


async def wait_async(value: Any) -> Any:
    """Resolve any async or compatibility future without blocking the event loop.

    Args:
        value: A coroutine, native awaitable, PyTRIO ``APIFuture``, object with
            ``result_async()``, or compatibility object with blocking ``result()``.

    Returns:
        The fully resolved remote result, including APIs with two async stages.
    """

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
    """Resolve one synchronous PyTRIO future.

    Args:
        value: A concrete result or object with ``result()``.

    Returns:
        The concrete result.

    Raises:
        RuntimeError: If resolving the value would require an event loop.
    """

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
    """Call and synchronously resolve a method on any PyTRIO client.

    Args:
        client: Training or sampling client instance.
        method: Public method name without an ``_async`` suffix.
        *args: Positional arguments forwarded unchanged.
        **kwargs: Keyword arguments forwarded unchanged.
    """

    function = getattr(client, method, None)
    if function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}().")
    return wait(function(*args, **kwargs))


async def call_async(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call and resolve a client method without blocking the caller's loop.

    Args:
        client: Training or sampling client instance.
        method: Base method name; ``<method>_async`` is preferred when present.
        *args: Positional arguments forwarded unchanged.
        **kwargs: Keyword arguments forwarded unchanged.

    A synchronous fallback runs in a worker thread for SDK versions that do
    not expose the async variant.
    """

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
    """Submit a remote operation while leaving its result future unresolved.

    Args:
        client: Training client instance.
        method: Forward/backward, optimizer, or save method to submit.
        *args: Positional arguments forwarded unchanged.
        **kwargs: Keyword arguments forwarded unchanged.

    Returns:
        The SDK future representing the queued remote operation. SFT uses this
        distinction to pipeline ordered training requests.
    """

    function = getattr(client, f"{method}_async", None) or getattr(client, method, None)
    if function is None:
        raise AttributeError(f"PyTRIO client does not expose {method}().")
    result = function(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def sdk_object(name: str, values: dict[str, Any], factory: Any = None) -> Any:
    """Build a small SDK value object without coupling tests to one SDK release.

    Args:
        name: Public SDK class name, such as ``SamplingParams`` or ``AdamParams``.
        values: Constructor keyword values.
        factory: Explicit class/factory used by tests or compatibility callers.

    Returns:
        The SDK object, or the original mapping when no supported SDK is installed.
    """

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


def sampling_params(config: Any, factory: Any = None, *, stop: Any = None) -> Any:
    """Materialize sampling parameters from a generation-aware config.

    Args:
        config: Object exposing an SDK-neutral ``sampling_params`` mapping.
        factory: Optional ``SamplingParams`` constructor.
        stop: Model-specific stop strings added by the owning algorithm.

    Returns:
        A PyTRIO ``SamplingParams`` instance or compatibility mapping.
    """

    values = dict(config.sampling_params)
    if stop is not None:
        values["stop"] = stop
    return sdk_object("SamplingParams", values, factory)


def optimizer_params(config: Any, factory: Any = None) -> Any:
    """Materialize remote Adam parameters from a trainer config.

    Args:
        config: Object exposing ``optimizer_params()``.
        factory: Optional ``AdamParams`` constructor.

    Returns:
        A PyTRIO ``AdamParams`` instance or compatibility mapping.
    """

    return sdk_object("AdamParams", config.optimizer_params(), factory)


__all__ = [
    "call",
    "call_async",
    "optimizer_params",
    "sampling_params",
    "sdk_object",
    "submit_async",
    "wait",
    "wait_async",
]
