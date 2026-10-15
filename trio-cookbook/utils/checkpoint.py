"""Cross-cutting checkpoint names and validated sync/async saves.

Save cadence and state publication belong to each algorithm. In particular,
SFT submits queued saves in its trainer to preserve the SDK's request ordering.
"""

from typing import Any

from ..api.client_api import call, call_async


def checkpoint_name(algorithm: str, step: int) -> str:
    return f"{algorithm}-step-{step:08d}"


def checkpoint_path(saved: Any, *, method: str = "save_state") -> str:
    path = getattr(saved, "path", None)
    if not isinstance(path, str) or not path:
        raise TypeError(f"PyTRIO {method}() returned no checkpoint path.")
    return path


def save_checkpoint(client: Any, algorithm: str, step: int) -> str:
    saved = call(client, "save_state", name=checkpoint_name(algorithm, step))
    return checkpoint_path(saved)


async def save_checkpoint_async(
    client: Any, algorithm: str, step: int, *, error_method: str = "save_state_async",
) -> str:
    saved = await call_async(client, "save_state", name=checkpoint_name(algorithm, step))
    return checkpoint_path(saved, method=error_method)
