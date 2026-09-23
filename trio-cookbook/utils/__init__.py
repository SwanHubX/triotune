"""Framework-agnostic PyTRIO client helpers.

Only utilities that can be reused by more than one training algorithm belong
here.  Algorithm contracts and GRPO data structures remain under
``trio-cookbook.trainer``.
"""

from .client import (
    call,
    call_async,
    optimizer_params,
    sampling_params,
    sdk_object,
    submit_async,
    wait,
    wait_async,
)

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
