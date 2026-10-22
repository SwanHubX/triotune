"""Bounded retries for transient model-service failures, never bad actions."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_retries: int = 2
    initial_delay: float = 0.5
    max_delay: float = 5.0

    def __post_init__(self):
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer.")
        for name in ("initial_delay", "max_delay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")

    def delay(self, retry_index):
        return min(self.max_delay, self.initial_delay * 2 ** min(retry_index, 30))


def is_transient(error: Exception) -> bool:
    """Use SDK metadata/statuses; auth, context and input errors fail fast."""
    code = getattr(error, "status_code", None)
    if code in (400, 401, 403, 404, 422):
        return False
    if isinstance(error, (ValueError, TypeError, PermissionError, FileNotFoundError)):
        return False
    return (
        isinstance(error, (ConnectionError, TimeoutError))
        or code in (408, 429, 500, 502, 503, 504)
        or getattr(error, "retryable", False) is True
    )


async def retry_call(operation, config: RetryConfig):
    """Return (value, retries performed); cancellation is never intercepted."""
    for attempt in range(config.max_retries + 1):
        try:
            return await operation(), attempt
        except Exception as error:
            if attempt == config.max_retries or not is_transient(error):
                raise
            await asyncio.sleep(config.delay(attempt))
