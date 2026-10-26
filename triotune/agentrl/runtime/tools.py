"""Execute registered Python tools independently of model and reward logic.

Tool definitions provide schemas for prompt rendering and callables for local
execution. This executor checks function argument binding, schedules synchronous
work off the event loop, and exposes recoverable model mistakes as observations.
Timeout policy and episode termination are owned by the runner.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..types import ToolActionError, ToolCall, ToolResult, ToolSystemError


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Pair a public tool description with its local Python implementation.

    ``parameters`` is an object-shaped schema advertised to the model; it does
    not perform runtime JSON Schema validation. The function validates domain
    types and ranges and raises ToolActionError for invalid model arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]
    retry_safe: bool = False
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.name or not callable(self.function):
            raise ValueError("A tool needs a nonempty name and callable function.")
        if self.parameters.get("type") != "object":
            raise ValueError("Tool parameters must declare an object schema.")
        if type(self.retry_safe) is not bool:
            raise ValueError("retry_safe must be a boolean.")
        if self.validator is not None and not callable(self.validator):
            raise ValueError("validator must be callable or None.")

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and optionally normalize model supplied tool arguments."""
        if self.validator is None:
            return arguments
        return self.validator(arguments)


class FunctionToolExecutor:
    """Execute tools without model calls, prompt rendering, rewards or retries.

    Function signatures validate argument binding. Tools validate their own
    domain constraints and raise ToolActionError for recoverable model errors.
    Other exceptions reject the episode as an infrastructure failure.
    """

    def __init__(self, tools: list[FunctionTool]):
        """Build an unambiguous name-to-function registry for shared execution.

        Tool functions may be invoked by concurrent episodes and should keep
        per-call state local or provide their own synchronization.
        """
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique.")

    def is_retry_safe(self, call: ToolCall) -> bool:
        """Only explicitly idempotent tools may repeat after a transient error."""
        tool = self._tools.get(call.name)
        return tool is not None and tool.retry_safe

    @property
    def schemas(self) -> list[Mapping[str, Any]]:
        """Expose function descriptions without serializing Python callables.

        The surrounding conversation renderer decides how these descriptors
        appear in a model's actual chat template.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one call and retain its identity in the returned observation.

        Signature binding failures are attributable to model arguments. Errors
        raised inside the function are system failures unless the function
        explicitly uses ToolActionError. This avoids hiding implementation bugs
        as ordinary task mistakes. Cancellation continues to propagate.

        Tool implementations can read current_execution_context() to obtain
        trusted candidate and interaction identity. asyncio.to_thread preserves
        that context for synchronous functions, so model arguments never need
        to carry session IDs or other execution authority.
        """
        try:
            tool = self._tools.get(call.name)
            if tool is None:
                raise ToolActionError(f"Unknown tool: {call.name}")
            try:
                inspect.signature(tool.function).bind(**call.arguments)
            except TypeError as exc:
                raise ToolActionError(str(exc)) from exc
            try:
                arguments = tool.validate_arguments(call.arguments)
            except (TypeError, ValueError) as exc:
                raise ToolActionError(str(exc)) from exc
            if inspect.iscoroutinefunction(tool.function):
                value = await tool.function(**arguments)
            else:
                # A synchronous tool must not block other episodes. Cancelling
                # this await does not forcibly terminate a running worker thread.
                value = await asyncio.to_thread(tool.function, **arguments)
                if inspect.isawaitable(value):
                    # Callable wrappers can return an awaitable even when the
                    # callable itself is not declared with async def.
                    value = await value
            content = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, allow_nan=False)
            )
            return ToolResult(call.call_id, call.name, content)
        except ToolActionError as exc:
            # Feeding a structured error back to the model preserves an
            # opportunity to recover within the remaining episode budget.
            return ToolResult(call.call_id, call.name, str(exc), is_error=True)
        except ToolSystemError:
            raise
        except Exception as exc:
            raise ToolSystemError(
                f"Tool {call.name} failed: {exc}",
                retryable=isinstance(exc, (ConnectionError, TimeoutError, OSError)),
            ) from exc
