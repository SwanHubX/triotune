"""Parse native OpenAI tool calls and format their observations.

Chat Completions and Responses expose different wire representations for the
same logical action.  The parser normalizes both forms into ``AgentAction`` and
``ToolCall`` records while preserving provider call IDs; the formatters perform
the inverse operation for tool results.  This module does not execute tools,
retry requests, or assign rewards, which keeps protocol parsing independent of
session and training lifecycle decisions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ...runtime.conversation import _reject_constant, _unique_object
from ...types import ActionParseError, AgentAction, ToolCall, ToolResult


def _value(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK model object or a plain mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class OpenAINativeActionParser:
    """Convert completed OpenAI responses into provider-neutral actions.

    Exactly one response choice is accepted for Chat Completions, while
    Responses items are scanned in order so reasoning, message, and function
    call items retain their original sequence.  Incomplete responses are
    rejected before an action can reach the tool executor because their token
    evidence cannot represent a completed decision.
    """

    def parse_chat_completion(self, response: Any) -> AgentAction:
        """Parse one Chat Completions choice and preserve native call IDs."""
        choices = _value(response, "choices", ())
        if not choices:
            raise ActionParseError("OpenAI chat completion has no choices.")
        if len(choices) != 1 or _value(choices[0], "finish_reason") in {
            "length",
            "content_filter",
        }:
            raise ActionParseError("A complete single OpenAI choice is required.")
        message = _value(choices[0], "message", {})
        content = _value(message, "content", None) or ""
        raw_calls = _value(message, "tool_calls", ()) or ()
        calls = tuple(
            self._tool_call(item, index) for index, item in enumerate(raw_calls)
        )
        assistant = {
            "role": "assistant",
            "content": content,
        }
        if raw_calls:
            assistant["tool_calls"] = [
                self._raw_chat_call(item, index) for index, item in enumerate(raw_calls)
            ]
        if calls:
            if len({call.call_id for call in calls}) != len(calls):
                raise ActionParseError("OpenAI tool call IDs must be unique.")
            return AgentAction(assistant, calls)
        if not isinstance(content, str) or not content:
            raise ActionParseError(
                "OpenAI chat completion has no content or tool call."
            )
        return AgentAction(assistant, final_text=content)

    def parse_response(self, response: Any) -> AgentAction:
        """Parse Responses output items into text or ordered function calls."""
        if _value(response, "status", "completed") != "completed":
            raise ActionParseError("OpenAI response is not completed.")
        output = _value(response, "output", ()) or ()
        calls = []
        text_parts = []
        for index, item in enumerate(output):
            kind = _value(item, "type", "")
            if kind == "function_call":
                arguments = _value(item, "arguments", "{}")
                try:
                    arguments = (
                        json.loads(
                            arguments,
                            parse_constant=_reject_constant,
                            object_pairs_hook=_unique_object,
                        )
                        if isinstance(arguments, str)
                        else arguments
                    )
                except (TypeError, ValueError) as exc:
                    raise ActionParseError(
                        "OpenAI function call arguments are invalid JSON."
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ActionParseError(
                        "OpenAI function call arguments must be an object."
                    )
                call_id, name = _value(item, "call_id"), _value(item, "name")
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or not isinstance(name, str)
                    or not name
                ):
                    raise ActionParseError(
                        "OpenAI function calls require a name and call_id."
                    )
                calls.append(
                    ToolCall(
                        call_id,
                        name,
                        arguments,
                    )
                )
            elif kind in {"message", "output_text"}:
                content = _value(item, "content", _value(item, "text", ""))
                if isinstance(content, list):
                    content = "".join(
                        str(_value(part, "text", ""))
                        for part in content
                        if _value(part, "type", "output_text") == "output_text"
                    )
                if content:
                    text_parts.append(str(content))
        if not output and _value(response, "output_text", None):
            text_parts.append(str(_value(response, "output_text")))
        if calls:
            if len({call.call_id for call in calls}) != len(calls):
                raise ActionParseError("OpenAI function call IDs must be unique.")
            return AgentAction({"role": "assistant", "content": ""}, tuple(calls))
        text = "".join(text_parts)
        if not text:
            raise ActionParseError("OpenAI response has no content or function call.")
        return AgentAction({"role": "assistant", "content": text}, final_text=text)

    @staticmethod
    def _tool_call(item: Any, index: int) -> ToolCall:
        """Validate and normalize one Chat Completions function call."""
        if _value(item, "type", "function") != "function":
            raise ActionParseError("Only OpenAI function tools are supported.")
        function = _value(item, "function", {})
        arguments = _value(function, "arguments", "{}")
        try:
            arguments = (
                json.loads(
                    arguments,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_unique_object,
                )
                if isinstance(arguments, str)
                else arguments
            )
        except (TypeError, ValueError) as exc:
            raise ActionParseError("OpenAI tool arguments are invalid JSON.") from exc
        if not isinstance(arguments, dict):
            raise ActionParseError("OpenAI tool arguments must be an object.")
        name = _value(function, "name", "")
        call_id = _value(item, "id")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(call_id, str)
            or not call_id
        ):
            raise ActionParseError("OpenAI tool calls require a name and ID.")
        return ToolCall(call_id, name, arguments)

    @staticmethod
    def _raw_chat_call(item: Any, index: int) -> dict[str, Any]:
        """Re-encode a parsed call for the next Chat Completions request."""
        function = _value(item, "function", {})
        return {
            "id": str(_value(item, "id", index)),
            "type": "function",
            "function": {
                "name": str(_value(function, "name", "")),
                "arguments": (
                    _value(function, "arguments")
                    if isinstance(_value(function, "arguments"), str)
                    else json.dumps(_value(function, "arguments", {}), allow_nan=False)
                ),
            },
        }


class OpenAIChatObservationFormatter:
    """Format ordered tool results as Chat Completions ``tool`` messages."""

    def format(self, results: Sequence[ToolResult]) -> list[dict[str, Any]]:
        """Attach each observation to its original provider call ID."""
        return [
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": result.content,
            }
            for result in results
        ]


class OpenAIResponsesObservationFormatter:
    """Format ordered tool results as Responses function-call outputs."""

    def format(self, results: Sequence[ToolResult]) -> list[dict[str, Any]]:
        """Return continuation items accepted by the Responses endpoint."""
        return [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": result.content,
            }
            for result in results
        ]
