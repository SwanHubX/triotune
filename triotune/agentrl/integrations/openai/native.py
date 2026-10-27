"""Keep model-output parsing separate from OpenAI response interpretation.

Chat Completions and Responses expose different wire representations for the
same logical action.  The parser normalizes both forms into ``AgentAction`` and
``ToolCall`` records while preserving provider call IDs; the formatters perform
the inverse operation for tool results.  This module does not execute tools,
retry requests, or assign rewards, which keeps protocol parsing independent of
session and training lifecycle decisions.

``Qwen35ToolParser`` implements the model-facing boundary and decodes original
``Generation`` IDs. ``OpenAINativeActionParser`` implements the API-facing
boundary and reads already structured provider responses. Keeping these entry
points distinct prevents raw tool syntax or reasoning from being mistaken for
an ordinary final answer in a synthetic API response.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from ...runtime.conversation import _reject_constant, _unique_object
from ...types import (
    ActionParseError,
    AgentAction,
    AgentContractError,
    Generation,
    ToolCall,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Parsed display fields alongside the exact, unmodified sampled generation.

    The OpenAI-compatible assistant message is a presentation of the action,
    not a substitute for ``generation.tokens`` or its behavior logprobs.
    ``raw_text`` is decoded with special tokens retained so model-specific end
    markers remain available for validation. Reasoning never becomes a final
    answer or a separate reward signal implicitly.
    """

    generation: Generation
    action: AgentAction
    assistant_message: dict[str, Any]
    reasoning: str
    raw_text: str


class ModelOutputParser(Protocol):
    """Decode one model grammar independently of the OpenAI wire protocol.

    Implementations return display fields but must retain the original
    ``Generation`` object. ``prompt_tokens`` supplies protocol state such as
    an opening reasoning delimiter already emitted by the chat template.
    OpenAI response parsers instead consume completed provider response objects
    and must never be used to interpret a raw sampled tool-call string.
    """

    def parse_generation(
        self, generation: Generation, *, tools, prompt_tokens
    ) -> ModelOutput: ...


class Qwen35ToolParser:
    """Parse Qwen3.5 XML-style function actions from original sampled IDs.

    This interface accepts a ``Generation`` rather than an OpenAI response.
    Qwen3.5 advertises ``<function=name>`` and ``<parameter=name>`` blocks in
    its tokenizer template. String parameters remain literal text; other JSON
    parameter types are decoded strictly with duplicate keys and NaN rejected.
    Only text, function tools, and the schema subset validated below are
    supported. Different model grammars must use a different explicit parser.
    """

    def __init__(self, tokenizer):
        """Bind the tokenizer used to decode, never re-encode, generated IDs."""
        self.tokenizer = tokenizer

    @classmethod
    def check_schema(cls, schema):
        """Reject schema features whose validation is not implemented here.

        The supported subset covers schemas emitted by ``function_tool``:
        JSON scalar types, arrays, string-keyed objects, anyOf, and enums.
        Rejecting constraints such as pattern or oneOf is safer than silently
        advertising constraints that the local parser cannot enforce.
        """
        if not isinstance(schema, dict) or set(schema) - {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "anyOf",
            "enum",
            "description",
            "default",
            "title",
        }:
            raise AgentContractError("Unsupported Qwen3.5 tool parameter schema.")
        kind = schema.get("type")
        if kind is not None and (
            not isinstance(kind, str)
            or kind
            not in {
                None,
                "string",
                "integer",
                "number",
                "boolean",
                "null",
                "array",
                "object",
            }
        ):
            raise AgentContractError("Unsupported Qwen3.5 schema type.")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        alternatives = schema.get("anyOf", [{}])
        if (
            not isinstance(properties, dict)
            or any(
                not isinstance(key, str)
                or re.fullmatch(r"[A-Za-z_][\w.-]*", key) is None
                for key in properties
            )
            or not isinstance(required, list)
            or any(
                not isinstance(key, str) or key not in properties for key in required
            )
            or not isinstance(alternatives, list)
            or not alternatives
            or not isinstance(schema.get("additionalProperties", True), (bool, dict))
            or "enum" in schema
            and (not isinstance(schema["enum"], list) or not schema["enum"])
            or "anyOf" in schema
            and set(schema) - {"anyOf", "description", "default", "title"}
            or kind != "object"
            and set(schema).intersection(
                {"properties", "required", "additionalProperties"}
            )
            or kind != "array"
            and "items" in schema
        ):
            raise AgentContractError("Malformed or unsupported Qwen3.5 schema layout.")
        for child in schema.get("properties", {}).values():
            cls.check_schema(child)
        for child in schema.get("anyOf", ()):
            cls.check_schema(child)
        for key in ("items", "additionalProperties"):
            if isinstance(schema.get(key), dict):
                cls.check_schema(schema[key])

    @classmethod
    def validate_schema(cls, value, schema):
        """Check supported JSON values without coercing booleans into numbers."""
        if "anyOf" in schema:
            return any(cls.validate_schema(value, branch) for branch in schema["anyOf"])
        if "enum" in schema and not any(
            type(value) is type(item) and value == item for item in schema["enum"]
        ):
            return False
        kind = schema.get("type")
        if kind == "object":
            if not isinstance(value, dict):
                return False
            properties = schema.get("properties", {})
            extra = schema.get("additionalProperties", True)
            return all(key in value for key in schema.get("required", ())) and all(
                cls.validate_schema(item, properties[key])
                if key in properties
                else extra is not False
                and (not isinstance(extra, dict) or cls.validate_schema(item, extra))
                for key, item in value.items()
            )
        if kind == "array":
            return isinstance(value, list) and all(
                cls.validate_schema(item, schema.get("items", {})) for item in value
            )
        if kind == "number":
            return type(value) in (int, float) and math.isfinite(value)
        if kind == "integer":
            return type(value) is int
        if kind in {"string", "boolean", "null"}:
            return (
                type(value)
                is {"string": str, "boolean": bool, "null": type(None)}[kind]
            )
        return kind is None

    @classmethod
    def _parameter_value(cls, text, schema):
        """Decode native parameter text according to its declared JSON type.

        The Qwen template writes strings literally but serializes objects and
        arrays as JSON. For a union or untyped enum, prefer a matching JSON
        value and otherwise accept literal text only when the schema permits
        a string. Numeric fields never coerce a quoted string into a number.
        """
        if schema.get("type") == "string":
            value = text
        else:
            try:
                value = json.loads(
                    text,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_unique_object,
                )
            except (ValueError, TypeError):
                value = text
                # Jinja's ``string`` filter renders Python scalars in the
                # native Qwen template. Accept those spellings only when the
                # declared field permits their actual boolean/null type. A
                # string field still retains its literal text, and numbers
                # never accept booleans through Python's int subclass rules.
                native_scalar = {"True": True, "False": False, "None": None}
                if text in native_scalar and cls.validate_schema(
                    native_scalar[text], schema
                ):
                    value = native_scalar[text]
            if not cls.validate_schema(value, schema) and cls.validate_schema(
                text, schema
            ):
                value = text
        if not cls.validate_schema(value, schema):
            raise ActionParseError(
                "Qwen3.5 parameter does not match its declared type."
            )
        return value

    def parse_generation(
        self, generation: Generation, *, tools, prompt_tokens
    ) -> ModelOutput:
        """Separate thinking, calls, and answer while retaining EOS evidence.

        The tokenizer may place the opening thinking marker in the prompt.
        Consequently a generated closing ``</think>`` is valid only when the
        prompt actually ends inside that thinking block. Truncated generations
        are not interpreted as completed calls, even if a partial XML block
        happens to resemble one.
        """
        generation.__post_init__()
        if generation.stop_reason in {"length", "max_tokens", "max_length"}:
            raise ActionParseError("Truncated Qwen3.5 output is not a complete action.")
        raw = self.tokenizer.decode(list(generation.tokens), skip_special_tokens=False)
        text = raw
        if "<|im_end|>" in text:
            text, trailing = text.split("<|im_end|>", 1)
            if trailing.strip():
                raise ActionParseError(
                    "Unexpected content after the Qwen3.5 end marker."
                )
        if "<|im_start|>" in text or "<|endoftext|>" in text:
            raise ActionParseError("Unexpected Qwen3.5 message boundary in output.")
        prompt = self.tokenizer.decode(list(prompt_tokens), skip_special_tokens=False)
        reasoning = ""
        if "</think>" in text:
            thought, text = text.split("</think>", 1)
            if thought.startswith("<think>"):
                thought = thought[len("<think>") :]
            elif not prompt.endswith("<think>\n"):
                raise ActionParseError("Qwen3.5 thinking block has no opening marker.")
            if "<think>" in thought:
                raise ActionParseError(
                    "Nested Qwen3.5 thinking blocks are unsupported."
                )
            reasoning = thought.strip()
        elif "<think>" in text or prompt.endswith("<think>\n"):
            raise ActionParseError("Qwen3.5 thinking block is incomplete.")
        if "<think>" in text or "</think>" in text:
            raise ActionParseError("Repeated Qwen3.5 thinking blocks are unsupported.")
        content = text.strip()
        functions = {tool["function"]["name"]: tool["function"] for tool in tools}
        calls = []
        if "<tool_call>" in content:
            before, blocks = content.split("<tool_call>", 1)
            remaining = "<tool_call>" + blocks
            while remaining.strip():
                match = re.match(
                    r"\s*<tool_call>\s*<function=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</function>\s*</tool_call>",
                    remaining,
                    re.DOTALL,
                )
                if match is None:
                    raise ActionParseError(
                        "Malformed Qwen3.5 function block or trailing text."
                    )
                name, parameters = match.groups()
                if name not in functions:
                    raise ActionParseError(f"Unknown Qwen3.5 function: {name}.")
                schema = functions[name].get("parameters", {})
                arguments = {}
                while parameters.strip():
                    parameter = re.match(
                        r"\s*<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>",
                        parameters,
                        re.DOTALL,
                    )
                    if parameter is None:
                        raise ActionParseError("Malformed Qwen3.5 parameter block.")
                    key, value = parameter.groups()
                    if key in arguments:
                        raise ActionParseError("Duplicate Qwen3.5 parameter.")
                    value = value.removeprefix("\n").removesuffix("\n")
                    field_schema = schema.get("properties", {}).get(
                        key, schema.get("additionalProperties", {})
                    )
                    if not isinstance(field_schema, dict):
                        field_schema = {}
                    arguments[key] = self._parameter_value(value, field_schema)
                    parameters = parameters[parameter.end() :]
                if not self.validate_schema(arguments, schema):
                    raise ActionParseError(
                        "Qwen3.5 arguments do not match the tool schema."
                    )
                calls.append(
                    {
                        "id": f"call_{uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments, ensure_ascii=False, allow_nan=False
                            ),
                        },
                    }
                )
                remaining = remaining[match.end() :]
            content = before.strip()
        elif any(
            marker in content
            for marker in (
                "<tool_call",
                "</tool_call",
                "<function",
                "</function",
                "<parameter",
                "</parameter",
            )
        ):
            raise ActionParseError("Malformed Qwen3.5 tool-call delimiters.")
        if not content and not calls:
            raise ActionParseError("Qwen3.5 output has no final answer or tool call.")
        message = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
        }
        if calls:
            message["tool_calls"] = calls
        action = AgentAction(
            message,
            tuple(
                ToolCall(
                    item["id"],
                    item["function"]["name"],
                    json.loads(item["function"]["arguments"]),
                )
                for item in calls
            ),
            None if calls else content,
        )
        return ModelOutput(generation, action, message, reasoning, raw)


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
        reasoning = _value(message, "reasoning_content", None)
        if isinstance(reasoning, str):
            assistant["reasoning_content"] = reasoning
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
