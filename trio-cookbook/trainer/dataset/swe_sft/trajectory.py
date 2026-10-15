"""Normalize and split software-engineering agent conversations.

The input contract follows common OpenAI-style agent traces. Records may hold
``messages`` directly or contain one or more entries under ``conversations``.
Tool definitions stay attached to the conversation and are passed to the
tokenizer rather than serialized into ordinary message text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .schema import SWEConversation, SWESFTConfig, SWESFTReport


_THINK_BLOCK = re.compile(
    r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class _AssistantSegment:
    """An assistant message together with its immediately following tools."""

    assistant_index: int
    end_index: int


@dataclass(slots=True)
class _Candidate:
    """Messages to tokenize and assistant ordinal positions to exclude."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    masked_assistant_indices: set[int]


def _normalize_thinking_tags(content: str) -> str:
    content = _THINK_OPEN.sub("<think>", content)
    return _THINK_CLOSE.sub("</think>", content)


def _without_thinking(content: str) -> str:
    return _THINK_BLOCK.sub("", _normalize_thinking_tags(content)).strip()


def _tool_calls(value: Any, *, location: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{location} must be a non-empty list.")
    calls = []
    for index, call in enumerate(value):
        if not isinstance(call, Mapping):
            raise TypeError(f"{location}[{index}] must be a mapping.")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise TypeError(f"{location}[{index}].function must be a mapping.")
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"{location}[{index}].function.name must be non-empty.")
        if not isinstance(arguments, (str, Mapping)):
            raise TypeError(
                f"{location}[{index}].function.arguments must be JSON text or a mapping."
            )
        normalized = {
            "type": call.get("type", "function"),
            "function": {"name": name, "arguments": arguments},
        }
        if "id" in call:
            normalized["id"] = call["id"]
        calls.append(normalized)
    return calls


def normalize_message(
    message: Mapping[str, Any],
    *,
    location: str,
    strip_thinking: bool,
) -> dict[str, Any]:
    """Keep fields used by chat templates and normalize reasoning content."""

    if not isinstance(message, Mapping):
        raise TypeError(f"{location} must be a mapping.")
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise TypeError(f"{location}.role must be a non-empty string.")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError(f"{location}.content must be text or null.")

    cleaned: dict[str, Any] = {"role": role}
    if role == "assistant":
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise TypeError(f"{location}.reasoning_content must be text when provided.")
        normalized_content = _normalize_thinking_tags(content or "")
        has_inline = bool(_THINK_BLOCK.search(normalized_content))
        if strip_thinking:
            normalized_content = _without_thinking(normalized_content)
        elif reasoning and reasoning.strip() and not has_inline:
            block = f"<think>\n{reasoning.strip()}\n</think>"
            normalized_content = (
                block
                if not normalized_content
                else f"{block}\n\n{normalized_content.lstrip()}"
            )
        cleaned["content"] = normalized_content or (
            None if message.get("tool_calls") else ""
        )
        if message.get("tool_calls"):
            cleaned["tool_calls"] = _tool_calls(
                message["tool_calls"], location=f"{location}.tool_calls"
            )
    else:
        cleaned["content"] = content or ""

    if role == "tool" and "tool_call_id" in message:
        cleaned["tool_call_id"] = message["tool_call_id"]
    if "name" in message:
        cleaned["name"] = message["name"]
    return cleaned


def conversations_from_record(
    record: Mapping[str, Any], *, record_index: int
) -> list[SWEConversation]:
    """Extract every conversation instead of silently dropping earlier ones."""

    if not isinstance(record, Mapping):
        raise TypeError(f"Record {record_index} must be a mapping.")
    raw_conversations = record.get("conversations")
    if raw_conversations is None or raw_conversations == []:
        raw_conversations = [record]
    if not isinstance(raw_conversations, (list, tuple)):
        raise TypeError(f"Record {record_index}.conversations must be a list.")

    conversations = []
    for conversation_index, raw in enumerate(raw_conversations):
        location = f"Record {record_index}.conversations[{conversation_index}]"
        if not isinstance(raw, Mapping):
            raise TypeError(f"{location} must be a mapping.")
        messages = raw.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError(f"{location}.messages must be a non-empty list.")
        if any(not isinstance(message, Mapping) for message in messages):
            raise TypeError(f"{location}.messages must contain only mappings.")
        tools = raw.get("tools", record.get("tools"))
        if tools is not None and not isinstance(tools, (list, tuple)):
            raise TypeError(f"{location}.tools must be a list when provided.")
        if tools is not None and any(not isinstance(tool, Mapping) for tool in tools):
            raise TypeError(f"{location}.tools must contain only mappings.")
        conversations.append(
            SWEConversation(
                messages=[dict(message) for message in messages],
                tools=[dict(tool) for tool in tools] if tools is not None else None,
            )
        )
    return conversations


def _segments(messages: list[dict[str, Any]]) -> list[_AssistantSegment]:
    segments = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "assistant":
            index += 1
            continue
        end = index + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        segments.append(_AssistantSegment(index, end))
        index = end
    return segments


def _has_tool_error(messages: list[dict[str, Any]], segment: _AssistantSegment) -> bool:
    return any(
        message.get("role") == "tool" and message.get("is_error") is True
        for message in messages[segment.assistant_index + 1 : segment.end_index]
    )


def _empty_tool_call(message: Mapping[str, Any]) -> bool:
    return bool(message.get("tool_calls")) and not (
        (message.get("content") or "").strip()
        or (message.get("reasoning_content") or "").strip()
    )


def _bare_tool_call(message: Mapping[str, Any]) -> bool:
    if not message.get("tool_calls"):
        return False
    content = _normalize_thinking_tags(message.get("content") or "")
    reasoning = message.get("reasoning_content") or ""
    has_thinking = any(match.group(1).strip() for match in _THINK_BLOCK.finditer(content))
    return bool(content.strip()) and not has_thinking and not reasoning.strip()


def _rejection_reason(
    messages: list[dict[str, Any]], segment: _AssistantSegment, config: SWESFTConfig
) -> str | None:
    assistant = messages[segment.assistant_index]
    if config.filter_error_segments and _has_tool_error(messages, segment):
        return "error"
    if config.filter_empty_tool_calls and _empty_tool_call(assistant):
        return "empty_tool_call"
    if config.filter_bare_tool_calls and _bare_tool_call(assistant):
        return "bare_tool_call"
    return None


def _count_rejection(report: SWESFTReport, reason: str) -> None:
    if reason == "error":
        report.error_segments += 1
    elif reason == "empty_tool_call":
        report.empty_tool_call_segments += 1
    else:
        report.bare_tool_call_segments += 1


def _clean_messages(
    messages: list[dict[str, Any]],
    *,
    target_assistant_index: int | None,
    config: SWESFTConfig,
) -> list[dict[str, Any]]:
    cleaned = []
    for index, message in enumerate(messages):
        is_assistant = message.get("role") == "assistant"
        is_target = target_assistant_index is None or index == target_assistant_index
        strip = (
            config.strip_target_thinking
            if is_target
            else config.strip_context_thinking
        )
        cleaned.append(
            normalize_message(
                message,
                location=f"messages[{index}]",
                strip_thinking=is_assistant and strip,
            )
        )
    return cleaned


def candidates_from_conversation(
    conversation: SWEConversation,
    config: SWESFTConfig,
    report: SWESFTReport,
) -> list[_Candidate]:
    """Create progressive pairs or one full-trajectory candidate."""

    raw_messages = conversation.messages
    segments = _segments(raw_messages)
    if not segments:
        return []

    if config.pre_split:
        target = segments[-1]
        reason = _rejection_reason(raw_messages, target, config)
        report.candidates += 1
        if reason is not None:
            _count_rejection(report, reason)
            return []
        return [
            _Candidate(
                _clean_messages(
                    raw_messages,
                    target_assistant_index=target.assistant_index,
                    config=config,
                ),
                conversation.tools,
                set(range(len(segments) - 1)),
            )
        ]

    if config.sample_mode == "pair":
        candidates = []
        for segment_index, segment in enumerate(segments):
            report.candidates += 1
            reason = _rejection_reason(raw_messages, segment, config)
            if reason is not None:
                _count_rejection(report, reason)
                continue
            # Exclude tool results following the target. They are environment
            # observations for the next decision and never target tokens.
            messages = raw_messages[: segment.assistant_index + 1]
            candidates.append(
                _Candidate(
                    _clean_messages(
                        messages,
                        target_assistant_index=segment.assistant_index,
                        config=config,
                    ),
                    conversation.tools,
                    set(range(segment_index)),
                )
            )
        return candidates

    report.candidates += 1
    masked = set()
    for segment_index, segment in enumerate(segments):
        reason = _rejection_reason(raw_messages, segment, config)
        if reason is not None:
            masked.add(segment_index)
            _count_rejection(report, reason)
    return [
        _Candidate(
            _clean_messages(
                raw_messages,
                target_assistant_index=None,
                config=config,
            ),
            conversation.tools,
            masked,
        )
    ]


def parse_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert JSON-string tool arguments for templates that require objects."""

    parsed = []
    for message in messages:
        current = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                copied = {**call, "function": dict(call["function"])}
                arguments = copied["function"].get("arguments")
                if isinstance(arguments, str):
                    try:
                        copied["function"]["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            "Tool call arguments are not valid JSON: " + arguments
                        ) from error
                calls.append(copied)
            current["tool_calls"] = calls
        parsed.append(current)
    return parsed


def records_from_source(source: Any) -> Iterable[Mapping[str, Any]]:
    """Read JSONL lazily or pass through an existing iterable of records."""

    if isinstance(source, Mapping):
        return (source,)
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        path = source.__fspath__() if hasattr(source, "__fspath__") else source

        def read_jsonl() -> Iterable[Mapping[str, Any]]:
            with open(path, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {path!s} at line {line_number}."
                        ) from error
                    if not isinstance(record, Mapping):
                        raise TypeError(
                            f"JSONL record at line {line_number} must be an object."
                        )
                    yield record

        return read_jsonl()
    if not isinstance(source, Iterable):
        raise TypeError("source must be a JSONL path or an iterable of records.")
    return source
