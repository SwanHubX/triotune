"""Normalize and split software-engineering agent conversations.

The input contract follows common OpenAI-style agent traces. Records may hold
``messages`` directly or contain one or more entries under ``conversations``.
Tool definitions stay attached to the conversation and are passed to the
tokenizer rather than serialized into ordinary message text.

Processing order: read records -> extract conversations -> group assistant/tool
segments -> apply selection policies -> normalize messages -> return candidates.
Candidates contain message-level supervision exclusions; prepare.py converts
them into token masks. This module does not call the tokenizer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .schema import SWEConversation, SWESFTConfig, SWESFTReport


# Accept both <think> and <thinking> spellings before normalizing their content.
_THINK_BLOCK = re.compile(
    r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE
)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


# ------------------------------------------------------------------
# Internal message segments and tokenization candidates
# ------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _AssistantSegment:
    """An assistant and its following tool messages, in [assistant_index, end_index).

    Tool errors are evaluated with the assistant decision that produced them.
    Both indices refer to the source message list; end_index is exclusive.
    """

    assistant_index: int
    end_index: int


@dataclass(slots=True)
class _Candidate:
    """Selected messages/tools plus assistant ordinals excluded from supervision.

    masked_assistant_indices counts assistant turns only (0, 1, ...), unlike
    _AssistantSegment indices into the complete message list. Tokens for these
    turns remain available as context even though their labels are masked.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    masked_assistant_indices: set[int]


# ------------------------------------------------------------------
# Reasoning and tool-call normalization for chat templates
# ------------------------------------------------------------------


def _normalize_thinking_tags(content: str) -> str:
    """Canonicalize supported reasoning tags without changing their inner text."""

    content = _THINK_OPEN.sub("<think>", content)
    return _THINK_CLOSE.sub("</think>", content)


def _without_thinking(content: str) -> str:
    """Remove complete reasoning blocks and trim surrounding whitespace."""

    return _THINK_BLOCK.sub("", _normalize_thinking_tags(content)).strip()


def _tool_calls(value: Any, *, location: str) -> list[dict[str, Any]]:
    """Copy tool calls after validating their function name and argument shape.

    Arguments may remain JSON strings or mappings; optional JSON decoding is
    performed later for templates that require argument objects.
    """

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
    """Copy template-facing message fields and normalize assistant reasoning.

    Args:
        message: Raw message; assistant tool-call content may be null.
        location: Source description included in field validation errors.
        strip_thinking: Remove assistant reasoning instead of preserving it.

    Separate reasoning_content is embedded in a canonical <think> block unless
    inline reasoning already exists. Tool-call IDs and names remain available
    for template linkage. Diagnostic fields such as is_error are inspected on
    raw segments before this function drops them from template-facing output.
    """

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
            # Avoid duplicating reasoning when the source already embedded a block.
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


# ------------------------------------------------------------------
# Record expansion and assistant/tool segment inspection
# ------------------------------------------------------------------


def conversations_from_record(
    record: Mapping[str, Any], *, record_index: int
) -> list[SWEConversation]:
    """Expand flat messages or nested conversations into SWEConversation objects.

    Missing/empty conversations uses the record itself as the conversation.
    Nested conversations inherit record-level tools unless they supply their
    own tools field. Field containers are validated and copied; individual
    message content is normalized later after segment filtering.
    """

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
    """Group each assistant with immediately following tools for error inspection."""

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
    """Check whether any following tool result explicitly sets is_error=True."""

    return any(
        message.get("role") == "tool" and message.get("is_error") is True
        for message in messages[segment.assistant_index + 1 : segment.end_index]
    )


def _empty_tool_call(message: Mapping[str, Any]) -> bool:
    """Detect a tool-calling assistant with neither visible content nor reasoning."""

    return bool(message.get("tool_calls")) and not (
        (message.get("content") or "").strip()
        or (message.get("reasoning_content") or "").strip()
    )


def _bare_tool_call(message: Mapping[str, Any]) -> bool:
    """Detect tool calls with content but no nonempty inline/separate reasoning.

    This is an optional dataset-quality filter. A tool call with empty content
    is handled by _empty_tool_call instead.
    """

    if not message.get("tool_calls"):
        return False
    content = _normalize_thinking_tags(message.get("content") or "")
    reasoning = message.get("reasoning_content") or ""
    has_thinking = any(match.group(1).strip() for match in _THINK_BLOCK.finditer(content))
    return bool(content.strip()) and not has_thinking and not reasoning.strip()


# ------------------------------------------------------------------
# Filtering decisions and target/context reasoning policies
# ------------------------------------------------------------------


def _rejection_reason(
    messages: list[dict[str, Any]], segment: _AssistantSegment, config: SWESFTConfig
) -> str | None:
    """Return the first enabled rejection reason for an assistant/tool segment.

    Priority is tool error, empty call, then bare call. Each inspected segment
    contributes at most one reason when the report is updated.
    """

    assistant = messages[segment.assistant_index]
    if config.filter_error_segments and _has_tool_error(messages, segment):
        return "error"
    if config.filter_empty_tool_calls and _empty_tool_call(assistant):
        return "empty_tool_call"
    if config.filter_bare_tool_calls and _bare_tool_call(assistant):
        return "bare_tool_call"
    return None


def _count_rejection(report: SWESFTReport, reason: str) -> None:
    """Increment the report counter for a reason produced by _rejection_reason."""

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
    """Apply distinct reasoning policies to the supervised turn and its context.

    A target index selects the final decision in a pair candidate. None applies
    the target policy to all assistant turns, as used for full trajectories.
    This choice is separate from which turns will eventually receive loss.
    """

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


# ------------------------------------------------------------------
# Candidate selection: pre-split pair, progressive pairs, full trajectory
# ------------------------------------------------------------------


def candidates_from_conversation(
    conversation: SWEConversation,
    config: SWESFTConfig,
    report: SWESFTReport,
) -> list[_Candidate]:
    """Select message sequences and assistant turns eligible for SFT loss.

    pre_split keeps one existing sequence and supervises its last assistant.
    pair creates a context prefix ending at each accepted assistant decision.
    trajectory keeps all messages and masks rejected assistant turns in place.

    Updates candidate/rejection counters on report. Conversations without an
    assistant produce no candidates. Source messages are inspected before
    normalization so tool-error metadata can inform each filtering decision.
    """

    raw_messages = conversation.messages
    segments = _segments(raw_messages)
    if not segments:
        return []

    if config.pre_split:
        # Earlier assistants remain context; only the last assistant is a target.
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

    # Full-trajectory mode preserves observations and rejected decisions as context.
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


# ------------------------------------------------------------------
# Template argument adaptation and trajectory source loading
# ------------------------------------------------------------------


def parse_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode JSON-string arguments on copied calls; preserve existing objects.

    Malformed JSON raises before tokenization. The source message/call mappings
    are not mutated, allowing reuse under a different template configuration.
    """

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
    """Read SWE records from a mapping, JSONL path, or existing iterable.

    This is the trajectory-specific source adapter. File rows are checked as
    JSON objects while iterating; existing iterables are passed through and
    conversations_from_record validates their records. The file generator is
    single-pass, whereas the source iterable retains its original semantics.
    """

    if isinstance(source, Mapping):
        return (source,)
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        path = source.__fspath__() if hasattr(source, "__fspath__") else source

        def read_jsonl() -> Iterable[Mapping[str, Any]]:
            """Yield nonempty JSONL records, reporting malformed lines at their source."""

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
