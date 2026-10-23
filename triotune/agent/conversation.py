"""Translate between model conversations and the agent's execution protocol.

The renderer owns chat-template serialization, the parser extracts decisions
from decoded output, and the formatter turns tool results into observations.
These responsibilities are separate so a model's native tool-call format can
replace the default JSON protocol without changing sampling or PPO code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from .contracts import (
    ActionParseError,
    AgentAction,
    Generation,
    ToolCall,
    ToolResult,
    validate_tokens,
)


class ChatTemplateRenderer:
    """Render exact request tokens; trajectory export checks prefix stability.

    Template kwargs may include model-specific flags such as enable_thinking.
    Native tool-call message formats can use a different ActionParser and
    ObservationFormatter without changing the runner or PPO interfaces.
    """

    def __init__(self, tokenizer: Any, **template_kwargs: Any):
        """Bind the tokenizer and model-specific template options.

        Generation-prefix and tokenization controls remain renderer-owned;
        allowing callers to override them would make request alignment unclear.
        """
        if {"tokenize", "add_generation_prompt", "tools"}.intersection(template_kwargs):
            raise ValueError(
                "Template controls tokenize, add_generation_prompt and tools are reserved."
            )
        self.tokenizer, self.template_kwargs = tokenizer, template_kwargs

    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> list[int]:
        """Render a complete request ending at the assistant generation prefix.

        Tool schemas are included only when tools are registered. Returning
        integer IDs here makes this serialization the actual sampling input;
        the trajectory exporter later checks that older output IDs survive.
        """
        kwargs = dict(self.template_kwargs)
        if tools:
            kwargs["tools"] = list(tools)
        tokens = list(
            self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        )
        validate_tokens(tokens, name="rendered prompt")
        return tokens


class JsonActionParser:
    """Parse {"tool_calls": [{"name": ..., "arguments": {...}}]} or {"final": ...}.

    Plain text is also a final answer. JSON tool calls remain verbatim assistant
    content, allowing ordinary text chat templates to preserve sampled tokens.
    This is an explicit application protocol, not a model-specific tool parser.
    """

    def parse(self, generation: Generation) -> AgentAction:
        """Extract a final answer or ordered calls while preserving message text.

        Leading/trailing whitespace is ignored only for interpretation. The
        assistant message retains the original decoded string to avoid changing
        the next request through JSON reformatting or whitespace normalization.
        """
        text = generation.text.strip()
        message = {"role": "assistant", "content": generation.text}
        if not text:
            raise ActionParseError("The model returned an empty action.")
        if not text.startswith("{"):
            # Plain-language answers remain useful for simple agents. Output
            # that starts an object must satisfy the explicit JSON protocol.
            return AgentAction(message, final_text=text)
        try:
            value = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ActionParseError("Action JSON is malformed.") from exc
        if not isinstance(value, dict):
            raise ActionParseError("An action must be a JSON object.")
        if set(value) == {"final"} and isinstance(value["final"], str):
            return AgentAction(message, final_text=value["final"])
        raw_calls = value.get("tool_calls")
        if (
            set(value) != {"tool_calls"}
            or not isinstance(raw_calls, list)
            or not raw_calls
        ):
            raise ActionParseError("Use either final or a nonempty tool_calls list.")
        calls = []
        for index, call in enumerate(raw_calls):
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not call["name"]
            ):
                raise ActionParseError("Every tool call needs a name.")
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                raise ActionParseError("Tool arguments must be an object.")
            # IDs identify observations within this generated turn; the turn
            # and episode records provide the enclosing identity in audit logs.
            calls.append(ToolCall(str(index), call["name"], arguments))
        return AgentAction(message, tuple(calls))


class JsonObservationFormatter:
    """Group ordered tool observations into one JSON text user message.

    This pairs with JsonActionParser, whose assistant content is ordinary text.
    A native tool-call parser can instead supply a formatter that emits tool
    messages with the role and call-ID fields required by its chat template.
    """

    def format(self, results: Sequence[ToolResult]) -> list[dict[str, Any]]:
        """Serialize observations with explicit IDs, names, and error flags.

        Non-finite JSON numbers are rejected rather than emitting an observation
        that another component might interpret inconsistently.
        """
        return [
            {
                "role": "user",
                "content": json.dumps(
                    {"tool_results": [asdict(result) for result in results]},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
        ]
