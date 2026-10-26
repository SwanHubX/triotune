"""Translate between model conversations and the agent's execution protocol.

The renderer owns chat-template serialization, the parser extracts decisions
from decoded output, and the formatter turns tool results into observations.
These responsibilities are separate so a model's native tool-call format can
replace the default JSON protocol without changing sampling or PPO code.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from ..types import (
    ActionParseError,
    AgentAction,
    AgentContractError,
    Generation,
    ToolCall,
    ToolResult,
    validate_tokens,
)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


class ChatTemplateRenderer:
    """Render exact request tokens; trajectory export checks prefix stability.

    Template kwargs may include model-specific flags such as enable_thinking.
    Native tool-call message formats can use a different ActionParser and
    ObservationFormatter without changing the runner or PPO interfaces.
    """

    def __init__(
        self, tokenizer: Any, *, advertise_tools: bool = True, **template_kwargs: Any
    ):
        """Bind the tokenizer and model-specific template options.

        Generation-prefix and tokenization controls remain renderer-owned;
        allowing callers to override them would make request alignment unclear.
        """
        if {"tokenize", "add_generation_prompt", "tools"}.intersection(template_kwargs):
            raise ValueError(
                "Template controls tokenize, add_generation_prompt and tools are reserved."
            )
        self.tokenizer, self.template_kwargs = tokenizer, template_kwargs
        self.advertise_tools = advertise_tools

    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> list[int]:
        """Render a complete request ending at the assistant generation prefix.

        Tool schemas are included only when tools are registered. Returning
        integer IDs here makes this serialization the actual sampling input;
        the trajectory exporter later checks that older output IDs survive.
        """
        kwargs = dict(self.template_kwargs)
        if tools and self.advertise_tools:
            kwargs["tools"] = list(tools)
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        # Transformers 5 can return BatchEncoding even without return_dict.
        tokens = list(
            rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        )
        validate_tokens(tokens, name="rendered prompt")
        return tokens


class TokenAppendRenderer(ChatTemplateRenderer):
    """Append template boundaries/observations to actual sampled token IDs.

    A unique placeholder identifies the end of the last assistant's content in
    the template. Only the suffix after that placeholder is tokenized. Previous
    prompts and generated IDs never pass through decode/encode. This supports
    templates that rewrite previous thinking content and noncanonical BPE IDs.
    Templates that duplicate/remove the placeholder are rejected explicitly.
    The JSON protocol uses ordinary assistant content, not native tool messages.
    """

    def render_continuation(self, episode, messages, tools) -> list[int]:
        previous = episode.turns[-1]
        index = max(
            i for i, message in enumerate(messages) if message["role"] == "assistant"
        )
        marker = "TRIOTUNE_ASSISTANT_BOUNDARY_7e39d2"
        probe = copy.deepcopy(list(messages))
        probe[index] = {"role": "assistant", "content": marker}
        kwargs = dict(self.template_kwargs)
        if tools and self.advertise_tools:
            kwargs["tools"] = list(tools)

        def suffix(items, generation_prompt):
            rendered = self.tokenizer.apply_chat_template(
                items, tokenize=False, add_generation_prompt=generation_prompt, **kwargs
            )
            if not isinstance(rendered, str) or rendered.count(marker) != 1:
                raise AgentContractError(
                    "Template must preserve one assistant boundary placeholder."
                )
            return rendered.split(marker, 1)[1]

        closing = suffix(probe[: index + 1], False)
        continuation = suffix(probe, True)
        if not closing or not continuation.startswith(closing):
            raise AgentContractError(
                "Template has no stable assistant closing boundary."
            )
        boundary = list(self.tokenizer.encode(closing, add_special_tokens=False))
        tokens = list(self.tokenizer.encode(continuation, add_special_tokens=False))
        if tokens[: len(boundary)] != boundary:
            raise AgentContractError(
                "Template boundary merges with observation tokens."
            )
        # Backends may include the stop marker, omit it, or return its trailing
        # newline too. Deduplicate only an exact suffix of the original IDs.
        output = previous.generation.tokens
        overlap = 0
        for size in range(1, min(len(output), len(boundary)) + 1):
            if tuple(boundary[:size]) == output[-size:]:
                overlap = size
        return list(previous.prompt_tokens + output) + tokens[overlap:]


class JsonActionParser:
    """Parse {"tool_calls": [{"name": ..., "arguments": {...}}]} or {"final": ...}.

    Plain text is also a final answer. JSON tool calls remain verbatim assistant
    content, allowing ordinary text chat templates to preserve sampled tokens.
    This is an explicit application protocol, not a model-specific tool parser.
    """

    def __init__(self, *, allow_plain_text: bool = True):
        self.allow_plain_text = allow_plain_text

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
            if self.allow_plain_text:
                return AgentAction(message, final_text=text)
            raise ActionParseError("The action must be a JSON object.")
        try:
            value = json.loads(
                text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
            )
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
            if set(call) != {"name", "arguments"} or not isinstance(arguments, dict):
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
