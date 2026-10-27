"""Translate between model conversations and the agent's execution protocol.

The renderer owns chat-template serialization, the parser extracts decisions
from decoded output, and the formatter turns tool results into observations.
These responsibilities are separate so a model's native tool-call format can
replace the default JSON protocol without changing sampling or PPO code.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..types import (
    ActionParseError,
    AgentAction,
    AgentContractError,
    EpisodeResult,
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


class PrefixCache:
    """Bound exact conversation serializations in a thread-safe LRU cache.

    Rendering the same system prompt and tool schema for many candidates can be
    material CPU work even when model sampling remains independent. The cache
    stores only tokenizer output; it never caches a sampled continuation or a
    behavior log probability. Keys are canonical JSON descriptions supplied by
    the renderer, and returned token lists are copied so callers cannot mutate
    cached provenance shared by concurrent episodes.
    """

    def __init__(self, max_entries: int = 256):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("PrefixCache max_entries must be a positive integer.")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[int] | None:
        """Return a detached cached prefix and update LRU order atomically."""
        with self._lock:
            value = self._entries.pop(key, None)
            if value is None:
                self.misses += 1
                return None
            self._entries[key] = value
            self.hits += 1
            return list(value)

    def put(self, key: str, tokens: Sequence[int]) -> None:
        """Store validated immutable IDs and evict the oldest exact request."""
        validate_tokens(tokens, name="cached prefix")
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = tuple(tokens)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop cached tokenizations without changing cumulative hit counters."""
        with self._lock:
            self._entries.clear()


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """Return compacted messages plus auditable interaction disposition."""

    messages: tuple[dict[str, Any], ...]
    retained_interaction_ids: tuple[str, ...]
    summarized_interaction_ids: tuple[str, ...]
    removed_messages: int


class ContextCompactor:
    """Summarize old tool exchanges while retaining selected interaction blocks.

    Messages are partitioned at assistant boundaries and paired with recorded
    generated interactions. Initial system/user instructions are always kept.
    The newest ``keep_recent_interactions`` blocks, explicitly preserved blocks,
    and interactions with tool errors remain literal. Consecutive omitted blocks
    become summary messages in their original position.

    A custom summarizer receives ``(interactions, previous_summary_messages)``
    and may be synchronous or asynchronous. It returns text, one message, or a
    message sequence. Summary messages are marked only inside the runtime; the
    renderer removes private marker keys before invoking the tokenizer. This
    permits later compactions to replace earlier summaries instead of growing an
    unbounded chain, without exposing control metadata to model templates.
    """

    def __init__(
        self,
        *,
        keep_recent_interactions: int = 2,
        summarizer=None,
        preserve=None,
        max_summary_chars: int = 4096,
    ):
        if type(keep_recent_interactions) is not int or keep_recent_interactions < 1:
            raise ValueError("keep_recent_interactions must be a positive integer.")
        if type(max_summary_chars) is not int or max_summary_chars < 1:
            raise ValueError("max_summary_chars must be a positive integer.")
        if summarizer is not None and not callable(summarizer):
            raise TypeError("Context summarizer must be callable or None.")
        if preserve is not None and not callable(preserve):
            raise TypeError("Interaction preservation hook must be callable or None.")
        self.keep_recent_interactions = keep_recent_interactions
        self.summarizer = summarizer
        self.preserve = preserve
        self.max_summary_chars = max_summary_chars

    @staticmethod
    def _default_summary(interactions, previous):
        """Produce a deterministic tool-result summary without model inference."""
        records = []
        for message in previous:
            records.append({"previous_summary": message.get("content", "")})
        for interaction in interactions:
            records.append(
                {
                    "interaction_id": interaction.interaction_id,
                    "tool_calls": [
                        {"call_id": call.call_id, "name": call.name}
                        for call in interaction.tool_calls
                    ],
                    "tool_results": [
                        asdict(result) for result in interaction.tool_results
                    ],
                }
            )
        return json.dumps(
            {"compacted_interactions": records},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    async def _summarize(self, interactions, previous):
        method = self.summarizer or self._default_summary
        if inspect.iscoroutinefunction(method):
            result = await method(tuple(interactions), tuple(previous))
        else:
            result = await asyncio.to_thread(
                method, tuple(interactions), tuple(previous)
            )
            if inspect.isawaitable(result):
                result = await result
        if isinstance(result, str):
            messages = [{"role": "user", "content": result}]
        elif isinstance(result, Mapping):
            messages = [dict(result)]
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            messages = [dict(message) for message in result]
        else:
            raise AgentContractError(
                "Context summarizer must return text or chat message mappings."
            )
        if not messages or any(
            message.get("role") not in {"system", "user", "tool"}
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise AgentContractError(
                "Context summaries require nonempty system/user/tool messages."
            )
        for message in messages:
            message["content"] = message["content"][: self.max_summary_chars]
            message["_triotune_summary"] = True
            message["_triotune_summarized_ids"] = tuple(
                dict.fromkeys(
                    interaction_id
                    for previous_message in previous
                    for interaction_id in previous_message.get(
                        "_triotune_summarized_ids", ()
                    )
                )
            ) + tuple(interaction.interaction_id for interaction in interactions)
        return messages

    async def compact(
        self, messages: Sequence[Mapping[str, Any]], episode: EpisodeResult
    ) -> CompactionPlan:
        """Replace eligible historical blocks and preserve chronological order."""
        original_items = [dict(message) for message in messages]
        previous_summaries = [
            message
            for message in original_items
            if message.get("_triotune_summary") is True
        ]
        previously_summarized_order = tuple(
            dict.fromkeys(
                interaction_id
                for message in previous_summaries
                for interaction_id in message.get("_triotune_summarized_ids", ())
            )
        )
        previously_summarized = set(previously_summarized_order)
        items = [
            message
            for message in original_items
            if message.get("_triotune_summary") is not True
        ]
        assistant_indices = [
            index
            for index, message in enumerate(items)
            if message.get("role") == "assistant"
        ]
        interactions = [
            interaction
            for interaction in episode.interactions
            if interaction.turn is not None
            and interaction.interaction_id not in previously_summarized
        ]
        if len(assistant_indices) != len(interactions):
            raise AgentContractError(
                "Conversation assistants do not align with recorded interactions."
            )
        if not interactions:
            return CompactionPlan(
                tuple(original_items), (), previously_summarized_order, 0
            )

        initial = items[: assistant_indices[0]]
        blocks = []
        for index, start in enumerate(assistant_indices):
            end = (
                assistant_indices[index + 1]
                if index + 1 < len(assistant_indices)
                else len(items)
            )
            blocks.append((interactions[index], items[start:end]))

        recent = {
            interaction.interaction_id
            for interaction, _ in blocks[-self.keep_recent_interactions :]
        }
        retained = set(recent)
        for interaction, _ in blocks:
            explicitly_preserved = False
            if self.preserve is not None:
                explicitly_preserved = (
                    await self.preserve(interaction)
                    if inspect.iscoroutinefunction(self.preserve)
                    else await asyncio.to_thread(self.preserve, interaction)
                )
                if inspect.isawaitable(explicitly_preserved):
                    explicitly_preserved = await explicitly_preserved
                if type(explicitly_preserved) is not bool:
                    raise AgentContractError(
                        "Interaction preservation hook must return a boolean."
                    )
            preserve = (
                any(result.is_error for result in interaction.tool_results)
                or explicitly_preserved
            )
            if preserve:
                retained.add(interaction.interaction_id)
        omitted = [
            interaction
            for interaction, _ in blocks
            if interaction.interaction_id not in retained
        ]
        if not omitted:
            return CompactionPlan(
                tuple(original_items),
                tuple(retained),
                previously_summarized_order,
                0,
            )

        output = list(initial)
        pending = []
        summary_history = list(previous_summaries)

        async def flush():
            nonlocal summary_history
            if not pending:
                return
            summary = await self._summarize(pending, summary_history)
            output.extend(summary)
            summary_history = []
            pending.clear()

        for interaction, block in blocks:
            if interaction.interaction_id in retained:
                await flush()
                output.extend(block)
            else:
                pending.append(interaction)
        await flush()
        removed = len(original_items) - len(output)
        if removed <= 0:
            raise AgentContractError(
                "Context compaction must remove at least one message."
            )
        return CompactionPlan(
            tuple(output),
            tuple(
                interaction.interaction_id
                for interaction in interactions
                if interaction.interaction_id in retained
            ),
            tuple(
                dict.fromkeys(
                    previously_summarized_order
                    + tuple(interaction.interaction_id for interaction in omitted)
                )
            ),
            removed,
        )


class ChatTemplateRenderer:
    """Render exact request tokens; trajectory export checks prefix stability.

    Template kwargs may include model-specific flags such as enable_thinking.
    Native tool-call message formats can use a different ActionParser and
    ObservationFormatter without changing the runner or PPO interfaces.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        advertise_tools: bool = True,
        prefix_cache: PrefixCache | None = None,
        **template_kwargs: Any,
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
        self.prefix_cache = prefix_cache

    @staticmethod
    def _public_messages(messages):
        """Remove runtime-only metadata before calling a model chat template."""
        return [
            {
                key: value
                for key, value in dict(message).items()
                if not key.startswith("_triotune_")
            }
            for message in messages
        ]

    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> list[int]:
        """Render a complete request ending at the assistant generation prefix.

        Tool schemas are included only when tools are registered. Returning
        integer IDs here makes this serialization the actual sampling input;
        the trajectory exporter later checks that older output IDs survive.
        """
        messages = self._public_messages(messages)
        kwargs = dict(self.template_kwargs)
        if tools and self.advertise_tools:
            kwargs["tools"] = list(tools)
        cache_key = None
        if self.prefix_cache is not None:
            try:
                cache_key = json.dumps(
                    {
                        "tokenizer_identity": id(self.tokenizer),
                        "template_kwargs": self.template_kwargs,
                        "messages": messages,
                        "tools": kwargs.get("tools"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                # Non-JSON template extensions remain renderable but cannot be
                # assigned a stable cross-task cache identity.
                cache_key = None
            if cache_key is not None:
                cached = self.prefix_cache.get(cache_key)
                if cached is not None:
                    return cached
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        # Transformers 5 can return BatchEncoding even without return_dict.
        tokens = list(
            rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        )
        validate_tokens(tokens, name="rendered prompt")
        if cache_key is not None:
            self.prefix_cache.put(cache_key, tokens)
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
        probe = copy.deepcopy(self._public_messages(messages))
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


class NativeToolContinuationRenderer(ChatTemplateRenderer):
    """Append native tool observations without re-encoding sampled actions.

    A native assistant action can contain reasoning and structured tool calls.
    Re-rendering those fields would canonicalize whitespace, JSON values, and
    token segmentation. Instead a unique placeholder replaces the whole last
    assistant action in a template probe. Only the closing boundary and new
    observations after that placeholder are encoded. The actual prefix always
    comes from the recorded sampling request and its unmodified output IDs.

    The caller must first verify that the message history and assistant action
    correspond to the selected parent interaction. This helper validates the
    tokenizer boundary, not request lineage or model tool-call syntax.
    """

    def render_after_tool(
        self, messages, tools, *, assistant_index, prompt_tokens, output_tokens
    ) -> list[int]:
        """Render a suffix and check its closing marker before appending IDs."""
        validate_tokens(prompt_tokens, name="parent prompt")
        validate_tokens(output_tokens, name="parent output")
        marker = "TRIOTUNE_NATIVE_ACTION_BOUNDARY_12ac69"
        probe = copy.deepcopy(self._public_messages(messages))
        probe[assistant_index] = {
            "role": "assistant",
            "content": marker,
            "reasoning_content": "",
        }
        kwargs = dict(self.template_kwargs)
        if tools and self.advertise_tools:
            kwargs["tools"] = list(tools)

        def suffix(items, generation_prompt):
            text = self.tokenizer.apply_chat_template(
                items, tokenize=False, add_generation_prompt=generation_prompt, **kwargs
            )
            if not isinstance(text, str) or text.count(marker) != 1:
                raise AgentContractError(
                    "Native template must preserve one action boundary."
                )
            return text.split(marker, 1)[1]

        closing = suffix(probe[: assistant_index + 1], False)
        continuation = suffix(probe, True)
        if not closing or not continuation.startswith(closing):
            raise AgentContractError("Native template has no stable closing boundary.")
        boundary = list(self.tokenizer.encode(closing, add_special_tokens=False))
        extra = list(self.tokenizer.encode(continuation, add_special_tokens=False))
        if not boundary or extra[: len(boundary)] != boundary:
            raise AgentContractError(
                "Native template boundary merges with observations."
            )
        # A sampled EOS may already contain all or part of the closing marker.
        # Deduplicate only identical token IDs; decoded text is never proof of
        # token identity, and no generated token is replaced by its encoding.
        overlap = 0
        for size in range(1, min(len(output_tokens), len(boundary)) + 1):
            if tuple(boundary[:size]) == tuple(output_tokens[-size:]):
                overlap = size
        tokens = list(prompt_tokens) + list(output_tokens) + extra[overlap:]
        expected = list(prompt_tokens) + list(output_tokens)
        if tokens[: len(expected)] != expected:
            raise AgentContractError("Native continuation changed the sampled prefix.")
        validate_tokens(tokens, name="native continuation")
        return tokens


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
