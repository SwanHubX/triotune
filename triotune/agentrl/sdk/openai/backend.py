"""Request backends for OpenAI-compatible agent execution.

The external backend forwards requests to an official OpenAI client.  The
PyTRIO backend instead binds one immutable ``SamplingSnapshot`` and converts
the sampled ``Generation`` into an OpenAI-shaped response for the existing
tool loop.  Keeping this boundary small prevents transport concerns from
changing session, reward, or PPO provenance rules.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from ...runtime.conversation import (
    ChatTemplateRenderer,
    NativeToolContinuationRenderer,
    _reject_constant,
    _unique_object,
)
from ...runtime.sampling import (
    PPO_TEMPERATURE,
    PPO_TOP_P,
    SamplingSnapshot,
)
from ...types import (
    ActionParseError,
    AgentAction,
    AgentContractError,
    ExecutionContext,
    validate_tokens,
)
from .native import ModelOutput, ModelOutputParser, Qwen35ToolParser


class OpenAIBackend(Protocol):
    """Resolve one validated endpoint request into a provider-shaped response.

    Backends are intentionally unaware of reward aggregation and tool-loop
    policy.  They receive a session so a training backend can verify snapshot
    provenance, but the client remains responsible for allocating the child
    interaction and recording the returned response.
    """

    async def complete(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        session: Any,
        method: Any | None = None,
    ) -> Any: ...

    def validate_request(
        self, endpoint: str, payload: Mapping[str, Any], *, session: Any
    ) -> None:
        """Validate request options before the client allocates an interaction."""
        ...

    async def close(self) -> None:
        """Release resources owned by the backend implementation."""
        ...


class ExternalOpenAIBackend:
    """Forward requests to an injected official OpenAI sync or async client.

    Async SDK methods are awaited directly.  Synchronous SDK methods run in a
    worker thread, and compatibility methods that return an awaitable after the
    thread call are awaited for a second stage.  The wrapper never closes an
    injected client unless the enclosing ``OpenAIClient`` created it.
    """

    def __init__(self, client: Any, *, owns_client: bool = False):
        """Bind a provider client and record whether this adapter owns it."""
        self.client = client
        self.owns_client = owns_client

    def validate_request(
        self, endpoint: str, payload: Mapping[str, Any], *, session: Any
    ) -> None:
        """External SDK validation remains owned by ``OpenAIClient``."""
        del endpoint, payload
        if getattr(session, "sampling_source_bound", False):
            raise AgentContractError(
                "A fixed-policy rollout cannot switch to an external provider."
            )

    async def complete(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        session: Any,
        method: Any | None = None,
    ) -> Any:
        """Call the SDK method without blocking an asyncio event loop."""
        if getattr(session, "sampling_source_bound", False):
            raise AgentContractError(
                "A fixed-policy rollout cannot switch to an external provider."
            )
        if method is None:
            method = (
                self.client.chat.completions.create
                if endpoint == "/v1/chat/completions"
                else self.client.responses.create
            )
        if inspect.iscoroutinefunction(method):
            return await method(**payload)
        response = await asyncio.to_thread(method, **payload)
        if inspect.isawaitable(response):
            return await response
        return response

    async def close(self) -> None:
        """Close only a client owned by this adapter."""
        if not self.owns_client:
            return
        result = self.client.close()
        if inspect.isawaitable(result):
            await result


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK model, dataclass-like object, or mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    """Validate a non-boolean integer sampling option."""
    if type(value) is not int or value < minimum:
        raise AgentContractError(f"{name} must be an integer >= {minimum}.")
    return value


def _response_messages(value: Any) -> list[dict[str, Any]]:
    """Convert common Responses items to portable chat-template messages.

    Responses reasoning and function-call records do not have one universal
    tokenizer representation, so this adapter retains their conversation
    boundaries while excluding provider-private reasoning summaries.  A model
    requiring a richer multimodal or native template should supply the
    backend's custom renderer instead of relying on this conservative mapping.
    """
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise AgentContractError("Responses input must be text or a sequence of items.")
    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise AgentContractError("Responses input items must be objects.")
        role = item.get("role")
        if isinstance(role, str):
            messages.append(dict(item))
            continue
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content", "")
            if isinstance(content, Sequence) and not isinstance(content, str):
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping)
                )
            messages.append({"role": item.get("role", "assistant"), "content": content})
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id", item.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            messages.append({"role": "tool", "content": str(item.get("output", ""))})
        elif item_type == "reasoning":
            # Reasoning items have no portable chat-template representation;
            # preserve their presence as an assistant boundary without copying
            # provider-private summaries into the model prompt.
            messages.append({"role": "assistant", "content": ""})
        else:
            raise AgentContractError(
                f"Unsupported Responses input item type: {item_type!r}."
            )
    if not messages:
        raise AgentContractError("Responses input must contain at least one item.")
    return messages


class PyTRIOBackend:
    """Sample through one fixed PyTRIO ``SamplingSnapshot``.

    The backend is intentionally training-only.  It requires the request
    session to carry the same snapshot and policy version, rejects sampling
    controls that would change PPO behavior-policy semantics, and never
    accepts a model name other than the one bound at construction time.

    Qwen/Qwen3.5-4B selects the native XML tool adapter automatically. A local
    alias can opt in through model_format='qwen3.5' with the matching tokenizer.
    Unknown model formats retain the earlier text-only sampling behavior and
    explicitly reject tools rather than treating raw calls as final answers.
    """

    def __init__(
        self,
        snapshot: SamplingSnapshot,
        *,
        model: str,
        max_tokens: int = 256,
        renderer: Any | None = None,
        output_parser: ModelOutputParser | None = None,
        model_format: str | None = None,
        template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind a fixed snapshot, tokenizer renderer, model, and output budget.

        ``output_parser`` is the raw-generation interface, not an OpenAI API
        response parser. ``template_kwargs`` supplies model-template flags such
        as enable_thinking; generation/tokenization controls remain reserved.
        A native tool run always uses the bound tokenizer renderer, so callers
        cannot override its actual input IDs with arbitrary request payloads.
        """
        if not isinstance(snapshot, SamplingSnapshot):
            raise AgentContractError("PyTRIO backend requires a SamplingSnapshot.")
        if not isinstance(model, str) or not model:
            raise ValueError("PyTRIO backend model must be a nonempty string.")
        self.snapshot = snapshot
        self.model = model
        self.max_tokens = _strict_int(max_tokens, name="max_tokens", minimum=1)
        self.renderer = renderer
        self.template_kwargs = dict(template_kwargs or {})
        # Reuse renderer validation for reserved tokenize/generation controls.
        ChatTemplateRenderer(snapshot.tokenizer, **self.template_kwargs)
        if model_format not in {None, "qwen3.5"}:
            raise AgentContractError("Unsupported model tool format.")
        self.native_tools = (
            model_format == "qwen3.5"
            or model == "Qwen/Qwen3.5-4B"
            or isinstance(output_parser, Qwen35ToolParser)
        )
        if self.native_tools:
            template = getattr(snapshot.tokenizer, "chat_template", "")
            if not isinstance(template, str) or any(
                marker not in template
                for marker in ("<tool_call>", "<function=", "<parameter=")
            ):
                raise AgentContractError(
                    "Qwen3.5 tools require the matching tokenizer template."
                )
            self._native_template = template
        self.output_parser = output_parser or (
            Qwen35ToolParser(snapshot.tokenizer) if self.native_tools else None
        )
        if snapshot.policy_version is None:
            raise AgentContractError(
                "PyTRIO backend requires a snapshot with policy_version."
            )

    def _validate_session(self, session: Any) -> ExecutionContext:
        """Require collector-owned training provenance before sampling.

        Checking both snapshot ID and policy version prevents a caller from
        reusing a valid session with a different sampler generation.  The
        returned context is the trusted identity that the surrounding client
        has already attached to the request interaction.
        """
        if not getattr(session, "training", False):
            raise AgentContractError("PyTRIO sampling requires a training session.")
        context = getattr(session, "context", None)
        if not isinstance(context, ExecutionContext):
            raise AgentContractError("PyTRIO sampling requires ExecutionContext.")
        if (
            context.snapshot_id != self.snapshot.snapshot_id
            or context.policy_version != self.snapshot.policy_version
        ):
            raise AgentContractError(
                "Session provenance does not match the bound PyTRIO snapshot."
            )
        session.bind_sampling_source(snapshot=self.snapshot, model=self.model)
        return context

    @staticmethod
    def _validate_sampling_options(payload: Mapping[str, Any]) -> None:
        """Sample the unwarped PPO behavior policy at temperature/top-p one.

        The existing PPO contract assumes behavior logprobs came from the
        fixed sampler configuration.  Requests that would introduce multiple
        choices, streaming state, background execution, or a different random
        policy are rejected before the PyTRIO sampler is called.
        """
        expected = {"temperature": PPO_TEMPERATURE, "top_p": PPO_TOP_P}
        for name, target in expected.items():
            if name in payload:
                value = payload[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise AgentContractError(
                        f"{name} must be exactly 1 in training mode."
                    )
                if not math.isfinite(float(value)) or float(value) != target:
                    raise AgentContractError(
                        f"{name} must be exactly 1 in training mode."
                    )
        if "stream" in payload and type(payload["stream"]) is not bool:
            raise AgentContractError("stream must be boolean in training mode.")
        if payload.get("stream"):
            raise AgentContractError("PyTRIO backend does not support streaming.")
        if "n" in payload and type(payload["n"]) is not int:
            raise AgentContractError("n must be exactly 1 in training mode.")
        if payload.get("n", 1) != 1:
            raise AgentContractError("PyTRIO backend requires n=1.")
        if "background" in payload and type(payload["background"]) is not bool:
            raise AgentContractError("background must be boolean in training mode.")
        if payload.get("background"):
            raise AgentContractError(
                "PyTRIO backend requires a synchronous completion."
            )
        if payload.get("seed") is not None:
            _strict_int(payload["seed"], name="seed")

    def _prompt_tokens(
        self, endpoint: str, payload: Mapping[str, Any], *, session: Any
    ) -> list[int]:
        """Render OpenAI input with the snapshot tokenizer into exact IDs.

        A reserved ``_triotune_input_ids`` field supports callers that already
        performed canonical rendering.  Otherwise Chat Completions messages or
        Responses items are rendered with the snapshot tokenizer, including
        tool schemas and Responses instructions where applicable.
        """
        supplied = payload.get("_triotune_input_ids")
        if supplied is not None:
            if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
                raise AgentContractError(
                    "_triotune_input_ids must be a token sequence."
                )
            tokens = list(supplied)
            validate_tokens(tokens, name="_triotune_input_ids")
            return tokens
        if self.renderer is not None:
            tokens = self.renderer(endpoint, payload)
            validate_tokens(tokens, name="rendered prompt")
            return list(tokens)
        if endpoint == "/v1/chat/completions":
            messages = payload.get("messages")
        else:
            messages = _response_messages(payload.get("input"))
            instructions = payload.get("instructions")
            if instructions:
                if not isinstance(instructions, str):
                    raise AgentContractError("Responses instructions must be text.")
                messages.insert(0, {"role": "system", "content": instructions})
        if endpoint == "/v1/chat/completions" and not isinstance(messages, Sequence):
            raise AgentContractError("Chat Completions messages must be a sequence.")
        tools = payload.get("tools", ()) or ()
        renderer = ChatTemplateRenderer(self.snapshot.tokenizer)
        tokens = renderer.render(messages, tools)
        # A continuation must begin with the exact parent request and sampled
        # output IDs. If the native tool layout cannot be represented by the
        # tokenizer template, reject it instead of re-encoding displayed text.
        validate_tokens(tokens, name="rendered prompt")
        return tokens

    @staticmethod
    def _canonical_messages(messages):
        """Validate supported text-only Chat messages into comparable values.

        JSON argument strings and dictionaries represent the same action, but
        multimodal content, arbitrary metadata, and missing tool-call IDs do
        not. This normalization is for lineage comparison and template input;
        it is never used to recover generated token IDs.
        """
        if not isinstance(messages, list) or not messages:
            raise AgentContractError("Native tools require nonempty text messages.")
        canonical = []
        for index, item in enumerate(messages):
            if not isinstance(item, Mapping):
                raise AgentContractError("Native messages must be objects.")
            # Official SDK model_dump() includes optional provider fields with
            # None values. Omitted/null optional fields are equivalent, but a
            # non-null audio/refusal/annotation payload is unsupported here.
            item = {
                key: value
                for key, value in item.items()
                if value is not None or key in {"role", "content"}
            }
            if set(item) - {
                "role",
                "content",
                "reasoning_content",
                "tool_calls",
                "tool_call_id",
            }:
                raise AgentContractError("Unsupported native message layout.")
            role = item.get("role")
            if (
                not isinstance(role, str)
                or role not in {"system", "user", "assistant", "tool"}
                or (role == "system" and index)
            ):
                raise AgentContractError(
                    "Unsupported native message role or system position."
                )
            content = item.get("content")
            if content is None and role == "assistant":
                content = ""
            if not isinstance(content, str):
                raise AgentContractError("Native message content must be text.")
            message = {"role": role, "content": content}
            if role == "assistant":
                reasoning = item.get("reasoning_content", "")
                if not isinstance(reasoning, str):
                    raise AgentContractError("Native reasoning_content must be text.")
                message["reasoning_content"] = reasoning
                if item.get("tool_calls"):
                    calls = []
                    for call in item["tool_calls"]:
                        if (
                            not isinstance(call, dict)
                            or call.get("type") != "function"
                            or not isinstance(call.get("id"), str)
                            or not call["id"]
                        ):
                            raise AgentContractError(
                                "Native assistant requires named function-call IDs."
                            )
                        function = call.get("function", {})
                        if not isinstance(function, dict):
                            raise AgentContractError(
                                "Native function descriptors must be objects."
                            )
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(
                                    arguments,
                                    parse_constant=_reject_constant,
                                    object_pairs_hook=_unique_object,
                                )
                            except (ValueError, TypeError) as exc:
                                raise AgentContractError(
                                    "Invalid historical tool arguments."
                                ) from exc
                        if not isinstance(arguments, dict) or not isinstance(
                            function.get("name"), str
                        ):
                            raise AgentContractError(
                                "Historical tool arguments must be objects."
                            )
                        calls.append(
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": function["name"],
                                    "arguments": arguments,
                                },
                            }
                        )
                    message["tool_calls"] = calls
            elif any(key in item for key in ("tool_calls", "reasoning_content")):
                raise AgentContractError(
                    "Only assistant messages can contain model actions."
                )
            if role == "tool":
                if (
                    not isinstance(item.get("tool_call_id"), str)
                    or not item["tool_call_id"]
                ):
                    raise AgentContractError(
                        "Native tool results require tool_call_id."
                    )
                message["tool_call_id"] = item["tool_call_id"]
            elif "tool_call_id" in item:
                raise AgentContractError("Only tool messages can carry tool_call_id.")
            canonical.append(message)
        # Third-party frameworks may submit parallel tool results in completion
        # order. OpenAI identifies them by call ID, whereas Qwen's template
        # renders positional tool_response blocks without IDs. Canonicalize
        # only observation order to the model's original call order; sampled
        # assistant tokens are still copied verbatim by the continuation renderer.
        index = 0
        while index < len(canonical):
            calls = canonical[index].get("tool_calls", [])
            end = index + 1
            if calls:
                while end < len(canonical) and canonical[end]["role"] == "tool":
                    end += 1
                order = {call["id"]: position for position, call in enumerate(calls)}
                values = canonical[index + 1 : end]
                identities = [value["tool_call_id"] for value in values]
                if (
                    len(order) != len(calls)
                    or len(set(identities)) != len(identities)
                    or any(identity not in order for identity in identities)
                ):
                    raise AgentContractError(
                        "Native observations require unique matching call IDs."
                    )
                canonical[index + 1 : end] = sorted(
                    values,
                    key=lambda value, ordering=order: ordering[value["tool_call_id"]],
                )
            index = end
        return canonical

    @staticmethod
    def _native_tools(payload):
        """Normalize function schemas and reject unsupported policy controls."""
        choice = payload.get("tool_choice", "auto")
        if choice not in ("auto", "none"):
            raise AgentContractError(
                "Qwen3.5 supports only tool_choice='auto' or 'none'."
            )
        raw = payload.get("tools", [])
        if not isinstance(raw, (list, tuple)):
            raise AgentContractError(
                "Native tools must be a sequence of function schemas."
            )
        tools = []
        for item in raw:
            if not isinstance(item, Mapping) or item.get("type") != "function":
                raise AgentContractError("Only native function tools are supported.")
            allowed = (
                {"type", "function"}
                if "function" in item
                else {"type", "name", "description", "parameters", "strict"}
            )
            if set(item) - allowed:
                raise AgentContractError("Unsupported native function tool layout.")
            function = copy.deepcopy(
                item.get(
                    "function",
                    {key: value for key, value in item.items() if key != "type"},
                )
            )
            if not isinstance(function, dict) or set(function) - {
                "name",
                "description",
                "parameters",
                "strict",
            }:
                raise AgentContractError("Unsupported native function descriptor.")
            name = function.get("name")
            if function.get("strict") not in (None, False):
                raise AgentContractError(
                    "Constrained strict tool generation is unsupported."
                )
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z_][\w.-]*", name) is None
            ):
                raise AgentContractError("Native function name is invalid.")
            schema = function.get("parameters", {})
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise AgentContractError(
                    "Native function parameters must be an object schema."
                )
            Qwen35ToolParser.check_schema(schema)
            function.pop("strict", None)
            tools.append({"type": "function", "function": function})
        if len({item["function"]["name"] for item in tools}) != len(tools):
            raise AgentContractError("Native function names must be unique.")
        try:
            json.dumps(tools, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise AgentContractError(
                "Native tools must contain finite JSON values."
            ) from exc
        return tools

    def _native_messages(self, endpoint, payload):
        """Convert full-history Responses items without losing call associations.

        Incremental provider-managed state and reasoning items without an
        inspectable summary cannot be rendered safely by this local backend.
        Those layouts are rejected explicitly; the external OpenAI backend is
        unaffected and continues to support its existing request contract.
        """
        if endpoint == "/v1/chat/completions":
            return self._canonical_messages(payload.get("messages"))
        if payload.get("previous_response_id") is not None:
            raise AgentContractError(
                "Qwen3.5 Responses requires full history, not previous_response_id."
            )
        value = payload.get("input")
        if isinstance(value, str):
            value = [{"role": "user", "content": value}]
        if not isinstance(value, list):
            raise AgentContractError(
                "Native Responses input must be text or a full item list."
            )
        messages = []
        for item in value:
            if not isinstance(item, dict):
                raise AgentContractError(
                    "Native Responses input items must be objects."
                )
            kind = item.get("type")
            if kind is not None and not isinstance(kind, str):
                raise AgentContractError("Native Responses item types must be strings.")
            layouts = {
                "message": {"type", "role", "content", "id", "status"},
                "reasoning": {"type", "summary", "id", "status"},
                "function_call": {
                    "type",
                    "name",
                    "arguments",
                    "call_id",
                    "id",
                    "status",
                },
                "function_call_output": {"type", "call_id", "output", "id", "status"},
            }
            if set(item) - layouts.get(kind or "message", set()):
                raise AgentContractError("Unsupported native Responses item fields.")
            role = item.get("role")
            if (
                isinstance(role, str)
                and role in {"system", "user", "assistant"}
                and kind
                in {
                    None,
                    "message",
                }
            ):
                content = item.get("content")
                if isinstance(content, list):
                    if any(
                        not isinstance(part, dict)
                        or part.get("type") not in {"input_text", "output_text"}
                        or not isinstance(part.get("text"), str)
                        for part in content
                    ):
                        raise AgentContractError(
                            "Only text Responses content is supported."
                        )
                    content = "".join(part["text"] for part in content)
                message = {"role": item["role"], "content": content}
                if (
                    messages
                    and messages[-1]["role"] == "assistant"
                    and item["role"] == "assistant"
                ):
                    messages[-1]["content"] += content
                else:
                    messages.append(message)
            elif kind == "reasoning":
                summary = item.get("summary")
                if not isinstance(summary, list) or any(
                    not isinstance(part, dict)
                    or part.get("type") != "summary_text"
                    or not isinstance(part.get("text"), str)
                    for part in summary
                ):
                    raise AgentContractError(
                        "Unsupported native Responses reasoning item."
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "".join(part["text"] for part in summary),
                    }
                )
            elif kind == "function_call":
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": ""})
                messages[-1].setdefault("tool_calls", []).append(
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments"),
                        },
                    }
                )
            elif kind == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": item.get("output"),
                    }
                )
            else:
                raise AgentContractError("Unsupported native Responses item layout.")
        if payload.get("instructions") is not None:
            if not isinstance(payload["instructions"], str):
                raise AgentContractError("Responses instructions must be text.")
            messages.insert(0, {"role": "system", "content": payload["instructions"]})
        return self._canonical_messages(messages)

    def _native_prompt(self, endpoint, payload, session, *, preflight: bool = False):
        """Verify parent messages, tools, and results before token-safe appending."""
        if self.snapshot.tokenizer.chat_template != self._native_template:
            raise AgentContractError("The bound native tokenizer template was changed.")
        if self.renderer is not None or "_triotune_input_ids" in payload:
            raise AgentContractError(
                "Native tools require the bound tokenizer renderer."
            )
        messages, tools = (
            self._native_messages(endpoint, payload),
            self._native_tools(payload),
        )
        renderer = NativeToolContinuationRenderer(
            self.snapshot.tokenizer, **self.template_kwargs
        )
        advertised_tools = tools if payload.get("tool_choice", "auto") == "auto" else []
        if preflight and session.interactions:
            # The client has not allocated this request yet, so parent lineage
            # is intentionally resolved by OpenAISession.start_request().
            # Validate the layout and template without guessing that parent.
            return renderer.render(messages, advertised_tools), messages, tools
        if not session.interactions:
            return renderer.render(messages, advertised_tools), messages, tools
        current = session.interactions[-1]
        parent_id = current.parent_interaction_id
        if parent_id is None:
            if any(item["role"] in {"assistant", "tool"} for item in messages):
                raise AgentContractError(
                    "Native continuation requires a verified parent interaction."
                )
            return renderer.render(messages, advertised_tools), messages, tools
        parent = session.get_interaction(parent_id)
        evidence = parent.response.get("_triotune_sampling", {})
        original_messages = evidence.get("model_messages")
        original_assistant = evidence.get("assistant_message")
        if (
            original_messages is None
            or original_assistant is None
            or evidence.get("model_tools") != tools
        ):
            raise AgentContractError(
                "Native parent lacks compatible model history or tool schemas."
            )
        if (
            evidence.get("tool_choice", "auto") != payload.get("tool_choice", "auto")
            or evidence.get("template_kwargs") != self.template_kwargs
        ):
            raise AgentContractError(
                "Native continuation changed template or tool-choice settings."
            )
        index = len(original_messages)
        if (
            messages[:index] != original_messages
            or len(messages) <= index
            or messages[index] != self._canonical_messages([original_assistant])[0]
        ):
            raise AgentContractError(
                "Native continuation changed the parent message prefix."
            )
        calls = messages[index].get("tool_calls", [])
        observations = messages[index + 1 :]
        if (
            not calls
            or len(observations) != len(calls)
            or any(item["role"] != "tool" for item in observations)
        ):
            raise AgentContractError(
                "Native continuation requires one result per parent tool call."
            )
        if [item["tool_call_id"] for item in observations] != [
            call["id"] for call in calls
        ]:
            raise AgentContractError(
                "Native tool results do not match parent call IDs/order."
            )
        if parent.tool_results:
            recorded = {item.get("call_id"): item for item in parent.tool_results}
            if len(recorded) != len(parent.tool_results) or set(recorded) != {
                call["id"] for call in calls
            }:
                raise AgentContractError(
                    "Recorded observations have incompatible tool-call IDs."
                )
            if any(
                recorded[item["tool_call_id"]].get("content") != item["content"]
                for item in observations
            ):
                raise AgentContractError(
                    "Native continuation changed recorded tool observations."
                )
        if parent.input_ids is None or parent.output_ids is None:
            raise AgentContractError(
                "Native parent is missing exact sampled token evidence."
            )
        tokens = renderer.render_after_tool(
            messages,
            advertised_tools,
            assistant_index=index,
            prompt_tokens=parent.input_ids,
            output_tokens=parent.output_ids,
        )
        return tokens, messages, tools

    def validate_request(
        self, endpoint: str, payload: Mapping[str, Any], *, session: Any
    ) -> None:
        """Run all deterministic preflight checks before request identity allocation."""
        if endpoint not in {"/v1/chat/completions", "/v1/responses"}:
            raise AgentContractError(f"Unsupported OpenAI endpoint: {endpoint}.")
        self._validate_session(session)
        if payload.get("model") != self.model:
            raise AgentContractError(
                "Request model must match the model bound to the PyTRIO snapshot."
            )
        self._validate_sampling_options(payload)
        stop = payload.get("stop")
        sampler_stop = getattr(self.snapshot.sampler, "stop", None)
        if stop is not None and stop != sampler_stop:
            raise AgentContractError(
                "Request stop settings must match the bound PyTRIO snapshot."
            )
        if not self.native_tools and (payload.get("tools") or "tool_choice" in payload):
            raise AgentContractError(
                "No native tool format is registered for this model."
            )
        if self.native_tools:
            self._native_prompt(endpoint, payload, session, preflight=True)
        else:
            self._prompt_tokens(endpoint, payload, session=session)
        self._max_tokens(endpoint, payload)

    def _max_tokens(self, endpoint: str, payload: Mapping[str, Any]) -> int:
        """Resolve the endpoint-specific output budget without ambiguity."""
        unsupported = (
            {"max_output_tokens"}
            if endpoint == "/v1/chat/completions"
            else {"max_tokens", "max_completion_tokens"}
        )
        if unsupported.intersection(payload):
            raise AgentContractError(
                "Output token limit does not belong to this OpenAI endpoint."
            )
        names = (
            ("max_completion_tokens", "max_tokens")
            if endpoint == "/v1/chat/completions"
            else ("max_output_tokens",)
        )
        values = [payload[name] for name in names if name in payload]
        if len(values) > 1 and values[0] != values[1]:
            raise AgentContractError("Conflicting output token limits.")
        return (
            _strict_int(values[0], name=names[0], minimum=1)
            if values
            else self.max_tokens
        )

    async def complete(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        session: Any,
        method: Any | None = None,
    ) -> Any:
        """Sample one request and expose original IDs/logprobs as trace evidence.

        The returned object deliberately follows the corresponding OpenAI
        response shape so the existing native parser and tool loop remain
        usable.  ``_triotune_sampling`` is a private, JSON-safe envelope read
        by ``OpenAISession``; it is never treated as model-generated content.
        """
        del method
        if endpoint not in {"/v1/chat/completions", "/v1/responses"}:
            raise AgentContractError(f"Unsupported OpenAI endpoint: {endpoint}.")
        self._validate_session(session)
        if payload.get("model") != self.model:
            raise AgentContractError(
                "Request model must match the model bound to the PyTRIO snapshot."
            )
        self._validate_sampling_options(payload)
        if not self.native_tools and (payload.get("tools") or "tool_choice" in payload):
            raise AgentContractError(
                "No native tool format is registered for this model."
            )
        if self.native_tools:
            allowed = {
                "model",
                "messages",
                "input",
                "instructions",
                "tools",
                "tool_choice",
                "max_tokens",
                "max_completion_tokens",
                "max_output_tokens",
                "seed",
                "temperature",
                "top_p",
                "n",
                "stream",
                "background",
                "stop",
                "extra_headers",
                "previous_response_id",
                "_triotune_input_ids",
            }
            if set(payload) - allowed:
                raise AgentContractError("Unsupported native tool request options.")
        if self.native_tools:
            prompt_tokens, model_messages, model_tools = self._native_prompt(
                endpoint, payload, session
            )
        else:
            prompt_tokens = self._prompt_tokens(endpoint, payload, session=session)
            model_messages, model_tools = None, None
        max_tokens = self._max_tokens(endpoint, payload)
        generation = await self.snapshot.sampler.generate(
            prompt_tokens,
            max_tokens=max_tokens,
            seed=payload.get("seed"),
        )
        generation.__post_init__()
        if len(generation.tokens) > max_tokens:
            raise AgentContractError("Sampler exceeded the requested output budget.")
        reason = generation.stop_reason
        truncated = reason in {"length", "max_tokens", "max_length"}
        parsed = None
        parse_error = None
        parse_error_type = None
        if not truncated and self.output_parser is not None:
            try:
                parsed = self.output_parser.parse_generation(
                    generation,
                    tools=(model_tools or ())
                    if payload.get("tool_choice", "auto") == "auto"
                    else (),
                    prompt_tokens=prompt_tokens,
                )
                if (
                    not isinstance(parsed, ModelOutput)
                    or parsed.generation is not generation
                ):
                    raise AgentContractError(
                        "Model output parser must return ModelOutput with the original Generation."
                    )
                if (
                    not isinstance(parsed.action, AgentAction)
                    or not isinstance(parsed.assistant_message, dict)
                    or parsed.action.assistant_message != parsed.assistant_message
                    or not isinstance(parsed.reasoning, str)
                    or not isinstance(parsed.raw_text, str)
                ):
                    raise AgentContractError(
                        "Model parser action and presentation fields disagree."
                    )
                parsed.action.__post_init__()
            except ActionParseError as exc:
                # Return sampled evidence even when the model action is
                # malformed. The recording client raises after committing it,
                # ensuring failed traces retain every original token/logprob.
                parse_error = str(exc)
                parse_error_type = "action"
                parsed = None
            except AgentContractError as exc:
                # A broken adapter is not a model-action error. Preserve the
                # sampled evidence with a distinct error type so the recording
                # client can reject the integration without losing the trace
                # or presenting the malformed result as a successful answer.
                parse_error = str(exc)
                parse_error_type = "contract"
                parsed = None
        if parsed is None:
            parsed = ModelOutput(
                generation,
                AgentAction(
                    {"role": "assistant", "content": generation.text},
                    final_text=generation.text,
                ),
                {"role": "assistant", "content": generation.text},
                "",
                self.snapshot.tokenizer.decode(
                    list(generation.tokens), skip_special_tokens=False
                )
                if self.native_tools
                else generation.text,
            )
        evidence = {
            "input_ids": list(prompt_tokens),
            "output_ids": list(generation.tokens),
            "logprobs": list(generation.logprobs),
            "stop_reason": reason,
            "sampling_source": getattr(
                self.snapshot.sampler, "sampling_source", "pytrio"
            ),
            "snapshot_id": self.snapshot.snapshot_id,
            "policy_version": self.snapshot.policy_version,
            "raw_text": parsed.raw_text,
            "reasoning": parsed.reasoning,
            "parse_error": parse_error,
            "parse_error_type": parse_error_type,
        }
        if model_messages is not None:
            evidence.update(
                {
                    "model_messages": model_messages,
                    "assistant_message": parsed.assistant_message,
                    "model_tools": model_tools,
                    "template_kwargs": copy.deepcopy(self.template_kwargs),
                    "tool_choice": payload.get("tool_choice", "auto"),
                }
            )
        usage = {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generation.tokens),
            "input_tokens": len(prompt_tokens),
            "output_tokens": len(generation.tokens),
            "total_tokens": len(prompt_tokens) + len(generation.tokens),
        }
        response_id = f"pytrio_{uuid4().hex}"
        if endpoint == "/v1/chat/completions":
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "message": (parsed.assistant_message),
                        "finish_reason": (
                            "length"
                            if truncated
                            else "tool_calls"
                            if parsed.action.tool_calls
                            else "stop"
                        ),
                    }
                ],
                "usage": usage,
                "_triotune_sampling": evidence,
            }
        return {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "model": self.model,
            "status": "incomplete" if truncated else "completed",
            "output": (
                (
                    [
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": parsed.reasoning}
                            ],
                        }
                    ]
                    if parsed.reasoning
                    else []
                )
                + (
                    [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": parsed.assistant_message["content"],
                                }
                            ],
                        }
                    ]
                    if parsed.action.tool_calls and parsed.assistant_message["content"]
                    else []
                )
                + (
                    [
                        {
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        }
                        for call in parsed.action.tool_calls
                    ]
                    if parsed.action.tool_calls
                    else [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "incomplete" if truncated else "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": parsed.action.final_text or "",
                                }
                            ],
                        }
                    ]
                )
            ),
            "usage": usage,
            "_triotune_sampling": evidence,
        }

    async def close(self) -> None:
        """Keep trainer-owned snapshots alive; the trainer controls lifecycle."""


__all__ = ["ExternalOpenAIBackend", "OpenAIBackend", "PyTRIOBackend"]
