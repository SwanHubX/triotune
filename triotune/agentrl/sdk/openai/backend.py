"""Request backends for OpenAI-compatible agent execution.

The external backend forwards requests to an official OpenAI client.  The
PyTRIO backend instead binds one immutable ``SamplingSnapshot`` and converts
the sampled ``Generation`` into an OpenAI-shaped response for the existing
tool loop.  Keeping this boundary small prevents transport concerns from
changing session, reward, or PPO provenance rules.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from ...runtime.conversation import ChatTemplateRenderer
from ...runtime.sampling import (
    PPO_TEMPERATURE,
    PPO_TOP_P,
    SamplingSnapshot,
)
from ...types import AgentContractError, ExecutionContext, validate_tokens


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

    async def complete(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        session: Any,
        method: Any | None = None,
    ) -> Any:
        """Call the SDK method without blocking an asyncio event loop."""
        del session
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
                    "content": item.get("arguments", ""),
                    "name": item.get("name", ""),
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
    """

    def __init__(
        self,
        snapshot: SamplingSnapshot,
        *,
        model: str,
        max_tokens: int = 256,
        renderer: Any | None = None,
    ) -> None:
        """Bind a fixed snapshot, tokenizer renderer, model, and output budget."""
        if not isinstance(snapshot, SamplingSnapshot):
            raise AgentContractError("PyTRIO backend requires a SamplingSnapshot.")
        if not isinstance(model, str) or not model:
            raise ValueError("PyTRIO backend model must be a nonempty string.")
        self.snapshot = snapshot
        self.model = model
        self.max_tokens = _strict_int(max_tokens, name="max_tokens", minimum=1)
        self.renderer = renderer
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
        return context

    @staticmethod
    def _validate_sampling_options(payload: Mapping[str, Any]) -> None:
        """Keep PPO behavior sampling deterministic at temperature/top-p one.

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
        if "seed" in payload:
            _strict_int(payload["seed"], name="seed")

    def _prompt_tokens(self, endpoint: str, payload: Mapping[str, Any]) -> list[int]:
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
        validate_tokens(tokens, name="rendered prompt")
        return tokens

    def _max_tokens(self, endpoint: str, payload: Mapping[str, Any]) -> int:
        """Resolve the endpoint-specific output budget without ambiguity."""
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
        prompt_tokens = self._prompt_tokens(endpoint, payload)
        max_tokens = self._max_tokens(endpoint, payload)
        stop = payload.get("stop")
        sampler_stop = getattr(self.snapshot.sampler, "stop", None)
        if stop is not None and stop != sampler_stop:
            raise AgentContractError(
                "Request stop settings must match the bound PyTRIO snapshot."
            )
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
        }
        usage = {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(generation.tokens),
            "input_tokens": len(prompt_tokens),
            "output_tokens": len(generation.tokens),
        }
        response_id = f"pytrio_{uuid4().hex}"
        if endpoint == "/v1/chat/completions":
            return {
                "id": response_id,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": generation.text},
                        "finish_reason": "length" if truncated else "stop",
                    }
                ],
                "usage": usage,
                "_triotune_sampling": evidence,
            }
        return {
            "id": response_id,
            "model": self.model,
            "status": "incomplete" if truncated else "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete" if truncated else "completed",
                    "content": [{"type": "output_text", "text": generation.text}],
                }
            ],
            "usage": usage,
            "_triotune_sampling": evidence,
        }

    async def close(self) -> None:
        """Keep trainer-owned snapshots alive; the trainer controls lifecycle."""


__all__ = ["ExternalOpenAIBackend", "OpenAIBackend", "PyTRIOBackend"]
