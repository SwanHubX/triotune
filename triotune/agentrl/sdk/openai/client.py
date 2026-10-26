"""Record OpenAI-compatible requests through a replaceable async backend.

``OpenAIClient`` owns request preparation and session bookkeeping while a
backend owns the actual completion mechanism.  The default backend forwards
to an injected official SDK client; ``PyTRIOBackend`` uses a fixed training
snapshot.  Both paths receive the same request ID header and interaction
context, so transport choice cannot alter provenance or failure cleanup.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from ...types import AgentEvent, _execution_context
from .backend import ExternalOpenAIBackend, OpenAIBackend
from .session import OpenAISession


class OpenAIClient:
    """Record non-streaming text/function requests using an SDK client.

    Official sync and async clients are accepted; sync operations run in a
    worker thread. Only a client created here is closed here. SDK retry and
    timeout options remain caller controlled; the tool loop never retries a
    tool merely because an API request failed.

    When ``backend`` is supplied, ``client`` is intentionally absent and all
    endpoint calls are delegated to that object.  Resource ownership follows
    the object that was constructed here: injected SDK clients and trainer
    snapshots are never closed by this adapter.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        session: OpenAISession | None = None,
        model: str | None = None,
        backend: OpenAIBackend | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Select an external SDK or custom backend without creating requests."""
        if backend is not None and client is not None:
            raise ValueError("Choose either an SDK client or a backend, not both.")
        self._owns_client = client is None and backend is None
        if client is None and backend is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(**client_kwargs)
        elif client_kwargs:
            raise ValueError("Client configuration belongs to the injected SDK client.")
        self.client = client
        self.backend = backend or ExternalOpenAIBackend(
            client, owns_client=self._owns_client
        )
        self.session = session or OpenAISession()
        self.model = model or getattr(self.backend, "model", None)

    async def create_chat_completion(
        self,
        *,
        messages,
        model=None,
        session=None,
        parent_interaction_id=None,
        **kwargs,
    ):
        """Submit one non-streaming Chat Completions request.

        The client inserts the configured model and lets ``_create`` validate
        headers, request cardinality, interaction lineage, and backend errors.
        The caller's message list remains owned by the caller.
        """
        return await self._create(
            "/v1/chat/completions",
            None if self.client is None else self.client.chat.completions.create,
            session or self.session,
            {
                "messages": messages,
                "model": self.model if model is None else model,
                **kwargs,
            },
            parent_interaction_id=parent_interaction_id,
        )

    async def create_response(
        self, *, input, model=None, session=None, parent_interaction_id=None, **kwargs
    ):
        """Submit one non-streaming Responses request with session identity."""
        return await self._create(
            "/v1/responses",
            None if self.client is None else self.client.responses.create,
            session or self.session,
            {"input": input, "model": self.model if model is None else model, **kwargs},
            parent_interaction_id=parent_interaction_id,
        )

    async def _create(
        self, endpoint, method, session, payload, *, parent_interaction_id=None
    ):
        """Validate, allocate, execute, and finalize one request interaction.

        Validation that can be completed without an interaction runs first so
        malformed headers or unsupported streaming options do not consume a
        session slot.  Once identity is allocated, every backend exception is
        recorded on that interaction and the session is terminalized when this
        client is the request owner.  The original exception is always
        re-raised to preserve SDK retry and cancellation behavior.
        """
        raw_headers = payload.get("extra_headers")
        raw_headers = {} if raw_headers is None else raw_headers
        if not isinstance(raw_headers, Mapping) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str)
            for key, value in raw_headers.items()
        ):
            raise ValueError("extra_headers must map nonempty strings to strings.")
        if not payload.get("model"):
            raise ValueError("An OpenAI model must be specified.")
        if payload.get("stream"):
            raise ValueError(
                "Streaming is not supported by the recorded inline client."
            )
        if payload.get("n", 1) != 1 or payload.get("background"):
            raise ValueError("Recorded requests require one synchronous completion.")
        # Copy caller-owned headers before recording an active request. Invalid
        # configuration must not leave a request that can never be finalized.
        headers = {
            key: value
            for key, value in raw_headers.items()
            if key.lower() != "x-client-request-id"
        }
        token = None
        interaction = session.start_request(
            endpoint, payload, parent_interaction_id=parent_interaction_id
        )
        try:
            headers["X-Client-Request-Id"] = interaction.request_id
            payload["extra_headers"] = headers
            token = _execution_context.set(interaction.context)
            response = await self.backend.complete(
                endpoint, payload, session=session, method=method
            )
            session.complete_request(interaction, response)
            return response
        except BaseException as exc:
            interaction.status = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            interaction.error = f"{type(exc).__name__}: {exc}"
            interaction.server_request_id = getattr(exc, "request_id", None)
            interaction.latency = time.monotonic() - interaction.started_at
            session.events.append(
                AgentEvent(
                    context=interaction.context,
                    kind="openai_failure",
                    status=interaction.status,
                )
            )
            if not session.running:
                try:
                    session.finish(interaction.status, error=interaction.error)
                except Exception as cleanup_error:  # noqa: BLE001 -- preserve the original request failure
                    # Preserve the request failure even if a custom session's
                    # cleanup fails; the original traceback remains actionable.
                    exc.add_note(f"Session finalization also failed: {cleanup_error}")
            raise
        finally:
            if token is not None:
                _execution_context.reset(token)

    async def close(self):
        """Close an owned backend while leaving injected resources untouched."""
        close = getattr(self.backend, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result

    async def __aenter__(self):
        """Enter an async context without changing backend ownership."""
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        """Close owned transport resources on context exit."""
        await self.close()
