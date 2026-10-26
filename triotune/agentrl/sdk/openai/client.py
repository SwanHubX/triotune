"""Async request recording around the official OpenAI Python SDK."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Mapping
from typing import Any

from ...types import AgentEvent, _execution_context
from .session import OpenAISession


class OpenAIClient:
    """Record non-streaming text/function requests using an SDK client.

    Official sync and async clients are accepted; sync operations run in a
    worker thread. Only a client created here is closed here. SDK retry and
    timeout options remain caller controlled; the tool loop never retries a
    tool merely because an API request failed.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        session: OpenAISession | None = None,
        model: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self._owns_client = client is None
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(**client_kwargs)
        elif client_kwargs:
            raise ValueError("Client configuration belongs to the injected SDK client.")
        self.client = client
        self.session = session or OpenAISession()
        self.model = model

    async def create_chat_completion(
        self,
        *,
        messages,
        model=None,
        session=None,
        parent_interaction_id=None,
        **kwargs,
    ):
        """Submit one Chat Completions request with caller or default identity."""
        return await self._create(
            "/v1/chat/completions",
            self.client.chat.completions.create,
            session or self.session,
            {"messages": messages, "model": model or self.model, **kwargs},
            parent_interaction_id=parent_interaction_id,
        )

    async def create_response(
        self, *, input, model=None, session=None, parent_interaction_id=None, **kwargs
    ):
        """Submit one Responses request with caller or default identity."""
        return await self._create(
            "/v1/responses",
            self.client.responses.create,
            session or self.session,
            {"input": input, "model": model or self.model, **kwargs},
            parent_interaction_id=parent_interaction_id,
        )

    async def _create(
        self, endpoint, method, session, payload, *, parent_interaction_id=None
    ):
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
            if inspect.iscoroutinefunction(method):
                response = await method(**payload)
            else:
                response = await asyncio.to_thread(method, **payload)
                if inspect.isawaitable(response):
                    response = await response
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
        """Close an internally created SDK client; injected clients stay owned by callers."""
        if self._owns_client:
            result = self.client.close()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()
