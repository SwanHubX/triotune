"""Async request recording around the official OpenAI Python SDK."""

from __future__ import annotations

import asyncio
import inspect
import time
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
        self, *, messages, model=None, session=None, **kwargs
    ):
        """Submit one Chat Completions request with caller or default identity."""
        return await self._create(
            "/v1/chat/completions",
            self.client.chat.completions.create,
            session or self.session,
            {"messages": messages, "model": model or self.model, **kwargs},
        )

    async def create_response(self, *, input, model=None, session=None, **kwargs):
        """Submit one Responses request with caller or default identity."""
        return await self._create(
            "/v1/responses",
            self.client.responses.create,
            session or self.session,
            {"input": input, "model": model or self.model, **kwargs},
        )

    async def _create(self, endpoint, method, session, payload):
        if not payload.get("model"):
            raise ValueError("An OpenAI model must be specified.")
        if payload.get("stream"):
            raise ValueError(
                "Streaming is not supported by the recorded inline client."
            )
        if payload.get("n", 1) != 1 or payload.get("background"):
            raise ValueError("Recorded requests require one synchronous completion.")
        interaction = session.start_request(endpoint, payload)
        headers = {
            key: value
            for key, value in (payload.get("extra_headers") or {}).items()
            if key.lower() != "x-client-request-id"
        }
        headers["X-Client-Request-Id"] = interaction.request_id
        payload["extra_headers"] = headers
        token = _execution_context.set(interaction.context)
        try:
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
            raise
        finally:
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
