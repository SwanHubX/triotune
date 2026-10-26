"""Optional local ASGI routes for recorded OpenAI SDK requests."""

from __future__ import annotations

from typing import Any

from .client import OpenAIClient
from .session import OpenAISession


class OpenAIProxyServer:
    """Forward non-streaming text/function API calls and return session identity.

    This is a local forwarding adapter, not a process supervisor or public
    deployment service. Clients associate calls using ``X-Triotune-Session-Id``;
    the generated session and request IDs are returned as response headers.
    Session traces are available to the embedding application via ``sessions``.
    The injected SDK client retains its caller-owned resource lifecycle.
    """

    def __init__(self, client: OpenAIClient, *, max_sessions: int = 1024):
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        self.client = client
        self.max_sessions = max_sessions
        self.sessions: dict[str, OpenAISession] = {}

    def _session(self, session_id):
        if session_id is not None and (not session_id or len(session_id) > 128):
            raise ValueError("Session IDs must contain 1 to 128 characters.")
        if session_id in self.sessions:
            return self.sessions[session_id]
        if len(self.sessions) >= self.max_sessions:
            raise ValueError(
                "Session capacity reached; release a completed session first."
            )
        session = OpenAISession(session_id=session_id)
        self.sessions[session.session_id] = session
        return session

    def release_session(self, session_id: str) -> OpenAISession:
        """Transfer a completed trace out of the server's bounded registry."""
        session = self.sessions[session_id]
        if any(item.status == "running" for item in session.interactions):
            raise ValueError("Cannot release a session with an active request.")
        return self.sessions.pop(session_id)

    async def chat_completions(
        self, payload: dict[str, Any], *, session_id=None
    ) -> Any:
        session = self._session(session_id)
        return await self.client.create_chat_completion(session=session, **payload)

    async def responses(self, payload: dict[str, Any], *, session_id=None) -> Any:
        session = self._session(session_id)
        return await self.client.create_response(session=session, **payload)

    def create_app(self):
        """Build an optional FastAPI application without launching a server."""
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse
        from openai import APIStatusError

        app = FastAPI()

        async def forward(endpoint, payload, session_id):
            # Select one session here and pass it explicitly. A shared client
            # is never mutated, so independent HTTP requests cannot mix traces.
            try:
                if "session" in payload:
                    raise ValueError("Supply session identity through the header.")
                session = self._session(session_id)
                response = await endpoint(session=session, **payload)
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except APIStatusError as exc:
                # Keep upstream API failures visible to downstream SDK retry
                # policy without turning them into successful trace entries.
                body = exc.body if isinstance(exc.body, dict) else {"message": str(exc)}
                return JSONResponse(
                    status_code=exc.status_code, content={"error": body}
                )
            interaction = session.interactions[-1]
            return JSONResponse(
                content=session._jsonable(response),
                headers={
                    "X-Triotune-Session-Id": session.session_id,
                    "X-Client-Request-Id": interaction.request_id,
                    "X-Request-Id": interaction.request_id,
                },
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(
            payload: dict[str, Any],
            session_id: str | None = Header(
                default=None, alias="X-Triotune-Session-Id"
            ),
        ):
            return await forward(
                self.client.create_chat_completion, payload, session_id
            )

        @app.post("/v1/responses")
        async def responses(
            payload: dict[str, Any],
            session_id: str | None = Header(
                default=None, alias="X-Triotune-Session-Id"
            ),
        ):
            return await forward(self.client.create_response, payload, session_id)

        return app
