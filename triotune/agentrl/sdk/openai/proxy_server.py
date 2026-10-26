"""Optional local ASGI routes for recorded OpenAI SDK requests."""

from __future__ import annotations

from collections import OrderedDict
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

    def __init__(
        self,
        client: OpenAIClient,
        *,
        max_sessions: int = 1024,
        max_released_session_ids: int = 1024,
    ):
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        if type(max_released_session_ids) is not int or max_released_session_ids < 1:
            raise ValueError("max_released_session_ids must be a positive integer.")
        self.client = client
        self.max_sessions = max_sessions
        self.max_released_session_ids = max_released_session_ids
        self.sessions: dict[str, OpenAISession] = {}
        self._released_session_ids: OrderedDict[str, None] = OrderedDict()

    def _session(self, session_id):
        if session_id is not None and (not session_id or len(session_id) > 128):
            raise ValueError("Session IDs must contain 1 to 128 characters.")
        if session_id in self._released_session_ids:
            raise ValueError("A released session ID cannot be reused.")
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
        session.release()
        # Tombstones reject late retries within a bounded FIFO retention
        # window. Evicted IDs can be reused; the old trace stays released.
        self._released_session_ids[session_id] = None
        if len(self._released_session_ids) > self.max_released_session_ids:
            self._released_session_ids.popitem(last=False)
        return self.sessions.pop(session_id)

    def finish_session(
        self, session_id: str, status: str = "completed", *, error: str | None = None
    ) -> None:
        """Seal an idle session before delayed rewards, export or release.

        Successful HTTP requests leave the session open for continuation.
        The embedding application decides when its external agent has ended;
        failed/cancelled SDK calls close their session automatically.
        """
        self.sessions[session_id].finish(status, error=error)

    async def chat_completions(
        self, payload: dict[str, Any], *, session_id=None, parent_interaction_id=None
    ) -> Any:
        return (
            await self._request(
                self.client.create_chat_completion,
                payload,
                session_id,
                parent_interaction_id,
            )
        )[0]

    async def responses(
        self, payload: dict[str, Any], *, session_id=None, parent_interaction_id=None
    ) -> Any:
        return (
            await self._request(
                self.client.create_response, payload, session_id, parent_interaction_id
            )
        )[0]

    def _discard_empty_session(self, session, is_new):
        """Reclaim only a newly allocated session with no execution evidence.

        Validation failures before request registration have no trace to keep.
        Existing sessions, failed requests and active work must survive so that
        callers can continue or inspect them. Discarding is not a release and
        therefore creates no tombstone that would block a corrected retry.
        """
        if is_new and not session.interactions and not session.running:
            self.sessions.pop(session.session_id, None)

    async def _request(
        self,
        endpoint,
        payload,
        session_id,
        parent_interaction_id,
        *,
        session=None,
        is_new=None,
    ):
        """Share allocation cleanup between Python handlers and ASGI routes."""
        if session is None:
            is_new = session_id not in self.sessions
            session = self._session(session_id)
        try:
            if {"session", "parent_interaction_id"}.intersection(payload):
                raise ValueError(
                    "Supply session identity through the header or handler arguments."
                )
            response = await endpoint(
                session=session, parent_interaction_id=parent_interaction_id, **payload
            )
            return response, session
        except BaseException:
            self._discard_empty_session(session, is_new)
            raise

    @staticmethod
    def _identity_headers(session, request_count):
        """Expose only the interaction registered by this specific HTTP call.

        A rejected continuation must not borrow the preceding interaction's
        request ID. Session identity is still returned for validation failures;
        request/interaction IDs are included only when new evidence exists.
        """
        if session is None:
            return {}
        headers = {"X-Triotune-Session-Id": session.session_id}
        if len(session.interactions) > request_count:
            interaction = session.interactions[request_count]
            headers.update(
                {
                    "X-Triotune-Interaction-Id": interaction.interaction_id,
                    "X-Client-Request-Id": interaction.request_id,
                    "X-Request-Id": interaction.request_id,
                }
            )
        return headers

    def create_app(self):
        """Build an optional FastAPI application without launching a server."""
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        app = FastAPI()

        async def forward(endpoint, payload, session_id, parent_interaction_id):
            # Select one session here and pass it explicitly. A shared client
            # is never mutated, so independent HTTP requests cannot mix traces.
            session, request_count = None, 0
            try:
                is_new = session_id not in self.sessions
                session = self._session(session_id)
                request_count = len(session.interactions)
                response, _ = await self._request(
                    endpoint,
                    payload,
                    session_id,
                    parent_interaction_id,
                    session=session,
                    is_new=is_new,
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                    headers=self._identity_headers(session, request_count),
                ) from exc
            except APIStatusError as exc:
                # Keep upstream API failures visible to downstream SDK retry
                # policy without turning them into successful trace entries.
                body = exc.body if isinstance(exc.body, dict) else {"message": str(exc)}
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"error": body},
                    headers=self._identity_headers(session, request_count),
                )
            except APIConnectionError as exc:
                return JSONResponse(
                    status_code=504 if isinstance(exc, APITimeoutError) else 502,
                    content={
                        "error": {"message": str(exc), "type": type(exc).__name__}
                    },
                    headers=self._identity_headers(session, request_count),
                )
            except Exception:  # noqa: BLE001 -- preserve identity for unexpected SDK failures
                # The client retains the original exception in its failed
                # interaction. Do not lose that trace behind a bare ASGI 500
                # or expose arbitrary internal exception text to HTTP callers.
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": "Proxy request failed.",
                            "type": "proxy_error",
                        }
                    },
                    headers=self._identity_headers(session, request_count),
                )
            return JSONResponse(
                content=session._jsonable(response),
                headers=self._identity_headers(session, request_count),
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(
            payload: dict[str, Any],
            session_id: str | None = Header(
                default=None, alias="X-Triotune-Session-Id"
            ),
            parent_interaction_id: str | None = Header(
                default=None, alias="X-Triotune-Parent-Interaction-Id"
            ),
        ):
            return await forward(
                self.client.create_chat_completion,
                payload,
                session_id,
                parent_interaction_id,
            )

        @app.post("/v1/responses")
        async def responses(
            payload: dict[str, Any],
            session_id: str | None = Header(
                default=None, alias="X-Triotune-Session-Id"
            ),
            parent_interaction_id: str | None = Header(
                default=None, alias="X-Triotune-Parent-Interaction-Id"
            ),
        ):
            return await forward(
                self.client.create_response, payload, session_id, parent_interaction_id
            )

        return app
