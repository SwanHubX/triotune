"""Expose optional local ASGI routes over the session-aware OpenAI client.

The proxy is deliberately an embedding adapter rather than a public server: it
forwards two non-streaming endpoints, keeps sessions in a bounded in-memory
registry, and returns identity headers that let callers continue or release a
trace.  Authentication, process supervision, durable storage, and policy
allocation remain responsibilities of the host application.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from .backend import OpenAIBackend
from .client import OpenAIClient
from .session import OpenAISession


class OpenAIProxyServer:
    """Forward non-streaming text/function API calls and return session identity.

    This is a local forwarding adapter, not a process supervisor or public
    deployment service. Clients associate calls using ``X-Triotune-Session-Id``;
    the generated session and request IDs are returned as response headers.
    Session traces are available to the embedding application via ``sessions``.
    The injected SDK client retains its caller-owned resource lifecycle.
    A ``session_factory`` can provide collector-bound training sessions when a
    PyTRIO backend is used; the default factory creates inference sessions.
    """

    def __init__(
        self,
        client: OpenAIClient | None = None,
        *,
        backend: OpenAIBackend | None = None,
        session_factory: Callable[[str | None], OpenAISession] | None = None,
        max_sessions: int = 1024,
        max_released_session_ids: int = 1024,
    ):
        """Configure transport/backend selection and bounded session storage.

        ``session_factory`` is the injection point for collector-owned training
        sessions.  Without it, newly allocated sessions are inference records;
        this prevents an HTTP session ID from being mistaken for a rollout or
        policy snapshot identity.
        """
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        if type(max_released_session_ids) is not int or max_released_session_ids < 1:
            raise ValueError("max_released_session_ids must be a positive integer.")
        if client is not None and backend is not None:
            raise ValueError("Choose either an OpenAI client or a backend.")
        self.client = OpenAIClient(backend=backend) if backend is not None else client
        self.session_factory = session_factory
        self.max_sessions = max_sessions
        self.max_released_session_ids = max_released_session_ids
        self.sessions: dict[str, OpenAISession] = {}
        self._released_session_ids: OrderedDict[str, None] = OrderedDict()

    def _session(self, session_id):
        """Resolve or allocate one session while enforcing capacity and tombstones."""
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
        session = (
            self.session_factory(session_id)
            if self.session_factory is not None
            else OpenAISession(session_id=session_id)
        )
        if not isinstance(session, OpenAISession):
            raise TypeError("session_factory must return an OpenAISession.")
        if session_id is not None and session.session_id != session_id:
            raise ValueError("session_factory must preserve the requested session ID.")
        self.sessions[session.session_id] = session
        return session

    def release_session(self, session_id: str) -> OpenAISession:
        """Transfer a completed trace out of the bounded registry.

        Release first freezes the session, then records a bounded tombstone so
        a late retry cannot silently append to the released trace.  The trace
        object returned to the caller remains exportable for diagnostics.
        """
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
        """Seal an idle session before delayed rewards, export, or release.

        Successful HTTP requests leave the session open for continuation.
        The embedding application decides when its external agent has ended;
        failed/cancelled SDK calls close their session automatically.
        """
        self.sessions[session_id].finish(status, error=error)

    async def chat_completions(
        self, payload: dict[str, Any], *, session_id=None, parent_interaction_id=None
    ) -> Any:
        """Forward one Chat Completions payload through a session registry entry."""
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
        """Forward one Responses payload through a session registry entry."""
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
        """Share allocation and cleanup rules between direct and HTTP handlers.

        Payload identity fields are rejected because headers and handler
        arguments are the trusted source of session and parent IDs.  A backend
        failure after interaction allocation is preserved in the session; only
        an empty session created for a preflight validation error is reclaimed.
        """
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
        """Build an optional FastAPI ASGI app without launching a server.

        The nested forwarding handler translates local validation failures into
        400 responses, preserves upstream HTTP status classes, and always
        includes the identity headers that the caller needs to inspect or
        release a failed request.
        """
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        app = FastAPI()

        async def forward(endpoint, payload, session_id, parent_interaction_id):
            """Execute one route request while retaining its precise identity."""
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
