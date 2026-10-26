"""Expose optional local ASGI routes over the session-aware OpenAI client.

The proxy is deliberately an embedding adapter rather than a public server: it
forwards two non-streaming endpoints, keeps sessions in a bounded in-memory
registry, and returns identity headers that let callers continue or release a
trace.  Authentication, process supervision, durable storage, and policy
allocation remain responsibilities of the host application.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from ...types import AgentContractError
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

    def create_client(self, *, session: OpenAISession, **options: Any):
        """Create a genuine AsyncOpenAI whose local transport pins session ID.

        This client is suitable for third-party asynchronous agent functions:
        they call the standard chat.completions/responses APIs without adding
        identity headers themselves. Transport-level injection also protects
        requests made through SDK with_options() copies: conflicting session
        headers are rejected instead of allocating a second session.

        The ASGI app is embedded in-process and never opens a network socket.
        Only the SDK client and its local HTTP transport are owned by this
        factory. The proxy's upstream adapter, tools, snapshots, and session
        lifecycle remain externally owned. Closing the SDK transport cancels
        and joins its own in-flight requests before session finalization.
        Automatic SDK retries are disabled
        because a failed recorded request terminalizes its session and cannot
        be replayed as a new candidate under the same identity.
        """
        import httpx
        from openai import AsyncOpenAI

        if not isinstance(session, OpenAISession):
            raise AgentContractError("A bound SDK client requires OpenAISession.")
        session.validate_identity()
        if self.client is None:
            raise AgentContractError(
                "A bound SDK client requires a forwarding adapter."
            )
        if len(session.session_id) > 128:
            raise AgentContractError(
                "Bound session IDs must contain at most 128 characters."
            )
        if session.status != "open" or session.released:
            raise AgentContractError("A bound SDK client requires an open session.")
        if {"http_client", "base_url"}.intersection(options):
            raise AgentContractError("The bound SDK factory owns its local connection.")
        if (
            type(options.get("max_retries", 0)) is not int
            or options.get("max_retries", 0) != 0
        ):
            raise AgentContractError("Recorded SDK clients require max_retries=0.")
        raw_headers = options.pop("default_headers", None)
        raw_headers = {} if raw_headers is None else raw_headers
        if not isinstance(raw_headers, Mapping) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str)
            for key, value in raw_headers.items()
        ):
            raise AgentContractError(
                "SDK default_headers must map nonempty strings to strings."
            )
        headers = dict(raw_headers)
        identity_header = "x-triotune-session-id"
        if any(
            key.lower() == identity_header and value != session.session_id
            for key, value in headers.items()
        ):
            raise AgentContractError("SDK headers conflict with the bound session.")
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() != identity_header
        }
        headers["X-Triotune-Session-Id"] = session.session_id
        existing = self.sessions.get(session.session_id)
        if existing is not None and existing is not session:
            raise AgentContractError(
                "Session ID is already bound to another session object."
            )
        if session.session_id in self._released_session_ids:
            raise AgentContractError("A released session ID cannot be rebound.")
        if existing is None and len(self.sessions) >= self.max_sessions:
            raise AgentContractError("Session capacity reached.")

        transport = httpx.ASGITransport(app=self.create_app())

        class BoundSessionTransport(httpx.AsyncBaseTransport):
            """Enforce identity even when per-request SDK headers override defaults."""

            def __init__(self):
                self.active = set()
                self.closed = False

            async def handle_async_request(self, request):
                if self.closed:
                    raise AgentContractError("The bound SDK transport is closed.")
                session.validate_identity()
                supplied = request.headers.get(identity_header)
                if supplied is not None and supplied != session.session_id:
                    raise AgentContractError(
                        "Request attempts to switch its bound session."
                    )
                request.headers[identity_header] = session.session_id
                # One explicit identity survives transport/API retries and is
                # independent of prompt contents. Replaying the same prepared
                # SDK request can therefore return its cached completion, while
                # a fresh application request receives a new identity.
                request.headers.setdefault(
                    "X-Triotune-Request-Id", f"sdk_{uuid4().hex}"
                )
                task = asyncio.current_task()
                self.active.add(task)
                try:
                    return await transport.handle_async_request(request)
                finally:
                    self.active.discard(task)

            async def aclose(self):
                # A third-party callback may fail while child request tasks are
                # still active. Cancel and join only this transport's work so
                # session finalization never leaves an orphan running record.
                # No caller-owned upstream client or unrelated task is closed.
                self.closed = True
                pending = [
                    task for task in self.active if task is not asyncio.current_task()
                ]
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                await transport.aclose()

        # Reserve the exact caller-provided session before the first request.
        # A session factory must not reconstruct its collector ExecutionContext.
        sdk = AsyncOpenAI(
            api_key=options.pop("api_key", "triotune-local"),
            base_url="http://triotune.local/v1",
            default_headers=headers,
            http_client=httpx.AsyncClient(transport=BoundSessionTransport()),
            **{"max_retries": 0, **options},
        )
        self.sessions[session.session_id] = session
        return sdk

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
        self,
        payload: dict[str, Any],
        *,
        session_id=None,
        parent_interaction_id=None,
        request_id=None,
    ) -> Any:
        """Forward one Chat Completions payload through a session registry entry."""
        return (
            await self._request(
                self.client.create_chat_completion,
                payload,
                session_id,
                parent_interaction_id,
                request_id,
            )
        )[0]

    async def responses(
        self,
        payload: dict[str, Any],
        *,
        session_id=None,
        parent_interaction_id=None,
        request_id=None,
    ) -> Any:
        """Forward one Responses payload through a session registry entry."""
        return (
            await self._request(
                self.client.create_response,
                payload,
                session_id,
                parent_interaction_id,
                request_id,
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
        request_id,
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
                session=session,
                parent_interaction_id=parent_interaction_id,
                request_id=request_id,
                **payload,
            )
            return response, session
        except BaseException:
            self._discard_empty_session(session, is_new)
            raise

    @staticmethod
    def _identity_headers(session, request_count, request_id=None):
        """Expose only the interaction registered by this specific HTTP call.

        A rejected continuation must not borrow the preceding interaction's
        request ID. Session identity is still returned for validation failures;
        request/interaction IDs are included only when new evidence exists.
        """
        if session is None:
            return {}
        headers = {"X-Triotune-Session-Id": session.session_id}
        interaction = None
        if len(session.interactions) > request_count:
            interaction = session.interactions[request_count]
        elif request_id in session.request_to_interaction:
            interaction = session.get_interaction(request_id=request_id)
        if interaction is not None:
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

        async def forward(
            endpoint, payload, session_id, parent_interaction_id, request_id
        ):
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
                    request_id,
                    session=session,
                    is_new=is_new,
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                    headers=self._identity_headers(session, request_count, request_id),
                ) from exc
            except APIStatusError as exc:
                # Keep upstream API failures visible to downstream SDK retry
                # policy without turning them into successful trace entries.
                body = exc.body if isinstance(exc.body, dict) else {"message": str(exc)}
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"error": body},
                    headers=self._identity_headers(session, request_count, request_id),
                )
            except APIConnectionError as exc:
                return JSONResponse(
                    status_code=504 if isinstance(exc, APITimeoutError) else 502,
                    content={
                        "error": {"message": str(exc), "type": type(exc).__name__}
                    },
                    headers=self._identity_headers(session, request_count, request_id),
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
                    headers=self._identity_headers(session, request_count, request_id),
                )
            return JSONResponse(
                content=session._jsonable(response),
                headers=self._identity_headers(session, request_count, request_id),
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
            request_id: str | None = Header(
                default=None, alias="X-Triotune-Request-Id"
            ),
        ):
            return await forward(
                self.client.create_chat_completion,
                payload,
                session_id,
                parent_interaction_id,
                request_id,
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
            request_id: str | None = Header(
                default=None, alias="X-Triotune-Request-Id"
            ),
        ):
            return await forward(
                self.client.create_response,
                payload,
                session_id,
                parent_interaction_id,
                request_id,
            )

        return app
