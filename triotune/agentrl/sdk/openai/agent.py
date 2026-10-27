"""Run an OpenAI conversation and execute locally registered tools.

``Agent`` is intentionally an orchestration façade rather than a model
runtime.  It delegates transport and request identity to ``OpenAIClient``,
delegates schema validation and invocation to ``FunctionToolExecutor``, and
delegates lineage, lifecycle, and delayed reward rules to ``OpenAISession``.
This separation lets the same loop use an external OpenAI client or a fixed
PyTRIO backend without changing tool semantics.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any

from ...config import RetryPolicy
from ...runtime.runner import RuntimeBoundary
from ...runtime.tools import FunctionToolExecutor
from ...types import (
    AgentContractError,
    AgentEvent,
    ExecutionContext,
    FailureClassifier,
    ToolCallEvent,
    ToolResultEvent,
    _execution_context,
)
from .client import OpenAIClient
from .native import (
    OpenAIChatObservationFormatter,
    OpenAINativeActionParser,
    OpenAIResponsesObservationFormatter,
)
from .session import OpenAISession, OpenAITrajectoryExport


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Return the final action text together with its owned session trace.

    ``response`` is the provider-native final response and is retained for
    callers that need usage or provider-specific fields.  ``session`` remains
    the authoritative source for interaction IDs, tool evidence, status, and
    delayed rewards.  Exporting this result produces the SDK trace format, not
    an automatically trainable PPO rollout.
    """

    text: str
    session: OpenAISession
    response: Any
    accepted: bool = True

    @property
    def context(self) -> ExecutionContext:
        """Expose immutable candidate identity to generic lifecycle hooks."""
        return self.session.context

    @property
    def status(self) -> str:
        """Expose terminal SDK session status without becoming EpisodeResult."""
        return self.session.status

    @property
    def interactions(self):
        """Expose recorded SDK interactions for trace-aware scorers."""
        return tuple(self.session.interactions)

    @property
    def trainable(self) -> bool:
        """Keep API traces separate from OpenAIEpisodeExecutor's PPO records."""
        return False

    @property
    def session_id(self) -> str:
        """Expose the conversation ID without conflating it with rollout IDs."""
        return self.session.session_id

    def export_trajectory(self) -> OpenAITrajectoryExport:
        """Export the terminal session after lifecycle validation."""
        return self.session.export_trajectory()


class Agent:
    """Execute local tools requested by an official OpenAI SDK client.

    Every invocation gets its own session unless one is supplied explicitly.
    API retries/timeouts belong to the SDK client. Tools run once per call and
    retain their ``retry_safe`` metadata; this loop never retries side effects.
    The optional tool timeout bounds awaiting a tool, but Python cannot stop a
    synchronous worker thread that has already started.

    Chat Completions continuation messages preserve assistant tool-call IDs;
    Responses continuation preserves every output item, including reasoning
    items, before appending function results.  The enclosing execution context
    is installed around API and tool calls and restored even when cancellation
    or cleanup raises.
    """

    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        tools=(),
        model: str | None = None,
        client: OpenAIClient | Any | None = None,
        max_turns: int = 8,
        api: str = "chat_completions",
        tool_timeout: float = 10.0,
        episode_timeout: float | None = None,
        max_generated_tokens: int | None = None,
        tool_retry: RetryPolicy | None = None,
        failure_classifier: FailureClassifier | None = None,
        resource_owner: str = "openai-agent",
    ) -> None:
        """Validate static agent configuration and construct protocol helpers."""
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(instructions, str)
        ):
            raise ValueError("Agent requires a name and string instructions.")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer.")
        if api not in {"chat_completions", "responses"}:
            raise ValueError("api must be chat_completions or responses.")
        if isinstance(tool_timeout, bool) or not 0 < tool_timeout < float("inf"):
            raise ValueError("tool_timeout must be finite and positive.")
        if episode_timeout is not None and (
            isinstance(episode_timeout, bool)
            or not isinstance(episode_timeout, (int, float))
            or not 0 < episode_timeout < float("inf")
        ):
            raise ValueError("episode_timeout must be finite and positive or None.")
        if max_generated_tokens is not None and (
            type(max_generated_tokens) is not int or max_generated_tokens < 1
        ):
            raise ValueError("max_generated_tokens must be positive or None.")
        self.name, self.instructions, self.model = name, instructions, model
        self.max_turns, self.api, self.tool_timeout = max_turns, api, tool_timeout
        self.episode_timeout = episode_timeout
        self.max_generated_tokens = max_generated_tokens
        self.tool_retry = tool_retry or RetryPolicy()
        self.boundary = RuntimeBoundary(
            retry=self.tool_retry,
            failure_classifier=failure_classifier,
            resource_owner=resource_owner,
        )
        self.executor = FunctionToolExecutor(list(tools))
        self.client = (
            client
            if isinstance(client, OpenAIClient)
            else OpenAIClient(client, model=model)
        )
        self.parser = OpenAINativeActionParser()

    async def run(
        self,
        input: str | list[dict[str, Any]],
        *,
        session: OpenAISession | None = None,
        context: ExecutionContext | None = None,
        training: bool = False,
        **request_options: Any,
    ) -> AgentRunResult:
        """Run a fresh conversation in one isolated session until a final answer.

        Chat messages echo native tool call IDs. Responses echoes every output
        item, including reasoning items, before appending function-call outputs.
        This preserves the API's required continuation state for reasoning models.

        ``context`` and ``training`` are provenance controls, not model-facing
        request options.  A supplied training session must already be bound to
        the collector's root context; the method refuses identity overrides in
        ``request_options`` so a model or caller cannot rewrite ownership.
        """
        if {
            "messages",
            "input",
            "tools",
            "instructions",
            "previous_response_id",
            "rollout_id",
            "episode_id",
            "group_id",
            "sample_index",
            "group_size",
            "snapshot_id",
            "policy_version",
            "task_id",
            "resource_owner",
            "interaction_id",
            "parent_interaction_id",
        }.intersection(request_options):
            raise ValueError("Conversation and tool fields are owned by Agent.run.")
        if type(training) is not bool:
            raise AgentContractError("training must be boolean.")
        await self.start()
        session = (
            session
            if session is not None
            else OpenAISession(context=context, training=training)
        )
        session.validate_identity(context)
        if training and not session.training:
            raise AgentContractError(
                "A training run requires a session bound in training mode."
            )
        if self.max_generated_tokens is not None or self.episode_timeout is not None:
            session.configure_limits(
                max_generated_tokens=self.max_generated_tokens,
                episode_timeout=self.episode_timeout,
            )
        session.begin_execution()
        if (
            session.running
            or session.interactions
            or session.status != "open"
            or session.released
        ):
            raise ValueError(
                "Agent.run requires an unused session for an independent conversation."
            )
        if not isinstance(input, (str, list)):
            raise TypeError("Agent input must be text or a list of messages.")
        items = (
            [{"role": "user", "content": input}]
            if isinstance(input, str)
            else copy.deepcopy(input)
        )
        if not items:
            raise ValueError("Agent input must not be empty.")
        if self.api == "chat_completions":
            items.insert(0, {"role": "system", "content": self.instructions})
        session.running = True
        root_token = _execution_context.set(session.context)
        terminal_status, terminal_error = "completed", None
        failure = None
        stage = "api"
        try:
            for _ in range(self.max_turns):
                remaining = session.remaining_generated_tokens()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Agent total generated-token budget exceeded.")
                kwargs = {"session": session, "model": self.model, **request_options}
                if remaining is not None:
                    option = (
                        "max_completion_tokens"
                        if self.api == "chat_completions"
                        else "max_output_tokens"
                    )
                    kwargs.setdefault(option, remaining)
                if self.api == "chat_completions":
                    if self.executor.schemas:
                        kwargs["tools"] = self.executor.schemas
                    response = await self.client.create_chat_completion(
                        messages=items, **kwargs
                    )
                    action = self.parser.parse_chat_completion(response)
                    formatter = OpenAIChatObservationFormatter()
                else:
                    if self.executor.schemas:
                        # Responses uses a flat function descriptor. Explicit
                        # non-strict mode retains optional Python defaults.
                        kwargs["tools"] = [
                            dict(type="function", strict=False, **tool["function"])
                            for tool in self.executor.schemas
                        ]
                    response = await self.client.create_response(
                        input=items, instructions=self.instructions, **kwargs
                    )
                    action = self.parser.parse_response(response)
                    formatter = OpenAIResponsesObservationFormatter()
                interaction = session.interactions[-1]
                if not action.tool_calls:
                    if interaction.status != "completed":
                        # A provider may return a text payload with a length,
                        # cancellation, or other non-terminal status. Preserve
                        # the partial response for diagnostics, but do not mark
                        # the session as a successful training candidate.
                        terminal_status = (
                            "cancelled"
                            if interaction.status == "cancelled"
                            else "failed"
                        )
                        terminal_error = (
                            f"OpenAI request ended with status {interaction.status!r}."
                        )
                    return AgentRunResult(action.final_text or "", session, response)
                if self.api == "chat_completions":
                    items.append(action.assistant_message)
                else:
                    items.extend(copy.deepcopy(interaction.response["output"]))
                token = _execution_context.set(interaction.context)
                stage = "tool"
                try:
                    results = []
                    for call in action.tool_calls:
                        session.events.append(
                            ToolCallEvent(
                                context=interaction.context,
                                call=call,
                                status="requested",
                            )
                        )
                        remaining_time = session.remaining_timeout()
                        timeout = self.tool_timeout
                        if remaining_time is not None:
                            if remaining_time <= 0:
                                raise TimeoutError("Agent episode timeout exceeded.")
                            timeout = min(timeout, remaining_time)
                        result = await self.boundary.operation(
                            session,
                            "tool",
                            len(session.interactions) - 1,
                            lambda call=call: self.executor.execute(call),
                            timeout=timeout,
                            retry_safe=self.tool_retry.retry_tools
                            and self.executor.is_retry_safe(call),
                            retry_policy=self.tool_retry,
                        )
                        results.append(result)
                        # Persist each result before running the next tool so
                        # a later timeout cannot erase completed side effects.
                        session.record_tool_results(interaction, [result])
                        session.events.append(
                            ToolResultEvent(
                                context=interaction.context,
                                result=result,
                                status="action_error"
                                if result.is_error
                                else "completed",
                            )
                        )
                except BaseException as exc:
                    interaction.status = (
                        "cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "tool_failure"
                    )
                    interaction.error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    _execution_context.reset(token)
                items.extend(formatter.format(results))
                stage = "api"
            raise RuntimeError(
                "OpenAI agent exceeded max_turns without a final answer."
            )
        except BaseException as exc:
            failure = exc
            terminal_status = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            terminal_error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, asyncio.CancelledError):
                classified = self.boundary.classify(stage, exc, session)
                if classified is not None:
                    session.failure_classification = classified
                    terminal_error = classified[1]
            session.events.append(
                AgentEvent(
                    context=session.context,
                    kind="agent_end",
                    status="cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed",
                )
            )
            raise
        finally:
            session.running = False
            try:
                session.finish(terminal_status, error=terminal_error)
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Session finalization also failed: {cleanup_error}")
            finally:
                _execution_context.reset(root_token)

    async def close(self):
        """Release resources owned by the configured client or backend."""
        await self.boundary.close()
        await self.client.close()

    async def start(self) -> None:
        """Start owned tool resources once without taking client ownership."""
        await self.boundary.start(self.executor)

    async def __aenter__(self):
        """Enter an async context without starting a conversation."""
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        """Close the client when leaving the surrounding async context."""
        await self.close()
