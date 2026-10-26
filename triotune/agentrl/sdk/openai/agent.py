"""Inline OpenAI agent execution using the existing local tool executor."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any

from ...runtime.tools import FunctionToolExecutor
from ...types import AgentEvent, ToolCallEvent, ToolResultEvent, _execution_context
from .client import OpenAIClient
from .native import (
    OpenAIChatObservationFormatter,
    OpenAINativeActionParser,
    OpenAIResponsesObservationFormatter,
)
from .session import OpenAISession, OpenAITrajectoryExport


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Final text and its request trace; API evidence is not PPO token data."""

    text: str
    session: OpenAISession
    response: Any

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def export_trajectory(self) -> OpenAITrajectoryExport:
        return self.session.export_trajectory()


class Agent:
    """Execute local tools requested by an official OpenAI SDK client.

    Every invocation gets its own session unless one is supplied explicitly.
    API retries/timeouts belong to the SDK client. Tools run once per call and
    retain their ``retry_safe`` metadata; this loop never retries side effects.
    The optional tool timeout bounds awaiting a tool, but Python cannot stop a
    synchronous worker thread that has already started.
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
    ) -> None:
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
        self.name, self.instructions, self.model = name, instructions, model
        self.max_turns, self.api, self.tool_timeout = max_turns, api, tool_timeout
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
        **request_options: Any,
    ) -> AgentRunResult:
        """Run a fresh conversation in one isolated session until a final answer.

        Chat messages echo native tool call IDs. Responses echoes every output
        item, including reasoning items, before appending function-call outputs.
        This preserves the API's required continuation state for reasoning models.
        """
        if {
            "messages",
            "input",
            "tools",
            "instructions",
            "previous_response_id",
        }.intersection(request_options):
            raise ValueError("Conversation and tool fields are owned by Agent.run.")
        session = session or OpenAISession()
        if session.running or session.interactions:
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
        terminal_status, terminal_error = "completed", None
        try:
            for _ in range(self.max_turns):
                kwargs = {"session": session, "model": self.model, **request_options}
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
                    return AgentRunResult(action.final_text or "", session, response)
                if self.api == "chat_completions":
                    items.append(action.assistant_message)
                else:
                    items.extend(copy.deepcopy(interaction.response["output"]))
                token = _execution_context.set(interaction.context)
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
                        result = await asyncio.wait_for(
                            self.executor.execute(call), self.tool_timeout
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
            raise RuntimeError(
                "OpenAI agent exceeded max_turns without a final answer."
            )
        except BaseException as exc:
            terminal_status = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            terminal_error = f"{type(exc).__name__}: {exc}"
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
            session.finish(terminal_status, error=terminal_error)

    async def close(self):
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()
