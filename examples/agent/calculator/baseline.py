"""Compose native tool and official SDK callback workflows for live acceptance.

Both paths use the same calculator, outcome scorer, fixed sampling snapshot,
and episode exporter. Only the application tool loop differs. No display text
is encoded back into training tokens; OpenAIEpisodeExecutor owns that boundary.
"""

from __future__ import annotations

import asyncio
import json

from triotune.agentrl import (
    AgentRolloutWorkflow,
    EpisodeLimits,
    FunctionTool,
    JsonlEpisodeLogger,
    OpenAIEpisodeExecutor,
    ToolCall,
)
from triotune.agentrl.integrations.openai import (
    Agent,
    OpenAIClient,
    OpenAIWorkflow,
    PyTRIOBackend,
)

from .agent import CalculatorScorer, calculate, calculator_tools

INSTRUCTIONS = (
    "Use the calculate tool to solve the question, even for simple arithmetic. "
    "After receiving the tool result, reply with the numerical answer only."
)
LIMITS = EpisodeLimits(
    max_turns=4,
    max_tokens_per_turn=128,
    max_generated_tokens=512,
    max_context_tokens=4096,
    sampling_timeout=120,
    tool_timeout=10,
    episode_timeout=360,
)


async def execute_calculator(sdk, question, *, model):
    """Run a bounded tool loop through the official AsyncOpenAI client.

    The workflow supplies a session-bound local transport backed by PyTRIO.
    Preserve SDK assistant messages and tool-call IDs verbatim when continuing
    the conversation so the bridge can associate observations with actions.
    Recoverable arithmetic errors are returned as tool observations; transport
    and implementation failures propagate to the collector.
    """
    tools = calculator_tools()
    messages = (
        [{"role": "user", "content": question}]
        if isinstance(question, str)
        else list(question)
    )
    messages = [{"role": "system", "content": INSTRUCTIONS}, *messages]
    for _ in range(LIMITS.max_turns):
        response = await sdk.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.schemas,
            max_tokens=LIMITS.max_tokens_per_turn,
            temperature=1.0,
            top_p=1.0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or ""
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            result = await asyncio.wait_for(
                tools.execute(
                    ToolCall(
                        call.id, call.function.name, json.loads(call.function.arguments)
                    )
                ),
                timeout=LIMITS.tool_timeout,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.content,
                }
            )
    # The executor's model-turn budget turns this partial conversation into a
    # max_turns episode rather than manufacturing a final answer.
    raise RuntimeError("Calculator callback reached max_turns.")


def make_workflow(kind, model, output):
    """Build a fresh workflow and backend for every collected candidate."""
    if kind not in {"native", "sdk"}:
        raise ValueError("Calculator baseline kind must be native or sdk.")

    def factory(snapshot):
        client = OpenAIClient(
            backend=PyTRIOBackend(
                snapshot,
                model=model,
                max_tokens=LIMITS.max_tokens_per_turn,
                template_kwargs={"enable_thinking": False},
            )
        )
        if kind == "sdk":
            return OpenAIWorkflow(client=client)
        return OpenAIWorkflow(
            Agent(
                name="calculator",
                instructions=INSTRUCTIONS,
                tools=[
                    FunctionTool(
                        **calculator_tools().schemas[0]["function"],
                        function=calculate,
                        retry_safe=True,
                    )
                ],
                model=model,
                client=client,
                max_turns=LIMITS.max_turns,
            )
        )

    async def execute(sdk, question):
        return await execute_calculator(sdk, question, model=model)

    return AgentRolloutWorkflow(
        OpenAIEpisodeExecutor(
            factory,
            scorer=CalculatorScorer(),
            limits=LIMITS,
            execute=execute if kind == "sdk" else None,
        ),
        max_concurrency=2,
        logger=JsonlEpisodeLogger(output / "episodes.jsonl"),
    )
