# OpenAI SDK tools and sessions

Install the optional SDK integration:

```bash
python -m pip install -e '.[openai]'
```

The inline agent supports text and local function tools through Chat Completions
and Responses. It uses the official Python SDK; model access, authentication,
request timeout and API retry configuration belong to that client.

## Register and run tools

```python
import asyncio
import os

from triotune.agentrl import Agent, function_tool
from triotune.agentrl.training.rewards import RewardAssignment


@function_tool(retry_safe=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


async def main():
    async with Agent(
        name="calculator",
        instructions="Use tools to solve math problems.",
        tools=[add],
        model=os.environ["OPENAI_MODEL"],
        api="chat_completions",  # Use "responses" for the Responses API.
    ) as agent:
        result = await agent.run("What is 2 + 3?")
        print(result.text, result.session_id)
        result.session.submit_reward(
            RewardAssignment("trajectory", 1.0, source="outcome"),
            submission_id="outcome-v1",
        )
        trace = result.export_trajectory()
        print(trace.logical_rollout_id)


if __name__ == "__main__":
    asyncio.run(main())
```

`OPENAI_API_KEY` is read by the SDK. A caller can inject an existing `OpenAI` or
`AsyncOpenAI` client via `OpenAIClient(client, model=...)`, then pass that adapter
to `Agent`. Injected SDK clients are closed by their caller. Internally created
clients are closed by `Agent.close()` or its async context manager.

The decorator derives an object schema from the function signature and a
description from the first docstring paragraph. It accepts sync/async functions,
JSON scalar types, `list[T]`, `dict[str, T]`, unions, `Literal`, and optional
defaults. Unsupported annotations, variadic and positional-only arguments fail
at registration. Recursive argument validation rejects incorrect values without
coercing strings or booleans into numbers. Invalid tool arguments become tool
error observations; implementation failures propagate as `ToolSystemError`.

`max_turns` bounds model requests and `tool_timeout` bounds each tool await.
The loop executes each requested tool once. `retry_safe` remains available to
other executors but does not enable automatic tool retries in this loop. A
cancelled synchronous worker thread may still finish its function.

## Session identity and evidence

Each `Agent.run()` creates an independent session. An unused `OpenAISession`
can instead be passed explicitly with rollout/group/sample/snapshot/policy
identity. Concurrent runs require separate sessions; an already used session
cannot start another independent agent conversation.

Every request allocates a request ID and a child `ExecutionContext` before SDK
execution. API and tool code can read `current_execution_context()`. Context
is task-local, including sync tools dispatched to worker threads. The trace
records local request IDs, provider request IDs, completion IDs, messages,
tool calls/results, usage, available logprobs, status and latency. Native tool
call IDs are preserved in observations. Responses continuation retains all
output items, including reasoning items, before appending function outputs.

Delayed finite scalar rewards, `EpisodeScore` and `RewardAssignment` submissions
are recorded in the session. Reusing a `submission_id` does not append a second
reward. Export returns a JSON-serializable `OpenAITrajectoryExport` API trace.
It is **not** a PPO `RolloutExport`: standard API responses do not provide exact
prompt/output token IDs and complete behavior logprobs for every generated tool
argument. Missing values stay missing. Trace metadata therefore reports
`trainable=False`; do not re-tokenize response text to manufacture PPO samples.
Existing PyTRIO agent training and its exact-token exporter remain unchanged.

## Optional local HTTP forwarding

`pip install -e '.[openai-proxy]'` additionally installs FastAPI.
`OpenAIProxyServer(adapter).create_app()` returns an ASGI application exposing
`POST /v1/chat/completions` and `POST /v1/responses`. It forwards API requests;
the high-level `Agent` is responsible for executing local tools.

Requests can use `X-Triotune-Session-Id`; generated session and local request
IDs are returned in `X-Triotune-Session-Id` and `X-Client-Request-Id` response
headers. The embedding application can inspect `proxy.sessions` and call
`release_session(id)` after exporting its trace. Session storage is bounded by
`max_sessions` (default 1024). One session cannot have concurrent requests.
Streaming, multiple chat choices, background Responses, subprocess execution,
multimodal agents and public server deployment are outside this initial adapter.

Official protocol reference:
[OpenAI function calling](https://platform.openai.com/docs/guides/function-calling).
