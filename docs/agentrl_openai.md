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

`extra_headers` must be a mapping of nonempty string keys to string values.
The adapter validates and copies headers before registering a request; invalid
configuration leaves no active interaction. Once a request is registered, all
preparation and SDK failures use the same failure/cancellation cleanup path.
Cleanup errors are attached to the original exception without replacing it.

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

`OpenAISession(context=collector_context, training=True)` binds the existing root
`ExecutionContext` without rebuilding it. This preserves rollout, episode, group,
sample index, group size, task, snapshot, policy version and resource owner.
Training provenance requires a context with a known policy version; individual
identity arguments cannot replace that requirement. Optional legacy identity
arguments may repeat the context's values but conflicting overrides are rejected.
An interaction context cannot be used as a session root.

Session IDs are independent of candidate identity. Without a supplied context,
the inference path generates separate session, episode, rollout, group and
external snapshot IDs. Its external snapshot ID is a trace label, not evidence
of a PyTRIO sampling snapshot. Read-only `session.context`, `session.session_id`,
`session.rollout_id` and `session.group_id` properties prevent accidental identity
replacement. To read episode, task or group-size fields, use `session.context`.
Callers that previously treated session ID as episode or snapshot identity must
use the corresponding context fields instead.

Both `Agent.run(..., context=context, training=True)` and
`OpenAIWorkflow.run(..., context=context, training=True)` accept collector context.
An explicitly supplied session must match it and be created in training mode.
`OpenAIWorkflow.run(..., context=context, snapshot=snapshot)` additionally checks
snapshot ID and policy version against a `SamplingSnapshot` before making an API
call, and enables training-context validation. No identity fields are forwarded
as SDK generation options or taken from model tool arguments.

```python
from triotune.agentrl import ExecutionContext
from triotune.agentrl.runtime.sampling import SamplingSnapshot
from triotune.agentrl.sdk.openai import OpenAISession, OpenAIWorkflow


async def record_candidate(
    workflow: OpenAIWorkflow,
    messages: list[dict],
    context: ExecutionContext,
    snapshot: SamplingSnapshot,
):
    # The existing collector supplies context and snapshot before execution.
    session = OpenAISession(context=context, training=True)
    result = await workflow.run(
        messages, context=context, session=session, snapshot=snapshot
    )
    return result.export_trajectory()
```

The existing collector already creates the complete context before executor
calls; no alternative candidate allocator is used. This SDK facade is not yet
an `EpisodeExecutor`: the example records external API evidence for a candidate,
not model actions sampled from PyTRIO. Snapshot validation checks identity only
and never invokes its sampler or selects its weights. Export always retains
`trainable=False`, even when `training_context=True`. To sample through the
current policy instead of an external provider, construct
`PyTRIOBackend(snapshot, model=...)` and pass it to `OpenAIClient(backend=...)`.
The backend requires a training session whose snapshot and policy version match,
enforces temperature/top-p equal to one, and records exact prompt IDs, output
IDs, behavior logprobs, stop reason, and `sampling_source="pytrio"`. It reuses
the snapshot's `PyTRIOAgentSampler`, so the existing `call_async()` and trainer
owned sampler lifecycle remain authoritative. The API trace remains separate
from PPO `RolloutExport`; an exporter still must convert these records into the
existing episode/trajectory contract before training.

Sessions start with status `open`. `Agent.run()` automatically finishes them
as `completed`, `failed`, or `cancelled`, including tool failures and turn-limit
exits. Successful direct client/proxy requests leave a session open for
continuation. The embedding application calls `session.finish()` or
`proxy.finish_session(session_id)` after its external agent finishes. Transport
failures and cancellation automatically terminate direct-client sessions.
Repeating the same finish status is idempotent; changing a terminal outcome or
starting another request after termination is rejected.

Rewards require a completed, unreleased session. Export and release require a
terminal session with no request or agent work in flight. Failed and cancelled
sessions retain their partial trace for export but reject reward submission.
Release freezes writes; the returned session can still export its trace for
inspection. Released proxy session IDs cannot be reused while retained in the
bounded released-ID cache described below.

Every request allocates a request ID and a child `ExecutionContext` before SDK
execution. API and tool code can read `current_execution_context()`. Context
is task-local, including sync tools dispatched to worker threads. The trace
records local request IDs, provider request IDs, completion IDs, messages,
tool calls/results, usage, available logprobs, status and latency. A PyTRIO
backend additionally records exact `input_ids`, `output_ids`, aligned behavior
`logprobs`, `stop_reason`, and `sampling_source`. Native tool call IDs are
preserved in observations. Responses continuation retains all output items,
including reasoning items, before appending function outputs.

Each child context changes only interaction/parent identity from the root;
candidate fields stay unchanged across API calls and local tools. Agent execution
restores the enclosing context after success, failure or cancellation. Session
boundaries validate original root/interaction identities before requests, result
recording, reward submission and export. Replacing a record's context or changing
its parent/lookup identity is rejected. Export metadata includes the complete
trusted root context and cannot override it with caller-provided metadata.
Request and response lookup tables are checked in both directions, so forged
aliases are rejected before they can redirect a reward or parent reference.
Explicit parents must belong to the same API endpoint; Responses continuations
should use `previous_response_id` or a complete validated history. A provider
response that is not completed is retained as failed/cancelled evidence and is
never treated as a successful candidate.

`session.get_interaction(request_id=...)` resolves a local request ID, and
`session.get_interaction(response_id=...)` resolves a provider completion or
response ID. `request_to_interaction` and `response_to_interaction` expose these
session-local indexes. Provider HTTP request IDs remain separate trace fields.

Parent identity is inferred only when a new request contains an earlier
request's messages and its actual output as a prefix. The longest unique
prefix wins; unrelated questions and repeated prompts are independent roots.
Ambiguous prefixes require an explicit `parent_interaction_id` in client calls,
or `X-Triotune-Parent-Interaction-Id` in proxy requests. Responses can also
reference a known response through `previous_response_id`. Unknown, conflicting,
cross-session or unfinished parents are rejected.

A candidate request that used `previous_response_id` contains only incremental
input; its local input/output is not a verified full conversation prefix. Such
candidates are excluded from automatic parent inference. To continue from one,
use its `previous_response_id` or an explicit parent interaction ID. Other
candidates with complete matching history can still be inferred normally.

Delayed finite scalar rewards, `EpisodeScore` and `RewardAssignment` submissions
are recorded in the session. Structured trajectory rewards remain session
scoped. Interaction/turn positions are zero-based request indexes; token
positions retain `(request_index, token_offset)` without constructing token
arrays. A targeted submission can alternatively specify `interaction_id`,
`request_id`, or `response_id`. If both a position and ID are supplied, they
must resolve to the same interaction. Non-trajectory assignments need a position
or ID; token assignments always need a token offset. Legacy scalar submissions
default to the final interaction.

`EpisodeScore` is a session-scoped submission and cannot be redirected by an
explicit interaction/request/response ID. Its outcome (`reward`, `success`,
`components`) stays at session scope. Each `process_rewards` vector is mapped
to token contributions at `(request_index, token_offset)`, and each nested
assignment uses the same target resolver as a top-level `RewardAssignment`.
Reward versus advantage mode, source, aggregation and discount are preserved;
the SDK neither applies GAE nor propagates reward backwards.

Every target is validated before any reward, event or submission ID is stored.
If one nested target is invalid, the whole score is rejected. The original
score is copied into the session's `rewards` exactly once; interaction bindings
and event `bindings` describe its contributions rather than additional scores
to add to that original. Idempotency compares the original payload together
with all resolved bindings, so a conflicting retry cannot partially overwrite
an earlier score. These API traces remain separate from PPO aggregation.
Each interaction retains all of its `reward_submissions` and exposes its latest
one through `reward`; these trace fields do not perform PPO reward aggregation.

Reusing a `submission_id` with the same content and resolved target is idempotent.
Reusing it with different content or a different target raises a contract error
without changing stored rewards. Export returns a JSON-serializable
`OpenAITrajectoryExport` API trace, including terminal status and error details.
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
When the adapter uses `PyTRIOBackend`, provide a `session_factory` that returns
the collector's training `OpenAISession(context=..., training=True)` for each
session ID. The default proxy session factory creates inference sessions and is
therefore intentionally rejected by the training-only PyTRIO backend.

Requests can use `X-Triotune-Session-Id`; generated session and local request
IDs are returned in `X-Triotune-Session-Id` and `X-Client-Request-Id` response
headers. `X-Triotune-Interaction-Id` exposes the corresponding interaction ID.
The embedding application can inspect `proxy.sessions`, call `finish_session(id)`
once its agent has ended, submit rewards, export the trace, then call
`release_session(id)`. Session trace storage is bounded by
`max_sessions` (default 1024). One session cannot have concurrent requests.

Validation failures that occur before a request is registered reclaim only a
newly created empty session, including its capacity slot. Corrected retries
may reuse that ID. Existing sessions and any registered request evidence are
preserved. HTTP errors return `X-Triotune-Session-Id` once a session was allocated;
if the call registered an interaction, its request and interaction identity
headers are also returned. Rejected continuations never return a preceding
interaction's IDs. Upstream HTTP status codes are preserved; connection failures,
timeouts and unexpected SDK failures return 502, 504 and 500 respectively. The
original exception remains in the failed interaction trace.

Released-ID retention is bounded separately by `max_released_session_ids`
(default 1024, positive integer). The oldest released ID is evicted first when
that limit is exceeded. Late retries using retained IDs are rejected; after
eviction, the same string can create a fresh session and must not be treated
as continuation of the old one. Old released trace objects remain read-only.
Applications requiring permanent ID uniqueness must enforce it outside this
in-memory adapter. No unbounded released-ID history is kept.
Streaming, multiple chat choices, background Responses, subprocess execution,
multimodal agents and public server deployment are outside this initial adapter.

Official protocol reference:
[OpenAI function calling](https://platform.openai.com/docs/guides/function-calling).
