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
an `EpisodeExecutor`: with the external backend, the example records provider
evidence rather than PyTRIO model actions. Workflow snapshot validation checks
identity only; model execution belongs to the configured backend. Export retains
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

The first PyTRIO request pins the live snapshot, sampler, tokenizer, underlying
client, model label, and stop settings to its session. Reusing snapshot/version
strings does not permit swapping those resources between requests. A session
cannot mix external-provider evidence with fixed-policy samples. Binding does
not transfer ownership or close resources, and the trainer must still keep
remote weights immutable throughout collection.

## Qwen3.5 native tools on PyTRIO

`PyTRIOBackend` selects the native adapter automatically for the project's
`Qwen/Qwen3.5-4B`. For a local model alias, specify `model_format="qwen3.5"`;
the bound tokenizer must still expose the Qwen3.5 tool template. Other model
names retain text-only sampling and explicitly reject tool requests. No model
weights or sampling clients are created by the adapter.

```python
from triotune.agentrl.sdk.openai import Agent, OpenAIClient, PyTRIOBackend


async def run_native_candidate(snapshot, context, tools):
    # Snapshot and candidate identity are allocated by the training owner.
    backend = PyTRIOBackend(
        snapshot,
        model="Qwen/Qwen3.5-4B",
        max_tokens=1024,
        template_kwargs={"enable_thinking": False},
    )
    agent = Agent(
        name="calculator",
        instructions="Use the available functions to answer the question.",
        tools=tools,
        client=OpenAIClient(backend=backend),
    )
    return await agent.run("What is 2 + 3?", context=context, training=True)
```

There are two distinct parsing boundaries. `ModelOutputParser.parse_generation`
consumes the original `Generation` and prompt IDs; `Qwen35ToolParser` implements
the Qwen3.5 XML-style `<tool_call><function=...><parameter=...>` grammar. It
separates reasoning (including a `<think>` opening already present in the prompt),
function calls, and final text. `OpenAINativeActionParser` instead consumes the
OpenAI-shaped Chat Completions/Responses objects produced after this conversion.
Multiple calls retain their order and receive distinct call IDs. Malformed
parameters, unknown functions, duplicate fields, and non-finite JSON numbers
cannot reach the tool executor. Parsing failures still retain the sampled IDs,
logprobs, raw text, stop reason, and error in the failed session trace.

Scalar parameters support both JSON boolean/null spellings and the
`True`/`False`/`None` forms emitted by Qwen's Jinja template, only when the
declared type allows them. String fields remain literal strings. A broken
custom parser is reported as a contract error, separately from a malformed
model action, while preserving the already sampled arrays in either case.

Messages and tool schemas are rendered with `SamplingSnapshot.tokenizer`.
Continuation requires a verified parent, unchanged messages and tool schemas,
the parent's actual assistant action, and one matching observation per call.
`NativeToolContinuationRenderer` encodes only the new template/observation
suffix, appending it to the exact parent prompt IDs and sampled output IDs.
Neither whitespace-normalized tool calls nor display text are re-encoded to
replace sampled actions. Noncanonical sampled BPE segmentations, reasoning, and
returned EOS IDs therefore remain in the causal prefix. Incompatible template
boundaries, changed histories, and mismatched tool results are rejected before
the next sample call.

The native adapter supports text-only Chat Completions and full-history
Responses input, function tools, and `tool_choice="auto"` or `"none"`.
Provider-managed `previous_response_id` continuation, forced/required tools,
constrained strict generation, multimodal messages, unsupported schema keywords,
and arbitrary request token-ID overrides are explicitly rejected on this path.
The external OpenAI backend retains its existing provider behavior. Native
sampling still exports an SDK trace with `trainable=False`, not an automatic
PPO `RolloutExport`.

## Session lifecycle and rewards

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

When exact PyTRIO output IDs are available, token offsets are bounded by that
original output and `EpisodeScore.process_rewards` vectors must cover it
exactly, including reasoning, tool syntax, and returned end markers. These
checks occur before any reward or idempotency key is committed. External traces
without original output IDs keep their existing deferred-validation behavior;
reported usage counts are not treated as token-alignment evidence.

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

## Standard SDK entry for third-party agents

The existing `openai-proxy` extra provides the dependencies for a session-bound
official `AsyncOpenAI` client. No `openai-agents` dependency or Triotune `Agent`
tool loop is needed. `adapter.as_async_openai(session=session)` and
`proxy.create_client(session=session)` both return a real `AsyncOpenAI` object;
requests use an in-process ASGI connection and never start a listening server.
Every request automatically uses the bound session ID, including calls through
SDK `with_options()` copies. Conflicting session headers are rejected by the
transport rather than creating a second conversation.

For a user-defined asynchronous agent, use a client-only workflow:

```python
import json

from triotune.agentrl.sdk.openai import OpenAIClient, OpenAIWorkflow


async def run_external_agent(upstream_sdk, model, question):
    async def execute(client, text):
        # This is ordinary third-party SDK code, not a Triotune Agent loop.
        messages = [{"role": "user", "content": text}]
        tools = [{
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }]
        first = await client.chat.completions.create(
            model=model, messages=messages, tools=tools
        )
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        for call in first.choices[0].message.tool_calls or []:
            arguments = json.loads(call.function.arguments)
            # External frameworks retain responsibility for tool registration,
            # argument/domain validation, retries and side-effect ownership.
            value = arguments["a"] + arguments["b"]
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": str(value)
            })
        final = await client.chat.completions.create(
            model=model, messages=messages, tools=tools
        )
        return final.choices[0].message.content

    workflow = OpenAIWorkflow(client=OpenAIClient(upstream_sdk, model=model))
    result = await workflow.run(question, execute=execute)
    return result.output, result.export_trajectory()
```

`execute` is awaited as `execute(client, input)` and may return any result type.
`SDKRunResult.output` preserves that value; its `session`, `session_id`, and
`export_trajectory()` expose the same trace and delayed reward lifecycle as the
built-in loop. `context`, `training`, and `snapshot` retain the existing
collector identity checks. The workflow restores the outer execution context
and finishes sessions as completed/failed/cancelled, including callback errors,
provider failures and cancellation. Callbacks must await their request work;
transport teardown cancels outstanding requests rather than leaving running
interactions behind. Injected upstream SDK clients and snapshots are not closed.

Subsequent Chat `tool` messages and Responses `function_call_output` items are
captured as tool observations on their original interactions. Association uses
call ID and verified session history, not the most recent request. Explicit
parent references retain the external-provider delta contract. Unknown/foreign
IDs, fabricated calls, duplicate results in one request, and conflicting result
replays fail before forwarding; full-history echoes do not duplicate stored
results or events. Valid observations are stored before the next provider call,
so a later failure does not erase already executed external tool work. Only
submitted observations are visible: the adapter cannot infer unreported side
effects, tool latency, retries, or a domain error flag from opaque result text.

The standard entry also works with a `PyTRIOBackend` adapter. Qwen observations
received in parallel-completion order are matched by ID and rendered in the
model's original call order while retaining the sampled token prefix unchanged.
The native backend's full-history, snapshot, schema and sampling constraints
remain in force; no API trace is automatically promoted to PPO data.

Bound SDK clients use `max_retries=0`: a failed recorded request cannot be
retried as a new request in a terminal session. Configure provider retry and
timeout behavior on the upstream SDK client instead. Direct bound-client usage
does not finish or release the session when the SDK client closes; the embedding
application calls `session.finish()` and then submits rewards/exports/releases.

`OpenAIClient` and `OpenAIWorkflow` also accept explicit `request_id` values.
Reusing one ID with identical endpoint, payload, and parent returns the stored
completed response without another provider call or duplicated tool events.
Reusing it with different content, while the original is running, or after a
failure is rejected. This identity-based completion rule is independent of
prompt-prefix lineage and is therefore safe for repeated identical prompts.

`Agent(episode_timeout=..., max_generated_tokens=...)` and third-party
`OpenAIWorkflow.run(..., execute=..., episode_timeout=...,
max_generated_tokens=...)` configure one session-owned total boundary. The
deadline covers backend requests and tool-loop work; generated-token accounting
uses original IDs when available and provider usage otherwise. A request that
would exceed the boundary is rejected before allocation, while a response that
crosses it remains in the failed trace. API retries are owned by the provider
client and default to zero for recorded SDK sessions; local tool retries remain
the existing runner policy and are never inferred from API retry settings.

`RuntimeBoundary` is the shared implementation for transient retries, custom
failure classification, and owned resource start/close order. `EpisodeRunner`
uses it for its existing sampler/tool/scorer stages; SDK `Agent` uses it only
for explicitly retry-safe tools, and `OpenAIWorkflow` uses it for optional
trace scorers, acceptance hooks, and callback resources. Provider/API retries
remain inside the provider client and never cause a completed local tool to run
again. Cancellation bypasses retry classification, while completed tool results
remain attached to their original interaction on later failures.

SDK scoring hooks receive `(input, result)` and interaction hooks receive
`(input, result, openai_interaction)`. They share timeout, retry, failure and
resource lifecycle mechanics with the runner, but do not pretend that SDK
records are `EpisodeResult`. SDK results expose read-only `context`, `status`,
`interactions`, and `trainable=False` views. A sample acceptance hook receives
`(result, None)`, matching the existing optional-export boundary. A false result
sets `accepted=False`; it does not create or submit a PPO row. Full reuse of
`EpisodeScorer`, `EpisodeResult`, `RolloutExport`, normalization, GAE and loss
weighting belongs to the explicit P2.5 training bridge.

For an existing proxy, use `proxy.create_client(session=session)` to register the
exact collector-created session without reconstruction. The connection transport
and base URL are factory-owned; optional SDK settings use `client_options` in
`workflow.run(..., execute=..., client_options=...)`.

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
