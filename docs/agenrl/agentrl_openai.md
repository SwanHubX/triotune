# Agents using OpenAI interfaces and PyTRIO training

[中文](agentrl_openai_CN.md)

Use this guide to run tool-using agents through Chat Completions or Responses,
or connect existing SDK code to PyTRIO training. For the basic training recipe,
start with [README](README.md).

## Choose an execution path

| Goal | Components | Result |
| --- | --- | --- |
| Run tools against an external provider | Agent + OpenAIClient | Final answer and session trace |
| Train with the current PyTRIO policy | PyTRIOBackend + OpenAIEpisodeExecutor + AgentTrainingLoop | Native episodes and PPO/GRPO trajectories |
| Reuse an existing async SDK agent | OpenAIWorkflow with execute | Callback output and session trace; add the PyTRIO bridge for training |
| Forward local HTTP requests | OpenAIProxyServer | API responses and session traces |

Agent.run and OpenAIWorkflow.run export diagnostic API traces with
trainable=False, including when sampling through PyTRIO. Use the explicit
OpenAIEpisodeExecutor bridge to construct training trajectories. Do not encode
response text to manufacture output token IDs or behavior log probabilities.

## Register and run tools

### Install dependencies

Run from the repository root:

```bash
uv sync --locked --extra openai
```

For training, also install ppo. For third-party SDK callbacks and local HTTP
forwarding, install openai-proxy, which includes FastAPI:

```bash
uv sync --locked --extra ppo --extra openai-proxy
```

### Define a Python tool

The decorator derives the parameter schema from annotations and the description
from the first docstring paragraph. The following add tool is reused below:

```python
from triotune.agentrl import function_tool


@function_tool(retry_safe=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
```

| Tool declaration | Behavior |
| --- | --- |
| Sync or async function | Both supported; synchronous work runs in a worker thread |
| JSON scalars, list[T], dict[str, T], unions, Literal, optional defaults | Converted into a tool schema |
| Variadic, positional-only, or unsupported annotations | Rejected at registration |
| Invalid model arguments | Tool error observation; values are not silently coerced |
| Tool implementation failure | ToolSystemError; not treated as a wrong model answer |

retry_safe declares that repeating the tool is safe. It does not enable retries
by itself; see [limits and retries](#configure-limits-and-retries).

### Run Chat Completions or Responses

For an external provider, configure OPENAI_API_KEY and set OPENAI_MODEL to an
available model. The SDK reads the key from the environment.

```python
import asyncio
import os

from triotune.agentrl.integrations.openai import Agent


async def run_tools(model, *, api="chat_completions", sdk=None):
    async with Agent(
        name="calculator",
        instructions="Use add to solve the problem. Answer with the number only.",
        tools=[add],
        model=model,
        client=sdk,
        api=api,
        max_turns=5,
        tool_timeout=10,
    ) as agent:
        result = await agent.run("What is 2 + 3?")
        print(result.text, result.session_id)
        return result


if __name__ == "__main__":
    asyncio.run(run_tools(os.environ["OPENAI_MODEL"]))
```

Call run_tools(model, api="responses") to use Responses. Agent executes local
functions and sends their results back with the original tool-call IDs. It
preserves Responses output items, including reasoning, when continuing.

Each run needs a fresh session. Reuse the Agent for separate conversations,
but never share one session between concurrent runs.

### Supply your own provider client

Configure provider authentication, request timeouts, and API retries on the
injected SDK client:

```python
from openai import AsyncOpenAI


async def run_with_client(model):
    async with AsyncOpenAI(timeout=60.0, max_retries=2) as sdk:
        return await run_tools(model, sdk=sdk)
```

OpenAIClient accepts either a synchronous OpenAI or an AsyncOpenAI client.
Injected clients remain caller-owned. Clients created internally by Agent are
closed by its async context manager.

| Resource | Who closes it |
| --- | --- |
| SDK client passed into Agent or OpenAIClient | The application that created it |
| SDK client created internally by Agent | Agent.close or its async context manager |
| Fresh workflow/Agent/client returned by a training factory | OpenAIEpisodeExecutor |
| Sampling client and tokenizer in a training snapshot | The trainer; closing an adapter does not close them |
| Local session-bound SDK transport | OpenAIWorkflow, or the caller when used directly |

## Train SDK executions with PyTRIO

### Bind every candidate to its sampling snapshot

Training uses PyTRIO credentials and does not require an OpenAI API key. Build
a fresh workflow, Agent, client, and backend for every candidate:

```python
from triotune.agentrl.integrations.openai import (
    Agent, OpenAIClient, OpenAIWorkflow, PyTRIOBackend,
)

MODEL = "Qwen/Qwen3.5-4B"


def make_workflow(snapshot):
    backend = PyTRIOBackend(snapshot, model=MODEL, max_tokens=1024)
    client = OpenAIClient(backend=backend)
    return OpenAIWorkflow(Agent(
        name="calculator",
        instructions="Use add. Answer with the number only.",
        tools=[add],
        model=MODEL,
        client=client,
    ))
```

The collector supplies snapshot. Do not replace it, switch a shared client's
backend, or modify its sampling resources midway through a conversation.

### Add the scorer and training loop

The scorer receives the complete original row. Only messages is sent to the
agent; answer remains available for scoring.

```python
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentRolloutWorkflow, AgentTrainingLoop, EpisodeLimits, EpisodeScore,
    EpisodeTrajectoryBuilder, OpenAIEpisodeExecutor,
)


class Outcome:
    async def score(self, row, episode):
        success = (
            episode.status == "completed"
            and episode.final_text is not None
            and episode.final_text.strip() == row["answer"]
        )
        return EpisodeScore(float(success), success)


async def train():
    runner = OpenAIEpisodeExecutor(
        make_workflow,
        scorer=Outcome(),
        limits=EpisodeLimits(
            max_turns=5,
            max_tokens_per_turn=1024,
            max_generated_tokens=2048,
        ),
        trajectory_builder=EpisodeTrajectoryBuilder(mode="individual"),
    )
    collector = AgentRolloutWorkflow(runner, max_concurrency=2)
    trainer = PPOTrainer(
        MODEL,
        args=PPOConfig.for_algorithm(
            "grpo", batch_size=1, num_generations=2, max_steps=1,
        ),
        train_dataset=[{"messages": "What is 2 + 3?", "answer": "5"}],
    )
    return await AgentTrainingLoop(trainer, collector).train()
```

Call await train() from an async entry point. To use Responses, set
api="responses" on the Agent returned by make_workflow.

| Training behavior | What to expect |
| --- | --- |
| Original prompt/output IDs and behavior logprobs | Preserved without re-encoding displayed text |
| Snapshot, policy, interaction, and tool identities | Carried into native episode and trajectory records |
| External provider, missing logprobs, or changed snapshot | Rejected before optimizer work |
| Malformed model action or bounded partial generation | Original actions retained for native partial-episode scoring |
| Infrastructure failure or rejected candidate | Excluded from training |
| Candidate workflow resources | Closed by the bridge; trainer-owned sampling clients and tokenizers stay trainer-owned |

Use the native scorer for outcome and success. SDK reward submissions are
additional contributions, so do not submit the same outcome through both paths.
Their source names are scoped under openai/assignment/. Export mode, process
rewards, group acceptance, normalization, GAE, and loss weighting follow the
[training-data guide](agentrl_training.md).

## Model-native tools on PyTRIO

PyTRIOBackend selects tool grammar from the snapshot tokenizer rather than a
model-name whitelist. Usually no model_format argument is needed, including for
new catalog models or aliases whose templates use an already-supported grammar:

```python
from triotune.agentrl.integrations.openai import (
    Agent, OpenAIClient, OpenAIWorkflow, PyTRIOBackend,
)


def make_native_workflow(snapshot, model):
    backend = PyTRIOBackend(
        snapshot,
        model=model,
        max_tokens=1024,
        template_kwargs={"enable_thinking": False},
    )
    return OpenAIWorkflow(Agent(
        name="calculator",
        instructions="Use add. Answer with the number only.",
        tools=[add],
        model=model,
        client=OpenAIClient(backend=backend),
    ))
```

Use this factory in OpenAIEpisodeExecutor. The model label must match the label
bound to the backend; the adapter does not load weights or create sampling
clients.

For the built-in Qwen XML adapter, execution follows this path:

```text
original PyTRIO output IDs
  → Qwen35ToolParser reads reasoning, function calls, and final text
  → OpenAI-shaped Chat Completions or Responses objects
  → Agent executes tools and appends observations
  → next request extends the original sampled token prefix
```

| Native path supports | Native path rejects |
| --- | --- |
| Text Chat Completions and full-history Responses input | Multimodal input and provider-managed previous_response_id continuation |
| Function tools with tool_choice="auto" or "none" | Forced/required tools and constrained strict generation |
| Matching Qwen3.5 templates and validated schemas | Unsupported schema keywords and arbitrary token-ID overrides |
| temperature=1 and top_p=1 | Sampling controls that change PPO behavior probabilities |

Unrecognized tokenizer templates retain text-only sampling and reject tool requests.
Parameters are validated before tool execution, including duplicate fields,
unknown functions, and non-finite values. Parsing failures keep original token
IDs, logprobs, and error details in the session trace.

Keep earlier messages, tool schemas, and actual assistant outputs unchanged.
Tool observations must match their original call IDs. The continuation renderer
encodes only the new observation/template suffix; it never replaces earlier
sampled actions with re-encoded display text.

## Session identity and evidence

### Bind a collector-created context

OpenAIEpisodeExecutor handles session creation for ordinary training. Use an
explicit session only when implementing your own execution integration:

```python
from triotune.agentrl.integrations.openai import OpenAISession


async def record_candidate(workflow, messages, context, snapshot):
    session = OpenAISession(context=context, training=True)
    result = await workflow.run(
        messages,
        context=context,
        session=session,
        snapshot=snapshot,
    )
    return result.export_trajectory()
```

context must be the collector's root ExecutionContext with a known policy_version,
not an interaction context. It must match snapshot. This helper returns an API
trace, not a native PPO export; use OpenAIEpisodeExecutor for training rows.

The caller owns the supplied workflow and closes its Agent/client and workflow
hooks when finished; record_candidate does not take ownership.

| Identity | Read it from |
| --- | --- |
| Conversation/session ID | session.session_id |
| Candidate and reward group | session.rollout_id and session.group_id |
| Episode, task, candidate position, and group size | session.context |
| Sampling policy | session.context.snapshot_id and policy_version |
| Request and parent | interaction.context |

Inference sessions create their own IDs. Their external snapshot labels are not
proof of PyTRIO sampling. Do not assign session IDs as episode or snapshot IDs,
or pass identity fields as model-generated tool arguments.

The first PyTRIO request pins the actual snapshot, sampler, tokenizer, client,
model, and stop settings. Matching identity strings do not allow substituting
those resources later or mixing provider and PyTRIO evidence in one session.

### Inspect requests and responses

```python
def inspect_requests(session):
    for item in session.interactions:
        print(item.request_id, item.response_id, item.status, item.latency)
        print(item.input_ids, item.output_ids, item.logprobs)
        resolved = session.get_interaction(request_id=item.request_id)
        assert resolved.interaction_id == item.interaction_id
```

External providers may not supply original token IDs or complete logprobs;
missing values stay missing. Provider HTTP request IDs and completion/response
IDs are separate fields. get_interaction(response_id=...) resolves the latter.

Parent inference requires an unchanged history prefix containing a previous
request and its actual output. Ambiguous histories need parent_interaction_id
on client calls or X-Triotune-Parent-Interaction-Id on HTTP requests. Unknown,
unfinished, conflicting, or cross-session parents are rejected.

On external Responses, previous_response_id can identify a continuation; incremental-only
requests are not treated as full prefixes for later automatic inference.

## Session lifecycle and rewards

### Finish, export, and release

Agent.run and OpenAIWorkflow.run finish their sessions automatically. Successful
direct OpenAIClient or proxy requests leave sessions open for more requests;
the application finishes them after the conversation ends:

```python
from triotune.agentrl.integrations.openai import OpenAISession


async def direct_request(adapter, model):
    session = OpenAISession()
    response = await adapter.create_chat_completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with 5."}],
        session=session,
        request_id="answer-request",
    )
    session.finish("completed")
    return session, response
```

| Session state | Allowed operations |
| --- | --- |
| open | Start requests; no concurrent requests within one session |
| completed and idle | Submit rewards, export, or release |
| failed/cancelled and idle | Export or release; no reward submission |
| released | Read/export the trace; no further writes |

Export and release require no request or agent work in flight. Finishing again
with the same status is harmless; reopening or changing a terminal outcome is
rejected. Registered request failures and cancellations preserve partial traces.

### Submit outcome or process rewards

Use a completed, unreleased session, such as result.session from run_tools:

```python
from triotune.agentrl import EpisodeScore, RewardAssignment


def score_session(session, success):
    outcome = EpisodeScore(float(success), success)
    session.submit_reward(outcome, submission_id="outcome-v1")
    session.submit_reward(outcome, submission_id="outcome-v1")  # Same retry.

    first = session.interactions[0]
    session.submit_reward(
        RewardAssignment("interaction", 0.1, source="process"),
        interaction_id=first.interaction_id,
        submission_id="process-v1",
    )
    trace = session.export_trajectory()
    session.release()
    return trace
```

| Reward form | Targeting rule |
| --- | --- |
| EpisodeScore | Session-scoped outcome; no explicit target ID |
| trajectory assignment | Session scope |
| interaction/turn assignment | Zero-based request position or interaction_id/request_id/response_id |
| token assignment | (request index, original output-token offset) |
| Legacy scalar | Final interaction unless explicitly targeted |

With exact PyTRIO output IDs, token offsets and process vectors must match the
original output, including reasoning, tool syntax, and end markers. Position and
ID must resolve to the same request if both are supplied. Invalid nested targets
reject the whole submission before any rewards or submission IDs are stored.

Repeating a submission ID is idempotent only for the same payload and target.
Changed content or targets raise a contract error. Reward, aggregation, discount,
and advantage mode are preserved; session traces do not run PPO aggregation or
GAE. Interaction reward fields and event bindings describe submissions rather
than extra rewards to sum again.

### Configure limits and retries

Set model-request, tool, and total-episode boundaries separately:

```python
from triotune.agentrl import RetryPolicy
from triotune.agentrl.integrations.openai import Agent


def make_bounded_agent(client, model):
    return Agent(
        name="calculator",
        instructions="Use add. Answer with the number only.",
        tools=[add],
        model=model,
        client=client,
        max_turns=5,
        tool_timeout=10,
        episode_timeout=120,
        max_generated_tokens=2048,
        tool_retry=RetryPolicy(max_attempts=2, retry_tools=True),
    )
```

| Boundary | Owner |
| --- | --- |
| Provider request timeout and API retries | Injected provider SDK client |
| Tool timeout and explicitly enabled safe retries | Agent tool_timeout and tool_retry |
| Total deadline and generated-token budget | Session, configured by Agent or OpenAIWorkflow.run |
| Training token/context/request/time limits | OpenAIEpisodeExecutor EpisodeLimits |

Generated-token budgets use original output IDs when available, otherwise
provider usage. The total deadline covers both requests and tool-loop work.

Tool retries require retry_tools=True, the tool's retry_safe=True, and an eligible
transient failure. Defaults do not retry tools. API retries do not replay
completed tools or restart conversations. Cancelling a synchronous tool's await
cannot stop a worker thread that has already started.

For SDK scoring hooks, OpenAIWorkflow accepts scorer, interaction_scorer,
scorer_timeout, scorer_retry, and accept_sample. SDK scorers receive
(input, result), and interaction scorers receive (input, result, interaction).
Acceptance receives (result, None); returning False rejects the result without
creating a PPO row. Native EpisodeScorer reuse belongs to OpenAIEpisodeExecutor.

## Replay and recovery

### Inspect, rescore, or compare a recorded session

These operations do not sample the model or create PPO training rows:

```python
from triotune.agentrl.integrations.openai import OpenAISession


async def inspect_recording(session, other_session):
    observations = await session.replay_tools()
    budget_score = await session.rescore(
        lambda row, trace: float(len(trace.interactions) <= row["max_requests"]),
        {"max_requests": 4},
    )
    suffix = session.branch(session.interactions[0].interaction_id)
    comparison = OpenAISession.compare(session, other_session)
    return observations, budget_score, suffix, comparison
```

| Operation | Result |
| --- | --- |
| replay_tools() | Detached recorded tool calls/results; no tool execution |
| replay_tools(executor) | Re-executes recorded calls with the supplied executor |
| rescore(scorer, row) | New score returned without appending a reward |
| branch(interaction_id) | Detached recorded suffix; not an EpisodeBranch or a new sampled path |
| compare(session_a, session_b) | Comparison of recorded output IDs/logprobs where available |

To re-execute safe tools, reuse a tool registry:

```python
from triotune.agentrl import FunctionToolExecutor


async def replay_calculator(session):
    return await session.replay_tools(FunctionToolExecutor([add]))
```

Supplying an executor really runs its functions again. Use replay_tools() without
an executor to inspect records when repeating side effects would be unsafe.
Neither form changes the original session.

### Recover after cancellation

Cancel and await the owning task before recovery. Completed tool results remain
in the trace; still-running interactions are marked cancelled:

```python
import asyncio


async def cancel_and_export(task, session):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    if session.status == "completed":
        return session.export_trajectory()
    return session.recover_inflight()
```

Use this before releasing the session. Recovery never restarts a request or
replays a tool. Failed/cancelled sessions reject delayed rewards; start a fresh
candidate if another attempt is needed.

## Standard SDK entry for third-party agents

### Keep an existing SDK tool loop

OpenAIWorkflow can call an async function as execute(client, input). client is a
real session-bound AsyncOpenAI using an in-process ASGI transport; no listening
server or Triotune Agent loop is required. Install openai-proxy first.

The following callback uses the same add tool registry while owning its request
and tool loop:

```python
import copy
import json

from triotune.agentrl import FunctionToolExecutor, ToolCall
from triotune.agentrl.integrations.openai import OpenAIWorkflow


async def execute(client, input):
    messages = (
        [{"role": "user", "content": input}]
        if isinstance(input, str) else copy.deepcopy(input)
    )
    executor = FunctionToolExecutor([add])
    for _ in range(5):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=executor.schemas,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content
        for call in message.tool_calls:
            result = await executor.execute(ToolCall(
                call.id, call.function.name, json.loads(call.function.arguments),
            ))
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": result.content,
            })
    raise RuntimeError("Agent exceeded its request limit")


async def run_existing_agent(adapter, question):
    async with OpenAIWorkflow(client=adapter) as workflow:
        result = await workflow.run(
            question,
            execute=execute,
            episode_timeout=120,
            max_generated_tokens=2048,
        )
        return result.output, result.export_trajectory()
```

MODEL is defined in the PyTRIO example above; use your provider's model label
for an external adapter. The caller owns adapter and its injected upstream
client. The workflow finishes its session and closes its local SDK transport.
Callback output is preserved as result.output.

Tool observations are recorded when the callback sends them back in a following
Chat tool message or Responses function_call_output. Preserve call IDs and
assistant output history. Unsubmitted side effects and tool latency cannot be
inferred from the SDK messages. Await callback requests before returning.

### Train the existing callback through PyTRIO

Use a client-only workflow factory, the same native scorer, and execute on the
training bridge:

```python
from triotune.agentrl import AgentRolloutWorkflow, AgentTrainingLoop, OpenAIEpisodeExecutor
from triotune.agentrl.integrations.openai import OpenAIClient, OpenAIWorkflow, PyTRIOBackend


def make_callback_workflow(snapshot):
    return OpenAIWorkflow(client=OpenAIClient(
        backend=PyTRIOBackend(snapshot, model=MODEL, max_tokens=1024),
    ))


async def train_existing_agent(trainer):
    runner = OpenAIEpisodeExecutor(
        make_callback_workflow,
        execute=execute,
        scorer=Outcome(),
    )
    collector = AgentRolloutWorkflow(runner, max_concurrency=2)
    return await AgentTrainingLoop(trainer, collector).train()
```

trainer must use MODEL and provide the task rows. Native Qwen template, schema,
snapshot, and sampling restrictions still apply. Session-bound SDK requests
always use max_retries=0; configure any external provider retry policy on the
upstream SDK client. Do not override the local SDK's factory-owned base_url or
http_client. Optional SDK settings can be passed through client_options on
OpenAIWorkflow.run.

For manual SDK execution, use adapter.as_async_openai(session=session) or
proxy.create_client(session=session). Closing that client does not finish or
release the session; call session.finish before rewards/export and release it
afterwards. with_options copies retain the same session binding.

Direct OpenAIClient requests accept request_id. While the session
remains open, an identical completed request ID returns the stored response
without another provider call. Changed payloads, endpoints, parents, running
requests, or failed requests cannot reuse that identity.

## Optional local HTTP forwarding

### Create the app and make a local request

The proxy forwards POST /v1/chat/completions and POST /v1/responses. The caller
runs tools and submits follow-up requests; the proxy does not execute tools.

```python
import httpx

from triotune.agentrl.integrations.openai import OpenAIClient, OpenAIProxyServer


def make_proxy(upstream_sdk, model):
    return OpenAIProxyServer(
        OpenAIClient(upstream_sdk, model=model),
        max_sessions=128,
        max_released_session_ids=128,
    )


async def call_local_proxy(proxy, model):
    app = proxy.create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://triotune.local",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Triotune-Session-Id": "demo-session"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with 5."}],
            },
        )
        response.raise_for_status()
    session_id = response.headers["X-Triotune-Session-Id"]
    proxy.finish_session(session_id)
    trace = proxy.sessions[session_id].export_trajectory()
    proxy.release_session(session_id)
    return response.json(), trace
```

This example uses an in-process connection. The caller owns upstream_sdk and
closes it after finishing proxy work.

| Header | Purpose |
| --- | --- |
| X-Triotune-Session-Id | Send a conversation ID, or read the allocated ID from the response |
| X-Triotune-Request-Id | Optional caller-supplied request identity |
| X-Triotune-Parent-Interaction-Id | Optional explicit continuation parent |
| X-Client-Request-Id | Local request ID returned in the response |
| X-Triotune-Interaction-Id | Recorded interaction ID returned in the response |

### Bind training sessions at the proxy boundary

A PyTRIOBackend rejects default inference sessions. Supply a factory using
collector-owned contexts instead of deriving training identity from HTTP input:

```python
from triotune.agentrl import AgentContractError
from triotune.agentrl.integrations.openai import (
    OpenAIClient, OpenAIProxyServer, OpenAISession, PyTRIOBackend,
)


def make_training_proxy(snapshot, contexts):
    def session_factory(session_id):
        if session_id not in contexts:
            raise AgentContractError("Unknown collector session")
        return OpenAISession(
            session_id=session_id, context=contexts[session_id], training=True,
        )

    return OpenAIProxyServer(
        OpenAIClient(backend=PyTRIOBackend(snapshot, model=MODEL)),
        session_factory=session_factory,
    )
```

contexts maps application session IDs to existing collector root contexts;
each must match the supplied snapshot. This records fixed-policy API evidence.
Use OpenAIEpisodeExecutor to turn executions into PPO training rows.

### Handle failures and session capacity

| Situation | Behavior |
| --- | --- |
| Registered request fails | Failed interaction retained; session identity returned in headers |
| Provider HTTP error | Original status preserved |
| Connection failure / timeout / unexpected SDK failure | HTTP 502 / 504 / 500 |
| Invalid request before registration | A newly allocated empty session is reclaimed; existing evidence is preserved |
| Session capacity reached | Finish and release completed sessions before allocating more |
| Released ID still retained | Late reuse rejected |

max_sessions and max_released_session_ids default to 1024. Released IDs use a
bounded FIFO cache; after eviction, the same string can identify a new session.
Enforce permanent uniqueness in the application if needed. Retained released
trace objects stay read-only.

Use nonempty string header names and string values for extra_headers. Streaming,
multiple chat choices, background Responses, multimodal agents, subprocess
execution, and public server deployment are outside this adapter's scope.
