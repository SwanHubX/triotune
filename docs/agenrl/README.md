# AgentRL training with PyTRIO

[中文说明](README_CN.md)

AgentRL in Triotune trains a model that must decide, call tools, read tool
observations, and finish a task. The project connects that loop to the existing
PPO family trainer. This solves a practical gap in ordinary text RL: a useful
reward is attached to a complete tool-using episode, while the optimizer still
needs the exact model output tokens, behavior log probabilities, masks, and
group identity for every candidate.

## What this project adds

The training path is built around PyTRIO as the model sampling, optimization,
metrics, and checkpoint interface.

- A fixed PyTRIO sampling snapshot is used for every candidate in one batch.
- Tool calls and observations stay in the conversation context; only original
  model output tokens receive policy loss.
- Outcome rewards and optional turn, interaction, or token rewards are checked
  and aligned before PPO or GRPO preparation.
- Candidate and group provenance is preserved, so relative-reward algorithms do
  not mix unrelated prompts or policy versions.
- Recoverable model mistakes become observations or failed episodes. System
  failures can be dropped instead of silently becoming zero-reward examples.
- The same workflow can collect training data, run evaluation, write JSONL
  episode traces, and publish PyTRIO metrics.

The [AReaL agentic RL tutorial](https://areal-ai.io/docs/en/tutorial/agentic_rl.html)
is a useful conceptual starting point. This repository goes further for the
PyTRIO backend: it specifies the exact-token trajectory contract, snapshot lifecycle,
reward alignment, group filtering, retry boundaries, and policy-age checks that
make that loop safe to optimize.

## Training flow

Use the following sequence for every experiment:

1. Install the PPO extra and make PyTRIO credentials available.
2. Build rows with a messages field and any private labels used only by the
   scorer.
3. Register tools with schemas and deterministic argument validation.
4. Create an episode scorer that returns a terminal score and, when useful,
   process rewards.
5. Compose an agent workflow with AgentConfig. The workflow renders prompts,
   samples with PyTRIO, executes tools, and exports aligned trajectories.
6. Construct PPOTrainer and run AgentTrainingLoop. The loop snapshots the
   current policy, collects a complete group, calls trainer.step, then refreshes
   the snapshot for the next update.
7. Evaluate with the same workflow and scorer, then use the checkpoint URI
   reported by PyTRIO.

The calculator example is a runnable reference for this sequence:
[train.py](../../examples/agent/calculator/train.py),
[agent.py](../../examples/agent/calculator/agent.py), and
[config.yaml](../../examples/agent/calculator/config.yaml).

## Minimal runnable recipe

First run the repository example. The tokenizer and probe modes catch template
and token alignment problems before a remote update is started.

```bash
uv sync --locked --extra ppo
uv run python examples/agent/calculator/train.py --mode tokenizer
uv run python examples/agent/calculator/train.py --mode probe
uv run python examples/agent/calculator/train.py --mode train
uv run python examples/agent/calculator/train.py --mode evaluate \
  --checkpoint "trio://YOUR_RUN/weights/YOUR_CHECKPOINT"
```

The essential application composition looks like this. The application owns
the task rows, tool registry, and scorer; PyTRIO owns policy execution and the
optimizer.

```python
import asyncio
import pytrio

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import AgentConfig, AgentTrainingLoop
from examples.agent.calculator.agent import calculator_tools, CalculatorScorer


async def train(rows, eval_rows):
    service = await asyncio.to_thread(pytrio.ServiceClient)
    trainer = PPOTrainer(
        args=PPOConfig.for_algorithm(
            "grpo",
            model={"name": "Qwen/Qwen3.5-4B", "lora_rank": 8},
            training={"batch_size": 2, "max_steps": 2, "seed": 42},
            rollout={
                "num_generations": 4,
                "max_tokens": 1024,
                "temperature": 1.0,
                "top_p": 1.0,
            },
        ),
        train_dataset=rows,
        eval_dataset=eval_rows,
        service_client=service,
    )
    agent = AgentConfig(
        limits={"max_turns": 5, "max_generated_tokens": 1024},
        retry={"max_attempts": 3, "retry_tools": True},
        export_mode="concat",
        log_path="output/agentrl/episodes.jsonl",
    )
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    workflow = agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=CalculatorScorer(),
    )
    loop = AgentTrainingLoop(trainer, workflow, config=agent)
    await loop.train()
```

For a new task, change only the row schema, tools, and scorer first. Keep the
sampling temperature and top-p at 1.0: the PPO contract uses the behavior
log-probabilities recorded by the fixed PyTRIO snapshot. Keep private answer
fields out of the prompt; the scorer receives the original row separately.

## Data, tools, and rewards

Rows normally contain messages and a task identifier. The scorer can inspect
the original row and the completed episode, including final text, tool results,
status, and interaction records.

```python
from triotune.agentrl import EpisodeScore


class AnswerScorer:
    async def score(self, row, episode):
        success = (
            episode.status == "completed"
            and episode.final_text == row["answer"]
        )
        return EpisodeScore(float(success), success)


rows = [
    {
        "task_id": "addition-1",
        "messages": [{"role": "user", "content": "What is 2 + 3?"}],
        "answer": "5",
    }
]
```

### Register tools and handle failures

Use FunctionTool and FunctionToolExecutor for local Python tools. This example
reuses the calculator tool registry; replace it with your own registry using the
same composition. snapshot comes from trainer.create_sampling_snapshot, and
AnswerScorer is defined above.

```python
from examples.agent.calculator.agent import calculator_tools
from triotune.agentrl import AgentConfig

agent = AgentConfig(
    retry={"max_attempts": 3, "retry_tools": True},
    on_system_failure="drop_group",
)


def make_workflow(snapshot):
    return agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )
```

The default JSON executor requires a system message describing tools and action
formats: for example, {"tool_calls":[{"name":"calculate","arguments":{"operation":"add","a":2,"b":3}}]}
for a tool call and {"final":"5"} to finish. The calculator example's
calculator_rows and generated_rows include the corresponding system prompt.

Tool retries require both retry_tools and the tool's own retry_safe declaration.
Do not retry side effects that cannot safely be repeated. drop_group discards
groups that fail the usable-candidate requirement while keeping failed attempts
in episode logs. Use raise to stop training on infrastructure failures.

### Add process rewards

Add a per-turn tool cost alongside terminal correctness. This scorer deducts a
cost for each returned tool result in a turn with model output. Replace
AnswerScorer() with ToolCostScorer() in the workflow to enable it.

```python
from triotune.agentrl import EpisodeScore, RewardAssignment


class ToolCostScorer(AnswerScorer):
    async def score(self, row, episode):
        outcome = await super().score(row, episode)
        costs = tuple(
            RewardAssignment(
                "turn",
                -0.01 * len(turn.tool_results),
                position=index,
                source="tool_cost",
            )
            for index, turn in enumerate(episode.turns)
            if turn.tool_results and turn.generation.tokens
        )
        return EpisodeScore(
            outcome.reward,
            outcome.success,
            assignments=costs,
        )
```

position is the zero-based index in episode.turns. Return raw rewards from the
scorer; the training pipeline handles alignment, normalization, and advantage
calculation. For delayed rewards submitted through RewardLedger, submit each
logical contribution once and reuse submission IDs for retries to avoid
counting rewards twice.

## Trajectory and rollout settings

### Choose an export mode

Choose the export mode for your objective and use that AgentConfig to build the
workflow:

- concat: one complete episode becomes one PPO row.
- individual: each generated turn becomes a row. Earlier model outputs and tool
  results are context; only the selected turn receives policy loss.
- branch: export interaction-tree paths to compare different continuations from
  the same state.

```python
from triotune.agentrl import AgentConfig

export_configs = {
    "episode": AgentConfig(export_mode="concat"),
    "turn": AgentConfig(export_mode="individual"),
    "tree": AgentConfig(
        export_mode="branch",
        shared_prefix_policy="weighted",
    ),
}
agent = export_configs["turn"]


def make_workflow(snapshot):
    return agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )
```

The example selects individual. This changes temporal reward modeling: GAE is
computed independently for each row, so it need not match concat.

Before using branch, your custom executor must collect branches and populate
episode.branches with EpisodeBranch records: fork position, interaction IDs,
branch reward, and completion status. Changing export_mode alone does not
create counterfactual branches.

Shared prefixes use deduplicate by default, counting each model action once.
repeat counts the action separately for each path. weighted distributes shared
action contributions using each branch's shared_prefix_weight; set those
weights on the branch records when choosing weighted. See the
[branch trajectory example](agentrl_training.md#branch-trees).

### Enable asynchronous collection

Verify tasks, tools, and rewards with default synchronous collection first.
Then replace the original training call with this function. trainer is the
PPOTrainer created earlier. At the end of training, the loop closes the workflow
and finalizes the trainer's reporting.

```python
from triotune.agentrl import AgentConfig, AgentTrainingLoop


async def train_with_prefetch(trainer):
    agent = AgentConfig(
        export_mode="concat",
        max_concurrency=4,
        async_rollouts=True,
        rollout_prefetch_capacity=1,
        max_policy_staleness=1,
    )
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    workflow = agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )
    return await AgentTrainingLoop(trainer, workflow, config=agent).train()
```

Prefetch capacity counts batches collected ahead of updates; max_concurrency
limits concurrent candidates. max_policy_staleness must be at least
rollout_prefetch_capacity. Each rollout keeps one snapshot throughout;
asynchronous mode overlaps later collection with the current update. Do not
call step concurrently on the same trainer.

### Shorten long conversation context

When tool observations grow long, enable context compaction before building the
workflow:

```python
agent = AgentConfig(
    context_compaction=True,
    compaction_keep_recent_interactions=2,
    compaction_summary_chars=2048,
    prefix_cache_size=256,
)
```

Compaction keeps initial instructions, recent interactions, and tool-error
interactions, replacing older content with summaries. Summaries change what the
model sees, so evaluate task success again after enabling them. Previously
sampled actions retain their original tokens and are exported as fragments.
prefix_cache_size caches only template tokenizations, never generations. See
[context compaction](agentrl_training.md#context-compaction-and-prefix-reuse)
for more settings.

## OpenAI interfaces

### Train with OpenAI-style tool calls

To register tools with function_tool and run an OpenAI-style Agent, bind
PyTRIOBackend to the current training snapshot and connect it to the same
training loop through OpenAIEpisodeExecutor:

```bash
uv sync --locked --extra ppo --extra openai
```

This example reuses AnswerScorer and rows defined above. Agent handles tool
calls here; unlike the default JSON executor, it does not require the model to
output {"final":"5"}.

```python
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentRolloutWorkflow,
    AgentTrainingLoop,
    EpisodeLimits,
    OpenAIEpisodeExecutor,
    function_tool,
)
from triotune.agentrl.integrations.openai import (
    Agent,
    OpenAIClient,
    OpenAIWorkflow,
    PyTRIOBackend,
)

MODEL = "Qwen/Qwen3.5-4B"


@function_tool(retry_safe=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def make_openai_workflow(snapshot):
    client = OpenAIClient(
        backend=PyTRIOBackend(snapshot, model=MODEL, max_tokens=256),
    )
    return OpenAIWorkflow(Agent(
        name="calculator",
        instructions="Use the add tool. Answer with the number only.",
        tools=[add],
        model=MODEL,
        client=client,
    ))


async def train_openai(rows):
    runner = OpenAIEpisodeExecutor(
        make_openai_workflow,
        scorer=AnswerScorer(),
        limits=EpisodeLimits(
            max_turns=5,
            max_tokens_per_turn=256,
            max_generated_tokens=1024,
        ),
    )
    workflow = AgentRolloutWorkflow(runner, max_concurrency=2)
    trainer = PPOTrainer(
        MODEL,
        args=PPOConfig.for_algorithm(
            "grpo", batch_size=1, num_generations=4, max_steps=1,
        ),
        train_dataset=rows,
    )
    return await AgentTrainingLoop(trainer, workflow).train()
```

Call await train_openai(rows) from your async entry point. This path samples and
trains through PyTRIO, so it requires PyTRIO credentials and no OpenAI API key.

Every make_openai_workflow call must create a fresh workflow, Agent, client, and
backend bound to the supplied snapshot. Do not reuse these objects between
candidates or switch a shared client's backend. The bridge closes these
candidate resources; the trainer manages its sampling clients.

The scorer still receives the original row; answer fields are not sent to the
model. Do not count the same terminal reward in both AnswerScorer and SDK reward
submissions. External OpenAI responses and provider traces cannot be passed
directly to PPO. Training requires original PyTRIO output tokens, behavior log
probabilities, and policy versions.

For an existing third-party asynchronous execution function, use the execute
parameter on OpenAIEpisodeExecutor. See [OpenAI tools and sessions](agentrl_openai.md)
for session binding, the standard AsyncOpenAI client, cancellation recovery, and
local HTTP forwarding. Install the openai-proxy extra as described there when
using third-party execution callbacks or HTTP forwarding.

## Diagnostics and evaluation

### Inspect a rollout group before training

Run the tokenizer and probe commands above first. In a custom application, call
this function with the same trainer, workflow, and AgentConfig before loop.train
to inspect a task group. It collects and scores without invoking the optimizer:

```python
async def inspect_rollout(trainer, workflow, agent, rows):
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    batch = await workflow.collect(
        rows[:1],
        snapshot,
        n_samples=2,
        is_eval=True,
        strict=False,
    )
    print(batch.metrics(prefix="probe"))
    print(batch.groups)
    for episode in batch.episodes:
        print(
            episode.status,
            episode.snapshot_id,
            episode.policy_version,
            episode.group_id,
        )
    for trajectory in batch.trajectories:
        print("action_tokens", sum(trajectory.effective_action_weights))
    return batch
```

strict=False retains failed candidates in the returned batch for inspection; it
does not make failed trajectories eligible for training. Inspect status for
format errors, tool failures, or timeouts, and batch.groups for candidate
completeness and acceptance. For within-group relative rewards such as GRPO,
ensure candidates have reward variance or a useful process signal. With
dynamic_sampling enabled, groups without either signal are filtered out.

### Save and read episode logs

Enable log_path on the AgentConfig used to build the workflow. After execution,
call show_episode_log to view episode status, policy provenance, groups, and
rewards:

```python
import json
from pathlib import Path
from triotune.agentrl import AgentConfig

agent = AgentConfig(log_path="output/agentrl/episodes.jsonl")


def make_logged_workflow(snapshot):
    return agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )


def show_episode_log(path="output/agentrl/episodes.jsonl"):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            score = episode.get("score") or {}
            print(
                episode["status"],
                episode["snapshot_id"],
                episode["policy_version"],
                episode["group_id"],
                score.get("reward"),
            )
```

log_path attaches a logger when agent.build_workflow is called. Changing the
configuration does not affect an already-built workflow. When constructing
AgentRolloutWorkflow manually, pass JsonlEpisodeLogger through its logger
parameter. Use different log paths for different processes to avoid concurrent
writes to the same file.

### Evaluate a checkpoint separately

Take the checkpoint URI from the training report and evaluate it with the same
tools and scorer:

```bash
uv run python examples/agent/calculator/train.py --mode evaluate \
  --checkpoint "trio://YOUR_RUN/weights/YOUR_CHECKPOINT"
```

Evaluation fixes one policy snapshot, collects, and scores without updating
weights. The calculator example saves evaluation metrics to
output/calculator-agent/evaluate-report.json. Compare success rate, tool error
rate, and generated token counts to check task completion and execution cost.
