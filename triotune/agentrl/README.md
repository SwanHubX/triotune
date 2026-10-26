# Agent RL

[中文](README_CN.md) | English

## Overview

triotune.agentrl connects tool-using agents to the existing PPOTrainer. It supports synchronous, multi-turn execution, trajectory collection, reward assignment, training updates and evaluation. PyTRIO handles model execution, optimization and checkpoint storage.

## How it works

1. Export one fixed policy snapshot for a batch of tasks.
2. Generate an action, parse tool calls, execute tools and append observations until the episode ends.
3. Preserve original output tokens and behavior log probabilities. Build a Trajectory with action masks and turn IDs; observations and template tokens provide context without policy loss.
4. Assign outcome and optional process rewards, then call trainer.step() to update the policy.
5. Refresh the snapshot for the next batch. Evaluate through the same execution and scoring logic.

Each episode receives provenance before execution: `rollout_id` identifies a logical task,
`episode_id` identifies one candidate attempt, `group_id` identifies the relative-reward group,
`sample_index` identifies the candidate position, and `snapshot_id`/`policy_version` identify
the sampling policy. The exporter carries these fields into each PPO `Trajectory` for scoring,
diagnostics, and mixed-policy checks.

Recoverable model action errors become observations or scored failures. Infrastructure failures never become zero-reward training examples. Retries are bounded, and tool retries require an explicit declaration that repeated execution is safe.

## Modules

| Location | Responsibility |
|---|---|
| config.py | Agent limits, retries, evaluation settings and adapter composition |
| types.py | Execution records, protocols and boundary validation |
| runtime/ | Conversation rendering, episode execution, sampling and tools |
| rollout/ | Grouped collection and exact-token trajectory construction |
| training/ | Training loop, reward composition and evaluation metrics |
| diagnostics/ | Episode logs and tokenizer preflight checks |

## Quick start

From the repository root, install dependencies and configure PyTRIO credentials:

```bash
uv sync --locked --extra ppo
uv run python examples/agent/calculator/train.py --mode tokenizer
uv run python examples/agent/calculator/train.py --mode probe
uv run python examples/agent/calculator/train.py --mode train
```

The calculator uses [config.yaml](../../examples/agent/calculator/config.yaml), defaults to Qwen/Qwen3.5-4B and runs two updates. Reports and episode logs are saved under output/calculator-agent/. Use the checkpoint URI from train-report.json for independent evaluation:

```bash
uv run python examples/agent/calculator/train.py --mode evaluate --checkpoint "trio://YOUR_RUN/weights/YOUR_CHECKPOINT"
```

## Custom agents

Create a PPOTrainer with training and evaluation rows, a tool executor and an episode scorer. Run this composition inside an async function, with trainer, tools and scorer supplied by the application:

```python
from triotune.agentrl import AgentConfig, AgentTrainingLoop

config = AgentConfig(
    limits={"max_turns": 5},
    retry={"max_attempts": 3, "retry_tools": True},
    log_path="output/agentrl/episodes.jsonl",
)
snapshot = await trainer.create_sampling_snapshot(stop=config.stop)
workflow = config.build_workflow(
    tokenizer=snapshot.tokenizer,
    tools=tools,
    scorer=scorer,
)
state = await AgentTrainingLoop(trainer, workflow, config=config).train()
```

The default input field is messages. Tools can use FunctionToolExecutor; scorers return EpisodeScore with a terminal reward, a success flag and optional token-aligned process rewards. CompositeScorer, ToolCostScorer and TurnRewardScorer combine outcome, cost and process signals.

The default adapters use JSON actions such as {"tool_calls":[{"name":"calculate","arguments":{"operation":"add","a":2,"b":3}}]} and {"final":"5"}. Explain the action protocol and tools in the system message. Set AgentConfig.prompt_key for another input field, or replace adapters through the public protocols.

## Current scope

Sampling uses temperature=1 and top_p=1 to preserve the PPO log-probability contract. The default renderer appends context to original sampled tokens; check tokenizer compatibility before using another model. Third-party SDK proxies, branching trajectories, context compaction and overlapping sampling with training are outside the current implementation.
