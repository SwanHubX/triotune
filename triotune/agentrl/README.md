# AgentRL

[中文说明](README_CN.md) | [完整训练文档](../../docs/agenrl/README.md)

The triotune.agentrl package connects a tool-using agent to the existing
PPO-family trainer. PyTRIO owns model sampling, optimization, metrics, and
checkpoint storage; this package owns the episode protocol, tools, scoring, and
exact-token trajectory export.

## What is trained

One candidate follows this loop:

1. PyTRIO exports one fixed sampling snapshot.
2. The agent generates an action, calls a registered tool, and reads its
   observation until the episode finishes.
3. The runner keeps original output tokens and behavior log probabilities.
   Prompt templates and tool observations are context, so they do not receive
   policy loss.
4. A scorer assigns outcome and optional process rewards.
5. AgentTrainingLoop passes validated trajectories to trainer.step, then takes a
   fresh snapshot for the next batch.

This preserves candidate identity, reward groups, policy versions, and action
alignment. It also separates recoverable model mistakes from infrastructure
failures so a service outage does not become a misleading zero-reward sample.

## Quick start

From the repository root:

```bash
uv sync --locked --extra ppo
uv run python examples/agent/calculator/train.py --mode tokenizer
uv run python examples/agent/calculator/train.py --mode probe
uv run python examples/agent/calculator/train.py --mode train
```

The runnable calculator workflow is in
[examples/agent/calculator](../../examples/agent/calculator). For the complete
training recipe, configuration choices, and diagnostics, read
[docs/agenrl/README.md](../../docs/agenrl/README.md).

## Package boundaries

| Package area | Responsibility |
| --- | --- |
| config.py | Limits, retries, renderers, export mode, evaluation, and workflow composition |
| runtime/ | Conversation rendering, sampling, episode execution, and tools |
| rollout/ | Grouped collection and exact-token trajectory construction |
| training/ | Agent training loop, reward composition, and evaluation metrics |
| diagnostics/ | Episode logs and tokenizer preflight checks |
| integrations/openai/ | OpenAI-compatible execution and the PyTRIO trajectory bridge |

Use the detailed contracts for reward assignment, branches, context
compaction, policy age, and group filtering in
[agentrl_training.md](../../docs/agenrl/agentrl_training.md). OpenAI session
ownership, replay, recovery, and third-party forwarding are documented in
[agentrl_openai.md](../../docs/agenrl/agentrl_openai.md).

Sampling uses temperature 1 and top-p 1 to preserve the PPO log-probability
contract. Begin with synchronous collection and a fixed snapshot; enable
bounded asynchronous collection only after the synchronous workflow passes its
tokenizer and probe checks.
