# Triotune

**PyTRIO-native training recipes and trainer building blocks.**

Triotune handles data preparation, rewards, advantages, token alignment, and training loops through the public PyTRIO client API. PyTRIO supplies model execution, tokenization, sampling, optimization, and checkpoint storage.

<p align="center">
  <a href="README_CN.md">中文文档</a>
</p>

## 🗂️ Contents

- [Introduction](#-introduction)
- [News](#-news)
- [Get Started](#-get-started)
  - [Installation](#-installation)
  - [Quick Start](#-quick-start)
- [Supported Features](#-supported-features)
  - [Algorithms](#-algorithms)
  - [Models](#-models)
- [Python API](#-python-api)
- [Agent RL](#agent-rl)
- [Data Contracts](#-data-contracts)
- [Token Usage Metrics](#token-usage-metrics)
- [Documentation](#documentation)
- [Repository Layout](#-repository-layout)
- [Current Scope](#-current-scope)

## 👋 Introduction

Choose from four PyTRIO-backed trainers:

- **PPO**: one asynchronous trainer for PPO, GRPO, RLOO, GSPO, SAPO, CISPO, DAPO, and related presets.
- **DPO**: chosen/rejected preference pairs, optional reference-policy scoring, and a local pairwise loss submitted through PyTRIO.
- **SFT**: text-only supervised fine-tuning with prompt/completion masking and an asynchronous training path.
- **OPD**: on-policy distillation in which a student samples completions and a fixed teacher supplies token-level scores.

PPO, DPO, and OPD include asynchronous training paths where the underlying PyTRIO client supports them. Shared utilities handle text, token-ID, and chat-message inputs, checkpoint naming, and SwanLab metrics.

For multi-turn agents, `triotune.agentrl` adds tool execution, trajectory export, scoring, and evaluation to the PPO trainer.

## 📰 News

- **0.1.3 · 2026-10-27:** Multi-turn Agent RL, tool execution, trajectory and reward alignment, OpenAI integration, and bilingual training examples.
- **0.1.2 · 2026-10-19:** Fixed package exports so the PyPI distribution and Python import both use `triotune`; updated examples and documentation.
- **0.1.1 · 2026-10-19:** Standardized dataset preprocessing, renamed the distribution and source layout to `triotune`, and added modular PPO, on-policy distillation, and YAML recipe configuration.
- **0.1.0 · 2026-10-12:** Initial release as `pytrio-cookbook` on [PyPI](https://pypi.org/project/triotune/).

## 🚀 Get Started

### 📦 Installation

- Python 3.12 (`>=3.12,<3.13`)
- Access to a PyTRIO-compatible training service and the models used by your recipe

Install [uv](https://docs.astral.sh/uv/) on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run the official installer in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then clone the repository and create the complete locked environment:

```bash
git clone https://github.com/SwanHubX/triotune.git
cd triotune
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

### 📚 Quick Start

Run the GSM8K PPO example with its YAML configuration:

```bash
uv run python examples/ppo/gsm8k_ppo.py --config examples/ppo/gsm8k_ppo_args.yaml
```

## 🔧 Supported Features

### 🧠 Algorithms

| Algorithm | Documentation | Example | Paper |
| --- | --- | --- | --- |
| **GRPO** | [📖 Docs](./docs/algorithms/grpo_series.md) | [🔧 Example](./examples/grpo) | [📄 Paper](https://arxiv.org/abs/2402.03300) |
| **PPO** | [📖 Docs](./docs/algorithms/ppo_series.md) | [🔧 Example](./examples/ppo) | [📄 Paper](https://arxiv.org/abs/1707.06347) |
| **DPO** | [📖 Docs](./docs/algorithms/dpo.md) | [🔧 Example](./examples/dpo) | [📄 Paper](https://arxiv.org/abs/2305.18290) |
| **OPD** | [📖 Docs](./docs/algorithms/opd.md) | [🔧 Example](./examples/opd) | [📄 Paper](https://arxiv.org/abs/2601.18734) |
| **SFT** | [📖 Docs](./docs/algorithms/sft.md) | [🔧 Example](./examples/sft) |  |

### 📖 Models

PyTRIO currently supports the following models:

- Qwen/Qwen3.5-4B
- Qwen/Qwen3.8-27B

## 🐍 Python API

The minimal GRPO training loop uses the default four generations per prompt:

```python
import asyncio

from triotune.trainer import PPOConfig, PPOTrainer


asyncio.run(
    PPOTrainer(
        model="your-pytrio-model-id",
        args=PPOConfig.for_algorithm("grpo"),
        reward_fn=lambda row, completion, **_: float(row["answer"] in completion),
        train_dataset=[{"messages": "What is 2 + 2?", "answer": "4"}],
    ).train()
)
```

## Agent RL

Install the `ppo` extra and configure PyTRIO credentials. Define a tool and an episode reward, then train with GRPO:

```python
import asyncio

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig, AgentTrainingLoop, EpisodeScore, FunctionToolExecutor, function_tool,
)


@function_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class AnswerScorer:
    async def score(self, row, episode):
        used_tool = any(
            not result.is_error
            for turn in episode.turns for result in turn.tool_results
        )
        success = (
            episode.status == "completed" and used_tool
            and episode.final_text.strip() == row["answer"]
        )
        return EpisodeScore(float(success), success)


async def main():
    trainer = PPOTrainer(
        model="Qwen/Qwen3.5-4B",
        args=PPOConfig.for_algorithm("grpo", batch_size=1, max_steps=1),
        train_dataset=[{
            "messages": [
                {"role": "system", "content": (
                    'Use add. Reply with JSON: '
                    '{"tool_calls":[{"name":"add","arguments":{"a":2,"b":3}}]}. '
                    'After tool_results, reply with {"final":"5"}.'
                )},
                {"role": "user", "content": "What is 2 + 3?"},
            ],
            "answer": "5",
        }],
    )
    agent = AgentConfig()
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    workflow = agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=FunctionToolExecutor([add]),
        scorer=AnswerScorer(),
    )
    state = await AgentTrainingLoop(trainer, workflow, config=agent).train()
    print(state.checkpoint_path)


asyncio.run(main())
```

See the [AgentRL guide](docs/agenrl/README.md) for evaluation and custom workflows.

## 🧩 Data Contracts

The trainers accept:

- a string;
- a non-empty list of token IDs; or
- a non-empty list of `{role, content}` chat messages.

Algorithm-specific fields are:

- **GRPO:** `messages` and a string `answer` used by the reward function;
- **DPO:** shared `messages`, plus `chosen` and `rejected` responses;
- **SFT:** `text`, `messages`, `prompt`/`completion`, or pre-tokenized `input_ids` (optionally with `labels`);
- **OPD:** prompt rows; the teacher scores completions sampled by the student.

See the [trainer guide](triotune/trainer/README.md) for validation rules and complete JSONL examples.

## Token Usage Metrics

[PyTRIO](https://pytrio.com/dashboard) records training, prefill, and sampler tokens. Triotune reports these counts and their running totals in SwanLab, using PyTRIO's sampling counters when available and submitted compute requests for training usage.

Fallback counting, failed requests, or retries may cause differences. PyTRIO's server-side records are authoritative.

## Documentation

- [Trainer guide](triotune/trainer/README.md)
- [AgentRL training guide](docs/agenrl/README.md)
- [PPO algorithm guide](docs/algorithms/ppo_series.md)
- [GRPO algorithm guide](docs/algorithms/grpo_series.md)

## 🗃 Repository Layout

```text
triotune/
├── api/           # Public configuration and client contracts
├── agentrl/       # Agent execution, trajectories, and training loops
├── dataset/       # Shared validation and tokenization
├── trainer/       # Algorithm trainers and components
└── utils/         # Shared runtime utilities
```

## 🎯 Current Scope

Triotune targets text-only training through PyTRIO. Custom model modules must implement PyTRIO's public client contract to use the trainer interfaces. Web UI, Megatron integration, multimodal training, and a general plugin registry are outside the current scope.
