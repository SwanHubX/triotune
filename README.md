# Triotune

**PyTRIO-native training recipes and trainer building blocks.**

Triotune provides text-training recipes, trainer components, and SWE agent workflows through the public PyTRIO client API. It handles data preparation, reward and advantage calculation, token alignment, and training-loop orchestration. PyTRIO supplies model execution, tokenization, sampling, and optimization.

<p align="center">
  <a href="README_CN.md">中文文档</a>
</p>


## 🗂️ Contents

- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#-installation)
- [Algorithms](#-algorithms)
- [Data contracts](#-data-contracts)
- [Token usage metrics](#token-usage-metrics)
- [Python API](#-python-api)
- [Triotune-SWE](#triotune-swe)
- [Documentation](#documentation)
- [Repository layout](#-repository-layout)
- [Current scope](#-current-scope)

## 👋 Introduction

Triotune provides a trainer-shaped interface for a few PyTRIO-backed training algorithms. The package is intentionally client-side: PyTRIO remains responsible for model weights, LoRA training clients, tokenizers, sampling, forward/backward execution, optimizer steps, and checkpoint storage.

The current trainers are:

- **PPO**: one asynchronous trainer for PPO, GRPO, RLOO, GSPO, SAPO, CISPO, DAPO, and related presets.
- **DPO**: chosen/rejected preference pairs, optional reference-policy scoring, and a local pairwise loss submitted through PyTRIO.
- **SFT**: text-only supervised fine-tuning with prompt/completion masking and an asynchronous training path.
- **OPD**: on-policy distillation in which a student samples completions and a fixed teacher supplies token-level scores.

PPO, DPO, and OPD include asynchronous training paths where the underlying PyTRIO client supports them. Shared utilities handle text, token-ID, and chat-message inputs, checkpoint naming, and SwanLab metrics.

Triotune-SWE connects model sampling, a coding agent, and disposable task environments. It turns issue descriptions into tested patches and exports repair episodes for supervised fine-tuning or reinforcement learning.

## 📰 News

- **2026-10-19:** Triotune 0.1.1 standardized dataset preprocessing and renamed the distribution and source layout to triotune, while adding modular PPO and on-policy distillation pipelines plus YAML-based recipe configuration.
- **2026-10-12:** Triotune 0.1.0 (pytrio-cookbook) was published to [PyPI](https://pypi.org/project/pytrio-cookbook/).

## 🚀 Get Started

### 📦 Installation

- Python 3.12 (the package currently declares >=3.12,<3.13)
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

## 🔧 Support Matrix

### 🧠 Algorithms

| Algorithm | Documentation | Example | Paper |
| --- | --- | --- | --- |
| **GRPO** | [📖 Docs](./docs/algorithms/grpo_series.md) | [🔧 Example](./examples/grpo) | [📄 Paper](https://arxiv.org/abs/2402.03300) |
| **PPO** | [📖 Docs](./docs/algorithms/ppo_series.md) | [🔧 Example](./examples/ppo) | [📄 Paper](https://arxiv.org/abs/1707.06347) |
| **DPO** | - | - | [📄 Paper](https://arxiv.org/abs/2305.18290) |
| **OPD** | - | - | [📄 Paper](https://arxiv.org/abs/2601.18734) |
| **SFT** | - | - |  |

### 📖 Models

PyTRIO currently supports the following models:

- Qwen/Qwen3.5-4B
- Qwen/Qwen3.5-9B
- Qwen/Qwen3.8-27B

## 🧩 Data contracts

The trainers accept a deliberately small set of input representations:

- a string;
- a non-empty list of token IDs; or
- a non-empty list of {role, content} chat messages.

Algorithm-specific fields are:

- **GRPO:** messages and a string answer used by the reward function;
- **DPO:** shared messages, plus chosen and rejected responses;
- **SFT:** text, messages, prompt/completion, or pre-tokenized input_ids (optionally with labels);
- **OPD:** prompt rows; the teacher scores completions sampled by the student.

See the [trainer guide](triotune/trainer/README.md) for validation rules and complete JSONL examples.

## Token usage metrics

PyTRIO records training tokens , prefill tokens , sampler tokens. Triotune reports these counts and their running totals in SwanLab, preferring PyTRIO's sampling counters and accumulating training usage from submitted compute requests. The counts normally agree, but fallback counting, failed requests, or retries may cause differences; PyTRIO's server-side records are authoritative.

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

## 🗃 Repository layout

```text
triotune/
├── api/           # Public configuration and client contracts
├── dataset/       # Shared validation and tokenization
├── swe/           # SWE sampling and training trajectory contracts
├── trainer/       # Algorithm trainers and components
└── utils/         # Shared runtime utilities
```

## 🎯 Current scope

Triotune currently targets text-only, PyTRIO-backed experiments. It does not claim to provide a Web UI, Megatron integration, multimodal training, or a universal plugin/registry system. If you need a custom model module, first adapt it to the public PyTRIO client contract and use the existing trainer interfaces; first-class custom-module registration is planned work, not a supported promise of version 0.1.2.
