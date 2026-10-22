# Triotune

**PyTRIO-native training recipes and trainer building blocks.**

Triotune is a small, text-only training cookbook built on the public PyTRIO
client API. The name combines **Trio**, from PyTRIO, with **tune**, inspired by
Torchtune. It keeps model execution, tokenization, sampling, and optimization
in PyTRIO, while providing the client-side data preparation, reward and
advantage calculation, token alignment, and training-loop orchestration.

<p align="center">
  <a href="README_CN.md">中文文档</a>
</p>

> **Project status:** Triotune is an early-stage cookbook. It currently offers
> built-in text-training workflows and examples; a general API for registering
> arbitrary custom model modules is not yet part of the project.

## 🗂️ Contents

- [Introduction](#introduction)
- [News](#news)
- [Installation](#installation)
- [Algorithms](#algorithms)
- [Data contracts](#data-contracts)
- [Token usage metrics](#token-usage-metrics)
- [Python API](#python-api)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Current scope](#current-scope)

## 👋 Introduction

Triotune provides a trainer-shaped interface for a few PyTRIO-backed training
algorithms. The package is intentionally client-side: PyTRIO remains
responsible for model weights, LoRA training clients, tokenizers, sampling,
forward/backward execution, optimizer steps, and checkpoint storage.

The current trainers are:

- **PPO**: one asynchronous trainer for PPO, GRPO, RLOO, GSPO, SAPO,
  CISPO, DAPO, and related presets.
- **DPO**: chosen/rejected preference pairs, optional reference-policy scoring,
  and a local pairwise loss submitted through PyTRIO.
- **SFT**: text-only supervised fine-tuning with prompt/completion masking and
  an asynchronous training path.
- **OPD**: on-policy distillation in which a student samples completions and a
  fixed teacher supplies token-level scores.

PPO, DPO, and OPD include asynchronous training paths where
the underlying PyTRIO client supports them. Shared utilities handle text,
token-ID, and chat-message inputs, checkpoint naming, and SwanLab metrics.

## 📰 News

- **2026-10-19:** Triotune 0.1.1 standardized dataset preprocessing and renamed
  the distribution and source layout to `triotune`, while adding modular PPO
  and on-policy distillation pipelines plus YAML-based recipe configuration.
- **2026-10-12:** Triotune 0.1.0 (`pytrio-cookbook`) was published to [PyPI](https://pypi.org/project/pytrio-cookbook/).

## 🚀 Get Started

### 📦 Installation

- Python 3.12 (the package currently declares `>=3.12,<3.13`)
- Access to a PyTRIO-compatible training service and the models used by your
  recipe

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

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.8-27B`

## 🧩 Data contracts

The trainers accept a deliberately small set of input representations:

- a string;
- a non-empty list of token IDs; or
- a non-empty list of `{role, content}` chat messages.

Algorithm-specific fields are:

- **GRPO:** `messages` and a string `answer` used by the reward function;
- **DPO:** shared `messages`, plus `chosen` and `rejected` responses;
- **SFT:** `text`, `messages`, `prompt`/`completion`, or pre-tokenized
  `input_ids` (optionally with `labels`);
- **OPD:** prompt rows; the teacher scores the student's sampled completion.

See the [trainer guide](triotune/trainer/README.md) for validation rules and
complete JSONL examples.

## Token usage metrics

Trainer-backed SwanLab runs report the tokens submitted to training compute as
`usage/train_tokens` and its running total as `usage/train_tokens_total`.
Sampling workflows additionally report `usage/sample_input_tokens`,
`usage/sample_output_tokens`, and their `_total` counterparts. Training usage
counts every actual remote training pass, including repeated PPO epochs and
minibatches. Sampling usage uses PyTRIO's response counters when available and
falls back to the tokenized prompt and returned sequences for compatible test
or older clients.

`usage/sample_input_tokens` is intentionally not labeled as either prefill or
cache usage. Prefix-cache hits are decided inside the serving system after
request routing: they depend on the selected replica, the prefixes currently
resident in that replica's cache, partial-prefix matching, and cache eviction.
The client sends the same prompt tokens whether they are recomputed or served
from cache, so prompt length alone cannot recover that split. PyTRIO 0.2.9
returns aggregate sampling input/output tokens but no cached-input or
uncached-prefill counters. Triotune therefore does not guess the split or
calculate a monetary cost from a hard-coded model price table.

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

## 🗃️ Repository layout

```text
triotune/
├── api/           # Public configuration and client contracts
├── dataset/       # Shared validation and tokenization
├── trainer/       # Algorithm trainers and components
└── utils/         # Shared runtime utilities
```

## 🎯 Current scope

Triotune currently targets text-only, PyTRIO-backed experiments. It does not
claim to provide a Web UI, Megatron integration, multimodal training, or a
universal plugin/registry system. If you need a custom model module, first
adapt it to the public PyTRIO client contract and use the existing trainer
interfaces; first-class custom-module registration is planned work,
not a supported promise of version 0.1.2.
