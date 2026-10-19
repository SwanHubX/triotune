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
- [Quick start](#quick-start)
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

- **2026-10-12:** Triotune 0.1.0 (`pytrio-cookbook`) was published to [PyPI](https://pypi.org/project/pytrio-cookbook/).

## 📦 Installation

### ✅ Requirements

- Python 3.12 (the package currently declares `>=3.12,<3.13`)
- Access to a PyTRIO-compatible training service and the models used by your
  recipe

Using [uv](https://docs.astral.sh/uv/) is recommended because `uv.lock` pins
the repository environment:

```bash
uv sync
uv run python -c "import triotune; print(triotune.__name__)"
```

Install the optional PyTorch dependency when using DPO:

```bash
uv sync --extra dpo
```

The published package can also be installed directly:

```bash
pip install triotune==0.1.2
```

For a source checkout with pip:

```bash
pip install -r requirements.txt
pip install -e .
```

The distribution name and Python import name are both `triotune`.

## 🚀 Quick start

The GSM8K recipe prepares a JSONL dataset and then runs the PPO trainer with
its GRPO preset. The preparation command does not create a PyTRIO client, so it is a
safe first check for the local environment:

```bash
# From the repository root
bash triotune/scripts/train_grpo_gms8k.sh --prepare-only

# Start training with the default YAML configuration
bash triotune/scripts/train_grpo_gms8k.sh
```

Useful overrides include `--steps`, `--batch-size`, `--group-size`,
`--max-examples`, and `--help`. The default configuration uses a Qwen model,
LoRA, SwanLab logging, and the GSM8K boxed-answer reward. Change the model and
training settings in `triotune/configs/grpo_gms8k.yaml` for your setup.

## 🔧 Support Matrix

### 🧠 Algorithms

| Algorithm | Documentation | Paper | Configuration |
| --- | --- | --- | --- |
| **GRPO** | - | [📄 Paper](https://arxiv.org/abs/2402.03300) | [🔗 GSM8K Example](triotune/configs/grpo_gms8k.yaml) |
| **PPO** | - | [📄 Paper](https://arxiv.org/abs/1707.06347) | [🔗 GSM8K Example](triotune/configs/ppo_gsm8k.yaml) |
| **DPO** | - | [📄 Paper](https://arxiv.org/abs/2305.18290) | [🔗 HH-RLHF Example](triotune/configs/dpo_rlhf.yaml) |
| **OPD** | - |  | [🔗 DeepMath-103K Example](triotune/configs/opd_deepmath.yaml) |
| **SFT** | - |  | [🔗 Huanhuan Chat Example](triotune/configs/sft_huanhuan.yaml) |

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

See the [trainer guide](triotune/trainer/README.md) and [script guide](triotune/scripts/README.md)
for validation rules and complete JSONL examples.

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

The public imports are grouped by algorithm:

```python
import asyncio

from triotune.trainer import PPOConfig, PPOTrainer

rows = [
    {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "answer": "4",
    },
]

trainer = PPOTrainer(
    model="your-pytrio-model-id",
    reward_fn=your_reward_function,
    args=PPOConfig.for_algorithm("grpo", num_generations=4, max_steps=100),
    train_dataset=rows,
)
asyncio.run(trainer.train())
```

## 🗃️ Repository layout

```text
triotune/
├── configs/       # YAML configurations for the built-in workflows
├── recipes/       # Trainer-based and standalone PyTRIO examples
├── scripts/       # Shell launchers and dataset preparation guides
└── trainer/       # Trainers plus shared config, dataset, and client contracts
```

## 🎯 Current scope

Triotune currently targets text-only, PyTRIO-backed experiments. It does not
claim to provide a Web UI, Megatron integration, multimodal training, or a
universal plugin/registry system. If you need a custom model module, first
adapt it to the public PyTRIO client contract and use the existing recipe or
trainer interfaces; first-class custom-module registration is planned work,
not a supported promise of version 0.1.2.
