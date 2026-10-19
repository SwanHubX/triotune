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

## Contents

- [Introduction](#introduction)
- [News](#news)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Algorithms and examples](#algorithms-and-examples)
- [Data contracts](#data-contracts)
- [Python API](#python-api)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Current scope](#current-scope)

## Introduction

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

## News

- **2026-10-12:** Triotune 0.1.0 (`pytrio-cookbook`) was published to [PyPI](https://pypi.org/project/pytrio-cookbook/).

## Installation

### Requirements

- Python 3.12 (the package currently declares `>=3.12,<3.13`)
- Access to a PyTRIO-compatible training service and the models used by your
  recipe

Using [uv](https://docs.astral.sh/uv/) is recommended because `uv.lock` pins
the repository environment:

```bash
uv sync
uv run python -c "import trio_cookbook; print(trio_cookbook.__name__)"
```

Install the optional PyTorch dependency when using DPO:

```bash
uv sync --extra dpo
```

The published package can also be installed directly:

```bash
pip install pytrio-cookbook==0.1.0
```

For a source checkout with pip:

```bash
pip install -r requirements.txt
pip install -e .
```

The distribution name is `pytrio-cookbook`; the Python import name is
`trio_cookbook`.

## Quick start

The GSM8K recipe prepares a JSONL dataset and then runs the PPO trainer with
its GRPO preset. The preparation command does not create a PyTRIO client, so it is a
safe first check for the local environment:

```bash
# From the repository root
bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only

# Start training with the default TOML configuration
bash trio-cookbook/scripts/train_grpo_gms8k.sh
```

Useful overrides include `--steps`, `--batch-size`, `--group-size`,
`--max-examples`, and `--help`. The default configuration uses a Qwen model,
LoRA, SwanLab logging, and the GSM8K boxed-answer reward. Change the model and
training settings in `trio-cookbook/configs/grpo_gms8k.toml` for your setup.

## Algorithms and examples

| Algorithm | Built-in example | Data | Entry point |
| --- | --- | --- | --- |
| PPO / GRPO | [GSM8K](trio-cookbook/recipes/grpo_gms8k_trio.py) | Prompt + reference answer | [`train_grpo_gms8k.sh`](trio-cookbook/scripts/train_grpo_gms8k.sh) |
| DPO | [HH-RLHF](trio-cookbook/recipes/dpo_rlhf_trio.py) | `messages`, `chosen`, `rejected` | [`train_dpo_rlhf.sh`](trio-cookbook/scripts/train_dpo_rlhf.sh) |
| SFT | [Huanhuan chat](trio-cookbook/recipes/sft_huanhuan_trio.py) | Text, messages, or prompt/completion | [`train_sft_huanhuan.sh`](trio-cookbook/scripts/train_sft_huanhuan.sh) |
| OPD | [DeepMath-103K](trio-cookbook/recipes/opd_deepmath_trio.py) | Prompt-only rows | [`train_opd_deepmath.sh`](trio-cookbook/scripts/train_opd_deepmath.sh) |

Every example has a TOML configuration under
[`trio-cookbook/configs/`](trio-cookbook/configs/). The `recipes/` directory
also contains standalone PyTRIO scripts that are useful as readable algorithm
baselines alongside the trainer-based versions.

## Data contracts

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

See the [trainer guide](trio-cookbook/trainer/README.md) and [script guide](trio-cookbook/scripts/README.md)
for validation rules and complete JSONL examples.

## Python API

The public imports are grouped by algorithm:

```python
import asyncio

from trio_cookbook.trainer import PPOConfig, PPOTrainer

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

`DPOTrainer`, `SFTTrainer`, and `OPDTrainer` follow the same configuration-
and-dataset-oriented style. Replace the example model ID and reward function
with values supported by your PyTRIO service.

## Documentation

- [Trainer internals and Python examples](trio-cookbook/trainer/README.md)
- [Recipes, configurations, and script commands](trio-cookbook/scripts/README.md)
- [Package-level README](trio-cookbook/README.md)
- [Framework evaluation notes](docs/README.md)

## Repository layout

```text
trio-cookbook/
├── configs/       # TOML configurations for the built-in workflows
├── recipes/       # Trainer-based and standalone PyTRIO examples
├── scripts/       # Shell launchers and dataset preparation guides
└── trainer/       # Trainers plus shared config, dataset, and client contracts
```

## Current scope

Triotune currently targets text-only, PyTRIO-backed experiments. It does not
claim to provide a Web UI, Megatron integration, multimodal training, or a
universal plugin/registry system. If you need a custom model module, first
adapt it to the public PyTRIO client contract and use the existing recipe or
trainer interfaces; first-class custom-module registration is planned work,
not a supported promise of version 0.1.0.
