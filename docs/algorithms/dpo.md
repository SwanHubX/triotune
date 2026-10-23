# DPO Example—GSM8K

[中文](dpo_CN.md)

DPO learns to prefer a chosen solution over a rejected solution relative to a fixed reference policy.

## Example Usage

### Environment Setup

Run from the repository root with Python 3.12 and access to a PyTRIO training service.

```bash
git clone https://github.com/SwanHubX/triotune.git
cd triotune
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

### Quick Start

```bash
uv run python examples/dpo/gsm8k_dpo.py --config examples/dpo/gsm8k_dpo_args.yaml
```

First complete [Prepare GSM8K Data](#prepare-gsm8k-data). The entry point accepts only `--config`; edit hyperparameters in YAML.

### Configuration Interfaces

| Interface | Purpose |
| --- | --- |
| `DPOConfig(**settings)` | Read the complete explicit configuration used by this example. |
| `DPOConfig.from_modules(settings)` | Read a modular mapping with some legacy field compatibility. |

There is no `algorithm` selector or algorithm-name wrapper; `load_yaml_config` resolves `${model.name}`. The complete configuration is [gsm8k_dpo_args.yaml](../../examples/dpo/gsm8k_dpo_args.yaml).

## Core Configuration Parameters

### Training and Rollout

The table lists example values, not all library defaults.

| Parameter | Example value | Purpose |
| --- | --- | --- |
| `model.name, model.lora_rank` | `Qwen/Qwen3.5-4B`, `32` | Trainable PyTRIO base model and LoRA rank. |
| `training.batch_size, max_steps, seed` | `2`, `100`, `42` | Batch size, maximum optimizer updates, and client seed. |
| `optimizer.learning_rate` | `1e-5` | Policy/student AdamW learning rate. |
| `optimizer.beta1, beta2, eps` | `0.9`, `0.95`, `1e-8` | Adam moments and numerical stabilizer. |
| `optimizer.weight_decay, grad_clip_norm` | `0.0`, `1.0` | Weight decay and gradient norm limit; zero clip norm disables clipping. |
| `train_dataset.path, valid_dataset.path` | See YAML | Prepared JSONL paths resolved relative to the repository root. |
| `train_dataset.split, valid_dataset.split` | `train`, `test` | Logical split names for the official GSM8K train/test files. |
| `train_dataset.max_examples, valid_dataset.max_examples` | `null`, `100` | Maximum rows after loading/shuffling; null keeps the entire file. |
| `train_dataset.shuffle, valid_dataset.shuffle` | `false`, `false` | Preserve source order; true shuffles once. |
| `train_dataset.seed, valid_dataset.seed` | `42`, `42` | Shuffle seed; null falls back to training.seed. |
| `train_dataset.loader_options, valid_dataset.loader_options` | `{}`, `{}` | Extra JSON-loader kwargs; do not duplicate data_files or split. |

max_steps counts optimizer updates. Batch units are preference pairs for DPO, prompts for OPD, and supervised sequences for SFT. Data is traversed once and exhaustion ends training early; there is no implicit cycling or repeated epoch. The example loader applies row limits and shuffling.

### Prepare GSM8K Data

The example reads local JSONL and does not download automatically. This Python code prepares the official train/test splits and reuses existing files.

GSM8K has no native preference pairs. This example uses the reference reasoning and final answer as chosen, and constructs rejected by adding 1 to the final number while retaining the reasoning. These are synthetic demonstration pairs, not human preference annotations.

```python
import json
from decimal import Decimal
from pathlib import Path
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)
for split in ("train", "test"):
    path = Path(f"data/gsm8k_dpo_{split}.jsonl")
    if path.exists():
        continue
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            reasoning, answer = row["answer"].rsplit("####", 1)
            answer = answer.strip().replace(",", "")
            wrong_answer = format(Decimal(answer) + 1, "f")
            item = {
                "messages": [{"role": "user", "content": row["question"]
                    + r" Provide a numerical answer without units, written inside \boxed{}."}],
                "chosen": reasoning.strip() + "\n" + rf"\boxed{{{answer}}}",
                "rejected": reasoning.strip() + "\n" + rf"\boxed{{{wrong_answer}}}",
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
```

Each JSONL line stores one record. The minimal format is:

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "chosen": "1 + 1 = 2.\n\\boxed{2}",
  "rejected": "1 + 1 = 2.\n\\boxed{3}"
}
```

### Reference and Preference Objective

| Parameter | Example value | Purpose |
| --- | --- | --- |
| `reference.name` | `${model.name}` | Frozen reference model; must share the policy token-to-ID vocabulary. |
| `data.max_length` | `2048` | Maximum prompt plus either response. Overlong pairs raise; null removes this limit. |
| `data.prompt_key` | `messages` | Shared prompt field. |
| `data.add_eos_token` | `true` | Append EOS when missing from a completion. |
| `data.chat_template_kwargs` | `{enable_thinking: false}` | Template options for rendering prompts. |
| `objective.beta` | `0.1` | Positive scale on the policy/reference preference margin. |

DPO sums completion logprobs and excludes prompt tokens. The reference is never updated. Finite data.max_length rejects rather than truncates overlong pairs; prefilter both alternatives using the actual tokenizer or set it to null.

## Core Concepts and Interfaces

### Trainer Lifecycle

| Interface | Contract |
| --- | --- |
| `DPOTrainer(args=config, train_dataset=rows, eval_dataset=valid_rows)` | Create clients; the example loads data, while the config does not open files. |
| `state = await trainer.train_async()` | Train, evaluate periodically, save final state, and finish reporting. |
| `await trainer.step_async(rows)` | Update one batch; custom loops must manage final saving and reporting lifecycle. |
| `await trainer.evaluate_async()` | Evaluate held-out records without optimizer updates. |

The example calls the async entry point with `asyncio.run`; calls on one trainer must not overlap.
Synchronous `train()`, `step()`, and `evaluate()` are also available; use the methods above in async applications.

## Triotune Implementation Notes

- **Evaluation:** `evaluation.steps: 20`, `before_train: true`, and `after_train: true`; `batch_size: 2` and `max_batches: 4` bound each pass. Set `valid_dataset: null` to disable validation. Current-policy/fixed-reference scoring reports eval/loss, eval/reward_accuracy, and eval/reward_margin; preference accuracy is not GSM8K answer accuracy.
- **Logging:** `reporting.report_to: swanlab` enables uploads; project, workspace, run_name, and tags identify the run. A null workspace uses the account default. Set report_to to null to disable uploads. log_dir is relative to the working directory and stores SwanLab artifacts and unique JSONL history. logging_step controls DPO/OPD aggregation windows; SFT publishes each step.
- **Saving:** `checkpoint.save_every: 20` counts updates; 0 still saves the final state. The URI is recorded in checkpoint/path and the returned state's checkpoint_path. There is no built-in resume/load entry point.

See the [configuration source](../../triotune/api/cli_args.py) and [trainer implementation](../../triotune/trainer/dpo_trainer.py) for more details.
