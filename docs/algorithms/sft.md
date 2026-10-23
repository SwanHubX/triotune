# SFT Example—GSM8K

[中文](sft_CN.md)

SFT learns the reference GSM8K solutions while masking the question tokens from the loss.

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
uv run python examples/sft/gsm8k_sft.py --config examples/sft/gsm8k_sft_args.yaml
```

First complete [Prepare GSM8K Data](#prepare-gsm8k-data). The entry point accepts only `--config`; edit hyperparameters in YAML.

### Configuration Interfaces

| Interface | Purpose |
| --- | --- |
| `SFTConfig(**settings)` | Read the complete explicit configuration used by this example. |
| `SFTConfig.from_modules(settings)` | Read a modular mapping with some legacy field compatibility. |

There is no `algorithm` selector or algorithm-name wrapper; `load_yaml_config` resolves `${model.name}`. The complete configuration is [gsm8k_sft_args.yaml](../../examples/sft/gsm8k_sft_args.yaml).

## Core Configuration Parameters

### Training and Rollout

The table lists example values, not all library defaults.

| Parameter | Example value | Purpose |
| --- | --- | --- |
| `model.name, model.lora_rank` | `Qwen/Qwen3.5-4B`, `32` | Trainable PyTRIO base model and LoRA rank. |
| `training.batch_size, max_steps, seed` | `4`, `100`, `42` | Batch size, maximum optimizer updates, and client seed. |
| `optimizer.learning_rate` | `1e-4` | Policy/student AdamW learning rate. |
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

Like AReaL's GSM8K SFT example, the question is masked and the reference solution is supervised. Here question/solution are rendered as chat prompt/completion pairs, and the final #### answer is converted to the same boxed format requested by GRPO/PPO.

```python
import json
from pathlib import Path
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)
for split in ("train", "test"):
    path = Path(f"data/gsm8k_sft_{split}.jsonl")
    if path.exists():
        continue
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            reasoning, answer = row["answer"].rsplit("####", 1)
            answer = answer.strip().replace(",", "")
            item = {
                "prompt": [{"role": "user", "content": row["question"]
                    + r" Provide a numerical answer without units, written inside \boxed{}."}],
                "completion": [{"role": "assistant", "content": reasoning.strip()
                    + "\n" + rf"\boxed{{{answer}}}"}],
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
```

Each JSONL line stores one record. The minimal format is:

```json
{
  "prompt": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "completion": [{"role": "assistant", "content": "1 + 1 = 2.\n\\boxed{2}"}]
}
```

### Supervision Masks and Asynchronous Execution

| Parameter | Example value | Purpose |
| --- | --- | --- |
| `data.max_length` | `1024` | Right-truncate full sequences; at least one supervised next-token target must survive. |
| `data.dataset_text_field` | `text` | Raw-text fallback key; unused by the paired GSM8K rows. |
| `data.completion_only_loss` | `true` | Mask question tokens; null auto-enables this only for paired rows. |
| `data.assistant_only_loss` | `false` | Optional assistant span masks; requires a compatible chat template when enabled. |
| `data.add_eos_token` | `true` | EOS for raw text; chat pairs use the template's termination tokens. |
| `data.chat_template_kwargs` | `{enable_thinking: false}` | Shared prompt/full-conversation template options. |
| `execution.max_in_flight` | `32` | Bound pending batches; 1 disables cross-batch pipelining. |

The paired chat format masks question tokens without requiring assistant-mask support. Full messages rows need assistant_only_loss and a compatible template to train only assistant spans. Requests preserve forward/backward → optimizer order; max_in_flight bounds queued batches, not independent optimizers.

## Core Concepts and Interfaces

### Trainer Lifecycle

| Interface | Contract |
| --- | --- |
| `SFTTrainer(args=config, train_dataset=rows, eval_dataset=valid_rows)` | Create clients; the example loads data, while the config does not open files. |
| `state = await trainer.train()` | Train, evaluate periodically, save final state, and finish reporting. |
| `await trainer.step(rows)` | Update one batch; custom loops must manage final saving and reporting lifecycle. |
| `await trainer.evaluate()` | Evaluate held-out records without optimizer updates. |

The example calls the async entry point with `asyncio.run`; calls on one trainer must not overlap.

## Triotune Implementation Notes

- **Evaluation:** `evaluation.steps: 20`, `before_train: true`, and `after_train: true`; `batch_size: 4` and `max_batches: 4` bound each pass. Set `valid_dataset: null` to disable validation. Forward-only validation reports supervised-token-weighted eval/loss and eval/perplexity, not generated-answer accuracy.
- **Logging:** `reporting.report_to: swanlab` enables uploads; project, workspace, run_name, and tags identify the run. A null workspace uses the account default. Set report_to to null to disable uploads. log_dir is relative to the working directory and stores SwanLab artifacts and unique JSONL history. logging_step controls DPO/OPD aggregation windows; SFT publishes each step.
- **Saving:** `checkpoint.save_every: 20` counts updates; 0 still saves the final state. The URI is recorded in checkpoint/path and the returned state's checkpoint_path. There is no built-in resume/load entry point.

See the [configuration source](../../triotune/api/cli_args.py) and [trainer implementation](../../triotune/trainer/sft_trainer.py) for more details.
