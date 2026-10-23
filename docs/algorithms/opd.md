# OPD Example—GSM8K

[中文](opd_CN.md)

OPD uses a fixed teacher to score student-generated GSM8K solutions and train the student.

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
uv run python examples/opd/gsm8k_opd.py --config examples/opd/gsm8k_opd_args.yaml
```

First complete [Prepare GSM8K Data](#prepare-gsm8k-data). The entry point accepts only --config; edit hyperparameters in YAML.

### Configuration Interfaces

| Interface | Purpose |
| --- | --- |
| OPDConfig(**settings) | Read the complete explicit configuration used by this example. |
| OPDConfig.from_modules(settings) | Read a modular mapping with some legacy field compatibility. |

There is no algorithm selector or algorithm-name wrapper; load_yaml_config resolves ${model.name}. The complete configuration is [gsm8k_opd_args.yaml](../../examples/opd/gsm8k_opd_args.yaml).

## Core Configuration Parameters

### Training and Rollout

The table lists example values, not all library defaults.

| Parameter | Example value | Purpose |
| --- | --- | --- |
| model.name, model.lora_rank | Qwen/Qwen3.5-4B, 32 | Trainable PyTRIO base model and LoRA rank. |
| training.batch_size, max_steps, seed | 2, 100, 42 | Batch size, maximum optimizer updates, and client seed. |
| optimizer.learning_rate | 4e-5 | Policy/student AdamW learning rate. |
| optimizer.beta1, beta2, eps | 0.9, 0.95, 1e-8 | Adam moments and numerical stabilizer. |
| optimizer.weight_decay, grad_clip_norm | 0.0, 1.0 | Weight decay and gradient norm limit; zero clip norm disables clipping. |
| train_dataset.path, valid_dataset.path | See YAML | Prepared JSONL paths resolved relative to the repository root. |
| train_dataset.split, valid_dataset.split | train, test | Logical split names for the official GSM8K train/test files. |
| train_dataset.max_examples, valid_dataset.max_examples | null, 100 | Maximum rows after loading/shuffling; null keeps the entire file. |
| train_dataset.shuffle, valid_dataset.shuffle | false, false | Preserve source order; true shuffles once. |
| train_dataset.seed, valid_dataset.seed | 42, 42 | Shuffle seed; null falls back to training.seed. |
| train_dataset.loader_options, valid_dataset.loader_options | {}, {} | Extra JSON-loader kwargs; do not duplicate data_files or split. |

max_steps counts optimizer updates, while training.batch_size counts prompts. Each prompt produces rollout.num_generations student completions before one update. The data is traversed once, so exhaustion can end training before max_steps; there is no implicit cycling or repeated epoch.

### Prepare GSM8K Data

The example reads local JSONL and does not download automatically. This Python code prepares the official train/test splits and reuses existing files.

This is the same prompt format and the same data files as the GRPO/PPO examples. The teacher scores the exact student-sampled tokens; answer is retained for consistency but is not consumed by OPD training or validation.

```python
import json
from pathlib import Path
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)
for split in ("train", "test"):
    path = Path(f"data/gsm8k_{split}.jsonl")
    if path.exists():
        continue
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            item = {
                "messages": [{"role": "user", "content": row["question"]
                    + r" Provide a numerical answer without units, written inside \boxed{}."}],
                "answer": row["answer"].rsplit("####", 1)[1].strip().replace(",", ""),
                "question": row["question"],
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
```

Each JSONL line stores one record. The minimal format is:

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "answer": "2"
}
```

### Teacher and Distillation

| Parameter | Example value | Purpose |
| --- | --- | --- |
| teacher.name, model_path | Qwen/Qwen3.8-27B, null | Fixed teacher and optional LoRA URI; teacher/student vocabularies must match exactly. |
| rollout.num_generations | 2 | Responses per question; two questions produce four completions per update. |
| rollout.max_completion_length, max_length | 512, 2048 | Completion and total token budgets; generation is bounded by remaining space. |
| rollout.prompt_key, add_generation_prompt | messages, true | Prompt field and assistant generation prefix. |
| rollout.chat_template_kwargs, stop | {enable_thinking: false}, null | Template settings and optional stop strings; null uses EOS behavior. |
| execution.max_concurrent_sampling, max_concurrent_scoring | 16, 32 | Independent sampling/scoring request concurrency limits. |
| objective.kl_coef | 1.0 | Positive scale on the teacher-minus-behavior logprob signal. |
| objective.discount_factor | 0.0 | Zero keeps token-local signal; up to 1 includes future token signals. |
| objective.importance_ratio_cap | 5.0 | Detached ratio cap for CISPO; null selects unbounded importance sampling. |

#### From Student Rollouts to Token Targets

Let $b$ be the behavior policy that sampled token $a_t$, $q$ the fixed teacher, and $\lambda$ objective.kl_coef. OPD constructs the detached token signal

$$
\delta_t=\lambda\left[\log q(a_t\mid s_t)-\log b(a_t\mid s_t)\right],
\qquad
G_t=\delta_t+\gamma G_{t+1},
$$

where $\gamma$ is objective.discount_factor. With the example value 0, $G_t=\delta_t$ and credit remains token-local. The current student $\pi_\theta$ is updated through the importance ratio $\rho_t=\pi_\theta(a_t\mid s_t)/b(a_t\mid s_t)$. CISPO uses $\operatorname{stopgrad}(\min(\rho_t,c))$ as the coefficient on the policy-gradient term, where $c$ is objective.importance_ratio_cap; null uses the uncapped ratio. This sampled score-function objective moves the student toward the teacher without transferring the teacher's full vocabulary distribution.

Sampling is fixed to temperature=1, top_p=1, and top_k=-1; these are not OPD YAML fields. Overlong prompts raise before sampling. Concurrency overlaps sampling/scoring within a batch, while optimizer steps use fresh rollouts in order.

## Core Concepts and Interfaces

### Trainer Lifecycle

| Interface | Contract |
| --- | --- |
| OPDTrainer(args=config, train_dataset=rows, eval_dataset=valid_rows) | Create clients; the example loads data, while the config does not open files. |
| state = await trainer.train_async() | Train, evaluate periodically, save final state, and finish reporting. |
| await trainer.step_async(rows) | Update one batch; custom loops must manage final saving and reporting lifecycle. |
| await trainer.evaluate_async() | Evaluate held-out records without optimizer updates. |

The example calls the async entry point with asyncio.run; calls on one trainer must not overlap.
Synchronous train(), step(), and evaluate() are also available; use the methods above in async applications.

## Triotune Implementation Notes

- **Evaluation:** evaluation.steps: 20, before_train: true, and after_train: true; batch_size: 2 and max_batches: 4 bound each pass. Set valid_dataset: null to disable validation. Fresh student sampling and teacher scoring report eval/reverse_kl_sampled and other distillation metrics, not GSM8K answer accuracy.
- **Logging:** reporting.report_to: swanlab enables uploads; project, workspace, run_name, and tags identify the run. A null workspace uses the account default. Set report_to to null to disable uploads. log_dir is relative to the working directory and stores SwanLab artifacts and unique JSONL history. logging_step controls DPO/OPD aggregation windows; SFT publishes each step.
- **Saving:** checkpoint.save_every: 20 counts updates; 0 still saves the final state. The URI is recorded in checkpoint/path and the returned state's checkpoint_path. There is no built-in resume/load entry point.

See the [configuration source](../../triotune/api/cli_args.py) and [trainer implementation](../../triotune/trainer/opd_trainer.py) for more details.
