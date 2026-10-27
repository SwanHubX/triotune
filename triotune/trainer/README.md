# Trainer module

PPO algorithm components are located in [ppo/](./ppo/): rl_trainer.py in the parent directory manages the shared asynchronous training loop, with configuration selecting
PPO, GRPO, RLOO, Dr.GRPO, LitePPO, GSPO, SAPO, CISPO, or DAPO.
Algorithm configuration and implementations reside in this directory; standard PPO creates independent PyTRIO actor/critic training clients by default. See the interface capability description for details.

Multi-turn tool-using agents use AgentTrainingLoop from triotune.agentrl to access the same training core through the public
PPOTrainer.create_sampling_snapshot, step, and publish_metrics methods.
Execution, action parsing, tools, scoring, and trajectory export interfaces are all exposed by triotune.agentrl.
For training steps and interface documentation, see the [AgentRL documentation](../../docs/agenrl/README.md).

## Directory structure and responsibilities


```text
triotune/
├── api/
│   ├── cli_args.py      # Public configuration and algorithm parameter contracts
│   └── client_api.py    # Training/Sampling client call contracts
├── dataset/
│   ├── prompt.py        # Rendering of text, token, and chat inputs
│   ├── prepare.py       # Validation, tokenization, masking, and length handling for fixed row formats
│   └── examples.py      # Standardized training examples shared across algorithms
├── utils/
│   ├── checkpoint.py    # Checkpoint naming, saving, and response validation
│   ├── dataloader.py    # Batch iteration over streaming datasets
│   ├── datums.py        # Construction of generic PyTRIO Datum wrappers
│   ├── reporting.py     # Process-isolated secondary-run reporter for legacy calls
│   ├── scoring.py       # Logprob scoring and alignment for fixed completions
│   └── metrics.py       # Metric normalization, JSONL history, window aggregation, and SwanLab submission
└── trainer/
    ├── __init__.py      # Public exports and compatibility with historical module paths
    ├── dpo_trainer.py   # DPO training orchestration
    ├── sft_trainer.py   # SFT training orchestration
    ├── opd_trainer.py   # OPD training orchestration
    ├── rl_trainer.py    # PPO-family training orchestration
    ├── dpo/
    │   ├── state.py          # Trainer state and step results
    │   ├── datum.py          # Causal alignment and masking for preference completions
    │   └── loss.py           # Pairwise DPO objective, implicit rewards, and evaluation metrics
    ├── sft/
    │   └── state.py
    ├── opd/
    │   ├── contracts.py      # OPD stage data and execution protocols
    │   ├── execution.py      # Immutable execution plans and concurrent transactions
    │   ├── rollout.py        # Student on-policy sampling
    │   ├── teacher_phase.py  # Fixed-teacher routing and token score aggregation
    │   ├── loss.py           # Sampled reverse-KL signals and remote surrogate
    │   ├── datum.py          # Next-token alignment and prompt masking
    │   └── targets.py        # Conversion of teacher scores into training targets
    └── ppo/                  # PPO-family algorithm components
        └── trajectory.py     # PPO trajectories, prepared batches, and trainer state
```

| Module | Responsibility |
| --- | --- |
| [Public entry point](./__init__.py) | Exports trainers, configuration, and types, and centrally maps historical flat module paths. |
| [Configuration contracts](../api/cli_args.py) | Defines composable configuration by responsibility: models, training/validation data sources, training loops, optimizers, saving, evaluation, logging, rollout, actor/critic, and reference/teacher. Top-level DPO, SFT, OPD, and PPO objects can represent complete experiments while retaining dependency injection compatibility. |
| [Client contracts](../api/client_api.py) | Unifies synchronous/asynchronous TrainingClient and SamplingClient calls, future resolution, and SDK parameter construction; each algorithm determines when to refresh clients. |
| [Input rendering](../dataset/prompt.py) | Validates text, token lists, and chat messages, and provides a common tokenizer invocation path. |
| [Data preparation](../dataset/prepare.py) | Converts rows with fixed formats into prompts, supervised sequences, or preference pairs, and validates tokens, masks, EOS, and lengths. |
| [Training examples](../dataset/examples.py) | Provides PromptExample, SupervisedExample, and PreferenceExample, separating data preparation from remote PyTRIO clients. |
| [Shared metrics](../utils/metrics.py) | Accepts arbitrary numerical metrics from algorithms and handles validation, naming, log_dir JSONL history, window aggregation, SwanLab submission, checkpoint URI recording, and run finalization. |
| [Shared saving](../utils/checkpoint.py) | Handles synchronous/asynchronous saving and path validation; each algorithm determines save cadence and state updates. |
| [Shared batching](../utils/dataloader.py) | Lazily batches examples from datasets and limits validation batches; does not handle tokenization or Datum construction. |
| [Shared Datum construction](../utils/datums.py) | Materializes tokens and loss inputs already aligned by an algorithm into PyTRIO Datum objects; does not determine GRPO/OPD training fields. |
| [Shared scoring](../utils/scoring.py) | Computes token logprobs for fixed completions through a sampling client and validates positional alignment between the full sequence and completion; reusable by DPO, OPD, and PPO. |
| [DPO orchestration](./dpo_trainer.py), [Datum](./dpo/datum.py), [loss](./dpo/loss.py) | The trainer invokes shared preference data preparation and manages fixed-reference scoring and remote updates; algorithm modules implement causal alignment of preference completions, the pairwise objective, and training/evaluation metrics. |
| [SFT orchestration](./sft_trainer.py) | Invokes shared SFT data preparation, constructs cross-entropy Datum objects, submits the pipeline, and publishes results in batch order. |
| [OPD orchestration](./opd_trainer.py), [execution plans](./opd/execution.py) | The trainer manages datasets, clients, metric reduction, updates, checkpoints, and the logging lifecycle; immutable execution plans compose replaceable rollout, teacher, target, and loss stages. |
| [PPO-family orchestration](./rl_trainer.py), [algorithm components](./ppo/) | Coordinates rollout, reference/value scoring, frozen targets, concurrent actor/critic updates, checkpoints, and the logging lifecycle; includes the GRPO preset. |
| [OPD rollout](./opd/rollout.py), [teacher phase](./opd/teacher_phase.py) | The student samples under the current behavior policy; fixed teachers within the routed group score the same token sequence, and their scores are aggregated into a weighted geometric ensemble. Calls can currently overlap across prompts, but the interface does not depend on colocated training and inference. |
| [OPD loss](./opd/loss.py), [targets](./opd/targets.py), [Datum](./opd/datum.py) | Computes the log p_teacher - log p_behavior token signal, supports optional discounted credit, preserves behavior logprobs, and constructs the next-token alignment fields required by PyTRIO policy loss; CISPO caps the IS ratio at 5.0 on one side by default. |
| [OPD contracts](./opd/contracts.py) | Isolates SDK responses and validates data between stages; the OPD trainer reduces distillation metrics using completion tokens as the denominator, and validation uses streaming totals without retaining the full list of remote Datum objects. |

The full SFT, DPO, OPD, and PPO-family training entry points display training-step progress; all evaluation
entry points display separate batch progress. Data sources with a known length show the exact total number of batches; streaming iterables show completed batches and throughput.

Metrics are aggregated using their natural sample counts: language-model loss is weighted by valid tokens, DPO by pairs, PPO policy/critic
by valid actions, and OPD distillation metrics by completion tokens. Standard deviations are combined using first and second moments,
rather than by directly averaging per-batch standard deviations. PPO/OPD KL and entropy fields containing sampled are estimates over sampled
actions and do not represent full-vocabulary distribution statistics.
SFT always reports perplexity from cross-entropy; train/token_accuracy is passed through only when the PyTRIO backend returns
token_accuracy/accuracy, because target-token logprobs alone cannot determine the argmax over the full vocabulary.

```text
dataset/ single example → utils/dataloader.py batch → algorithm alignment → utils/datums.py Datum
```

dataset/ loads raw rows from mappings, JSONL, or existing iterables and converts them into standardized examples; batching and Datum wrapper construction belong to the runtime
data pipeline. PPO-family/OPD logprobs, advantages, teacher scores, and prompt masks are still determined by the algorithm
modules.



## Trainer input data formats

```python
# PPO-family / OPD: each row supplies one generation prompt.
# The field defaults to messages and can be changed through PPOConfig.rollout.prompt_key
# or OPDConfig.rollout.prompt_key. Values support plain text, token IDs, or chat messages.
prompt_rows = [
    {"messages": "Explain why the sky is blue."},
    {"messages": [151644, 8948, 198]},
    {
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
    },
]

# DPO: each row supplies a shared prompt, a chosen response, and a rejected response.
# messages supports plain text, token IDs, or chat messages; chosen/rejected supports plain text,
# token IDs, or assistant message lists paired with the chat prompt.
preference_rows = [
    {"messages": "What is 2 + 2?", "chosen": "4", "rejected": "5"},
    {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
    },
    {"messages": [101, 102], "chosen": [103, 104], "rejected": [105, 106]},
]

# SFT: a dataset can use any one of the following row formats.
sft_text_rows = [{"text": "Question: 2 + 2? Answer: 4"}]
sft_chat_rows = [
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
]
sft_pair_rows = [{"prompt": "What is 2 + 2?", "completion": "4"}]
sft_token_rows = [
    {
        "input_ids": [101, 102, 103, 104],
        "labels": [-100, -100, 103, 104],
        "loss_mask": [0, 0, 1, 1],
    }
]
```

## Standard format passed from Dataset to Trainer

```python
from triotune.dataset import prepare_rows

# PPO-family / OPD
prompt_examples = prepare_rows(prompt_rows, tokenizer, format="prompt")
# PromptExample(text: str, tokens: list[int])

# SFT
supervised_examples = prepare_rows(sft_pair_rows, tokenizer, format="sft")
# SupervisedExample(input_ids: list[int], labels: list[int])
# -100 in labels indicates that the token does not contribute to loss.
# When sent to PyTRIO:
# model_input = input_ids[:-1]
# target_tokens = labels[1:]

# DPO
preference_examples = prepare_rows(
    preference_rows,
    tokenizer,
    format="preference",
)
# PreferenceExample(prompt: list[int], chosen: list[int], rejected: list[int])
# chosen/rejected are each appended to the prompt, with prompt labels set to -100.
```

## Dataset JSONL transfer example

```jsonl
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
{"messages":[{"role":"user","content":"What is 3 + 3?"},{"role":"assistant","content":"6"}]}
```


## SFT training

```python
import asyncio

from triotune.trainer import EvaluationConfig, SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3.5-4B",
    args=SFTConfig(
        evaluation=EvaluationConfig(steps=50, after_train=False),
    ),
    train_dataset=train_rows,
    eval_dataset=eval_rows,
)
state = asyncio.run(trainer.train())
```

## GRPO training

```python
import asyncio

from triotune.trainer import PPOConfig, PPORolloutConfig, PPOTrainer, TrainingConfig

trainer = PPOTrainer(
    model="Qwen/Qwen3.5-4B",
    reward_fn=my_reward,
    args=PPOConfig.for_algorithm(
        "grpo",
        training=TrainingConfig(max_steps=100),
        rollout=PPORolloutConfig(num_generations=4),
    ),
    train_dataset=prompt_rows,
)
state = asyncio.run(trainer.train())
```

## DPO training

```bash
uv sync --extra dpo
```

```python
from triotune.trainer import (
    DPOConfig,
    DPOObjectiveConfig,
    DPOTrainer,
    TrainingConfig,
)

trainer = DPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=DPOConfig(
        training=TrainingConfig(batch_size=1, max_steps=1),
        objective=DPOObjectiveConfig(beta=0.1),
    ),
    train_dataset=preference_rows,
)
state = trainer.train()
```

## OPD training

```python
from triotune.trainer import OPDConfig, OPDTrainer, TrainingConfig

trainer = OPDTrainer(
    model="student-base-model-id",
    teacher_model="teacher-base-model-id",
    # teacher_model_path="trio://...",  # Optional teacher LoRA checkpoint.
    args=OPDConfig(training=TrainingConfig(batch_size=4, max_steps=100)),
    train_dataset=prompt_rows,
)
state = trainer.train()
```

The default OPD execution chain is:

```text
student rollout
  → routed fixed-teacher token scoring
  → weighted geometric teacher ensemble
  → sampled reverse-KL token advantage
  → prompt-masked PyTRIO policy Datum
  → CISPO capped-importance-ratio forward/backward
  → optimizer step
```

A single teacher continues to use the compatible interface above. Multiple teachers are configured through explicit specs and groups;
data rows carry only group names and do not interact with model clients or algorithm fields:

```python
from triotune.trainer import OPDConfig, OPDTrainer
from triotune.trainer.opd import OPDTeacherGroup, OPDTeacherSpec

teachers = [
    OPDTeacherSpec("math", "org/math-teacher", "checkpoint://math-lora"),
    OPDTeacherSpec("code", "org/code-teacher"),
]
groups = [
    OPDTeacherGroup("reasoning", {"math": 2.0, "code": 1.0}),
    OPDTeacherGroup("code-only", {"code": 1.0}),
]
rows = [
    {"messages": "Solve this proof.", "teacher_group": "reasoning"},
    {"messages": "Implement a parser.", "teacher_group": "code-only"},
]
trainer = OPDTrainer(
    model="student-base-model-id",
    teachers=teachers,
    teacher_groups=groups,
    args=OPDConfig(training=TrainingConfig(batch_size=2, max_steps=100)),
    train_dataset=rows,
)
```
