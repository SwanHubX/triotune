# PyTRIO GRPO trainer

This directory is the client-side, text-generation-only GRPO design requested
for the cookbook.  It deliberately does not import or modify `trl` or the
original Tinker sources.

The modules follow the useful boundaries from `trl/trl/trainer/grpo_trainer.py`
while replacing local-model assumptions with PyTRIO public clients:

| Module | Responsibility |
| --- | --- |
| `config.py` | GRPO-specific sampling/optimizer settings |
| `rollout.py` | `SamplingClient.sample`, sequence/logprob validation |
| `rewards.py` | TRL-compatible callable rewards; no local reward model loading |
| `advantages.py` | group-relative GRPO advantage |
| `datum.py` | one-token shift and `Datum` validation/materialization |
| `utils/client.py` | algorithm-independent sync/async PyTRIO future and SDK-object adaptation |
| `grpo_trainer.py` | synchronous/asynchronous training state machine |

Example:

```python
from trio_cookbook.trainer import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model_name="Qwen/Qwen3.5-4B",
    reward_funcs=my_reward,
    args=GRPOConfig(num_generations=4, max_steps=100, save_every=10),
    train_dataset=rows,                     # rows contain a `messages` field
)
trainer.train()
```

When `save_every` is greater than zero, the trainer calls PyTRIO
`TrainingClient.save_state()` after every configured number of completed steps.
It also saves the final step if training ends between intervals. The returned
checkpoint URI is available as `TrainerState.checkpoint_path` and contains
both model weights and optimizer state.

SwanLab logging is disabled by default. Enable it through `GRPOConfig` when
experiment tracking is needed:

```python
args = GRPOConfig(
    num_generations=4,
    max_steps=100,
    mode="online",  # or "local" / "offline"
    project="my-grpo-project",
    workspace="my-organization",  # None (default) uses SwanLab's default
    tags=["grpo", "gsm8k"],       # None (default) uses SwanLab's default
)
```

The trainer logs aggregate reward, learning rate, loss, sampled-policy entropy,
sample-to-train KL, timing, datum counts, skipped groups, and remote gradient
norm when PyTRIO provides it. A terminal progress bar shows
the core metrics for synchronous and asynchronous training.
At the end of a SwanLab run, `checkpoint/path` records the final saved URI as
text when checkpointing is enabled. `sample/text` shows one inference result
for each of up to four prompts from the last training batch, using the final
model weights. No checkpoint archive download is needed for this inference.
The final step, selected metrics, and checkpoint URI are printed to the
terminal and recorded together as `trainer/final_summary` before the run ends.

Dataset rows are text-only. The `messages` value may be a string, a non-empty
integer token list, or chat messages such as:

```python
{"messages": [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is 2 + 2?"},
]}
```

Chat messages are passed through the tokenizer's `apply_chat_template` with a
generation prompt before sampling. Message content must be a string because
this first trainer accepts text inputs only.

The default dataset key is `messages`. Use `GRPOConfig(prompt_key="prompt")`
when reading an existing dataset with a `prompt` column.

The source directory keeps the repository's historical `trio-cookbook` name,
while the build configuration installs it under the valid Python package name
`trio_cookbook`.

The GSM8K recipe uses a [TOML configuration](configs/grpo_gms8k.toml) and a
shell launcher. From the repository root, run
`bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only` to inspect the
data before training; omit `--prepare-only` to start the configured run.
The standalone [GSM8K GRPO recipe](recipes/grpo_gsm8k.py) can use the same
prepared JSONL for metric comparisons; see the [script guide](scripts/README.md).
