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
    train_dataset=rows,                     # rows contain a `prompt` field
)
trainer.train()
```

When `save_every` is greater than zero, the trainer calls PyTRIO
`TrainingClient.save_state()` after every configured number of completed steps.
The returned checkpoint URI is available as `TrainerState.checkpoint_path` and
contains both model weights and optimizer state.

SwanLab logging is disabled by default. Enable it through `GRPOConfig` when
experiment tracking is needed:

```python
args = GRPOConfig(
    num_generations=4,
    max_steps=100,
    mode="online",  # or "local" / "offline"
    project="my-grpo-project",
)
```

The trainer logs reward, advantage, training metrics, datum counts, skipped
groups, and checkpoint events once per completed step, then closes the run at
the end of synchronous or asynchronous training.

The source directory uses the repository's requested `trio-cookbook` name;
projects may expose it under their preferred import alias (for example
`trio_cookbook`) when packaging it.
