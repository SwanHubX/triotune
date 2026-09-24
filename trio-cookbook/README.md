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
    args=GRPOConfig(num_generations=4, max_steps=100),
    train_dataset=rows,                     # rows contain a `prompt` field
)
trainer.train()
```

A custom `rollout_func` must always return `groups`. It may add `metrics` and
`metadata` only when they are needed:

```python
return {
    "groups": rollout_groups,
    "metrics": {"rollout/latency": latency},
    "metadata": {"request_ids": request_ids},
}
```

`groups` must contain one `GRPOBatch` per input prompt. Numeric `metrics` are
included in the step metrics, while arbitrary `metadata` is exposed through
`GRPOResult.metadata` and the final `TrainerState.metadata`. The default
rollout therefore returns only `{"groups": rollout_groups}`.

The source directory uses the repository's requested `trio-cookbook` name;
projects may expose it under their preferred import alias (for example
`trio_cookbook`) when packaging it.
