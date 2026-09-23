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

Example (the concrete SDK may provide `pytrio.Datum`, `pytrio.AdamParams` and
`pytrio.SamplingParams` factories):

```python
from trio_cookbook.trainer import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    training_client=training_client,       # created by ServiceClient
    sampling_client=sampling_client,
    prompt_tokenizer=tokenizer,
    reward_funcs=my_reward,
    args=GRPOConfig(num_generations=4, max_steps=100),
    train_dataset=rows,                     # rows contain a `prompt` field
)
trainer.train()
```

The source directory uses the repository's requested `trio-cookbook` name;
projects may expose it under their preferred import alias (for example
`trio_cookbook`) when packaging it.
