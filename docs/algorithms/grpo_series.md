# PPO, GRPO, and Related Algorithms

[中文](grpo_series_CN.md)

Triotune uses one asynchronous `PPOTrainer` for PPO and GRPO-family algorithms:
select a configuration preset, then reuse the same data, reward, and training interfaces.
GRPO learns from rewards relative to other responses to the same prompt, without a critic.
The [configuration matrix](#algorithm-configuration-matrix) lists the supported presets.

This guide follows the section structure of
[AReaL's GRPO-series guide](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/grpo_series.md),
but all commands and parameter names below belong to Triotune.

## Example Usage

Run commands from the repository root with Python 3.12 and access to a configured
PyTRIO training service. Remote model execution is provided by PyTRIO.

```bash
uv sync --locked --extra ppo
```

Before running training commands, complete [Prepare GSM8K data](#prepare-gsm8k-data).

### Switching Algorithms from the Command Line

The example accepts **only `--config`**, rather than Hydra-style overrides.
The command below selects a preset through `PPOConfig.for_algorithm`, writes a
complete temporary YAML file, and runs a two-step GRPO smoke experiment.
Replace `grpo` in the first command with any preset from the matrix to switch algorithms.

```bash
uv run python - grpo /tmp/triotune-grpo.yaml <<'PY'
import sys
from dataclasses import asdict
from pathlib import Path
import yaml
from triotune.trainer import PPOConfig
from triotune.utils.config import load_yaml_config

settings = load_yaml_config("examples/ppo/gsm8k_ppo_args.yaml")
# Remove explicit PPO choices so that the selected preset can supply them.
settings.pop("actor")
settings.pop("critic")
settings["training"].update(max_steps=2, batch_size=2)
settings["rollout"].update(num_generations=4, max_tokens=512)
settings["checkpoint"]["save_every"] = 0
settings["valid_dataset"] = None
settings["reporting"].update(
    report_to=None, workspace=None, run_name=f"gsm8k-{sys.argv[1]}-smoke",
    tags=[sys.argv[1], "gsm8k"], log_dir="output/grpo-series-smoke",
)
config = PPOConfig.for_algorithm(sys.argv[1], **settings)
Path(sys.argv[2]).write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
PY
uv run python examples/ppo/gsm8k_ppo.py --config /tmp/triotune-grpo.yaml
```

The model remains `Qwen/Qwen3.5-9B`, LoRA rank 32, learning rate `4e-5`, and seed 42.
Two prompts × four responses produce eight trajectories per update.
To run the checked-in 100-step PPO configuration with validation and SwanLab:

```bash
uv run swanlab login
uv run python examples/ppo/gsm8k_ppo.py --config examples/ppo/gsm8k_ppo_args.yaml
```

Set `reporting.project`, `workspace`, and `run_name` for your account before running.
For a longer GRPO experiment, adjust `training.max_steps`, `valid_dataset`,
`evaluation`, and `reporting` in the generated YAML, then pass it to the same entry point.

### Configuration interfaces

| Interface | When to use it |
| --- | --- |
| `PPOConfig.for_algorithm("grpo", **modules)` | Apply a preset, followed by explicit overrides. Pass actor/critic overrides as dictionaries to merge with preset defaults. |
| `PPOConfig.from_modules(settings)` | Read a modular mapping with an optional top-level `algorithm` selector (default: `grpo`). |
| `PPOConfig(**settings)` | Load a complete explicit configuration, as the GSM8K example does; no `algorithm` field is accepted. |

For example, `PPOConfig.from_modules({"algorithm": "gspo", "rollout":
{"num_generations": 4}})` selects sequence importance sampling.
Explicit fields override presets: retaining `critic.enabled: true` or
`actor.reward_norm: null` from a PPO file changes the intended GRPO configuration.
Typed `PPOActorConfig`/`PPOCriticConfig` instances replace their entire module;
use dictionaries when you want partial preset overrides.

## Core Configuration Parameters

Names in the tables are nested YAML paths and Python configuration attributes.
Defaults below are library defaults unless an algorithm preset overrides them.

### Training and rollout

| Parameter | Default | Purpose |
| --- | --- | --- |
| `model.name`, `model.lora_rank` | `null`, `32` | PyTRIO base-model ID and actor LoRA rank; supply a model before training. |
| `training.batch_size`, `max_steps`, `seed` | `1`, `null`, `null` | Prompts per update, update limit, and client/sampling seed. |
| `optimizer.learning_rate` | `1e-5` | Actor AdamW learning rate; `beta1`, `beta2`, `eps`, `weight_decay`, and `grad_clip_norm` are also available. |
| `rollout.num_generations`, `max_tokens` | `4`, `512` | Responses per prompt and maximum completion length. |
| `rollout.temperature`, `top_p` | `1.0`, `1.0` | Both must stay at 1: target-token training logprobs cannot reconstruct modified sampling probabilities. |
| `rollout.prompt_key`, `chat_template_kwargs` | `messages`, `{}` | Prompt field and tokenizer template options, e.g. `enable_thinking: false`. |
| `actor.update_epochs`, `ppo_n_minibatches` | `1`, `1` | Reuse a frozen rollout batch and partition actor updates while preserving prompt groups. |
| `actor.loss_backend` | `custom` | Full local Torch loss callback through PyTRIO; `native` supports a restricted ordinary token-level PPO subset. |

`max_steps` counts rollout updates, not optimizer steps. Training consumes the supplied
iterable once and can end earlier; `update_epochs` repeats frozen rollout targets,
not dataset epochs. Actor minibatches cannot outnumber the available prompt groups.
Dataset path, limits, and shuffling are recipe-side loading responsibilities;
the current GSM8K entry point applies `valid_dataset.max_examples`, but does not
apply `train_dataset.max_examples` or shuffle training rows.

### Prepare GSM8K data

The [GSM8K example](../../examples/ppo/gsm8k_ppo.py) reads local JSONL files;
it does not download data automatically. Each row needs `messages` and `answer`.
The following command prepares the official training and test splits, requesting a
final numerical answer inside `\boxed{}`. Existing files are reused.

```bash
uv run python - <<'PY'
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
PY
```

This preparation uses a single user message; existing experiment data may include
few-shot messages. Keep identical data files when comparing algorithms.
The reward function in the example gives 1 for a matching final boxed answer and 0 otherwise.

Each JSONL line contains one record. The minimal format is shown expanded for readability:

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "answer": "2"
}
```

### Policy-gradient Loss Aggregation (`actor.loss_aggregation`)

| Value | Reduction |
| --- | --- |
| `token_mean` (default) | Average over all active generated tokens. |
| `seq_mean` | Average each response's token mean equally. |
| `prompt_mean` | Average each prompt group's token mean equally. |
| `constant` | Divide summed terms by trajectory count × `loss_aggregation_divisor`; set a positive divisor. |

### Reward and Advantage Normalization (`actor.reward_norm`, `actor.adv_norm`)

Both accept `null` to disable normalization, or a `NormConfig` mapping:

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: group
    mean_leave1out: false
    std_unbiased: true
    eps: 0.00001
  adv_norm:
    mean_level: batch
    std_level: batch
```

`mean_level` and `std_level` independently accept `group`, `batch`, or `null`.
Group means compare responses to the same prompt; `mean_leave1out: true` excludes
the current response. All presets retain batch/batch advantage normalization unless
overridden. Plain `PPOConfig()` disables reward normalization; the GRPO preset enables it.

### Clipping Strategy (`actor.eps_clip*`)

`eps_clip: 0.2` gives the ordinary ratio range `[0.8, 1.2]`.
`eps_clip_higher: 0.28` changes its upper bound to `1.28`.
`c_clip` optionally enables dual clipping for negative advantages.
`loss_type` selects `ppo`, `sapo`, or `cispo`; their clipping behavior differs.

### Importance Sampling Level (`actor.importance_sampling_level`)

`token` is the default; `sequence` uses the geometric mean ratio across active
response tokens and is selected by GSPO. `use_decoupled_loss` separates behavior
and proximal policies; `recompute_logprob` refreshes proximal scores before targets
are frozen. Optional `rejection_lower`/`rejection_upper` mask tokens by the
proximal-to-behavior ratio and require `use_decoupled_loss: true`.

## Algorithm Configuration Matrix

These are **implemented presets**, not claims of reproducing every paper setting.
Unless listed otherwise: token importance sampling, PPO loss, token-mean aggregation,
batch/batch advantage normalization, clipping 0.2, and no critic.

| Preset | One-sentence description | Reward mean / std | Distinct settings |
| --- | --- | --- | --- |
| `ppo` | Uses a separately trained critic for value-based advantages. | Disabled | `critic.enabled=true` |
| `grpo` | Compares rewards within each prompt's response group. | group / group | Base group-relative preset |
| `drgrpo` | Removes group standard-deviation scaling. | group / none | Aggregation remains `token_mean` |
| `liteppo` | Centers within groups and scales across the batch. | group / batch | Mixed normalization |
| `rloo` | Uses the other responses' mean reward as baseline. | leave-one-out group / none | `mean_leave1out=true` |
| `reinforce` | Uses outcome rewards without a group baseline. | Disabled | No critic |
| `reinforce_baseline` | Subtracts a group mean without standard-deviation scaling. | group / none | Same current preset as `drgrpo` |
| `gspo` | Applies a sequence-level importance ratio. | group / group | `importance_sampling_level=sequence` |
| `sapo` | Replaces hard clipping with soft gates. | group / group | `loss_type=sapo` |
| `cispo` | Clips a detached importance weight in the policy objective. | group / group | `loss_type=cispo`, clip low/high `1.0/4.0` |
| `dapo` | Combines asymmetric clipping, informative-group refill, and a length penalty. | group / group | Upper clip `0.28`, dynamic sampling, `overlong_tokens=128` |

## Algorithm-Specific Options

### Vanilla PPO

`critic.enabled: true` creates an independent PyTRIO LoRA training client.
Its `optimizer`, `lora_rank`, `eps_clip`, `update_epochs`, and `n_minibatches`
are separate from the actor's. Values are derived from two marker-token logprobs:
`value_scale * (logp(positive) - logp(negative)) + value_bias`.
Default markers are single-token encodings of `"1"` and `"0"`; set `critic.token_ids`
explicitly if necessary. This is a trainable logprob-based critic, not a native value head.

### GRPO, Dr.GRPO, LitePPO, and RLOO

Switch through `for_algorithm` or `from_modules`; keep at least two responses per
prompt for useful group comparisons and leave-one-out baselines. Change
`actor.adv_norm` explicitly if your comparison requires unnormalized advantages.

### GSPO

The preset sets `actor.importance_sampling_level: sequence`.
Set `actor.loss_aggregation: seq_mean` separately if equal response weighting is desired.

### SAPO and CISPO

SAPO exposes `sapo_tau_pos` (1.0) and `sapo_tau_neg` (1.05), and requires
`use_decoupled_loss: false`. CISPO requires token importance sampling and
`use_decoupled_loss: false`; its preset ratio bounds are `[0, 5]`.

### DAPO

`dynamic_sampling: true` discards groups with equal rewards and reads more source
prompts to refill the batch. `max_empty_batches` (32) limits consecutive empty refills;
dataset exhaustion can still yield a smaller final batch. `overlong_tokens` (128)
defines the tail of `rollout.max_tokens` where a penalty grows, scaled by
`overlong_penalty_factor` (1.0); the tail cannot exceed the completion budget.

### IcePop and KPop

There are no IcePop or KPop presets. The ratio rejection options above do not
implement AReaL's full rejection-sampling interface or KPop's binary-KL mask.

## Core Concepts and Interfaces

### Trainer lifecycle

| Interface | Contract |
| --- | --- |
| `PPOTrainer(args=config, train_dataset=rows, reward_fn=fn, eval_dataset=valid_rows)` | Construct lazily; alternatively supply `model` directly or inject clients/tokenizer/critic. |
| `await trainer.initialize()` | Create actor, optional critic, tokenizer, and reporter once. |
| `state = await trainer.train()` | Collect, score, update, evaluate, save final weights, and finish reporting. |
| `metrics = await trainer.step(trajectories)` | Update one custom frozen rollout batch; do not call concurrently on one trainer. |
| `await trainer.evaluate()` | Sample and score validation prompts without reference/critic scoring or optimizer updates. |
| `await trainer.save_checkpoint()` | Save actor and owned critic states; return actor URI. |
| `await trainer.close()` | Finish reporting after a custom lifecycle. |

Call an async entry point with `asyncio.run(...)`. `train()` handles reporting cleanup;
when driving `step()` manually, manage saving and `close()` yourself.
`PPOState` exposes `global_step`, `optimizer_steps`, `critic_optimizer_steps`,
`metrics`, `checkpoint_path`, and `critic_checkpoint_path`.

### Reward interface

`reward_fn(*, row, completion, completion_ids)` can be synchronous or asynchronous.
Return a finite scalar, or `{"reward": total, "correctness": score, ...}` with
finite component scores. Components appear under `reward/<name>/mean` and
`eval/reward_<name>_mean`. The GSM8K example supplies this function; the trainer
does not interpret `answer` itself. Prompt values can be text, token IDs, or chat messages.

### GAE timestep units and custom trajectories

`actor.discount` and `gae_lambda` default to 1.0; `gae_lambda` is a fixed scalar.
`gae_timestep_unit` supports `token` or `turn`. Turn mode requires custom
`Trajectory.turn_ids`; the stock GSM8K rollout does not populate them.

Import `Trajectory` from `triotune.trainer.ppo` and pass a list to `step()`.
Required fields are `prompt_tokens`, `completion_tokens`, behavior `logprobs`,
scalar `reward`, and `group_id`. Response-aligned optional fields include
`loss_mask`, `reference_logprobs`, `token_rewards`, and `turn_ids`;
`values` has response length + 1 for bootstrap. In turn mode, active tokens
need nondecreasing integer turn IDs, and masked tokens use `-1`.
No dynamic-lambda callable interface is implemented.

## Triotune Implementation Notes

- **Execution:** prompt sampling is concurrent; each update freezes targets before
  actor and critic optimization can run concurrently on independent clients.
  Rollout collection and policy updates remain sequential across training steps.
- **KL:** `actor.kl_ctl` defaults to 0; positive values score a frozen reference
  (`reference.name`, or the actor base model) with `kl_estimator: k1|k2|k3`.
  KL is included in rewards before token GAE; turn mode keeps it token-local in actor advantages.
- **Evaluation:** `evaluation.steps`, `before_train`, `after_train`, `batch_size`,
  and `max_batches` control reward evaluation on the supplied replayable validation data.
  It uses the current rollout sampling settings, including group size.
- **Logging:** `reporting.report_to: swanlab` enables uploads. Actor and critic
  share one experiment with `reward/*`, `train/*`, `policy/*`, `critic/*`,
  `eval/*`, `time/*`, and `usage/*` namespaces. Some metrics require the corresponding
  component or option. `reporting.log_dir` also writes a unique JSONL history;
  PPO publishes each step regardless of `reporting.logging_step`.
- **Token usage:** `usage/actor_train_tokens` and `usage/critic_train_tokens`
  count input tokens processed by each component per step; `usage/train_tokens`
  is their sum, and `_total` variants accumulate across steps. Counts include
  repeated epochs and logprob/value preparation forwards. Custom loss counts
  both its forward and forward-backward passes; native actor loss counts one pass.
- **Checkpoints:** `checkpoint.save_every` counts rollout updates; 0 saves only
  at the end. URIs are recorded as `checkpoint/actor_path` and
  `checkpoint/critic_path`. The trainer has no built-in resume/load entry point.
- **Scope:** PyTRIO owns model placement and distributed execution. AReaL's cluster
  backends, workflow API, and Hydra configuration are not Triotune interfaces.

See the [implementation and testing report](../ppo_pytrio-use_eval.md),
[configuration source](../../triotune/api/cli_args.py), and
[trainer modules](../../triotune/trainer/ppo/) for further details.
