# GRPO Example—gms8k

[中文](grpo_series_CN.md)

Triotune runs GRPO with one asynchronous `PPOTrainer`: select a configuration preset,
then reuse the same data, reward, and training interfaces. GRPO learns from relative
rewards among multiple responses to the same prompt without a critic. Preset parameters
are listed in the [algorithm configuration matrix](#algorithm-configuration-matrix).


## Example Usage

### Environment Setup

Install the environment before training. Triotune provides a suitable uv environment,
so you can install it quickly with the following commands:

```bash
git clone https://github.com/SwanHubX/triotune.git
cd triotune
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

### Quick Start

Run the following command to start training directly on PyTRIO:

```bash
uv run python examples/grpo/gsm8k_grpo.py --config examples/grpo/gsm8k_grpo_args.yaml
```

Before running the training command, complete [Prepare GSM8K data](#prepare-gsm8k-data).


### Configuration Interfaces

| Interface | Purpose |
| --- | --- |
| `PPOConfig.for_algorithm("grpo", **modules)` | Apply the algorithm preset first, then apply explicit overrides. |
| `PPOConfig.from_modules(settings)` | Read a modular configuration dictionary with an algorithm selector. |
| `PPOConfig(**settings)` | Read a complete, explicit configuration directly. |


Explicit configuration takes precedence over presets: keeping `critic.enabled: true` or
`actor.reward_norm: null` from a PPO file changes the intended GRPO settings. For example,
the following configuration uses GRPO in the current project:

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
    mean_leave1out: false
    std_unbiased: true
    eps: 0.00001

critic:
  enabled: false
```


## Core Configuration Parameters

### Training and Rollout

| Parameter | Default | Purpose |
| --- | --- | --- |
| `model.name`, `model.lora_rank` | `null`, `32` | PyTRIO base-model ID and actor LoRA rank; a model must be supplied before training. |
| `optimizer.learning_rate` | `1e-5` | Actor AdamW learning rate; `beta1`, `beta2`, `eps`, `weight_decay`, and `grad_clip_norm` are also supported. |
| `rollout.num_generations`, `max_tokens` | `4`, `512` | Number of responses generated per prompt and the response length limit. |
| `rollout.temperature`, `top_p` | `1.0`, `1.0` | Both must be 1; training logprobs for only the target tokens cannot reconstruct modified sampling probabilities. |
| `actor.update_epochs`, `ppo_n_minibatches` | `1`, `1` | Reuse the same frozen rollout targets and divide actor updates by complete prompt groups. |
| `actor.loss_backend` | `custom` | Use the complete local Torch loss callback through PyTRIO; `native` supports only a restricted ordinary token-level PPO subset. |

`max_steps` counts rollout updates, not optimizer steps. Training traverses the supplied
data once and ends early when the data is exhausted; `update_epochs` repeats frozen rollout
targets, not dataset epochs. The number of actor minibatches cannot exceed the actual number
of prompt groups.

The example's data-loading code is responsible for data paths, sample limits, and shuffling.
The current GSM8K entry point applies `valid_dataset.max_examples`, but does not apply
`train_dataset.max_examples` or shuffle the training data.

### Prepare GSM8K Data

The example reads local JSONL files and does not download data automatically. Each dataset
row must contain at least `messages` and `answer`. The following Python code prepares the
official training and test splits, asks the model to put its final numerical answer inside
`\boxed{}`, and reuses files that already exist.

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

This preparation code uses a single user message; existing experiment data may contain
few-shot examples. Use the same data files when comparing algorithms. The example reward
function checks the final boxed answer and returns 1 when it is correct, otherwise 0.

Each JSONL file stores one record per line. The minimal format is:

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "answer": "2"
}
```


### Reward and Advantage Normalization (actor.reward_norm, actor.adv_norm)

Each can be set to null to disable it, or given a NormConfig to configure normalization;
specify the corresponding fields in YAML.

- `reward_norm: null`: do not subtract a mean or scale the result reward for each response. The reward still goes through `reward_bias`, `reward_scaling`, and `reward_clip` before the subsequent advantage calculation.
- `adv_norm: null`: keep the advantages for active response tokens produced by GAE without another mean subtraction or scaling. Disabling normalization does not disable GAE, the critic, or a configured KL penalty.

When a NormConfig is supplied, reward_norm processes each response's scalar result reward
before advantage calculation, and adv_norm processes active response-token advantages after
GAE. Both subtract the mean over the range specified by mean_level, then calculate the scale
over the range specified by std_level and divide by that scale plus eps. For example, the
following uses same-prompt response-group statistics for rewards and the full rollout-batch
statistics for advantages:

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

mean_level and std_level can independently be group, batch, or null.
group means the response group for one prompt, while batch means all responses in the rollout;
advantage statistics use only active response tokens.

When mean_level is null, no mean is subtracted. When std_level is null, no scale is applied.
`mean_leave1out: true` excludes the current value when calculating the mean; for a scalar
result reward per response, this means excluding the current response.

`std_unbiased: true` uses the corrected sample standard deviation. eps prevents a zero
denominator during scaling.

All presets retain batch/batch advantage normalization unless explicitly overridden. Directly
constructing `PPOConfig()` disables reward normalization by default, while the GRPO preset
enables it.

### GRPO's Within-Group Relative Advantage

Sample $G$ responses for the same prompt $q$, with $G$ at least 2. Let $R_i$ be the result
reward for response $i$ after reward bias, scaling, and clipping. With the default
`mean_leave1out: false` and `std_unbiased: true`, the group mean, sample standard deviation,
and relative reward are:

$$
\mu_q = \frac{1}{G}\sum_{j=1}^{G}R_j,
\qquad
s_q = \sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(R_j-\mu_q)^2},
\qquad
a_i = \frac{R_i-\mu_q}{s_q+\eta}.
$$

Here $\eta$ corresponds to `reward_norm.eps`. Responses above the group mean receive a
positive relative reward, and those below it receive a negative relative reward. When all
group rewards are equal, the relative reward is 0. With `std_unbiased: false`, the variance
denominator changes from $G-1$ to $G$.

In the current GSM8K GRPO example, no critic is used, discount and gae_lambda are both 1,
kl_ctl is 0, and a result reward is given only at the end of the response. Therefore, before
additional advantage normalization, every active response token has advantage $a_i$. The
example also enables batch/batch advantage normalization, so the final advantage used for
policy updates is:

$$
\hat{A}_{i,t} = \frac{a_i-\mu_A}{s_A+\eta_A}.
$$

Here $\mu_A$ and $s_A$ are the mean and sample standard deviation of advantages over active
response tokens in the full rollout batch, and $\eta_A$ corresponds to `adv_norm.eps`.
Every active token participates in the statistics, so response length affects the weighting.
Set adv_norm to null to skip this step; the final advantage is then $a_i$.

### Clipping Strategy (actor.eps_clip*)

GRPO reuses PPO's clipped policy objective. The current example does not enable recomputation,
decoupling, or rejection sampling, so the token importance-sampling ratio is:

$$
\rho_{i,t}(\theta)
= \frac{\pi_\theta(o_{i,t}\mid q_i,o_{i,<t})}
       {\pi_{\mathrm{old}}(o_{i,t}\mid q_i,o_{i,<t})}
= \exp\!\left(\log\pi_\theta(o_{i,t}\mid q_i,o_{i,<t})
              -\log\pi_{\mathrm{old}}(o_{i,t}\mid q_i,o_{i,<t})\right).
$$

$\pi_\theta$ is the actor being updated, and $\pi_{\mathrm{old}}$ is the old policy that
sampled the response; $q_i$ is the prompt and $o_{i,t}$ is token $t$ in the response.
`eps_clip: 0.2` gives the clipping interval [0.8, 1.2]; `eps_clip_higher: 0.28` changes
the upper bound to 1.28. Let $\epsilon_-$ correspond to eps_clip. When eps_clip_higher is
null, $\epsilon_+$ also equals eps_clip; otherwise it equals eps_clip_higher.

With `loss_type: ppo`, `c_clip: null`, and the default token_mean aggregation, the actor loss
to minimize is:

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta)
= -\frac{1}{N}\sum_i\sum_{t\in\mathcal{M}_i}
\min\!\left(
  \rho_{i,t}(\theta)\hat{A}_{i,t},
  \operatorname{clip}\!\left(\rho_{i,t}(\theta),1-\epsilon_-,1+\epsilon_+\right)
  \hat{A}_{i,t}
\right),
\qquad N=\sum_i|\mathcal{M}_i|.
$$

Here $i$ ranges over all responses in the rollout batch, $\mathcal{M}_i$ is the set of active
token positions in response $i$, and $N$ is the total number of active tokens. A positive
advantage encourages increasing the probability of the corresponding token, while a negative
advantage encourages decreasing it. Clipping limits the objective's gain from continuing an
improving update; it does not force the actual ratio to remain within the clipping interval.
The default averages over all active tokens. Set `loss_aggregation: seq_mean` to give every
response equal weight.

`c_clip` can enable dual clipping for negative advantages. `loss_type` supports `ppo`, `sapo`,
and `cispo`; the equation above is for ppo, while sapo uses soft gates and cispo uses a
stop-gradient clipping weight, so their objectives differ. The current example has kl_ctl at
0. If KL penalty is enabled, this implementation first adds it to token rewards and then
calculates advantages through GAE.

### Importance-Sampling Level (actor.importance_sampling_level)

By default, token uses the per-token ratio above. sequence uses the geometric mean of ratios
over the active response tokens that remain:

$$
\rho_i^{\mathrm{seq}}(\theta)
= \exp\!\left(\frac{1}{|\mathcal{K}_i|}
  \sum_{t\in\mathcal{K}_i}\log\rho_{i,t}(\theta)\right)
= \left(\prod_{t\in\mathcal{K}_i}\rho_{i,t}(\theta)\right)^{1/|\mathcal{K}_i|}.
$$

$\mathcal{K}_i$ is the set of tokens retained after the mask and any optional rejection
thresholds; the equation assumes it is nonempty. sequence mode also averages the advantages
of retained tokens before calculating that response's objective. GSPO selects this setting.

`use_decoupled_loss` distinguishes the behavior policy that generated the response from the
proximal policy used as the clipping-ratio denominator. When enabled, the ratio above uses
proximal scores in its denominator. `recompute_logprob` recalculates proximal scores before
freezing targets. Optional `rejection_lower`/`rejection_upper` mask tokens based on the
proximal-to-behavior probability ratio and require `use_decoupled_loss: true`.


## Core Concepts and Interfaces

### Trainer Lifecycle

| Interface | Contract |
| --- | --- |
| `PPOTrainer(args=config, train_dataset=rows, reward_fn=fn, eval_dataset=valid_rows)` | Lazily initialized; may also receive `model` directly or injected clients, tokenizer, and critic. |
| `await trainer.initialize()` | Creates the actor, optional critic, tokenizer, and reporter; initializes only once. |
| `state = await trainer.train()` | Completes sampling, scoring, updating, evaluation, final saving, and reporting cleanup. |
| `metrics = await trainer.step(trajectories)` | Updates one batch of custom frozen trajectories; the same trainer cannot be called concurrently. |
| `await trainer.evaluate()` | Samples and scores validation prompts without reference/critic calculation or optimizer updates. |
| `await trainer.save_checkpoint()` | Saves actor and framework-managed critic state and returns the actor URI. |
| `await trainer.close()` | Finishes reporting for a custom lifecycle. |

Run the asynchronous entry point with `asyncio.run(...)`. train() handles reporting cleanup;
when calling step() manually, manage saving and close() yourself.
PPOState contains global_step, optimizer_steps, critic_optimizer_steps,
metrics, checkpoint_path, and critic_checkpoint_path.

### Reward Interface

`reward_fn(\*, row, completion, completion_ids)` supports synchronous and asynchronous
functions. It returns a finite scalar, or a mapping with finite components such as
`{"reward": total, "correctness": score, ...}`. Components are recorded as
reward/\<name\>/mean and eval/reward_\<name\>_mean. The GSM8K example provides the reward
function; the trainer does not interpret answer itself. prompt accepts text, token ID lists,
or chat message lists.

### GAE Timestep Units and Custom Trajectories

`actor.discount` and `gae_lambda` default to 1.0; gae_lambda is a fixed scalar.
gae_timestep_unit supports token and turn. Turn mode requires custom
Trajectory.turn_ids; the built-in GSM8K rollout does not generate this field.

Import Trajectory from triotune.trainer.ppo and pass a list of trajectories to step().
Required fields are prompt_tokens, completion_tokens, behavior logprobs, scalar reward,
and group_id. Optional loss_mask, reference_logprobs, token_rewards, and turn_ids align with
response tokens; values has response length + 1, with the extra element used for bootstrap.
In turn mode, active-token integer turn IDs must be nondecreasing, and masked tokens use -1.
Dynamic lambda function interfaces are not currently supported.

## Triotune Implementation Notes

- **Execution:** Prompts are sampled concurrently. Each update first freezes targets, then independent clients optimize the actor and critic concurrently. Rollout and policy updates remain sequential across training steps.
- **KL:** actor.kl_ctl defaults to 0. When set to a positive value, a frozen reference is scored (reference.name, or the actor base model when unspecified), with kl_estimator: k1|k2|k3 supported. Token mode adds KL to rewards before calculating GAE; turn mode keeps it as a token-local term in actor advantages.
- **Evaluation:** evaluation.steps, before_train, after_train, batch_size, and max_batches control reward evaluation over a replayable validation set. The current rollout sampling settings, including responses per prompt, are used.
- **Logging:** reporting.report_to: swanlab enables uploads. Actor and critic share one experiment; metrics are under reward/&#42;, train/&#42;, policy/&#42;, critic/&#42;, eval/&#42;, time/&#42;, and usage/&#42;. Some metrics require the corresponding component or option. reporting.log_dir also writes a separate JSONL history; PPO publishes metrics every step regardless of reporting.logging_step.
- **Token Usage:** usage/actor_train_tokens and usage/critic_train_tokens count the input tokens processed by each component per step; usage/train_tokens is their sum, and the _total suffix means cumulative across steps. Counts include repeated epochs and forward passes used to prepare logprobs/values. Custom loss counts forward and forward-backward separately; native actor loss counts once.
- **Saving:** checkpoint.save_every counts rollout updates; 0 saves only at the end. URIs are recorded as checkpoint/actor_path and checkpoint/critic_path. The trainer currently has no built-in resume/load entry point.
- **Scope:** PyTRIO manages model deployment and distributed computation. AReaL cluster backends, workflow API, and Hydra configuration are not Triotune interfaces.

See the [configuration source](../../triotune/api/cli_args.py) and [algorithm components](../../triotune/trainer/ppo/) for more details.
