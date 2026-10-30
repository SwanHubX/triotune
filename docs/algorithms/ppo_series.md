# Understanding and Choosing PPO-Family Algorithms in Triotune

[中文](ppo_series_CN.md)

PPO, GRPO, RLOO, GSPO, and related algorithms share one asynchronous PPOTrainer. Their differences center on three questions: how to construct advantages, at what granularity to compute policy ratios, and how to limit the size of an update.

From this perspective, choosing an algorithm means replacing a few stages within the same data pipeline rather than replacing the training workflow:

```text
prompt
  └─ rollout: generate several responses per prompt and record behavior logprobs
       └─ reward: task rewards, length penalty, optional KL
            └─ advantage: critic/group baseline → GAE → optional normalization
                 └─ actor loss: token or sequence ratio → PPO/SAPO/CISPO
                      └─ PyTRIO: remote forward pass, backward pass, and optimizer updates
```

This guide first explains that pipeline, then covers algorithm selection and algorithm-specific configuration.

## What Happens in One Update

Suppose a training batch contains $B$ prompts, with $G$ responses sampled per prompt. One rollout produces
$B\times G$ trajectories. Each trajectory contains at least the response tokens, sampling logprobs, a scalar reward, and a prompt-group ID; when the critic is enabled, values are also estimated for the response states.

The trainer processes this information in the following order:

1. Apply the length penalty, bias, scaling, and clipping to outcome rewards.
2. Construct a baseline according to the algorithm preset, such as PPO's critic value or GRPO's mean reward across responses to the same prompt.
3. Use GAE to propagate the final outcome reward to active response tokens.
4. Optionally normalize advantages again across the complete rollout batch.
5. Freeze the targets for this round and update the actor over one or more minibatches and update epochs.
6. If enabled, update the independent critic client using the same frozen returns.

Keep two counts distinct: training.max_steps counts rollout updates,

$$actor.update_epochs × actor.ppo_n_minibatches$$

while the expression above determines how many actor optimizer steps may occur per round. Training data is consumed only once, so the actual number of rollout steps can be lower than max_steps if the dataset runs out early.

## PPO Training Example

### Environment and Data

Install the environment from the repository root:

```bash
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

The PPO and GRPO examples use the same GSM8K JSONL format, with at least `messages` and `answer` in each row. Follow
[Prepare GSM8K Data](grpo_series.md#prepare-gsm8k-data) to generate
`data/gsm8k_train.jsonl` and `data/gsm8k_test.jsonl`.

### Start Training

```bash
uv run python examples/ppo/gsm8k_ppo.py \
  --config examples/ppo/gsm8k_ppo_args.yaml
```

This command uses an independent critic and assigns a 0/1 reward based on whether the answer in the final `\boxed{}` is correct.


## Three Components to Consider When Choosing an Algorithm

Algorithm names are useful experiment labels, but understanding the configuration is easier when you look at which component each algorithm changes.

### 1. Where the Advantage Baseline Comes From

- PPO learns a critic whose state value estimates the expected return from the current state; the advantage then measures how much better the actual outcome is than expected.
- GRPO, Dr.GRPO, LitePPO, RLOO, and REINFORCE with a baseline compare responses to the same prompt without training a critic.
- Original REINFORCE uses neither a critic nor a reward baseline, propagating task rewards through GAE with zero values.

Group-based methods remove the critic, but they have a cost: each prompt requires multiple generations, and a group with identical rewards provides no relative ranking signal. DAPO's dynamic sampling filters out these entire prompt groups and samples new prompts to replace them.

### 2. Where the New and Old Policies Are Compared

By default, PPO computes an importance ratio for each active token:

$$
\rho_{i,t}(\theta)
=\exp\!\left(\log\pi_\theta(o_{i,t}\mid x_i,o_{i,<t})
-\log\pi_{\mathrm{prox}}(o_{i,t}\mid x_i,o_{i,<t})\right).
$$

GSPO first averages the log-ratios over retained tokens in a response, then exponentiates the result, producing a geometric-mean sequence ratio. The whole response therefore shares the same update scale, rather than having an extreme token ratio independently determine clipping at that position.

### 3. How Policy Movement Is Constrained

- PPO applies hard clipping to the surrogate term associated with the ratio.
- DAPO uses a wider upper bound, allowing more room to improve positive-advantage samples.
- SAPO replaces the corner in hard clipping with a smooth gate.
- CISPO clips the ratio and stops its gradient, letting the current logprob carry the gradient.
- Optional rejection thresholds directly mask tokens with excessive behavior/proximal deviation.

The following table describes the defaults actually set by Triotune presets; it does not claim to reproduce the complete training system of each paper.

| Preset | Advantage baseline | Ratio granularity | Additional changes |
| --- | --- | --- | --- |
| ppo | Independent critic | token | Enable the critic and disable reward normalization |
| grpo | Group mean (group standard deviation is used for scaling) | token | group/group reward normalization |
| drgrpo | Group mean | token | No scaling by group standard deviation |
| liteppo | Group mean | token | Scale using the complete batch |
| rloo | Mean of other responses to the same prompt | token | Leave-one-out baseline |
| reinforce | No baseline | token | Propagate rewards through GAE with zero values |
| reinforce_baseline | Group mean | token | Group centering only |
| gspo | GRPO defaults | sequence | Geometric-mean ratio and response-mean advantage |
| sapo | GRPO defaults | token | Smoothly gated loss |
| cispo | GRPO defaults | token | Clipped weights with stopped gradients |
| dapo | GRPO defaults | token | Asymmetric clipping, dynamic sampling, and length penalty |

All presets retain batch/batch normalization of advantages after GAE unless actor.adv_norm is explicitly overridden.

## Keep the Two Configuration Methods Distinct

### Start from a Preset

In Python, select the algorithm directly and supply the common modules:

```python
from triotune.trainer import PPOConfig

config = PPOConfig.for_algorithm(
    "drgrpo",
    model={"name": "Qwen/Qwen3.5-9B", "lora_rank": 32},
    training={"batch_size": 4, "max_steps": 100, "seed": 42},
    rollout={"num_generations": 4, "max_tokens": 1024},
)
```

actor and critic settings are merged with the preset defaults, with explicit values taking precedence; passing an already constructed PPOActorConfig or PPOCriticConfig instead replaces the corresponding module entirely.

Modular YAML can use a top-level algorithm selector, but it must be read through from_modules:

```yaml
algorithm: dapo

rollout:
  num_generations: 4
  max_tokens: 1024

actor:
  overlong_penalty_factor: 0.5
```

```python
from triotune.trainer import PPOConfig
from triotune.utils.config import load_yaml_config

settings = load_yaml_config("experiment.yaml")
config = PPOConfig.from_modules(settings)
```

### Use a Complete Explicit Configuration

`examples/ppo/gsm8k_ppo_args.yaml` maps to PPOConfig(**settings). This approach suits fixed configuration files that record the final behavior in full; for example, critic.enabled: true explicitly selects PPO. Do not add algorithm to such a file, because direct construction does not parse the algorithm selector.

## Algorithm-Specific Options

### PPO: Critic and GAE

Triotune's built-in critic is an independent PyTRIO training client. It does not use a native value head; instead, it reads an unbounded scalar from the difference between two marker-token logprobs in the causal context of each state:

$$
V_t=s\left(\log p(\text{positive}\mid s_t)-\log p(\text{negative}\mid s_t)\right)+b.
$$

critic.value_scale and critic.value_bias correspond to $s$ and $b$. By default, the trainer attempts to encode each of the digit characters 1 and 0 as a single token; if the tokenizer does not satisfy this condition, explicitly set two distinct token IDs through critic.token_ids.

```yaml
critic:
  enabled: true
  lora_rank: null          # null inherits the actor rank
  optimizer:
    learning_rate: 0.00004
  token_ids: null
  value_scale: 1.0
  value_bias: 0.0
  eps_clip: 0.2
  update_epochs: 1
  n_minibatches: 1
```

The critic has its own optimizer, minibatch count, and update epoch count. In value regression, eps_clip limits the change in the current value relative to the frozen old value; it is independent of the actor's ratio clipping width.

### GRPO: Using Responses to the Same Prompt as Each Other's Baseline

For $G$ responses to the same prompt, first obtain the shaped scalar rewards $R_i$. The default GRPO preset computes:

$$
\widetilde R_i=\frac{R_i-\mu_g}{s_g+\varepsilon},\qquad
\mu_g=\frac{1}{G}\sum_{j=1}^{G}R_j.
$$

The default std_unbiased: true makes $s_g$ the sample standard deviation. Responses above the group mean receive a positive signal, while those below it receive a negative signal; if all rewards in a group are equal, the centered signal is zero. Group boundaries come from trajectory group_id values, so groups of unequal sizes after filtering are still processed separately. For example, a GRPO configuration is:

```yaml
rollout:
  num_generations: 4

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

critic:
  enabled: false
```

### Dr.GRPO, LitePPO, and RLOO: Changing the Baseline Scale

These three presets change neither the trainer nor the default token-level PPO loss; they only replace the normalization of outcome rewards:

```python
drgrpo = PPOConfig.for_algorithm("drgrpo", **common)
liteppo = PPOConfig.for_algorithm("liteppo", **common)
rloo = PPOConfig.for_algorithm("rloo", **common)
```

- Dr.GRPO uses mean_level: group, std_level: null during reward normalization. It subtracts only the group mean without scaling by the group standard deviation, avoiding amplification of groups with small reward differences. After GAE, advantages are still normalized across the complete batch by default.

  ```yaml
  actor:
    reward_norm:
      mean_level: group
      std_level: null
      mean_leave1out: false
    adv_norm:
      mean_level: batch
      std_level: batch
  ```

- LitePPO first centers rewards within each group, then uses std_level: batch to apply a common scale across the batch. Different prompts share a scaling factor but determine their own positive and negative directions.

  ```yaml
  actor:
    reward_norm:
      mean_level: group
      std_level: batch
      mean_leave1out: false
      std_unbiased: true
      eps: 0.00001
    adv_norm:
      mean_level: batch
      std_level: batch
  ```

- RLOO excludes response $i$ from its own baseline, giving $b_i=\frac{1}{G-1}\sum_{j\ne i}R_j$, and does not scale by a standard deviation.

  ```yaml
  actor:
    reward_norm:
      mean_level: group
      std_level: null
      mean_leave1out: true
    adv_norm:
      mean_level: batch
      std_level: batch
  ```

In practice, within-group comparison requires at least two responses per prompt; when dynamic sampling is enabled, configuration validation also enforces rollout.num_generations >= 2.


### REINFORCE: Keeping the Most Direct Gradient Signal

reinforce disables both the critic and reward normalization. Outcome rewards still pass through GAE and are affected by discount, gae_lambda, KL, and subsequent adv_norm, so this is the trainer's REINFORCE configuration rather than a separate implementation that bypasses the advantage pipeline.

The core reinforce configuration is:

```yaml
actor:
  reward_norm: null
  adv_norm:
    mean_level: batch
    std_level: batch
```

During reward normalization, reinforce_baseline subtracts only the group mean across responses to the same prompt, without scaling by the group standard deviation. After GAE, advantages are still normalized across the complete batch by default:

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: null
    mean_leave1out: false
  adv_norm:
    mean_level: batch
    std_level: batch
```

### GSPO: Controlling Updates with Response-Level Ratios

The GSPO preset sets importance_sampling_level: sequence. For the set of retained tokens $\mathcal K_i$ in a response, the implementation computes:

$$
\rho_i^{\mathrm{seq}}
=\exp\!\left(\frac{1}{|\mathcal K_i|}
\sum_{t\in\mathcal K_i}\log\rho_{i,t}\right),
\qquad
\bar A_i=\frac{1}{|\mathcal K_i|}\sum_{t\in\mathcal K_i}A_{i,t}.
$$

Both quantities are broadcast back to the active tokens of that response. Sequence ratios do not automatically give responses of different lengths equal weight in the total loss; for equal response weights, also set actor.loss_aggregation to seq_mean.

The core GSPO configuration is:

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
  importance_sampling_level: sequence
  loss_aggregation: token_mean
```

### SAPO and CISPO: Replacing the Clipped Objective

SAPO uses sapo_tau_pos for positive advantages and sapo_tau_neg for non-positive advantages, applying a sigmoid gate to smoothly adjust the ratio's contribution to the surrogate objective. It requires use_decoupled_loss: false. The core configuration is:

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: group
  adv_norm:
    mean_level: batch
    std_level: batch
  loss_type: sapo
  sapo_tau_pos: 1.0
  sapo_tau_neg: 1.05
  use_decoupled_loss: false
```

CISPO clips the ratio and stops its gradient, then combines it with the advantage as a weight for the current logprob. It requires token-level ratios and an explicit upper clipping width. The preset uses the wider interval [0, 5]. The core configuration is:

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: group
  adv_norm:
    mean_level: batch
    std_level: batch
  loss_type: cispo
  importance_sampling_level: token
  eps_clip: 1.0
  eps_clip_higher: 4.0
```

### DAPO: Filling the Batch with Informative Prompt Groups

The DAPO preset combines three options:

- eps_clip: 0.2 and eps_clip_higher: 0.28 set the PPO clipping interval to [0.8, 1.28].
- dynamic_sampling: true drops entire prompt groups with identical raw rewards and reads new prompts to refill the current update.
- overlong_tokens: 128 gradually increases the penalty over the final 128 tokens of the maximum generation budget.

The core configuration is:

```yaml
rollout:
  num_generations: 4
  max_tokens: 1024

actor:
  reward_norm:
    mean_level: group
    std_level: group
  adv_norm:
    mean_level: batch
    std_level: batch
  loss_type: ppo
  eps_clip: 0.2
  eps_clip_higher: 0.28
  dynamic_sampling: true
  max_empty_batches: 32
  overlong_tokens: 128
  overlong_penalty_factor: 1.0
```

overlong_penalty_factor controls the maximum length penalty, which is applied before reward bias, scaling, and clipping.
overlong_tokens cannot exceed rollout.max_tokens. If consecutive refill attempts yield no informative prompts, max_empty_batches stops collection with an error; dataset exhaustion can leave a smaller final batch.

### Behavior Rejection and the IcePop/KPop Boundary

Decoupled mode distinguishes the behavior policy that sampled a response from the proximal policy used in the PPO ratio denominator.
recompute_logprob can refresh proximal logprobs before training targets are frozen; tokens can then be masked using the proximal/behavior ratio when the deviation is too large:

```yaml
actor:
  use_decoupled_loss: true
  recompute_logprob: true
  rejection_lower: 0.5
  rejection_upper: 5.0
```

Setting either rejection threshold requires use_decoupled_loss: true. Triotune currently provides a general token rejection mechanism, with no preset named icepop and no implementation of KPop's binary-KL mask; PPOConfig.for_algorithm() does not accept either name.

## From Rewards to Actor Loss

### Reward Calculation and GAE

Scalar outcome rewards are processed in the following order:

$$
R' = \operatorname{clip}\left((R+\text{reward\_bias})
\cdot\text{reward\_scaling},-\text{reward\_clip},\text{reward\_clip}\right).
$$

When enabled, the overlong penalty is applied before the expression above. The normalized outcome reward is added to the final active response token; Trajectory.token_rewards can additionally provide process rewards.

The token-level GAE recurrence is:

$$
\delta_t=r_t+\gamma V_{t+1}-V_t,
\qquad
A_t=\delta_t+\gamma\lambda A_{t+1},
\qquad
G_t=A_t+V_t.
$$

Here $r_t$ includes the task reward and optional KL penalty at that position. The index $t$ advances over active response tokens; masked positions do not consume a timestep.

actor.discount and actor.gae_lambda correspond to $\gamma$ and $\lambda$. Without a critic, all values are zero; both parameters default to 1, so the final outcome reward contributes equally to the advantages of all active tokens before subsequent advantage normalization. Normally terminated responses bootstrap from zero, while responses that reach the length limit use an extra terminal-state value. With mask_no_eos_with_zero: true, task rewards for truncated responses are zeroed, while KL and critic bootstrapping are retained.

gae_timestep_unit: turn treats each nonempty generated turn as one timestep. This mode depends on Trajectory.turn_ids; the stock GSM8K rollout labels the whole response as turn 0, while multi-turn trajectories require custom IDs. Task rewards are summed within a turn, and the value of the state corresponding to its first active token is used in the recurrence. Advantages and returns are then broadcast back to the active tokens in that turn. KL penalties are added back to actor advantages as token-local terms and do not participate in turn-level return calculation.

### Reward Normalization and Advantage Normalization Are Different

actor.reward_norm processes each response's scalar outcome reward before GAE, determining how responses are compared;
actor.adv_norm processes active-token advantages after GAE, primarily controlling the final gradient scale.

Both use NormConfig:

| Field | Options | Purpose |
| --- | --- | --- |
| mean_level | group, batch, null | Subtract the mean over the selected scope |
| std_level | group, batch, null | Scale by the standard deviation over the selected scope |
| mean_leave1out | true, false | Exclude the current response when computing the mean |
| std_unbiased | true, false | Whether to apply the sample standard deviation correction |
| eps | Positive float | Stabilizer added to the standard deviation |

Setting the entire field to null skips that stage; setting only mean_level or std_level to null disables only centering or scaling. Advantage statistics at the batch/group level include only tokens whose loss_mask is true.

### PPO Objective and Weight Aggregation

Ordinary loss_type: ppo minimizes the negative clipped surrogate:

$$
\mathcal L(\theta)
=-\operatorname{Reduce}\left[
\min\!\left(\rho_{i,t}A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon_-,1+\epsilon_+)A_{i,t}\right)
\right].
$$

eps_clip determines the lower clipping width; with eps_clip_higher: null, the upper width is the same. Clipping limits the surrogate gains from continuing to move in a favorable direction; it does not guarantee that the actual ratios after training remain inside the interval.
c_clip > 1 enables dual clipping for negative-advantage terms.

Reduce determines which units receive equal weight within a batch:

| loss_aggregation | Weighting |
| --- | --- |
| token_mean | Equal weight per active token; the default |
| seq_mean | Average within each response first, then give responses equal weight |
| prompt_mean | Give each prompt group's active tokens the same total weight |
| constant | Divide the sum by the product of the active response count and loss_aggregation_divisor |

Long responses naturally carry more weight under token_mean, while seq_mean removes this length weighting. Choose the aggregation method according to what the experiment should average, rather than treating it only as a numerical scaling parameter. prompt_mean preserves original prompt-group weights before minibatch splitting; empty groups contribute zero. constant requires a positive divisor.

### KL with a Fixed Reference Model

actor.kl_ctl and actor.reference_kl_coef both default to 0. Reference scoring is skipped when both are zero. With either coefficient enabled, the trainer uses reference.name; when no reference is configured, it uses the actor's unmodified base model. kl_estimator supports k1, k2, and k3 for KL-as-reward.

Token GAE adds the KL penalty to token rewards before the recurrence. Turn GAE uses only task rewards to compute critic targets, adding KL back to actor advantages as a token-local term to avoid accumulating fine-grained KL into turn returns.

reference_kl_coef instead enables an independent low-variance K3 actor objective:

$$
d_t = \log \pi_\theta(a_t|s_t)-\log \pi_{\mathrm{ref}}(a_t|s_t),
\qquad
K_3(d_t)=e^{-d_t}+d_t-1.
$$

The custom loss adds reference_kl_coef * weighted_mean(K3) directly to the policy objective in every minibatch. The mean uses loss_mask and effective action weights. It does not modify rewards, frozen advantages, or critic returns. This path requires loss_backend: custom and cannot be enabled together with kl_ctl; PyTRIO's custom logprob-gradient bridge is sufficient, so no full-vocabulary logits or server-side loss change is required. Metrics are reported as policy/reference_kl and policy/reference_kl_loss.

## Training Scale and Execution Boundaries

The following parameters are especially likely to affect experiment cost or semantics:

| Parameter | Default | Meaning |
| --- | --- | --- |
| training.batch_size | 1 | Prompts per round, not a device microbatch |
| rollout.num_generations | 4 | Responses generated per prompt |
| rollout.max_tokens | 512 | Maximum response tokens, excluding the prompt |
| actor.update_epochs | 1 | Number of passes over the same frozen actor targets |
| actor.ppo_n_minibatches | 1 | Number of actor update partitions; prompt groups remain intact |
| actor.loss_backend | custom | Use the complete local objective; native supports only an ordinary token PPO subset |

The current PPO logprob contract requires rollout.temperature: 1.0 and top_p: 1.0. The training side receives only target-token logprobs and cannot reconstruct the full normalized distribution after temperature changes or nucleus truncation, so configuration validation rejects other values.

Prompts are sampled concurrently. Each round fixes the rollout data before updating the actor and optional critic; different rollout steps still proceed sequentially. PyTRIO manages model deployment, distributed execution, and optimizers. Triotune configuration does not include other frameworks' cluster backends or Hydra override semantics.

## Custom Trajectories and Manual Lifecycle Management

In addition to the full workflow managed by train(), prepared trajectories can be passed to step():

```python
from triotune.trainer.ppo import Trajectory

metrics = await trainer.step([
    Trajectory(
        prompt_tokens=[...],
        completion_tokens=[...],
        logprobs=[...],
        reward=1.0,
        group_id="prompt-0",
        values=[...],  # completion length + 1 when the critic is enabled
    )
])
```

loss_mask, reference_logprobs, token_rewards, and turn_ids all align with completion tokens. Call initialize() once before step(); when managing the lifecycle manually, also call save_checkpoint() and close(). Do not run multiple step() calls concurrently on the same trainer.

Evaluation is controlled by evaluation.steps, before_train, after_train, batch_size, and max_batches. checkpoint.save_every counts rollout steps; 0 means saving only the final checkpoint. Actor and critic URIs are recorded in checkpoint/actor_path and checkpoint/critic_path, respectively.

Metrics are grouped under reward/*, train/*, policy/*, critic/*, eval/*, time/*, and usage/*. usage/actor_train_tokens and usage/critic_train_tokens include repeated epochs and forward passes used to prepare logprobs/values; usage/train_tokens is their sum, and metrics with the _total suffix accumulate across rollout steps.

## Common Configuration Pitfalls

- **GRPO has no relative rewards:** check whether actor.reward_norm: null from a PPO file was accidentally used to override the GRPO preset.
- **Group rewards are always zero:** with binary rewards, a group may contain only correct or only incorrect responses; increase response diversity or use DAPO dynamic sampling to filter uninformative groups.
- **Optimizer steps exceed max_steps:** this is expected when update epochs and minibatches expand each rollout update.
- **GSPO still favors long responses:** sequence ratios and loss aggregation are separate dimensions; use seq_mean for equal response weights.
- **Rejection configuration raises an error:** also set use_decoupled_loss: true and ensure the lower threshold does not exceed the upper threshold.
- **The critic still trains after switching algorithms:** check whether an explicit critic.enabled setting overrides the preset.
- **All tokens are masked:** there is no usable policy gradient for this round, so the trainer rejects the update rather than silently performing a zero-gradient step.

For implementation details, see the [configuration definitions](../../triotune/api/cli_args.py),
[PPO trainer components](../../triotune/trainer/ppo/), and
[GSM8K PPO recipe](../../examples/ppo/gsm8k_ppo_args.yaml).
