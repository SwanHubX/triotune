# Configuring learning-rate schedules

[中文](lr_scheduler_CN.md)

Choose a learning-rate schedule under optimizer.scheduler. SFT, DPO, OPD, and PPO share this configuration; the default constant policy preserves the original fixed learning rate.

## 1. Enable a schedule in your experiment

Merge the following fields into an existing experiment YAML:

```yaml
training:
  max_steps: 100

optimizer:
  learning_rate: 0.0001
  scheduler:
    type: cosine
    warmup_steps: 10
    total_steps: null
    min_lr: 0.00001
```

learning_rate is the base learning rate. Each optimizer update computes and submits its own LR while leaving the base configuration value unchanged.

| Parameter | Usage |
| --- | --- |
| type | constant, linear, or cosine. |
| warmup_steps | Number of optimizer updates for warmup, defaulting to 0; works with all three schedules. |
| total_steps | Total optimizer updates for decay; inferred from the training plan when omitted. constant does not use this field. |
| min_lr | Learning-rate floor; linear/cosine require a positive value and default to 1% of the base LR when omitted. The floor cannot exceed the base LR. |

For warmup followed by a fixed LR, use constant and set warmup_steps.

## 2. Understand the LR at each update

Step indices start at 1 and count actual optimizer updates.

- Without warmup, the first update of a multi-step linear/cosine schedule uses the base LR.
- During warmup, the LR rises to the base LR as base LR × current step / warmup_steps, bounded by the learning-rate floor.
- linear/cosine use min_lr at the update indexed by total_steps and retain that floor afterward.
- A decay plan with only one update uses min_lr directly. Decay plans require warmup_steps to be less than total_steps.

For example, with base LR 0.0001, linear warmup over 2 steps, 5 optimizer updates in total, and a floor of 0.00001:

| Optimizer step | LR |
| --- | --- |
| 1 | 0.00005 |
| 2 | 0.0001 |
| 3 | 0.00007 |
| 4 | 0.00004 |
| 5 | 0.00001 |

## 3. Set the total update count

When setting scheduler.total_steps explicitly, supply the optimizer update count directly. When omitted:

1. training.max_steps takes priority as the training update budget.
2. If max_steps is unset, use the batch count of a sized dataset.
3. PPO then converts that budget into optimizer updates using each component's own epoch and minibatch counts.

When both dataset length and max_steps are unknown, constant and warmup remain available. linear/cosine require explicit total_steps; dynamic sampling cannot infer usable batch counts from raw dataset length and also needs max_steps or explicit total_steps.

## 4. Configure the PPO actor and critic separately

PPO global_step counts rollout updates. Each component's optimizer step count also includes updates across multiple epochs and minibatches.

```yaml
training:
  max_steps: 100
  batch_size: 4

optimizer:
  learning_rate: 0.00004
  scheduler:
    type: linear
    warmup_steps: 8
    min_lr: 0.000004

actor:
  update_epochs: 2
  ppo_n_minibatches: 4

critic:
  enabled: true
  update_epochs: 1
  n_minibatches: 2
  optimizer:
    learning_rate: 0.00008
    scheduler:
      type: cosine
      warmup_steps: 4
      min_lr: 0.000008
```

The actor budget is 100 × 2 × 4 = 800 optimizer updates; the critic budget is 100 × 1 × 2 = 200. Each component counts updates and runs warmup independently.

When critic.optimizer is omitted from YAML, it copies the actor optimizer's initial configuration and then computes LR using its own step indices.

## 5. Continue the original schedule after a pause

Enable checkpoint.state_dir and restore with checkpoint.resume_from as described in the [resume tutorial](checkpoint_resume.md).

Keep the base LR, scheduler configuration, and resolved total update count the same. Resuming continues with the next optimizer update without restarting warmup. The SFT pipeline also assigns indices in submission order.

If you plan to increase max_steps later, set scheduler.total_steps explicitly before the first run and keep that value unchanged when resuming. A decay plan inferred automatically from max_steps requires the original max_steps. constant supports extending the training target.

P0-02 fixed-LR manifests can be resumed directly with the default constant policy and no warmup; new manifests also save each component's schedule horizon, optimizer count, and most recent LR.

## 6. View the current learning rate

The train/learning_rate metric reports the LR submitted for the corresponding training update. The PPO critic uses critic/learning_rate; when a rollout contains multiple optimizer updates, these metrics show the last value.
