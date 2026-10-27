# 配置学习率调度

[English](lr_scheduler.md)

在 optimizer.scheduler 中选择学习率调度。SFT、DPO、OPD 和 PPO 使用同一套配置；默认 constant 保持原来的固定学习率。

## 1. 在实验配置中启用

将下面的字段合并到已有实验 YAML：

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

learning_rate 是基础学习率。每次 optimizer 更新都会计算并提交该步的 LR，基础配置值保持不变。

| 参数 | 用法 |
| --- | --- |
| type | constant、linear 或 cosine。 |
| warmup_steps | 预热的 optimizer 更新次数，默认 0；可搭配三种调度。 |
| total_steps | 衰减的总 optimizer 更新次数；省略时从训练计划推导。constant 不使用该字段。 |
| min_lr | 学习率下限；linear/cosine 需要正数，省略时使用基础 LR 的 1%。下限不能超过基础 LR。 |

只需要预热后保持固定 LR 时，使用 constant 并设置 warmup_steps。

## 2. 理解每一步的 LR

步号从 1 开始，按实际 optimizer 更新计数。

- 无预热时，多步 linear/cosine 调度的第 1 步使用基础 LR。
- 预热期间按基础 LR × 当前步号 / warmup_steps 升到基础 LR，并受学习率下限约束。
- linear/cosine 在 total_steps 对应的更新使用 min_lr；之后保持该下限。
- 只有一个更新的衰减计划直接使用 min_lr。衰减计划要求 warmup_steps 小于 total_steps。

例如，基础 LR 为 0.0001，linear 预热 2 步，总共 5 个 optimizer 更新，下限为 0.00001：

| optimizer 步号 | LR |
| --- | --- |
| 1 | 0.00005 |
| 2 | 0.0001 |
| 3 | 0.00007 |
| 4 | 0.00004 |
| 5 | 0.00001 |

## 3. 设置总更新次数

显式设置 scheduler.total_steps 时，直接填写 optimizer 更新次数。省略时：

1. 优先使用 training.max_steps 作为训练更新预算。
2. 未设置 max_steps 时，使用有长度数据集的 batch 数。
3. PPO 再按各组件自己的 epoch 数和 minibatch 数换算为 optimizer 更新次数。

数据长度和 max_steps 都未知时，可以使用 constant 和预热。linear/cosine 需要显式 total_steps；dynamic sampling 的有效 batch 数不能由原始数据长度推导，也需要 max_steps 或显式 total_steps。

## 4. 分别配置 PPO actor 和 critic

PPO 的 global_step 统计 rollout 更新。actor 和 critic 各自的 optimizer step 还包含多 epoch、多 minibatch 的更新。

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

此配置的 actor 预算为 100 × 2 × 4 = 800 次 optimizer 更新；critic 为 100 × 1 × 2 = 200 次。两者分别计数和预热。

YAML 中省略 critic.optimizer 时，会复制 actor optimizer 的初始配置，随后按各自的步号计算 LR。

## 5. 暂停后继续原来的调度

启用 checkpoint.state_dir 并使用 checkpoint.resume_from 恢复，操作见 [断点续训教程](checkpoint_resume_CN.md)。

保持基础 LR、scheduler 配置以及推导出的总更新次数一致，恢复会从下一次 optimizer 更新继续，预热不会重新开始。SFT 流水线也按提交次序分配步号。

如果之后需要提高 max_steps，在第一次训练前显式设置 scheduler.total_steps，并在恢复时保持该值不变。自动从 max_steps 推导的衰减计划需要保持原来的 max_steps。constant 可以继续扩展训练目标。

P0-02 的固定 LR 清单可以直接恢复，使用默认 constant 且不启用预热；新的清单会同时保存各组件的调度总步数、optimizer 计数和最近一次 LR。

## 6. 查看当前学习率

训练指标中的 train/learning_rate 记录对应训练更新实际提交的 LR。PPO critic 使用 critic/learning_rate；一个 rollout 包含多个 optimizer 更新时显示最后一次的值。
