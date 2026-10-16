# PyTRIO PPO 算法族

把 AReaL 的“同一训练循环，通过配置替换优势估计和损失”的设计映射到 PyTRIO。
所有算法实现和开关都在本目录；沿用仓库的 dataset、客户端适配、日志与保存工具。
这是文本、token 级 GAE 的实现，不是 AReaL 分布式运行时的完整移植。
已有 `GRPOTrainer` 保持原行为；这里提供另一套统一的 PPO 算法族入口。

## 模块与数据流

| 文件 | 职责 | AReaL 对应实现 |
| --- | --- | --- |
| `config.py` | 配置校验、算法预设、独立 mean/std 范围 | `PPOActorConfig`、各算法 YAML |
| `contracts.py` | 不补齐的轨迹、冻结后的训练 batch、运行状态 | rollout 数据和 batch metadata |
| `rollout.py` | 并发采样、异步/同步 reward、固定参考评分、动态补采 | workflow、rollout 调度 |
| `normalization.py` | batch/group 均值和标准差、leave-one-out | `Normalization` |
| `advantages.py` | 奖励缩放、长度惩罚、KL、GAE、优势归一化 | actor `_compute_advantages` |
| `gae.py` | 跳过工具位置的 GAE、截断 bootstrap | `_compute_token_level_gae` |
| `loss.py` | PPO、双重裁剪、GSPO、SAPO、CISPO、聚合和行为过滤 | `functional.py` 策略损失 |
| `datum.py` | next-token 对齐、prompt/tool mask、原生 PPO 字段 | token shift、训练数据准备 |
| `actor.py` | 旧策略重算、冻结目标、多 minibatch/epoch 更新 | `PPOActor` |
| `critic.py` | 独立 PyTRIO critic 的评分、回归更新和保存，保留外部协议 | `PPOCritic` |
| `critic_loss.py` | 两个 marker logprob 读出无界 value、clipped regression、因果对齐 | value head 的另一种参数化 |
| `reporting.py` | 为 critic 创建独立 SwanLab 日志进程 | 双模型日志 |
| `trainer.py` | 串联 rollout → reference/value → 冻结目标 → 并发更新 actor/critic → 保存/日志 | `RLTrainer` |

一次 rollout update 冻结 reward、advantage、return 和 proximal logprob。
`ppo_n_minibatches` 把完整 prompt group 分成多个 optimizer update，
`update_epochs` 重复使用这些冻结目标。为保证组归一化和 `prompt_mean` 的含义，
这里不会把同一个 prompt group 拆到不同 minibatch；minibatch 数不能超过有效 group 数。
`max_steps`、`save_every` 和 SwanLab step 统计 rollout update；
`state.optimizer_steps`、`state.critic_optimizer_steps` 分别统计实际完成的 actor/critic Adam 更新。

## 使用与切换

默认自定义损失需要本地 CPU PyTorch：

```bash
uv sync --extra ppo
```

```python
import asyncio
from trio_cookbook.trainer.ppo import PPOConfig, PPOTrainer


def reward(*, row, completion, completion_ids):
    # 根据任务替换奖励；也支持 async def。
    return float(row["answer"] in completion)


async def main():
    args = PPOConfig.for_algorithm(
        "ppo",  # 默认创建两个独立训练客户端；改为 grpo/rloo/gspo 等可关闭 critic。
        batch_size=2,
        num_generations=4,
        max_steps=10,
        max_tokens=256,
        learning_rate=1e-5,
        seed=42,
        report_to="swanlab",
        project="ppo-family",
    )
    rows = [
        {"messages": [{"role": "user", "content": "1+1=?"}], "answer": "2"},
        {"messages": [{"role": "user", "content": "2+2=?"}], "answer": "4"},
    ]
    trainer = PPOTrainer("Qwen/Qwen3.5-4B", args, rows, reward_fn=reward)
    state = await trainer.train()
    print(state.checkpoint_path, state.critic_checkpoint_path)


asyncio.run(main())
```

`train()` 单遍消费 iterable，`max_steps` 是上限；上面的两条数据仅产生一次 update。
要运行十次，应提供至少二十条 prompt 行，可自行重复 iterable。
`messages` 也支持纯文本或 token ID，字段名由 `prompt_key` 配置。
奖励函数接收原始行、completion 文本和 token ID，不会自动执行任务判断。
模型必须是服务端可用的模型；初始化与训练均通过协程调用。

| 预设 | 共享计算中的主要变化 |
| --- | --- |
| `ppo` | 独立 PyTRIO critic 计算 value + GAE；不归一化 outcome reward |
| `grpo` | 无 critic，组内 reward 减均值、除标准差 |
| `drgrpo` | 组内 reward 减均值，不除标准差 |
| `liteppo` | 组内 reward 减均值，用 batch 标准差缩放 |
| `rloo` | 用同组其他回答的平均 reward 作 baseline |
| `reinforce` | 无 critic，不归一化 outcome reward |
| `reinforce_baseline` | 无 critic，组均值 baseline |
| `gspo` | GRPO 优势；ratio 改为序列 log-ratio 均值的指数，advantage 同样取序列均值 |
| `sapo` | GRPO 优势；用正负 advantage 对应的 sigmoid gate 替换硬裁剪 |
| `cispo` | GRPO 优势；裁剪并 detach ratio，让梯度继续经过当前 logprob |
| `dapo` | GRPO 优势；动态补采、非对称裁剪、超长回答惩罚 |

预设体现 AReaL 的配置组合，不表示完全复刻每篇论文的默认超参或官方实验。
特别是 `drgrpo` 对应 AReaL 示例 YAML 的 reward 标准差开关；所有预设默认仍有
`adv_norm=NormConfig("batch", "batch")`，可通过 `adv_norm=None` 关闭。
`reinforce` 也保留 AReaL 公用的 PPO surrogate；第一次 on-policy 更新梯度等于
REINFORCE，重复更新时仍会发生裁剪，不能视为无裁剪的多轮 REINFORCE。
DAPO 默认 `overlong_tokens=128`，缩短 `max_tokens` 时须相应缩小该值。

`loss_aggregation` 支持：

- `token_mean`：总 token loss / 有效 action token 数。
- `seq_mean`：每条回答先取 token 均值，再对回答取平均。
- `prompt_mean`：每个 prompt 内汇总所有回答的 token loss，除以该组有效 token 数，
  再对 prompt 平均。minibatch 使用完整 group 和按总 group 数计算的缩放系数。
- `constant`：总 token loss / (有效回答数 × `loss_aggregation_divisor`)。

`use_decoupled_loss=True` 时重算 proximal 策略，保留 rollout 的 behavior logprob；
ratio 使用 current/proximal。`rejection_lower/upper` 可选择 token 级行为概率比过滤，
同时乘 detached proximal/behavior 校正权重。被过滤的 token 仅从分子移除，保留原始分母。
普通 `recompute_logprob=True` 则同时替换 KL reward 和 ratio 分母中的旧 logprob。
这些配置允许接收外部旧轨迹，但内置循环每轮都刷新策略，没有实现异步 stale rollout 调度。

## Critic 实现和自定义 rollout

`await trainer.step(trajectories)` 可直接训练自定义工作流产生的 `Trajectory`。
prompt 与 completion 分开保存；completion 的 logprob、mask、reference 和 token reward
必须逐 token 对齐。`group_id` 标识同一 token 化 prompt 的不同回答，组大小可以不同。
多轮工具返回可以保留在 completion 的上下文中，用 `loss_mask=False` 排除其梯度及 GAE timestep。
`token_rewards` 在这里表示直接进入 token GAE 的奖励增量；没有移植 AReaL 的 PRM/turn reward 折叠规则。

标准 PPO 默认创建独立的 actor 和 critic LoRA 训练客户端，同一个基座模型、同一 tokenizer，
参数和 Adam 状态相互独立。actor 使用 `seed`，critic 使用 `seed+1`。
critic 以两个既有单 token（默认 `"1"`、`"0"`）的 logprob 读出标量：

```text
V(s) = critic_value_scale * (log p(positive|s) - log p(negative|s)) + critic_value_bias
```

log-softmax 的归一化项抵消，因此这是两个输出 logit 的差，可表示正负无界 value。
每条轨迹构造两个 Datum，输入始终为真实 prompt+completion；marker 只放在 target，
不会插入上下文。`auto_shift=False` 时，从 prompt 最后一个位置开始读取 T+1 个值，
对应 action 之前的状态和最终 bootstrap，不会提前看到待评估的 action。

用这批旧 value 计算并冻结 GAE advantage/return，然后并发更新两个独立客户端。
actor 优化 PPO surrogate；critic 用 `forward_backward_custom` 对 clipped value MSE 求导，
SDK 将两个 marker 的相反梯度转换成远端 CE 权重，再调用 critic 自己的 `optim_step`。
本地仅处理 logprob/梯度，不加载两个模型；这是可训练的 critic，不是固定打分器。

`critic_learning_rate`、`critic_lora_rank` 未设置时继承 actor；
`critic_token_ids=(positive_id, negative_id)` 可替换默认 marker（必须是既有、不同的 token）。
其余配置为 `critic_value_scale=1`、`critic_value_bias=0`、`critic_eps_clip=0.2`、
`critic_n_minibatches=1`、`critic_update_epochs=1`。critic minibatch 按轨迹切分，
不得超过有有效 action 的轨迹数。注入 `critic_training_client=` 时必须与 actor 的 tokenizer 一致。

也可以显式传入实现 `Critic` 协议的对象：

```python
args = PPOConfig.for_algorithm("ppo", discount=0.99, gae_lambda=0.95)
trainer = PPOTrainer("your-service-model", args, critic=your_external_critic)
await trainer.step(your_trajectories)
```

`compute_values()` 返回每条回答的 T+1 个 value，前 T 个对应各 action 之前的状态，
最后一个用于截断 bootstrap。正常终止的 bootstrap 为零。
`update()` 接收冻结的 return 和 mask，自行训练并管理 critic optimizer/checkpoint；
可使用 `value_loss()` 实现本地 torch clipped value regression。
外部 critic 的生命周期由调用者管理；内置 critic 的 checkpoint 和 reporter 由 trainer 管理。
`save_checkpoint()` 保存两份远端状态，返回 actor 路径；critic 路径见
`state.critic_checkpoint_path`。仅调用 `step()` 的工作流应在结束时自行调用
`save_checkpoint()` 和 `close()`。两份保存不是原子事务，目前没有联合加载/恢复接口。

## PyTRIO 能力与边界

已核对本仓库锁定的 **PyTRIO 0.2.9**，以及
[Trio-Train](https://github.com/SwanHubX/Trio-Train) 的
`api/loss_types.py`、`losses/ppo.py`、`losses/cross_entropy.py`。

| 功能 | 实现或限制 |
| --- | --- |
| PPO 原生策略损失 | `loss_backend="native"` 使用 `forward_backward(loss_fn="ppo")`；阈值字段为 `clip_low_threshold/high_threshold`，prompt/tool label 为 `-100` |
| PPO、GSPO、SAPO、CISPO 等可导 logprob 目标 | 默认 `custom`，通过 `forward_backward_custom` 本地求梯度，SDK 将负梯度作为远端 CE 权重；本地不加载模型 |
| 原生损失的归一化 | Trio-Train 的 native PPO backward normalization 是服务端运行配置；不能通过当前请求参数保证与 custom 数值完全相同，比较实验建议统一 custom |
| 固定 reference KL | `SamplingClient.compute_logprobs`，completion 从完整序列的 `len(prompt)` 位置切片；k1/k2/k3 限幅到 [-10, 10] 后加入奖励 |
| 标准 PPO critic | 通过第二个训练客户端的两个 marker logprob 读出 value，自定义回归损失更新；无需扩展 SDK。没有原生 value head，与 AReaL value head 的参数化不同 |
| 温度/top-p | 当前实现要求二者均为 1；目标 token logprob 不足以重建变换后的全词表归一化，未实现任意温度下的等价 AReaL actor loss |
| 精确 entropy / 全词表 KL | 当前自定义接口只暴露目标 token logprob，没有 logits；未提供这类目标 |
| 分布式 engine、GPU/offload、异步 rollout 调度 | 由 PyTRIO 服务控制，客户端不能复刻 AReaL 的 FSDP/SGLang、资源调度和权重版本协议 |
| 尚未移植的客户端功能 | turn 级/自定义 lambda GAE、PRM shaping、多模态、序列级/其他模式 rejection sampling、评估循环及联合恢复 actor+critic 训练；这些不都属于 SDK 的硬限制 |

`mask_no_eos_with_zero=True` 将截断样本的 outcome/process reward 置零，仍保留 KL 和 critic
bootstrap。长度惩罚先作用于 outcome reward，再做 bias/scaling/clip 和组归一化。
这里使用直接的 token 轨迹语义，没有 AReaL rollout-export 中 logical group、分段轨迹
和某些超长惩罚的特殊归一化处理，不承诺与这些高级路径逐位一致。

训练指标通过公共 `MetricsAPI` 输出：`reward/mean/std`、`train/loss`（custom）、
`policy/ratio/clip_fraction/rejected_fraction`、`train/action_tokens/optimizer_steps`、
`time/prepare/update/rollout/step`，保存远端 actor checkpoint 后记录 `checkpoint/path`。
内置 critic 输出 `critic/loss`、`critic/value_mean`、`critic/return_mean`、`critic/clip_fraction`、
`critic/optimizer_steps`、`critic/learning_rate`、`critic/action_tokens` 及 `time/critic_update`。
启用 SwanLab 时分别创建 `<run_name>-actor` 和 `<run_name>-critic` 两个 run，
critic run 记录回归指标和自己的 `checkpoint/path`；actor run 同时包含合并后的 critic 指标。
SwanLab 0.10.1 每进程仅允许一个活动 run，故仅 critic 日志放在子进程，两个训练客户端仍在主进程。
原生 actor 损失指标取决于服务端；外部 critic 的指标由其 `update()` 返回。

## 参考与验证

设计对照本地 AReaL checkout `b2f2f56` 的
`areal/trainer/rl_trainer.py`、`areal/trainer/ppo/{actor,critic,gae,loss_reduction}.py`、
`areal/utils/functional/functional.py` 和 `examples/math/gsm8k_*.yaml`。

```bash
uv run --extra ppo python -m unittest discover -s tests -p 'test_ppo*.py' -v
```

测试覆盖优势/GAE、目标值和梯度、不同长度的 reduction、异步两阶段 future、动态补采、
固定 reference、内置/外部 critic、native 参数、因果 value 对齐、真实回归学习、两个独立优化器，
以及安装的真实 SDK 如何将两个 marker 的梯度转换为 CE 权重。当前共 39 项专项测试。
另已用 Qwen/Qwen3.5-4B 完成两轮真实 PyTRIO 训练：actor/critic 各更新两次并保存，
同一批状态的 critic value 在每次更新后发生变化；两个 SwanLab run 离线正常结束。
这只证明训练链路可用，不代表与 AReaL 的收敛效果或性能等价，详见
[测试记录](../../../docs/ppo_pytrio-use_eval.md)。
