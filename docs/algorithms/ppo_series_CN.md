# 在 Triotune 中理解和选择 PPO 系列算法

[English](ppo_series.md)

PPO、GRPO、RLOO、GSPO 等算法共用一个异步 PPOTrainer，差别集中在三个问题上：用什么构造优势、用多大粒度计算策略比率，以及如何限制一次更新的幅度。

从这个角度看，选择算法不是替换训练流程，而是在同一条数据链路上替换少数环节：

```text
prompt
  └─ rollout：每题生成若干回答并记录 behavior logprob
       └─ reward：任务奖励、长度惩罚、可选 KL
            └─ advantage：critic/组内 baseline → GAE → 可选归一化
                 └─ actor loss：token 或 sequence ratio → PPO/SAPO/CISPO
                      └─ PyTRIO：远端前向、反向与优化器更新
```

本文先解释这条链路，再给出算法选择方法和专用配置。

## 一次更新究竟发生了什么

假设一个训练 batch 有 $B$ 道题，每道题采样 $G$ 个回答。一次 rollout 会产生
$B\times G$ 条轨迹。每条轨迹至少包含回答 tokens、采样时的 logprobs、标量奖励和题目组ID；启用 critic 时，还会为回答的各个状态估计 value。

Trainer 会按以下顺序处理这些信息：

1. 对结果奖励施加长度惩罚、偏移、缩放和裁剪；
2. 按算法预设构造 baseline，例如 PPO 的 critic value 或 GRPO 的同题回答均值；
3. 用 GAE 将末尾的结果奖励传播到有效回答 token；
4. 可选地在整个 rollout batch 中再次归一化优势；
5. 冻结本轮目标，在一个或多个 minibatch、多个 update epoch 上更新 actor；
6. 若启用 critic，用同一批冻结的 return 更新独立的 critic client。

这里有两个计数不要混淆：training.max_steps 统计 rollout 更新次数，

$$actor.update_epochs × actor.ppo_n_minibatches$$

上述公式决定每轮可能发生多少次 actor optimizer step。训练数据只遍历一次，因此数据提前耗尽时，实际 rollout step 可以少于 max_steps。

## PPO 训练案例

### 环境与数据

在仓库根目录安装环境：

```bash
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

PPO 示例与 GRPO 示例共用 GSM8K JSONL 格式，每行至少包含 `messages` 和 `answer`。可按
[准备 GSM8K 数据](grpo_series_CN.md#准备-gsm8k-数据)生成
`data/gsm8k_train.jsonl` 和 `data/gsm8k_test.jsonl`。

### 启动训练

```bash
uv run python examples/ppo/gsm8k_ppo.py \
  --config examples/ppo/gsm8k_ppo_args.yaml
```

该命令使用独立 critic，并以最后一个 `\boxed{}` 中的答案是否正确作为 0/1 奖励。


## 选算法时看三个模块

算法名称适合作为实验标签，但理解配置时，更有用的是看它改变了哪一个模块。

### 1. 优势的参照物来自哪里

- PPO 学习一个 critic，用状态 value 估计当前状态的期望回报，再由 advantage 衡量实际结果比预期好多少。
- GRPO、Dr.GRPO、LitePPO、RLOO 和带 baseline 的 REINFORCE 在同一道题的多个回答之间建立参照，不训练 critic。
- 原始 REINFORCE 不使用 critic 或奖励 baseline，通过零 value 的 GAE 传播任务奖励。

组内方法省掉了 critic，但并非没有代价：每道题需要多次生成，而且一组回答如果奖励
完全相同，就没有相对排序信号。DAPO 的动态采样会过滤这样的整个题目组并补采新题。

### 2. 新旧策略在哪里比较

默认 PPO 为每个有效 token 计算重要性比率：

$$
\rho_{i,t}(\theta)
=\exp\!\left(\log\pi_\theta(o_{i,t}\mid x_i,o_{i,<t})
-\log\pi_{\mathrm{prox}}(o_{i,t}\mid x_i,o_{i,<t})\right).
$$

GSPO 将一条回答中保留 token 的 log-ratio 先取平均，再取指数，相当于使用几何平均的
序列级 ratio。这样一条回答共享同一个更新尺度，不会因为某个 token 的 ratio 特别极端
而单独改变该位置的裁剪判断。

### 3. 如何约束策略移动

- PPO 对 ratio 对应的 surrogate 项做硬裁剪；
- DAPO 使用更宽的上界，允许正优势样本有更大的提升空间；
- SAPO 用平滑门控取代硬裁剪的拐点；
- CISPO 裁剪 ratio 后停止其梯度，再让当前 logprob 承担梯度；
- 可选的 rejection 阈值直接屏蔽 behavior/proximal 偏差过大的 token。

下表只描述 Triotune preset 实际设置的默认值，不表示复现了同名论文的全部训练系统。

| Preset | 优势参照 | ratio 粒度 | 额外改变 |
| --- | --- | --- | --- |
| ppo | 独立 critic | token | 开启 critic，关闭奖励归一化 |
| grpo | 组均值（组标准差用于缩放） | token | group/group 奖励归一化 |
| drgrpo | 组均值 | token | 不按组标准差缩放 |
| liteppo | 组均值 | token | 用整个 batch 的尺度缩放 |
| rloo | 其他同题回答的均值 | token | leave-one-out baseline |
| reinforce | 无 baseline | token | 通过零 value 的 GAE 传播奖励 |
| reinforce_baseline | 组均值 | token | 只做组内中心化 |
| gspo | GRPO 默认值 | sequence | 几何平均 ratio 与回答平均优势 |
| sapo | GRPO 默认值 | token | 平滑门控损失 |
| cispo | GRPO 默认值 | token | 停止梯度的裁剪权重 |
| dapo | GRPO 默认值 | token | 非对称裁剪、动态采样、长度惩罚 |

所有 preset 默认仍会对 GAE 后的优势做 batch/batch 归一化，除非显式覆盖
actor.adv_norm。

## 两种配置方式不要混用

### 从 preset 开始

在 Python 中可以直接选择算法，再补充公共模块：

```python
from triotune.trainer import PPOConfig

config = PPOConfig.for_algorithm(
    "drgrpo",
    model={"name": "Qwen/Qwen3.5-9B", "lora_rank": 32},
    training={"batch_size": 4, "max_steps": 100, "seed": 42},
    rollout={"num_generations": 4, "max_tokens": 1024},
)
```

actor、critic 会与 preset 默认值合并，并以显式值为准；如果传入已经
构造好的 PPOActorConfig 或 PPOCriticConfig，则会替换整个对应模块。

模块化 YAML 可以在顶层使用算法选择器，但必须通过 from_modules 读取：

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

### 使用完整显式配置

`examples/ppo/gsm8k_ppo_args.yaml` 会映射到 PPOConfig(**settings)。这种方式适合固定配置文件，配置文件会完整记录最终行为，比如 critic.enabled: true 明确表示使用 PPO。不要在这种文件里额外添加 algorithm，因为直接构造不解析算法选择器。

## 算法专用选项

### PPO：critic 与 GAE

Triotune 内置的 critic 是独立的 PyTRIO training client。它不使用原生 value head，而是在每个状态的因果上下文上，用两个标记 token 的 logprob 差读出一个无界标量：

$$
V_t=s\left(\log p(\text{positive}\mid s_t)-\log p(\text{negative}\mid s_t)\right)+b.
$$

critic.value_scale 和 critic.value_bias 分别对应 $s$ 与 $b$。默认会尝试将数字字符 1 和 0 分别编码为单个 token；若当前 tokenizer 不满足这个条件，需要通过 critic.token_ids 显式设置两个不同的 token ID。

```yaml
critic:
  enabled: true
  lora_rank: null          # null 表示继承 actor rank
  optimizer:
    learning_rate: 0.00004
  token_ids: null
  value_scale: 1.0
  value_bias: 0.0
  eps_clip: 0.2
  update_epochs: 1
  n_minibatches: 1
```

critic 有自己的 optimizer、minibatch 数和 update epoch 数。eps_clip 在 value regression
中限制当前 value 相对冻结旧 value 的变化，与 actor 的 ratio 裁剪宽度无关。

### GRPO：把同题回答变成彼此的参照

对同一道题的 $G$ 个回答，先得到经过奖励整形的标量 $R_i$。默认 GRPO preset 计算：

$$
\widetilde R_i=\frac{R_i-\mu_g}{s_g+\varepsilon},\qquad
\mu_g=\frac{1}{G}\sum_{j=1}^{G}R_j.
$$

默认 std_unbiased: true，因此 $s_g$ 是样本标准差。高于组均值的回答获得正信号，低于
均值的回答获得负信号；如果整组奖励相同，中心化后的信号为零。这里的统计边界由轨迹的
group_id 决定，过滤后大小不同的回答组仍能分别计算。比如 GRPO 的配置：

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

### Dr.GRPO、LitePPO 与 RLOO：改变 baseline 的尺度

这三个 preset 不改变 trainer，也不改变默认的 token 级 PPO loss，只替换结果奖励的
归一化方式：

```python
drgrpo = PPOConfig.for_algorithm("drgrpo", **common)
liteppo = PPOConfig.for_algorithm("liteppo", **common)
rloo = PPOConfig.for_algorithm("rloo", **common)
```

- Dr.GRPO 在奖励归一化阶段使用 mean_level: group, std_level: null，只减去组均值，不按组标准差缩放，从而避免奖励差距很小的组被放大。GAE 后的优势仍按默认配置在整个 batch 中归一化。

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

- LitePPO 先在组内中心化，再用 std_level: batch 统一整批尺度。不同题目共享缩放因子，但仍各自决定正负方向。

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

- RLOO 计算第 $i$ 个回答的 baseline 时排除它自己，即 $b_i=\frac{1}{G-1}\sum_{j\ne i}R_j$，并且不做标准差缩放。

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

组内比较在实践中至少需要每题两个回答；动态采样开启时，配置层也会强制令 rollout.num_generations >= 2。


### REINFORCE：保留最直接的梯度信号

reinforce 同时关闭 critic 和奖励归一化。结果奖励仍会经过 GAE，并受
discount、gae_lambda、KL 以及后续 adv_norm 影响，所以它表示的是本 trainer 中的
REINFORCE 配置，而不是绕开优势管线的另一套实现。

reinforce 的核心配置如下：

```yaml
actor:
  reward_norm: null
  adv_norm:
    mean_level: batch
    std_level: batch
```

reinforce_baseline 在奖励归一化阶段只减去同题回答的组均值，不按组标准差缩放；GAE 后
的优势仍按默认配置在整个 batch 中归一化：

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

### GSPO：用回答级 ratio 控制更新

GSPO preset 设置 importance_sampling_level: sequence。对一条回答中保留下来的 token 集合
$\mathcal K_i$，实现计算：

$$
\rho_i^{\mathrm{seq}}
=\exp\!\left(\frac{1}{|\mathcal K_i|}
\sum_{t\in\mathcal K_i}\log\rho_{i,t}\right),
\qquad
\bar A_i=\frac{1}{|\mathcal K_i|}\sum_{t\in\mathcal K_i}A_{i,t}.
$$

两者都会广播回该回答的有效 token。注意，序列级 ratio 不会自动让不同长度的回答在总
loss 中等权；如需每条回答等权，还要将 actor.loss_aggregation 设置为 seq_mean。

GSPO 的核心配置如下：

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

### SAPO 与 CISPO：替换裁剪目标

SAPO 对正优势使用 sapo_tau_pos，对非正优势使用 sapo_tau_neg，以 sigmoid 门控平滑调整
ratio 对代理目标的贡献。它要求 use_decoupled_loss: false，核心配置如下：

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

CISPO 将 ratio 裁剪后停止其梯度，再与优势一起作为当前 logprob 的权重。它要求 token
级 ratio，且必须设置上裁剪宽度。preset 使用较宽的区间 [0, 5]，核心配置如下：

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

### DAPO：让有效训练组占满 batch

DAPO preset 组合了三个选项：

- eps_clip: 0.2、eps_clip_higher: 0.28 将 PPO 裁剪区间设为 [0.8, 1.28]；
- dynamic_sampling: true 丢弃组内原始奖励完全相同的整个题目组，并继续读取新题补充本轮更新；
- overlong_tokens: 128 在最大生成长度之前的最后 128 个 token 区间内逐渐增加惩罚。

核心配置如下：

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

长度惩罚最大值由 overlong_penalty_factor 控制，且在奖励偏移、缩放和裁剪之前应用。
overlong_tokens 不能大于 rollout.max_tokens。连续补不到有效题目时，
max_empty_batches 会终止补采样并报错；数据耗尽则可能留下一个较小的最终 batch。

### Behavior rejection，以及 IcePop/KPop 的边界

解耦模式区分采样回答的 behavior 策略和 PPO ratio 分母使用的 proximal 策略。
recompute_logprob 可以在冻结训练目标前重新计算 proximal logprob；随后可按
proximal/behavior 比率屏蔽偏差过大的 token：

```yaml
actor:
  use_decoupled_loss: true
  recompute_logprob: true
  rejection_lower: 0.5
  rejection_upper: 5.0
```

设置任一 rejection 阈值都要求 use_decoupled_loss: true。Triotune 当前提供的是通用
token rejection 机制，没有名为 icepop 的 preset，也没有实现 KPop 的 binary-KL mask；
PPOConfig.for_algorithm() 不接受这两个名称。

## 从奖励到 actor loss

### 奖励计算和 GAE

标量结果奖励按以下次序处理：

$$
R' = \operatorname{clip}\left((R+\text{reward\_bias})
\cdot\text{reward\_scaling},-\text{reward\_clip},\text{reward\_clip}\right).
$$

启用超长惩罚时，惩罚发生在上式之前。归一化后的结果奖励会加到最后一个有效回答
token；Trajectory.token_rewards 可另外提供过程奖励。

token 级 GAE 的递推为：

$$
\delta_t=r_t+\gamma V_{t+1}-V_t,
\qquad
A_t=\delta_t+\gamma\lambda A_{t+1},
\qquad
G_t=A_t+V_t.
$$

这里 $r_t$ 包含该位置的任务奖励及可选 KL 惩罚，$t$ 按有效回答 token 推进，被 mask 的位置不消耗时间步。

actor.discount 和 actor.gae_lambda 分别是 $\gamma$、$\lambda$。无 critic 时 value
全部为零；默认二者都为 1，因此在后续优势归一化之前，末尾结果奖励对所有有效 token
的优势贡献相同。正常结束的回答以零 bootstrap，达到长度上限的回答使用额外的末端状态
value。若 mask_no_eos_with_zero: true，截断回答的任务奖励会被清零，但 KL 和 critic bootstrap
仍保留。

gae_timestep_unit: turn 可将每个非空生成 turn 当作时间步。该模式依赖
Trajectory.turn_ids；仓库自带的 GSM8K rollout 将整条回答标记为 turn 0，多轮轨迹需要
自定义该字段。turn 内任务奖励求和，turn 首个有效 token 对应状态的 value 用于递推，
优势和 return 再广播回该 turn 的有效 token。KL 惩罚作为 token 局部项加回 actor 优势，
不参与 turn 级 return 的计算。

### 奖励归一化与优势归一化不是一回事

actor.reward_norm 在 GAE 之前处理每条回答的标量结果奖励，决定回答之间如何比较；
actor.adv_norm 在 GAE 之后处理有效 token 的优势，主要控制最终梯度尺度。

二者都使用 NormConfig：

| 字段 | 可选值 | 作用 |
| --- | --- | --- |
| mean_level | group、batch、null | 在指定范围减均值 |
| std_level | group、batch、null | 在指定范围按标准差缩放 |
| mean_leave1out | true、false | 计算均值时排除当前回答 |
| std_unbiased | true、false | 是否使用样本标准差修正 |
| eps | 正浮点数 | 加在标准差后的稳定项 |

将整个字段设为 null 会跳过对应阶段；只将 mean_level 或 std_level 设为 null，则
仅关闭中心化或缩放。优势的 batch/group 统计只包含 loss_mask 为真的 token。

### PPO 目标和权重聚合

普通 loss_type: ppo 最小化负的裁剪 surrogate：

$$
\mathcal L(\theta)
=-\operatorname{Reduce}\left[
\min\!\left(\rho_{i,t}A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon_-,1+\epsilon_+)A_{i,t}\right)
\right].
$$

eps_clip 决定下界宽度；eps_clip_higher: null 时上界宽度与它相同。裁剪限制的是继续
沿有利方向提高 surrogate 所带来的收益，并不保证训练后的实际 ratio 都落在区间内。
c_clip > 1 可为负优势项增加 dual clip。

Reduce 决定谁在 batch 中拥有同等权重：

| loss_aggregation | 权重含义 |
| --- | --- |
| token_mean | 每个有效 token 等权；默认值 |
| seq_mean | 先在回答内部平均，再让回答等权 |
| prompt_mean | 让每道题的整组有效 token 总权重相同 |
| constant | 总和除以有效回答数与 loss_aggregation_divisor 的乘积 |

长回答在 token_mean 中天然占更大权重，而 seq_mean 会消除这种长度权重。选择聚合方式
应与实验希望平均的对象一致，不应只把它当作数值缩放参数。prompt_mean 在 minibatch
切分前保留原题目组权重；空组贡献为零。constant 要求 divisor 为正数。

### 固定参考模型 KL

actor.kl_ctl 和 actor.reference_kl_coef 默认都为 0。两者均为零时不会请求 reference
评分；任一系数启用后，trainer 使用 reference.name，未配置 reference 时使用 actor
的未修改基座模型。kl_estimator 的 k1、k2、k3 用于 KL-as-reward。

token GAE 会把 KL 惩罚加入 token 奖励后再递推。turn GAE 则只用任务奖励计算 critic
target，KL 作为 token 局部项加回 actor advantage，避免将细粒度 KL 累加成 turn return。

reference_kl_coef 启用独立的低方差 K3 actor objective：

$$
d_t = \log \pi_\theta(a_t|s_t)-\log \pi_{\mathrm{ref}}(a_t|s_t),
\qquad
K_3(d_t)=e^{-d_t}+d_t-1.
$$

custom loss 会在每个 minibatch 将 reference_kl_coef * weighted_mean(K3) 直接加到策略
目标。均值使用 loss_mask 和有效 action weight，不会修改奖励、冻结的 advantage 或 critic
return。该路径要求 loss_backend: custom，且不能与 kl_ctl 同时启用；PyTRIO 的自定义
logprob 梯度桥已经足够，不需要完整词表 logits 或服务端 loss 改动。指标记录为
policy/reference_kl 和 policy/reference_kl_loss。

## 训练规模与执行边界

下面这些参数最容易影响实验成本或语义：

| 参数 | 默认值 | 实际含义 |
| --- | --- | --- |
| training.batch_size | 1 | 每轮取多少道题，不是设备 microbatch |
| rollout.num_generations | 4 | 每道题生成多少条回答 |
| rollout.max_tokens | 512 | 单条回答 token 上限，不含 prompt |
| actor.update_epochs | 1 | 同一批冻结 actor 目标复用几次 |
| actor.ppo_n_minibatches | 1 | actor 切成几份更新，题目组不会被拆开 |
| actor.loss_backend | custom | 使用完整本地目标；native 只支持普通 token PPO 子集 |

当前 PPO logprob 契约要求 rollout.temperature: 1.0 和 top_p: 1.0。训练端只拿到目标
token 的 logprob，无法重建温度变化或 nucleus 截断后的完整归一化分布，因此其他值会被
配置校验拒绝。

采样由题目并发执行。每轮先固定 rollout 数据，再更新 actor 和可选 critic；不同 rollout
step 仍按顺序推进。模型部署、分布式执行和 optimizer 由 PyTRIO 管理，Triotune 配置中
不包含其他框架的集群后端或 Hydra override 语义。

## 自定义轨迹与手动生命周期

除了 train() 管理的完整流程，也可以将准备好的轨迹交给 step()：

```python
from triotune.trainer.ppo import Trajectory

metrics = await trainer.step([
    Trajectory(
        prompt_tokens=[...],
        completion_tokens=[...],
        logprobs=[...],
        reward=1.0,
        group_id="prompt-0",
        values=[...],  # critic 开启时长度为 completion 长度 + 1
    )
])
```

loss_mask、reference_logprobs、token_rewards 和 turn_ids 都与 completion token
对齐。调用 step() 前先执行一次 initialize()；手动管理生命周期时，还要自行调用
save_checkpoint() 和 close()。同一个 trainer 不应并发执行多个 step()。

评估由 evaluation.steps、before_train、after_train、batch_size 和 max_batches
控制。checkpoint.save_every 按 rollout step 计数，设为 0 表示只保存最终 checkpoint。
actor 与 critic URI 分别记录在 checkpoint/actor_path 和
checkpoint/critic_path。

日志指标按 reward/*、train/*、policy/*、critic/*、eval/*、time/* 和
usage/* 分组。usage/actor_train_tokens 与 usage/critic_train_tokens 包含重复 epoch
以及准备 logprob/value 的前向计算；usage/train_tokens 是二者之和，带 _total 的指标
表示跨 rollout step 累计。

## 常见配置误区

- **GRPO 没有相对奖励：**检查是否误把 PPO 文件中的 actor.reward_norm: null 覆盖到了
  GRPO preset。
- **组内奖励总是零：**二值奖励下，一组回答可能全对或全错；增加回答多样性，或使用
  DAPO 动态采样过滤无信息组。
- **optimizer step 比 max_steps 多：**这是 update epoch 和 minibatch 展开后的预期结果。
- **GSPO 仍偏向长回答：**sequence ratio 与 loss 聚合是两个维度；需要回答等权时使用
  seq_mean。
- **开启 rejection 后配置报错：**同时设置 use_decoupled_loss: true，并确认下阈值不大于
  上阈值。
- **切换算法后 critic 仍在训练：**检查显式 critic.enabled 是否覆盖了 preset。
- **所有 token 都被 mask：**这一轮没有可用的策略梯度，trainer 会拒绝执行更新，而不是
  静默产生零梯度 step。

实现细节可继续查看[配置定义](../../triotune/api/cli_args.py)、
[PPO trainer 组件](../../triotune/trainer/ppo/)和
[GSM8K PPO 配方](../../examples/ppo/gsm8k_ppo_args.yaml)。
