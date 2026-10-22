# GRPO 案例——gms8k

[English](grpo_series.md)

Triotune 使用同一个异步 PPOTrainer 运行 GRPO 算法：选择配置预设后，
即可复用数据、奖励和训练接口。GRPO 通过同一道题的多个回答之间的相对奖励学习，无需 critic。预设参数配置可见[算法配置矩阵](#算法配置矩阵)。


## 运行示例

### 环境安装

在训练开始前需要安装好环境，在 triotune 中我已经准备好合适的 uv 环境，可以直接运行下面的代码实现快速安装👇：

```bash
git clone https://github.com/SwanHubX/triotune.git
cd triotune
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

### 快速启动

运行如下命令可以直接在 pytrio 上启动训练👇：

```bash
uv run python examples/grpo/gsm8k_grpo.py --config examples/grpo/gsm8k_grpo_args.yaml
```

运行训练命令前，先完成[准备 GSM8K 数据](#准备-gsm8k-数据)。


### 配置接口

| 接口 | 用途 |
| --- | --- |
| PPOConfig.for_algorithm("grpo", **modules) | 先应用算法预设，再应用显式覆盖 |
| PPOConfig.from_modules(settings) | 读取模块化配置字典，支持 algorithm 选择器 |
| PPOConfig(**settings) | 直接读取完整、显式的配置 |


显式配置优先于预设：若保留 PPO 文件中的 critic.enabled: true 或
actor.reward_norm: null，就会改变预期的 GRPO 设置。例如，在当前项目中可以通过下面的配置使用 GRPO ：

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


## 核心配置参数

### 训练与 rollout

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| model.name、model.lora_rank | null、32 | PyTRIO 基座模型 ID 和 actor LoRA rank；训练前必须提供模型。 |
| optimizer.learning_rate | 1e-5 | actor AdamW 学习率；还支持 beta1、beta2、eps、weight_decay、grad_clip_norm。 |
| rollout.num_generations、max_tokens | 4、512 | 每题生成的回答数，以及回答长度上限。 |
| rollout.temperature、top_p | 1.0、1.0 | 两者必须为 1；仅有目标 token 的训练 logprob 无法还原修改后的采样概率。 |
| actor.update_epochs、ppo_n_minibatches | 1、1 | 复用同一批冻结的 rollout 目标，以及按完整题目组划分 actor 更新。 |
| actor.loss_backend | custom | 通过 PyTRIO 使用完整的本地 Torch loss 回调；native 只支持受限的普通 token 级 PPO。 |

max_steps 统计 rollout 更新次数，不是 optimizer 次数。训练只遍历传入的数据一次，
数据耗尽会提前结束；update_epochs 表示重复使用冻结的 rollout 目标，不是数据集 epoch。
actor 的 minibatch 数不能超过实际题目组数。

数据路径、样本上限和 shuffle 由示例的数据加载代码负责；当前 GSM8K 入口会应用
valid_dataset.max_examples，但不会应用 train_dataset.max_examples，也不会打乱训练数据。

### 准备 GSM8K 数据

会读取本地 JSONL，不会自动下载数据。要求数据集中每行至少包含 `messages` 和 `answer`。以下 Python 代码准备官方训练集和测试集，要求模型把最终数字答案放入 `\boxed{}`，已有文件会直接复用。

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

这段准备代码使用单条 user 消息；已有实验数据可能包含 few-shot 示例。
比较算法时请使用相同的数据文件。示例奖励函数检查最后一个 boxed 答案，正确为 1，否则为 0。

JSONL 文件每行保存一条记录，最小格式如下：

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "answer": "2"
}
```


### 奖励与优势归一化（actor.reward_norm、actor.adv_norm）

两者可以分别设为 null 关闭，或传入 NormConfig 配置归一化方式；在 YAML 中填写对应字段即可。

- reward_norm 设为 null：不对每个回答的结果奖励减均值或做标准差缩放，奖励仍会经过 reward_bias、reward_scaling 和 reward_clip 处理，再参与后续优势计算。
- adv_norm 设为 null：保留 GAE 计算出的有效回答 token 的优势值，不再额外减均值或缩放。关闭归一化不会关闭 GAE、critic 或已配置的 KL 惩罚。

传入 NormConfig 时，reward_norm 在计算优势前处理每个回答的标量结果奖励，
adv_norm 在 GAE 之后处理有效回答 token 的优势值。两者都会按 mean_level 指定的范围
减去均值，再按 std_level 指定的范围计算缩放量，并以缩放量加 eps 作为除数。
例如，下面对奖励使用同一道题回答组内的统计量，对优势使用整个 rollout batch 的统计量：

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

mean_level 和 std_level 可以分别选择 group、batch、null。
group 表示同一道题的回答组，batch 表示本次 rollout 的全部回答；优势统计只使用有效回答 token。

mean_level 为 null 时不减均值，std_level 为 null 时不做尺度缩放。
mean_leave1out: true 在计算均值时排除当前值；对每个回答只有一个标量的结果奖励而言，就是排除当前回答。

std_unbiased: true 使用样本标准差的修正，eps 用于避免缩放时分母为零。

所有预设默认保留 batch/batch 优势归一化，除非显式覆盖。直接构造 PPOConfig() 默认关闭奖励归一化，GRPO 预设则会开启。

### GRPO 的组内相对优势

对同一道题 $q$ 采样 $G$ 个回答，$G$ 至少为 2。令 $R_i$ 为第 $i$ 个回答经过奖励偏移、缩放和裁剪后的结果奖励。
在 mean_leave1out: false、std_unbiased: true 的默认设置下，组均值、样本标准差和相对奖励为：

$$
\mu_q = \frac{1}{G}\sum_{j=1}^{G}R_j,
\qquad
s_q = \sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(R_j-\mu_q)^2},
\qquad
a_i = \frac{R_i-\mu_q}{s_q+\eta}.
$$

其中 $\eta$ 对应 reward_norm.eps。高于组均值的回答得到正相对奖励，低于组均值的回答得到负相对奖励；
组内奖励完全相同时，相对奖励为 0。std_unbiased: false 时，方差的分母由 $G-1$ 改为 $G$。

在当前 GSM8K GRPO 示例中，不使用 critic，discount 和 gae_lambda 均为 1，
kl_ctl 为 0，且只在回答末尾给予结果奖励。因此，额外优势归一化之前，每个有效回答 token 的优势都等于 $a_i$。
示例还开启了 batch/batch 优势归一化，最终用于策略更新的是：

$$
\hat{A}_{i,t} = \frac{a_i-\mu_A}{s_A+\eta_A}.
$$

这里 $\mu_A$ 和 $s_A$ 是整个 rollout batch 中有效回答 token 的优势均值与样本标准差，
$\eta_A$ 对应 adv_norm.eps。统计时每个有效 token 都参与计算，因此回答长度会影响统计权重。
将 adv_norm 设为 null 后，不执行这一步，此时最终优势就是 $a_i$。

### 裁剪策略（actor.eps_clip*）

GRPO 复用 PPO 的裁剪策略目标。当前示例未开启重算、解耦或拒绝采样，token 的重要性采样比率为：

$$
\rho_{i,t}(\theta)
= \frac{\pi_\theta(o_{i,t}\mid q_i,o_{i,<t})}
       {\pi_{\mathrm{old}}(o_{i,t}\mid q_i,o_{i,<t})}
= \exp\!\left(\log\pi_\theta(o_{i,t}\mid q_i,o_{i,<t})
              -\log\pi_{\mathrm{old}}(o_{i,t}\mid q_i,o_{i,<t})\right).
$$

$\pi_\theta$ 表示正在更新的 actor，$\pi_{\mathrm{old}}$ 表示采样回答时的旧策略；$q_i$ 是题目，$o_{i,t}$ 是回答中的第 $t$ 个 token。
eps_clip: 0.2 对应裁剪区间 [0.8, 1.2]；eps_clip_higher: 0.28 将上界改为 1.28。
令 $\epsilon_-$ 对应 eps_clip，$\epsilon_+$ 在 eps_clip_higher 为 null 时也取 eps_clip，否则取 eps_clip_higher。

在 loss_type: ppo、c_clip: null 和默认 token_mean 聚合下，最小化的 actor 损失为：

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

这里 $i$ 遍历本次 rollout batch 的全部回答，$\mathcal{M}_i$ 是第 $i$ 个回答的有效 token 位置集合，$N$ 是有效 token 总数。
正优势鼓励提高对应 token 的概率，负优势鼓励降低概率；裁剪限制继续沿有利方向更新时的目标收益，
不会强制实际 ratio 始终处于裁剪区间内。默认按全部有效 token 平均；若希望每条回答等权，可设置 loss_aggregation: seq_mean。

c_clip 可为负优势启用 dual clipping。loss_type 支持 ppo、sapo、cispo，
上式对应 ppo；sapo 使用平滑门控，cispo 使用停止梯度的裁剪权重，目标形式不同。
当前示例的 kl_ctl 为 0；若启用 KL 惩罚，本实现先将其加入 token 奖励，再通过 GAE 计算优势。

### 重要性采样层级（actor.importance_sampling_level）

默认 token 使用上面的逐 token 比率；sequence 使用保留的有效回答 token 上比率的几何均值：

$$
\rho_i^{\mathrm{seq}}(\theta)
= \exp\!\left(\frac{1}{|\mathcal{K}_i|}
  \sum_{t\in\mathcal{K}_i}\log\rho_{i,t}(\theta)\right)
= \left(\prod_{t\in\mathcal{K}_i}\rho_{i,t}(\theta)\right)^{1/|\mathcal{K}_i|}.
$$

$\mathcal{K}_i$ 表示通过 mask 和可选拒绝阈值后保留的 token 集合，上式针对非空集合。
sequence 模式同时对保留 token 的优势取平均，再用于该回答的目标计算；GSPO 选择此设置。

use_decoupled_loss 区分生成回答的 behavior 策略和用于裁剪比率分母的 proximal 策略；
启用后，上式比率的分母使用 proximal 分数。recompute_logprob 在冻结目标前重新计算 proximal 分数。
可选的 rejection_lower/rejection_upper 根据 proximal 与 behavior 的概率比屏蔽 token，要求 use_decoupled_loss: true。


## 核心概念与接口

### Trainer 生命周期

| 接口 | 功能与约定 |
| --- | --- |
| PPOTrainer(args=config, train_dataset=rows, reward_fn=fn, eval_dataset=valid_rows) | 延迟初始化；也可直接传入 model，或注入客户端、tokenizer、critic。 |
| await trainer.initialize() | 创建 actor、可选 critic、tokenizer 和日志器，只初始化一次。 |
| state = await trainer.train() | 完成采样、评分、更新、评估、最终保存和日志收尾。 |
| metrics = await trainer.step(trajectories) | 更新一批自定义冻结轨迹；同一 trainer 不可并发调用。 |
| await trainer.evaluate() | 对验证题目采样、评分，不计算 reference/critic，也不更新优化器。 |
| await trainer.save_checkpoint() | 保存 actor 和框架管理的 critic 状态，返回 actor URI。 |
| await trainer.close() | 为自定义生命周期结束日志记录。 |

通过 `asyncio.run(...)` 执行异步入口。train() 负责日志清理；
手动调用 step() 时，需要自行管理保存和 close()。
PPOState 包含 global_step、optimizer_steps、critic_optimizer_steps、
metrics、checkpoint_path 和 critic_checkpoint_path。

### 奖励接口

`reward_fn(\*, row, completion, completion_ids)` 支持同步或异步函数，
返回有限标量，或含有限分项的 {"reward": total, "correctness": score, ...}。
分项记录为 reward/\<name\>/mean 和 eval/reward_\<name\>_mean。
GSM8K 示例提供奖励函数；trainer 本身不会解释 answer。
prompt 支持文本、token ID 列表或聊天消息列表。

### GAE 时间单位与自定义轨迹

`actor.discount` 和 `gae_lambda` 默认为 1.0；gae_lambda 为固定标量。
gae_timestep_unit 支持 token 和 turn。turn 模式需要自定义
Trajectory.turn_ids，自带 GSM8K rollout 不会生成此字段。

从 triotune.trainer.ppo 导入 Trajectory，将轨迹列表交给 step()。
必填字段为 prompt_tokens、completion_tokens、behavior logprobs、
标量 reward 和 group_id。可选的 loss_mask、reference_logprobs、
token_rewards、turn_ids 与回答 token 对齐；values 的长度为回答长度 + 1，
多出的元素用于 bootstrap。turn 模式下，有效 token 的整数 turn ID 必须单调不减，
被 mask 的 token 使用 -1。当前不支持动态 lambda 函数接口。

## Triotune 实现说明

- **执行方式：** 题目间并发采样，每次更新先冻结目标，再由独立客户端并发优化 actor 和 critic。
  不同训练 step 之间的 rollout 与策略更新仍顺序执行。
- **KL：** actor.kl_ctl 默认 0；设置正数后，使用冻结 reference 评分
  （reference.name，未指定时使用 actor 基座模型），支持 kl_estimator: k1|k2|k3。
  token 模式将 KL 加入奖励后计算 GAE；turn 模式将其保留为 actor 优势中的 token 局部项。
- **评估：** evaluation.steps、before_train、after_train、batch_size、max_batches
  控制在可重复遍历的验证集上评估奖励；使用当前 rollout 采样配置，包括每题回答数。
- **日志：** reporting.report_to: swanlab 开启上传。actor 和 critic 共用一个实验，
  指标位于 reward/&#42;、train/&#42;、policy/&#42;、critic/&#42;、eval/&#42;、time/&#42;、usage/&#42;。
  部分指标需要启用对应组件或选项。reporting.log_dir 还会写入独立 JSONL 历史；
  PPO 每步发布指标，不受 reporting.logging_step 控制。
- **Token 用量：** usage/actor_train_tokens 和 usage/critic_train_tokens
  分别统计每步各组件处理的输入 token，usage/train_tokens 为两者之和；
  _total 后缀表示跨 step 累计。统计包含重复 epoch 和准备 logprob/value 的前向计算。
  custom loss 的 forward 和 forward-backward 分别计数，native actor loss 计一次。
- **保存：** checkpoint.save_every 按 rollout 更新计数，0 表示只在结束时保存。
  URI 记录为 checkpoint/actor_path 和 checkpoint/critic_path。
  trainer 当前没有内置的 resume/load 入口。
- **边界：** 模型部署和分布式计算由 PyTRIO 管理；AReaL 的集群后端、workflow API、
  Hydra 配置不是 Triotune 的接口。

更多细节见[配置源码](../../triotune/api/cli_args.py)和[算法组件](../../triotune/trainer/ppo/)。
