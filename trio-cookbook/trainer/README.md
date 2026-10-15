# Trainer 模块

`trainer` 包含 PyTRIO 原生 GRPO、DPO、SFT 和 OPD 训练实现。GRPO 将采样、奖励和组内 advantage 对齐为 PyTRIO `Datum`；DPO 使用偏好数据和固定参考策略，在客户端计算成对损失；SFT 用远端交叉熵训练文本样本；OPD 用学生生成的序列及教师评分进行在线蒸馏。GRPO、DPO 和 OPD 提供同步和异步训练路径，SFT 提供异步训练路径。


## 目录与职责


```text
trainer/
├── __init__.py          # 公开导出与历史模块路径兼容
├── common/
│   ├── config.py        # 公共配置契约与各算法配置
│   ├── logging.py       # SwanLab 初始化、指标窗口计算和结束记录
│   └── checkpoint.py    # checkpoint 命名、保存和响应校验
├── dataset/
│   ├── prompt.py        # 文本、token 和聊天输入渲染
│   ├── prepare.py       # 固定行格式的校验、分词、mask 与长度处理
│   ├── examples.py      # 各算法共享的标准化训练样本
│   └── swe_sft/         # 软件工程智能体轨迹的 SFT 数据处理
├── grpo/
│   ├── trainer.py
│   ├── types.py
│   ├── advantages.py
│   ├── rewards.py
│   ├── rollout.py
│   ├── datum.py
│   └── metrics.py
├── dpo/
│   ├── trainer.py
│   ├── types.py
│   ├── loss.py
│   └── metrics.py
├── sft/
│   ├── trainer.py
│   └── types.py
└── opd/
    ├── trainer.py
    ├── types.py
    └── distillation.py
```

| 模块 | 职责 |
| --- | --- |
| [公开入口](./__init__.py) | 导出 trainer、配置和类型，并集中映射历史的平铺模块路径。 |
| [配置契约](./common/config.py) | 定义 `BaseTrainerConfig` 以及 GRPO、DPO、SFT、OPD 配置；集中校验优化器、运行控制和日志字段。 |
| [输入渲染](./dataset/prompt.py) | 校验文本、token 列表和聊天消息，统一调用 tokenizer。 |
| [数据处理](./dataset/prepare.py) | 将固定格式的数据行统一处理成 prompt、监督序列或偏好对，并完成 token、mask、EOS 与长度校验。 |
| [训练样本](./dataset/examples.py) | 提供 `PromptExample`、`SupervisedExample` 和 `PreferenceExample`，隔离数据准备与 PyTRIO 远端客户端。 |
| [SWE 轨迹](./dataset/swe_sft/) | 将包含工具调用、工具结果和多轮决策的软件工程智能体轨迹转换为 PyTRIO SFT 数据。 |
| [日志公共能力](./common/logging.py) | 初始化 SwanLab、累计与平均指标、结束运行；指标命名和记录频率由各算法决定。 |
| [保存公共能力](./common/checkpoint.py) | 同步／异步保存与路径校验；保存频率及状态更新由各算法决定。 |
| [GRPO 编排](./grpo/trainer.py) | 组织采样、奖励、advantage、Datum、远端更新以及同步／异步训练。 |
| [GRPO 奖励](./grpo/rewards.py)、[优势](./grpo/advantages.py) | 适配奖励函数、聚合加权奖励，计算组内 centered advantage。 |
| [GRPO 采样](./grpo/rollout.py)、[对齐](./grpo/datum.py)、[指标](./grpo/metrics.py) | 解析分组采样结果，完成 token shift，计算 rollout 统计和策略变化估计。 |
| [DPO 编排](./dpo/trainer.py)、[损失](./dpo/loss.py)、[指标](./dpo/metrics.py) | trainer 调用通用偏好数据处理，并负责 Datum、固定参考评分、远端反向；其余模块实现成对损失与指标映射。 |
| [SFT 编排](./sft/trainer.py) | 调用通用 SFT 数据处理，然后构造交叉熵 Datum、提交流水线并按批次顺序发布结果。 |
| [OPD 编排](./opd/trainer.py)、[蒸馏](./opd/distillation.py) | 组织学生生成和教师评分；蒸馏模块负责评分校验、token advantage 和 Datum 对齐。 |

`common/` 是 trainer 内部的公共接口与数据契约层，集中放置稳定配置、日志和
checkpoint 能力。算法配置继承 `BaseTrainerConfig`，共享 batch、step、Adam、
保存和报告字段，同时保留各算法原有的公开构造参数。GRPO 的分组采样依赖 `GRPOBatch`，
因此保留在 `grpo/rollout.py`；PyTRIO future 与客户端调用适配继续位于
[utils/client.py](../utils/client.py)。各算法保留自己的同步／异步训练时序，
SFT 的排队保存仍在 trainer 中提交，以保持 SDK 请求顺序。用户输入的校验、
分词、EOS、长度和 loss mask 统一由 `dataset/` 负责；算法目录只保留 Datum
构造及其训练目标特有的对齐。


## 数据输入格式

训练器会根据算法选择对应的数据处理路径。数据模块不会根据内容猜测算法，
因为同一个 `messages` 字段既可能是 SFT 对话，也可能是 GRPO/OPD 的生成
prompt。只要每行符合下面的固定格式，导入 trainer 后就会自动完成校验、分词、
mask 和 PyTRIO next-token 对齐：

| 数据类型 | 使用算法 | 固定字段 |
| --- | --- | --- |
| prompt-only | GRPO、OPD | `messages`，可配置其他 prompt 字段名 |
| 监督文本 | SFT | `text`、`messages`、`prompt`/`completion`，或 `input_ids` |
| 偏好对 | DPO | `messages`、`chosen`、`rejected` |

prompt、completion 和 preference response 可以是字符串、非空整数 token 列表，
或相应的聊天消息。聊天消息中的每一项都必须包含非空字符串 `role` 和字符串
`content`。

```python
rows = [
    {
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "answer": "4",
    }
]
```

### 标准化 token 与 mask


需要在 trainer 外预览或缓存处理结果时，可以直接使用公开数据 API：

```python
from trio_cookbook.trainer.dataset import prepare_rows

prepared = prepare_rows(
    rows,
    tokenizer,
    format="sft",  # 还支持 "prompt" 和 "preference"
    max_length=1024,
)
cached_rows = [example.as_dict() for example in prepared]
```

### 软件工程智能体轨迹


```python
from trio_cookbook.trainer import SFTTrainer
from trio_cookbook.trainer.dataset import SWESFTConfig, prepare_swe_sft_dataset

dataset = prepare_swe_sft_dataset(
    "swe-trajectories.jsonl",
    tokenizer,
    SWESFTConfig(sample_mode="pair", max_length=8192),
)

trainer = SFTTrainer(
    model="Qwen/Qwen3.5-4B",
    train_dataset=dataset,
)
```





## GRPO训练

```python
from trio_cookbook.trainer import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model_name="Qwen/Qwen3.5-4B",
    reward_funcs=my_reward,
    args=GRPOConfig(num_generations=4, max_steps=100),
    train_dataset=rows,
)
trainer.train()
```


## DPO 训练

DPO 需要本机安装 PyTorch 以计算成对损失和梯度：`uv sync --extra dpo`。数据集每行包含 `messages`、`chosen`、`rejected`；`messages` 支持文本、token ID 或聊天消息，回答支持文本、token ID，或与聊天消息匹配的末轮 assistant 消息列表。聊天模板生成的完整对话必须以输入消息的 token 为前缀，否则会明确报错，避免训练目标错位。

```python
from trio_cookbook.trainer import DPOConfig, DPOTrainer

rows = [
    {"messages": "What is 2 + 2?", "chosen": "4", "rejected": "5"},
]
trainer = DPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=DPOConfig(beta=0.1, batch_size=1, max_steps=1, report_to="swanlab"),
    train_dataset=rows,
)
state = trainer.train()
print(state.checkpoint_path)
```

## SFT 训练

`text` 和 `messages` 行默认训练全部 next-token 标签；`prompt`/`completion` 行默认只训练 completion。预分词行可传 `input_ids` 和 `labels`，用 `-100` 屏蔽不参与 loss 的 token。`assistant_only_loss=True` 需要能返回 assistant token mask 的聊天模板。SFT 只支持文本，暂不支持 packing 和多模态数据。

```python
import asyncio

from trio_cookbook.trainer import SFTConfig, SFTTrainer

rows = [{"prompt": "What is 2 + 2?", "completion": "4"}]
trainer = SFTTrainer(
    model="Qwen/Qwen3.5-4B",
    args=SFTConfig(batch_size=1, max_steps=1, save_every=1),
    train_dataset=rows,
)
state = asyncio.run(trainer.train())
print(state.checkpoint_path)
```

## OPD 训练

DeepMath 的完整框架版案例、配置和启动命令见
[OPD 异步脚本说明](../scripts/README.md#deepmath-opd-异步案例)。

OPD（On-Policy Distillation）只需要 prompt 数据。每一步保存当前学生权重并创建采样器，由学生生成 completion，固定教师对完全相同的 prompt + completion token 序列调用 `compute_logprobs`。教师不生成训练答案，也不参与优化。

### 训练目标和对齐

记学生采样时的 token logprob 为 `log p_t`，教师对该 token 的 logprob 为 `log q_t`：

```text
a_t = kl_coef * (log q_t - log p_t)
A_t = a_t + discount_factor * A_(t+1)     # 从后向前计算，末尾未来项为 0
L = -mean_completion[exp(log p_theta,t - log p_t) * stop_gradient(A_t)]
```

默认 `discount_factor=0`，使用每个学生访问到的上下文中的 token 级反向 KL 梯度代理；设为 `1` 时加入后续 token 的完整 KL 信号，用于序列级信用分配。这里是采样 token 的估计，不计算全词表 KL，也不使用教师 top-k logits。advantage 不做 GRPO 式组内中心化，因此每个 prompt 只生成一条回答也能训练。

`Datum.model_input` 为 `(prompt + completion)[:-1]`，`target_tokens` 为 `[-100] * (len(prompt)-1) + completion`。prompt 的 logprob 和 advantage 填零，`-100` 同时将其排除在 PyTRIO 有效 token 的归一化分母之外。教师评分包含整个输入序列（首 token 可以为 `None`），只取 `[len(prompt):]`；学生采样评分必须与 completion 等长。远端使用 `importance_sampling` 和 `auto_shift=False`，只进行一次 next-token shift，无需本地 PyTorch 或自定义 backward。

学生采样固定为 `temperature=1.0, top_p=1.0, top_k=-1`，保证采样评分对应训练时的学生分布。当前 config 不开放温度和截断采样调节；支持这些模式需要另行设计分布修正。生成保留实际采样到的 EOS，不补写 EOS；到达长度上限时训练生成前缀。`max_length` 限制 prompt + completion 长度，不截断 prompt，而是缩减该 prompt 的生成预算；没有剩余空间则报错。

SwanLab 的 `train/loss`（原名 `train/surrogate_loss`）映射远端 `loss_mean`，两者数值相同；PyTRIO 返回的 `loss_mean = loss_sum / token_count`。这是 `importance_sampling` 的代理损失，不是完整词表 KL。`distill/reverse_kl_sampled` 记录采样 token 的学生与教师 logprob 差值均值；蒸馏效果还需结合独立验证集评估。


### 使用示例


```python
from trio_cookbook.trainer import OPDConfig, OPDTrainer

rows = [{"messages": [{"role": "user", "content": "Explain why the sky is blue."}]}]
trainer = OPDTrainer(
    model="student-base-model-id",        # 替换成服务可用的学生模型 ID
    teacher_model="teacher-base-model-id",  # 替换成词表一致的教师模型 ID
    # teacher_model_path="trio://...",    # 可选的教师 LoRA checkpoint
    args=OPDConfig(batch_size=4, max_steps=100, report_to="swanlab"),
    train_dataset=rows,
)
state = trainer.train()
print(state.checkpoint_path)
```
