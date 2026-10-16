# Trainer 模块

## 目录与职责


```text
trio_cookbook/
├── api/
│   ├── cli_args.py      # 公共配置与算法参数契约
│   └── client_api.py    # Training/Sampling client 调用契约
├── dataset/
│   ├── prompt.py        # 文本、token 和聊天输入渲染
│   ├── prepare.py       # 固定行格式的校验、分词、mask 与长度处理
│   ├── examples.py      # 各算法共享的标准化训练样本
│   └── swe_sft/         # 软件工程智能体轨迹的 SFT 数据处理
├── utils/
│   ├── checkpoint.py    # checkpoint 命名、保存和响应校验
│   ├── dataloader.py    # 流式数据集的 batch 迭代
│   ├── datums.py        # 通用 PyTRIO Datum 外壳构造
│   └── metrics.py       # 指标规范化、窗口聚合与 SwanLab 提交
└── trainer/
    ├── __init__.py      # 公开导出与历史模块路径兼容
    ├── grpo_trainer.py  # GRPO 训练编排
    ├── dpo_trainer.py   # DPO 训练编排
    ├── sft_trainer.py   # SFT 训练编排
    ├── opd_trainer.py   # OPD 训练编排
    ├── grpo/
    │   ├── contracts.py
    │   ├── advantages.py
    │   ├── rewards.py
    │   ├── rollout.py
    │   ├── datum.py
    │   └── metrics.py
    ├── dpo/
    │   ├── state.py
    │   └── loss.py
    ├── sft/
    │   └── state.py
    └── opd/
        ├── state.py
        └── distillation.py
```

| 模块 | 职责 |
| --- | --- |
| [公开入口](./__init__.py) | 导出 trainer、配置和类型，并集中映射历史的平铺模块路径。 |
| [配置契约](../api/cli_args.py) | 定义 `BaseTrainerConfig` 以及 GRPO、DPO、SFT、OPD 配置；集中校验优化器、运行控制和日志字段。 |
| [客户端契约](../api/client_api.py) | 统一 TrainingClient、SamplingClient 的同步／异步调用、future 解析及 SDK 参数构造；client 刷新时机由算法决定。 |
| [输入渲染](../dataset/prompt.py) | 校验文本、token 列表和聊天消息，统一调用 tokenizer。 |
| [数据处理](../dataset/prepare.py) | 将固定格式的数据行统一处理成 prompt、监督序列或偏好对，并完成 token、mask、EOS 与长度校验。 |
| [训练样本](../dataset/examples.py) | 提供 `PromptExample`、`SupervisedExample` 和 `PreferenceExample`，隔离数据准备与 PyTRIO 远端客户端。 |
| [SWE 轨迹](../dataset/swe_sft/) | 将包含工具调用、工具结果和多轮决策的软件工程智能体轨迹转换为 PyTRIO SFT 数据。 |
| [指标公共实现](../utils/metrics.py) | 接收算法产生的任意数值指标，统一完成校验、命名、窗口聚合、SwanLab 提交、文本记录和 run 结束。 |
| [保存公共实现](../utils/checkpoint.py) | 同步／异步保存与路径校验；保存频率及状态更新由各算法决定。 |
| [批处理公共实现](../utils/dataloader.py) | 对 dataset 输出的样本做惰性 batch 切分，并限制验证 batch 数；不负责分词或 Datum。 |
| [Datum 公共实现](../utils/datums.py) | 将已完成算法对齐的 token 与 loss inputs 物化为 PyTRIO Datum；不决定 GRPO/OPD 的训练字段。 |
| [GRPO 编排](./grpo_trainer.py) | 组织采样、奖励、advantage、Datum、远端更新以及同步／异步训练。 |
| [GRPO 奖励](./grpo/rewards.py)、[优势](./grpo/advantages.py) | 适配奖励函数、聚合加权奖励，计算组内 centered advantage。 |
| [GRPO 采样](./grpo/rollout.py)、[对齐](./grpo/datum.py)、[指标](./grpo/metrics.py) | 解析分组采样结果，完成 token shift，计算 rollout 统计和策略变化估计。 |
| [DPO 编排](./dpo_trainer.py)、[损失](./dpo/loss.py) | trainer 调用通用偏好数据处理，并负责 Datum、固定参考评分、远端反向；损失模块实现成对目标与算法指标。 |
| [SFT 编排](./sft_trainer.py) | 调用通用 SFT 数据处理，然后构造交叉熵 Datum、提交流水线并按批次顺序发布结果。 |
| [OPD 编排](./opd_trainer.py)、[蒸馏](./opd/distillation.py) | 组织学生生成和教师评分；蒸馏模块负责评分校验、token advantage 和 Datum 对齐。 |

```text
dataset/ 单条样本 → utils/dataloader.py batch → 算法对齐 → utils/datums.py Datum
```

`dataset/` 负责从 mapping、JSONL 或已有 iterable 加载原始行，并转换成标准样本；batch 切分、Datum 外壳构造属于运行时
数据管线。GRPO/OPD 的 logprob、advantage、teacher score 和 prompt mask 仍由算法
模块决定。



## Trainer 输入数据格式

```python
# GRPO / OPD：每行传入一个生成 prompt。
# 字段名默认是 messages，可分别通过 GRPOConfig.prompt_key 和
# OPDConfig.prompt_key 修改。字段值支持纯文本、token ID 或聊天消息。
prompt_rows = [
    {"messages": "Explain why the sky is blue."},
    {"messages": [151644, 8948, 198]},
    {
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
    },
]

# DPO：每行传入共享 prompt、优选回答和拒绝回答。
# messages 支持纯文本、token ID 或聊天消息；chosen/rejected 支持纯文本、
# token ID，或与聊天 prompt 配套的 assistant 消息列表。
preference_rows = [
    {"messages": "What is 2 + 2?", "chosen": "4", "rejected": "5"},
    {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
    },
    {"messages": [101, 102], "chosen": [103, 104], "rejected": [105, 106]},
]

# SFT：同一数据集可使用以下任一种行格式。
sft_text_rows = [{"text": "Question: 2 + 2? Answer: 4"}]
sft_chat_rows = [
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
]
sft_pair_rows = [{"prompt": "What is 2 + 2?", "completion": "4"}]
sft_token_rows = [
    {
        "input_ids": [101, 102, 103, 104],
        "labels": [-100, -100, 103, 104],
        "loss_mask": [0, 0, 1, 1],
    }
]
```

## Dataset 向 Trainer 传输的标准格式

```python
from trio_cookbook.dataset import prepare_rows

# GRPO / OPD
prompt_examples = prepare_rows(prompt_rows, tokenizer, format="prompt")
# PromptExample(text: str, tokens: list[int])

# SFT
supervised_examples = prepare_rows(sft_pair_rows, tokenizer, format="sft")
# SupervisedExample(input_ids: list[int], labels: list[int])
# labels 中的 -100 表示该 token 不参与 loss。
# 发送给 PyTRIO 时：
# model_input = input_ids[:-1]
# target_tokens = labels[1:]

# DPO
preference_examples = prepare_rows(
    preference_rows,
    tokenizer,
    format="preference",
)
# PreferenceExample(prompt: list[int], chosen: list[int], rejected: list[int])
# chosen/rejected 会分别拼接 prompt，并将 prompt 对应标签设为 -100。
```

## 数据集 JSONL 传输示例

```jsonl
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
{"messages":[{"role":"user","content":"What is 3 + 3?"},{"role":"assistant","content":"6"}]}
```


## 检查数据集配置

```python
from trio_cookbook.trainer import SFTTrainer
from trio_cookbook.dataset import SWESFTConfig, prepare_swe_sft_dataset

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

## SFT 训练

```python
from trio_cookbook.trainer import EvaluationConfig, SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3.5-4B",
    args=SFTConfig(
        evaluation=EvaluationConfig(
            before_train=True,
            every_steps=100,
            after_train=True,
            batch_size=None,  # None 表示复用训练 batch_size。
            max_batches=20,
        )
    ),
    train_dataset=sft_pair_rows,
    valid_dataset=sft_pair_rows,  # 周期验证时必须可以重新迭代。
)
```

## GRPO 训练

```python
from trio_cookbook.trainer import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model_name="Qwen/Qwen3.5-4B",
    reward_funcs=my_reward,
    args=GRPOConfig(num_generations=4, max_steps=100),
    train_dataset=prompt_rows,
)
state = trainer.train()
```

## DPO 训练

```bash
uv sync --extra dpo
```

```python
from trio_cookbook.trainer import DPOConfig, DPOTrainer

trainer = DPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=DPOConfig(beta=0.1, batch_size=1, max_steps=1),
    train_dataset=preference_rows,
)
state = trainer.train()
```

## OPD 训练

```python
from trio_cookbook.trainer import OPDConfig, OPDTrainer

trainer = OPDTrainer(
    model="student-base-model-id",
    teacher_model="teacher-base-model-id",
    # teacher_model_path="trio://...",  # 可选的教师 LoRA checkpoint。
    args=OPDConfig(batch_size=4, max_steps=100),
    train_dataset=prompt_rows,
)
state = trainer.train()
```
