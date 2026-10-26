# Trainer 模块

PPO 算法族的组件位于 [`ppo/`](./ppo/)：上一级的 `rl_trainer.py` 负责同一异步训练循环，配置可切换
PPO、GRPO、RLOO、Dr.GRPO、LitePPO、GSPO、SAPO、CISPO 和 DAPO。
其算法配置及模块实现保存在该目录；标准 PPO 默认创建独立的 PyTRIO actor/critic 训练客户端，详见接口能力说明。

多轮工具智能体使用 [`triotune.agentrl`](../agentrl/__init__.py) 中的 `AgentTrainingLoop`，
通过公开的 `PPOTrainer.create_sampling_snapshot()`、`step()` 和 `publish_metrics()`
接入相同训练核心。执行、动作解析、工具、评分与轨迹导出接口均由
`triotune.agentrl` 子包公开。
本地应用和接口说明见 `output/agentrl/` 中的计算器与 Agent RL 文档。

## 目录与职责


```text
triotune/
├── api/
│   ├── cli_args.py      # 公共配置与算法参数契约
│   └── client_api.py    # Training/Sampling client 调用契约
├── dataset/
│   ├── prompt.py        # 文本、token 和聊天输入渲染
│   ├── prepare.py       # 固定行格式的校验、分词、mask 与长度处理
│   └── examples.py      # 各算法共享的标准化训练样本
├── utils/
│   ├── checkpoint.py    # checkpoint 命名、保存和响应校验
│   ├── dataloader.py    # 流式数据集的 batch 迭代
│   ├── datums.py        # 通用 PyTRIO Datum 外壳构造
│   ├── reporting.py     # 兼容旧调用的进程隔离 secondary-run reporter
│   ├── scoring.py       # 固定 completion 的 logprob 评分与对齐
│   └── metrics.py       # 指标规范化、JSONL 历史、窗口聚合与 SwanLab 提交
└── trainer/
    ├── __init__.py      # 公开导出与历史模块路径兼容
    ├── dpo_trainer.py   # DPO 训练编排
    ├── sft_trainer.py   # SFT 训练编排
    ├── opd_trainer.py   # OPD 训练编排
    ├── rl_trainer.py    # PPO-family 训练编排
    ├── dpo/
    │   ├── state.py          # trainer 状态与 step 结果
    │   ├── datum.py          # 偏好 completion 的因果对齐与 mask
    │   └── loss.py           # 成对 DPO 目标、隐式奖励与评估指标
    ├── sft/
    │   └── state.py
    ├── opd/
    │   ├── contracts.py      # OPD 阶段数据与执行协议
    │   ├── execution.py      # 不可变执行计划与并发事务
    │   ├── rollout.py        # 学生 on-policy 采样
    │   ├── teacher_phase.py  # 路由固定 teacher 并聚合 token score
    │   ├── loss.py           # sampled reverse-KL 信号与远端 surrogate
    │   ├── datum.py          # next-token 对齐与 prompt mask
    │   └── targets.py        # teacher score 到训练 target 的转换
    └── ppo/                  # PPO-family 算法组件
        └── trajectory.py     # PPO 轨迹、准备批次和 trainer 状态
```

| 模块 | 职责 |
| --- | --- |
| [公开入口](./__init__.py) | 导出 trainer、配置和类型，并集中映射历史的平铺模块路径。 |
| [配置契约](../api/cli_args.py) | 按模型、训练／验证数据源、训练循环、优化器、保存、评估、日志、rollout、actor/critic、reference/teacher 等职责定义可组合配置；DPO、SFT、OPD、PPO 顶层对象可表示完整实验，同时保留依赖注入兼容性。 |
| [客户端契约](../api/client_api.py) | 统一 TrainingClient、SamplingClient 的同步／异步调用、future 解析及 SDK 参数构造；client 刷新时机由算法决定。 |
| [输入渲染](../dataset/prompt.py) | 校验文本、token 列表和聊天消息，统一调用 tokenizer。 |
| [数据处理](../dataset/prepare.py) | 将固定格式的数据行统一处理成 prompt、监督序列或偏好对，并完成 token、mask、EOS 与长度校验。 |
| [训练样本](../dataset/examples.py) | 提供 `PromptExample`、`SupervisedExample` 和 `PreferenceExample`，隔离数据准备与 PyTRIO 远端客户端。 |
| [指标公共实现](../utils/metrics.py) | 接收算法产生的任意数值指标，统一完成校验、命名、`log_dir` JSONL 历史、窗口聚合、SwanLab 提交、checkpoint URI 记录和 run 结束。 |
| [保存公共实现](../utils/checkpoint.py) | 同步／异步保存与路径校验；保存频率及状态更新由各算法决定。 |
| [批处理公共实现](../utils/dataloader.py) | 对 dataset 输出的样本做惰性 batch 切分，并限制验证 batch 数；不负责分词或 Datum。 |
| [Datum 公共实现](../utils/datums.py) | 将已完成算法对齐的 token 与 loss inputs 物化为 PyTRIO Datum；不决定 GRPO/OPD 的训练字段。 |
| [评分公共实现](../utils/scoring.py) | 通过 sampling client 为固定 completion 计算 token logprob，并统一校验完整序列与 completion 的位置对齐；DPO、OPD 和 PPO 可复用。 |
| [DPO 编排](./dpo_trainer.py)、[Datum](./dpo/datum.py)、[损失](./dpo/loss.py) | trainer 调用通用偏好数据处理并负责固定参考评分与远端更新；算法模块分别实现偏好 completion 的因果对齐、成对目标和训练／评估指标。 |
| [SFT 编排](./sft_trainer.py) | 调用通用 SFT 数据处理，然后构造交叉熵 Datum、提交流水线并按批次顺序发布结果。 |
| [OPD 编排](./opd_trainer.py)、[执行计划](./opd/execution.py) | Trainer 负责数据集、client、指标归约、update、checkpoint 和日志生命周期；不可变执行计划组合可替换的 rollout、teacher、target 与 loss 阶段。 |
| [PPO-family 编排](./rl_trainer.py)、[算法组件](./ppo/) | 组织 rollout、reference/value、冻结目标、actor/critic 并发更新、checkpoint 和日志生命周期；包括 GRPO 预设。 |
| [OPD rollout](./opd/rollout.py)、[teacher phase](./opd/teacher_phase.py) | 学生按当前 behavior policy 采样；路由组内的固定 teacher 给同一 token 序列评分，并聚合为加权几何 ensemble。当前可跨 prompt 重叠调用，但接口不依赖训练／推理 colocate。 |
| [OPD loss](./opd/loss.py)、[target](./opd/targets.py)、[Datum](./opd/datum.py) | 计算 `log p_teacher - log p_behavior` token 信号，可选 discounted credit，保留 behavior logprob，并构造 PyTRIO policy loss 所需的 next-token 对齐字段；默认通过 CISPO 将 IS ratio 单侧截断到 5.0。 |
| [OPD contracts](./opd/contracts.py) | 隔离 SDK response 并校验阶段间数据；OPD trainer 以 completion token 为分母归约蒸馏指标，验证阶段使用流式 totals，不保留完整远端 Datum 列表。 |

SFT、DPO、OPD 和 PPO-family 的完整训练入口均显示 train step 进度；所有 evaluation
入口按 batch 显示独立进度。数据源可求长度时展示准确总 batch 数，流式 iterable 则展示已完成 batch 与速率。

指标按其自然样本数聚合：语言模型损失按有效 token、DPO 按 pair、PPO policy/critic
按有效 action、OPD 蒸馏指标按 completion token 加权；标准差通过一二阶矩合并，而不是
直接平均各 batch 的标准差。PPO/OPD 的 KL 与 entropy 字段带有 `sampled` 时仅表示已采样
action 上的估计，不代表完整词表分布统计。
SFT 始终从交叉熵报告 perplexity；`train/token_accuracy` 仅在 PyTRIO 后端返回
`token_accuracy`/`accuracy` 时透传，因为目标 token logprob 本身不足以判断全词表 argmax。

```text
dataset/ 单条样本 → utils/dataloader.py batch → 算法对齐 → utils/datums.py Datum
```

`dataset/` 负责从 mapping、JSONL 或已有 iterable 加载原始行，并转换成标准样本；batch 切分、Datum 外壳构造属于运行时
数据管线。PPO-family/OPD 的 logprob、advantage、teacher score 和 prompt mask 仍由算法
模块决定。



## Trainer 输入数据格式

```python
# PPO-family / OPD：每行传入一个生成 prompt。
# 字段名默认是 messages，可分别通过 PPOConfig.rollout.prompt_key 和
# OPDConfig.rollout.prompt_key 修改。字段值支持纯文本、token ID 或聊天消息。
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
from triotune.dataset import prepare_rows

# PPO-family / OPD
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


## SFT 训练

```python
import asyncio

from triotune.trainer import EvaluationConfig, SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3.5-4B",
    args=SFTConfig(
        evaluation=EvaluationConfig(steps=50, after_train=False),
    ),
    train_dataset=train_rows,
    eval_dataset=eval_rows,
)
state = asyncio.run(trainer.train())
```

## GRPO 训练

```python
import asyncio

from triotune.trainer import PPOConfig, PPORolloutConfig, PPOTrainer, TrainingConfig

trainer = PPOTrainer(
    model="Qwen/Qwen3.5-4B",
    reward_fn=my_reward,
    args=PPOConfig.for_algorithm(
        "grpo",
        training=TrainingConfig(max_steps=100),
        rollout=PPORolloutConfig(num_generations=4),
    ),
    train_dataset=prompt_rows,
)
state = asyncio.run(trainer.train())
```

## DPO 训练

```bash
uv sync --extra dpo
```

```python
from triotune.trainer import (
    DPOConfig,
    DPOObjectiveConfig,
    DPOTrainer,
    TrainingConfig,
)

trainer = DPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=DPOConfig(
        training=TrainingConfig(batch_size=1, max_steps=1),
        objective=DPOObjectiveConfig(beta=0.1),
    ),
    train_dataset=preference_rows,
)
state = trainer.train()
```

## OPD 训练

```python
from triotune.trainer import OPDConfig, OPDTrainer, TrainingConfig

trainer = OPDTrainer(
    model="student-base-model-id",
    teacher_model="teacher-base-model-id",
    # teacher_model_path="trio://...",  # 可选的教师 LoRA checkpoint。
    args=OPDConfig(training=TrainingConfig(batch_size=4, max_steps=100)),
    train_dataset=prompt_rows,
)
state = trainer.train()
```

默认 OPD 执行链是：

```text
student rollout
  → routed fixed-teacher token scoring
  → weighted geometric teacher ensemble
  → sampled reverse-KL token advantage
  → prompt-masked PyTRIO policy Datum
  → CISPO capped-importance-ratio forward/backward
  → optimizer step
```

单 teacher 继续使用上面的兼容接口。多 teacher 通过显式 spec 和 group 配置，
数据行只携带组名，不接触模型 client 或算法字段：

```python
from triotune.trainer import OPDConfig, OPDTrainer
from triotune.trainer.opd import OPDTeacherGroup, OPDTeacherSpec

teachers = [
    OPDTeacherSpec("math", "org/math-teacher", "checkpoint://math-lora"),
    OPDTeacherSpec("code", "org/code-teacher"),
]
groups = [
    OPDTeacherGroup("reasoning", {"math": 2.0, "code": 1.0}),
    OPDTeacherGroup("code-only", {"code": 1.0}),
]
rows = [
    {"messages": "Solve this proof.", "teacher_group": "reasoning"},
    {"messages": "Implement a parser.", "teacher_group": "code-only"},
]
trainer = OPDTrainer(
    model="student-base-model-id",
    teachers=teachers,
    teacher_groups=groups,
    args=OPDConfig(training=TrainingConfig(batch_size=2, max_steps=100)),
    train_dataset=rows,
)
```
