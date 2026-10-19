# Trainer 模块

PPO 算法族的组件位于 [`ppo/`](./ppo/)：上一级的 `rl_trainer.py` 负责同一异步训练循环，配置可切换
PPO、GRPO、RLOO、Dr.GRPO、LitePPO、GSPO、SAPO、CISPO 和 DAPO。
其算法配置及模块实现保存在该目录；标准 PPO 默认创建独立的 PyTRIO actor/critic 训练客户端，详见接口能力说明。

## 目录与职责


```text
triotune/
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
│   ├── reporting.py     # 进程隔离的 SwanLab secondary-run reporter
│   ├── scoring.py       # 固定 completion 的 logprob 评分与对齐
│   └── metrics.py       # 指标规范化、窗口聚合与 SwanLab 提交
└── trainer/
    ├── __init__.py      # 公开导出与历史模块路径兼容
    ├── dpo_trainer.py   # DPO 训练编排
    ├── sft_trainer.py   # SFT 训练编排
    ├── opd_trainer.py   # OPD 训练编排
    ├── rl_trainer.py    # PPO-family 训练编排
    ├── dpo/
    │   ├── state.py
    │   └── loss.py
    ├── sft/
    │   └── state.py
    ├── opd/
    │   ├── contracts.py      # 阶段数据契约、trainer 状态与 step 结果
    │   ├── execution.py      # 不可变执行计划与并发事务
    │   ├── rollout.py        # 学生 on-policy 采样
    │   ├── teacher_phase.py  # 路由固定 teacher 并聚合 token score
    │   ├── loss.py           # sampled reverse-KL 信号与远端 surrogate
    │   ├── datum.py          # next-token 对齐与 prompt mask
    │   ├── targets.py        # teacher score 到训练 target 的转换
    │   └── metrics.py        # OPD 指标归约
    └── ppo/                  # PPO-family 算法组件
```

| 模块 | 职责 |
| --- | --- |
| [公开入口](./__init__.py) | 导出 trainer、配置和类型，并集中映射历史的平铺模块路径。 |
| [配置契约](../api/cli_args.py) | 定义 `BaseTrainerConfig` 以及 DPO、SFT、OPD、PPO 配置；GRPO 使用 `PPOConfig.for_algorithm("grpo")`。 |
| [客户端契约](../api/client_api.py) | 统一 TrainingClient、SamplingClient 的同步／异步调用、future 解析及 SDK 参数构造；client 刷新时机由算法决定。 |
| [输入渲染](../dataset/prompt.py) | 校验文本、token 列表和聊天消息，统一调用 tokenizer。 |
| [数据处理](../dataset/prepare.py) | 将固定格式的数据行统一处理成 prompt、监督序列或偏好对，并完成 token、mask、EOS 与长度校验。 |
| [训练样本](../dataset/examples.py) | 提供 `PromptExample`、`SupervisedExample` 和 `PreferenceExample`，隔离数据准备与 PyTRIO 远端客户端。 |
| [SWE 轨迹](../dataset/swe_sft/) | 将包含工具调用、工具结果和多轮决策的软件工程智能体轨迹转换为 PyTRIO SFT 数据。 |
| [指标公共实现](../utils/metrics.py) | 接收算法产生的任意数值指标，统一完成校验、命名、窗口聚合、SwanLab 提交、文本记录和 run 结束。 |
| [保存公共实现](../utils/checkpoint.py) | 同步／异步保存与路径校验；保存频率及状态更新由各算法决定。 |
| [批处理公共实现](../utils/dataloader.py) | 对 dataset 输出的样本做惰性 batch 切分，并限制验证 batch 数；不负责分词或 Datum。 |
| [Datum 公共实现](../utils/datums.py) | 将已完成算法对齐的 token 与 loss inputs 物化为 PyTRIO Datum；不决定 GRPO/OPD 的训练字段。 |
| [评分公共实现](../utils/scoring.py) | 通过 sampling client 为固定 completion 计算 token logprob，并统一校验完整序列与 completion 的位置对齐；DPO、OPD 和 PPO 可复用。 |
| [DPO 编排](./dpo_trainer.py)、[损失](./dpo/loss.py) | trainer 调用通用偏好数据处理，并负责 Datum、固定参考评分、远端反向；损失模块实现成对目标与算法指标。 |
| [SFT 编排](./sft_trainer.py) | 调用通用 SFT 数据处理，然后构造交叉熵 Datum、提交流水线并按批次顺序发布结果。 |
| [OPD 编排](./opd_trainer.py)、[执行计划](./opd/execution.py) | Trainer 负责数据集、client、update、checkpoint 和日志生命周期；不可变执行计划组合可替换的 rollout、teacher、target、loss 与 metric 阶段。 |
| [PPO-family 编排](./rl_trainer.py)、[算法组件](./ppo/) | 组织 rollout、reference/value、冻结目标、actor/critic 并发更新、checkpoint 和日志生命周期；包括 GRPO 预设。 |
| [OPD rollout](./opd/rollout.py)、[teacher phase](./opd/teacher_phase.py) | 学生按当前 behavior policy 采样；路由组内的固定 teacher 给同一 token 序列评分，并聚合为加权几何 ensemble。当前可跨 prompt 重叠调用，但接口不依赖训练／推理 colocate。 |
| [OPD loss](./opd/loss.py)、[target](./opd/targets.py)、[Datum](./opd/datum.py) | 计算 `log p_teacher - log p_behavior` token 信号，可选 discounted credit，保留 behavior logprob，并构造 PyTRIO policy loss 所需的 next-token 对齐字段；默认通过 CISPO 将 IS ratio 单侧截断到 5.0。 |
| [OPD contracts](./opd/contracts.py)、[metrics](./opd/metrics.py) | 隔离 SDK response、校验阶段间数据，并以 completion token 为分母归约蒸馏指标；验证阶段使用流式 totals，不保留完整远端 Datum 列表。 |

```text
dataset/ 单条样本 → utils/dataloader.py batch → 算法对齐 → utils/datums.py Datum
```

`dataset/` 负责从 mapping、JSONL 或已有 iterable 加载原始行，并转换成标准样本；batch 切分、Datum 外壳构造属于运行时
数据管线。PPO-family/OPD 的 logprob、advantage、teacher score 和 prompt mask 仍由算法
模块决定。



## Trainer 输入数据格式

```python
# PPO-family / OPD：每行传入一个生成 prompt。
# 字段名默认是 messages，可分别通过 PPOConfig.prompt_key 和
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


## 检查数据集配置

```python
from triotune.trainer import SFTTrainer
from triotune.dataset import SWESFTConfig, prepare_swe_sft_dataset

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
import asyncio

from triotune.trainer import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3.5-4B",
    args=SFTConfig(
        eval_steps=50,
        eval_after_train=False,
    ),
    train_dataset=train_rows,
    eval_dataset=eval_rows,
)
state = asyncio.run(trainer.train())
```

## GRPO 训练

```python
import asyncio

from triotune.trainer import PPOConfig, PPOTrainer

trainer = PPOTrainer(
    model="Qwen/Qwen3.5-4B",
    reward_fn=my_reward,
    args=PPOConfig.for_algorithm("grpo", num_generations=4, max_steps=100),
    train_dataset=prompt_rows,
)
state = asyncio.run(trainer.train())
```

## DPO 训练

```bash
uv sync --extra dpo
```

```python
from triotune.trainer import DPOConfig, DPOTrainer

trainer = DPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=DPOConfig(beta=0.1, batch_size=1, max_steps=1),
    train_dataset=preference_rows,
)
state = trainer.train()
```

## OPD 训练

```python
from triotune.trainer import OPDConfig, OPDTrainer

trainer = OPDTrainer(
    model="student-base-model-id",
    teacher_model="teacher-base-model-id",
    # teacher_model_path="trio://...",  # 可选的教师 LoRA checkpoint。
    args=OPDConfig(batch_size=4, max_steps=100),
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
    args=OPDConfig(batch_size=2, max_steps=100),
    train_dataset=rows,
)
```

group 权重保持未归一化：loss 使用
`Σj wj log p_teacher_j - (Σj wj) log p_behavior`，而 teacher logprob 指标使用
归一化后的加权均值，便于比较不同权重规模的实验。

`OPDConfig.importance_ratio_cap` 默认为 `5.0`。它通过 CISPO 使用
`detach(clamp(exp(log p_current - log p_behavior), 0, cap))` 作为
score-function 系数；设为 `None` 可恢复原来的 unbounded
`importance_sampling`，用于兼容或消融实验。

研究代码也可以只替换一个执行阶段：

```python
from dataclasses import replace

from triotune.trainer.opd import OPDExecutionPlan, OPDTeacherPhase


class ExperimentalTeacherPhase(OPDTeacherPhase):
    async def score(self, **kwargs):
        target = await super().score(**kwargs)
        # Apply an experimental target transformation here.
        return target


config = OPDConfig(batch_size=4, max_steps=100)
plan = replace(
    OPDExecutionPlan.from_config(config),
    teacher_phase=ExperimentalTeacherPhase(),
)
trainer = OPDTrainer(
    model="student-base-model-id",
    teacher_model="teacher-base-model-id",
    args=config,
    execution_plan=plan,
    train_dataset=prompt_rows,
)
```
