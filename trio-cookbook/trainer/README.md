# Trainer 模块

`trainer` 包含 PyTRIO 原生 GRPO、DPO、SFT 和 OPD 训练实现。GRPO 将采样、奖励和组内 advantage 对齐为 PyTRIO `Datum`；DPO 使用偏好数据和固定参考策略，在客户端计算成对损失；SFT 用远端交叉熵训练文本样本；OPD 用学生生成的序列及教师评分进行在线蒸馏。GRPO、DPO 和 OPD 提供同步和异步训练路径，SFT 提供异步训练路径。

## 训练流程

一次训练 step 大致按以下顺序执行：

1. `GRPOTrainer` 从数据行读取默认的 `messages` 字段。
2. `prompts.py` 校验并渲染字符串、整数 token 列表或文本聊天消息。
3. `rollout.py` 通过 `SamplingClient` 为每个输入采样多个 completion，并校验 token、log-probability 和文本。
4. `rewards.py` 批量调用奖励函数，聚合各个奖励并生成奖励指标。
5. `advantages.py` 在每个 prompt 对应的 completion 组内计算 centered advantage。
6. `datum.py` 完成 prompt 与 completion 的 one-token shift，构造 PyTRIO `Datum`。
7. `GRPOTrainer` 调用 `forward_backward` 和 `optim_step`，记录指标并按配置保存状态。

## 文件说明

| 文件 | 主要对象 | 用处 |
| --- | --- | --- |
| [`__init__.py`](./__init__.py) | `GRPOTrainer`、`DPOTrainer`、配置、类型和渲染函数 | 定义模块的公开导出，提供稳定的导入入口。 |
| [`advantages.py`](./advantages.py) | `group_advantages` | 对一个完整的 GRPO completion 组计算均值、标准差、centered advantage 和退化组指标。支持不缩放和组内标准差缩放。 |
| [`grpo_config.py`](./grpo_config.py) | `GRPOConfig` | 保存采样、优化、奖励缩放、数据输入和 SwanLab 配置；在初始化时校验数值范围和不兼容选项，并生成采样器和 Adam 参数。 |
| [`dpo_config.py`](./dpo_config.py) | `DPOConfig` | 配置 DPO 系数、批大小、长度、优化、checkpoint 和 SwanLab。 |
| [`dpo_trainer.py`](./dpo_trainer.py) | `DPOTrainer`、`DPOState`、`DPOResult` | 用 PyTRIO 的参考采样器和自定义反向接口训练偏好对。 |
| [`sft_config.py`](./sft_config.py) | `SFTConfig` | 配置批大小、截断、标签掩码、优化器和日志。 |
| [`sft_trainer.py`](./sft_trainer.py) | `SFTTrainer`、`SFTState` | 将文本或聊天数据转成 next-token 标签，调用 PyTRIO 交叉熵训练并保存状态。 |
| [`opd_config.py`](./opd_config.py) | `OPDConfig` | 配置学生生成、异步并发、反向 KL 信号、优化、checkpoint 和日志；类注释逐项说明原理与默认值。 |
| [`opd_trainer.py`](./opd_trainer.py) | `OPDTrainer`、`OPDState`、`OPDResult` | 编排当前学生采样、固定教师评分、远端更新和同步／异步训练。 |
| [`opd_distillation.py`](./opd_distillation.py) | `distillation_advantages`、`build_opd_datum` | 校验 logprob、提取教师 completion 评分、计算 token advantage，并对齐和屏蔽 prompt。 |
| [`datum.py`](./datum.py) | `build_aligned_datum`、`validate_datum`、`to_client_datum` | 将 prompt token、completion token、旧策略 log-probability 和 advantage 对齐为训练字段，检查长度和有限值，再物化为 PyTRIO 或兼容客户端的 `Datum`。 |
| [`grpo_trainer.py`](./grpo_trainer.py) | `GRPOTrainer`、`TrainerState` | 训练主编排器。管理客户端、数据行、采样、奖励、advantage、Datum、前向反向、优化、checkpoint、同步/异步训练和 SwanLab 日志。 |
| [`prompts.py`](./prompts.py) | `RenderedPrompt`、`render_prompt`、`as_prompt_text` | 校验 text-only 输入并调用 tokenizer。聊天消息使用 `apply_chat_template`，整数列表直接作为 token，同时生成采样和奖励需要的文本表示。 |
| [`rewards.py`](./rewards.py) | `RewardEvaluator`、`RewardFunc` | 适配同步或异步的 TRL 风格奖励函数，展开数据列，聚合加权奖励，校验返回数量和有限值，并输出每个奖励函数的指标。 |
| [`rollout.py`](./rollout.py) | `SamplingAdapter`、`parse_sample` | 适配 PyTRIO 采样客户端的同步/异步调用，读取不同响应对象中的 sequences，校验 token 与 log-probability，并生成 `SequenceSample`。 |
| [`types.py`](./types.py) | `SequenceSample`、`GRPOBatch`、`GRPODatum`、`GRPOResult` | 定义采样结果、prompt 分组、训练 Datum 和 step 结果等数据结构，连接采样、奖励和优化阶段。 |

## 数据输入格式

- 字符串；
- 非空整数 token 列表；
- 每条消息都包含字符串 `role` 和 `content` 的聊天消息列表。

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
