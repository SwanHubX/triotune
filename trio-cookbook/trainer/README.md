# Trainer 模块

`trainer` 包含 PyTRIO 原生 GRPO、DPO 和 SFT 训练实现。GRPO 将采样、奖励和组内 advantage 对齐为 PyTRIO `Datum`；DPO 使用偏好数据和固定参考策略，在客户端计算成对损失；SFT 用远端交叉熵训练文本样本。GRPO 和 DPO 提供同步和异步训练路径，SFT 提供异步训练路径。

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

当 `report_to="swanlab"` 时，`SFTConfig` 的以下字段传给 `swanlab.init`。这些字段默认为 `None`，可在 recipe 的 TOML `[swanlab]` 表中配置。

| 字段 | 作用 |
| --- | --- |
| `workspace` | 指定 SwanLab 工作区；`None` 使用 SDK 默认值。 |
| `tags` | 设置运行标签列表。 |
| `run_name` | 设置运行名称；`None` 使用 `<模型名>-sft`。 |
| `log_dir` | 指定本地 SwanLab 日志目录；`None` 使用 SDK 默认值。 |

`step()`、`train()` 和 checkpoint 保存采用异步 PyTRIO 调用，等待远端结果时让出事件循环。每步仍按反向、优化器更新、保存的顺序执行，不并发更新同一模型。下面使用 `asyncio.run` 启动训练；在已有事件循环中使用 `await trainer.train()`。客户端构造和 tokenization 仍是同步操作，异步应用可用 `asyncio.to_thread` 构造 trainer。

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
