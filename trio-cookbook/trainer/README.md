# Trainer 模块

`trainer` 包含 PyTRIO 原生 GRPO 和 DPO 训练实现。GRPO 将采样、奖励和组内 advantage 对齐为 PyTRIO `Datum`；DPO 使用偏好数据和固定参考策略，在客户端计算成对损失，通过 PyTRIO 完成远端反向传播。两者均提供同步和异步训练路径。

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

异步场景可以调用 `step_async()` 或 `train_async()`。训练器要求每个 GRPO 组至少包含两个 completion；`loss_fn` 当前支持 `importance_sampling`，模型加载、采样和优化由 PyTRIO 客户端负责。

## DPO 训练

DPO 需要本机安装 PyTorch 以计算成对损失和梯度：`uv sync --extra dpo`。数据集每行包含 `messages`、`chosen`、`rejected`；`messages` 支持文本、token ID 或聊天消息，回答支持文本、token ID，或与聊天消息匹配的末轮 assistant 消息列表。聊天模板生成的完整对话必须以输入消息的 token 为前缀，否则会明确报错，避免训练目标错位。

```python
from trio_cookbook.trainer import DPOConfig, DPOTrainer

rows = [
    {"messages": "What is 2 + 2?", "chosen": "4", "rejected": "5"},
]
trainer = DPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=DPOConfig(beta=0.1, batch_size=1, max_steps=1, mode="local"),
    train_dataset=rows,
)
state = trainer.train()
print(state.checkpoint_path)
```

默认参考策略是未经 LoRA 更新的同一基座模型；`ref_model` 可指定另一个 PyTRIO 基座模型，但须使用相同 token ID 语义。DPO 每步记录损失、偏好奖励与准确率、优化指标到 SwanLab，并在训练结束保存 PyTRIO checkpoint。`step()`、`step_async()`、`train_async()` 也可单独使用。`max_length` 超限时抛错，不会静默截断偏好数据。
聊天数据可通过 `DPOConfig(chat_template_kwargs={"enable_thinking": False})` 向 tokenizer 的聊天模板传参。
旧数据如使用 `prompt` 列，可显式设置 `DPOConfig(prompt_key="prompt")`；默认列名为 `messages`。
