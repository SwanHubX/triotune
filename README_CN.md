# Triotune

**基于 PyTRIO 的训练案例与训练器组件。**

Triotune 基于 PyTRIO 公共客户端 API，负责数据准备、奖励与优势计算、token 对齐和训练循环。PyTRIO 提供模型执行、分词、采样、优化和 checkpoint 存储。

<p align="center">
  <a href="README.md">English</a>
</p>

## 🗂️ 目录

- [项目简介](#-项目简介)
- [近期动态](#-近期动态)
- [开始使用](#-开始使用)
  - [安装](#-安装)
  - [快速开始](#-快速开始)
- [支持能力](#-支持能力)
  - [算法](#-算法)
  - [模型](#-模型)
- [Python API](#-python-api)
- [Agent RL](#agent-rl)
- [数据契约](#-数据契约)
- [Token 用量指标](#token-用量指标)
- [文档](#文档)
- [仓库结构](#-仓库结构)
- [当前范围](#-当前范围)

## 👋 项目简介

提供四类基于 PyTRIO 的训练器：

- **PPO**：一个异步 trainer 支持 PPO、GRPO、RLOO、GSPO、SAPO、CISPO、DAPO 等预设。
- **DPO**：chosen/rejected 偏好对、可选的 reference policy 打分，以及通过 PyTRIO 提交的本地成对损失。
- **SFT**：纯文本监督微调，支持 prompt/completion mask，并提供异步训练路径。
- **OPD**：在线蒸馏，由学生模型采样 completion，再由固定教师模型提供 token 级分数。

在底层 PyTRIO 客户端支持的前提下，PPO、DPO 和 OPD 提供异步训练路径。公共工具统一处理文本、token ID 和聊天消息输入，以及 checkpoint 命名和 SwanLab 指标记录。

`triotune.agentrl` 在 PPO trainer 上提供多轮智能体的工具执行、轨迹导出、评分和评估能力。

## 📰 近期动态

- **0.1.3 · 2026-10-27：** 多轮 Agent RL、工具执行、轨迹与奖励对齐、OpenAI 接入及中英文训练示例。
- **0.1.2 · 2026-10-19：** 修复包导出，将 PyPI 发行包名与 Python 导入名统一为 `triotune`，同步更新示例和文档。
- **0.1.1 · 2026-10-19：** 统一数据预处理，将发行包和源码目录重命名为 `triotune`，新增模块化 PPO、在策略蒸馏和 YAML 训练配置。
- **0.1.0 · 2026-10-12：** 以 `pytrio-cookbook` 名称首次发布到 [PyPI](https://pypi.org/project/triotune/)。

## 🚀 开始使用

### 📦 安装

- Python 3.12（`>=3.12,<3.13`）
- 可用的 PyTRIO 训练服务，以及该案例所使用模型的访问权限

在 macOS 或 Linux 上安装 [uv](https://docs.astral.sh/uv/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在 Windows PowerShell 中运行官方安装命令：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

然后克隆仓库并创建由锁文件约束的完整环境：

```bash
git clone https://github.com/SwanHubX/triotune.git
cd triotune
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

### 📚 快速开始

通过 YAML 配置运行 GSM8K PPO 示例：

```bash
uv run python examples/ppo/gsm8k_ppo.py --config examples/ppo/gsm8k_ppo_args.yaml
```

## 🔧 支持能力

### 🧠 算法

| 算法 | 文档 | 案例 | 论文 |
| --- | --- | --- | --- |
| **GRPO** | [📖 文档](./docs/algorithms/grpo_series_CN.md) | [🔧 案例](./examples/grpo) | [📄 论文](https://arxiv.org/abs/2402.03300) |
| **PPO** | [📖 文档](./docs/algorithms/ppo_series_CN.md) | [🔧 案例](./examples/ppo) | [📄 论文](https://arxiv.org/abs/1707.06347) |
| **DPO** | [📖 文档](./docs/algorithms/dpo_CN.md) | [🔧 案例](./examples/dpo) | [📄 论文](https://arxiv.org/abs/2305.18290) |
| **OPD** | [📖 文档](./docs/algorithms/opd_CN.md) | [🔧 案例](./examples/opd) | [📄 论文](https://arxiv.org/abs/2601.18734) |
| **SFT** | [📖 文档](./docs/algorithms/sft_CN.md) | [🔧 案例](./examples/sft) |  |

### 📖 模型

PyTRIO 当前支持以下模型：

- Qwen/Qwen3.5-4B
- Qwen/Qwen3.8-27B

## 🐍 Python API

最简 GRPO 训练流程会使用默认配置，为每个 prompt 采样 4 个回复：

```python
import asyncio

from triotune.trainer import PPOConfig, PPOTrainer


asyncio.run(
    PPOTrainer(
        model="your-pytrio-model-id",
        args=PPOConfig.for_algorithm("grpo"),
        reward_fn=lambda row, completion, **_: float(row["answer"] in completion),
        train_dataset=[{"messages": "What is 2 + 2?", "answer": "4"}],
    ).train()
)
```

## Agent RL

安装 `ppo` 扩展并配置 PyTRIO 凭据。定义工具和 episode 奖励，再用 GRPO 训练：

```python
import asyncio

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig, AgentTrainingLoop, EpisodeScore, FunctionToolExecutor, function_tool,
)


@function_tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class AnswerScorer:
    async def score(self, row, episode):
        used_tool = any(
            not result.is_error
            for turn in episode.turns for result in turn.tool_results
        )
        success = (
            episode.status == "completed" and used_tool
            and episode.final_text.strip() == row["answer"]
        )
        return EpisodeScore(float(success), success)


async def main():
    trainer = PPOTrainer(
        model="Qwen/Qwen3.5-4B",
        args=PPOConfig.for_algorithm("grpo", batch_size=1, max_steps=1),
        train_dataset=[{
            "messages": [
                {"role": "system", "content": (
                    'Use add. Reply with JSON: '
                    '{"tool_calls":[{"name":"add","arguments":{"a":2,"b":3}}]}. '
                    'After tool_results, reply with {"final":"5"}.'
                )},
                {"role": "user", "content": "What is 2 + 3?"},
            ],
            "answer": "5",
        }],
    )
    agent = AgentConfig()
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    workflow = agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=FunctionToolExecutor([add]),
        scorer=AnswerScorer(),
    )
    state = await AgentTrainingLoop(trainer, workflow, config=agent).train()
    print(state.checkpoint_path)


asyncio.run(main())
```

评估和自定义 workflow 见 [AgentRL 说明](docs/agenrl/README_CN.md)。

## 🧩 数据契约

训练器支持以下基础输入：

- 字符串；
- 非空 token ID 列表；
- 非空的 `{role, content}` 聊天消息列表。

各算法的字段要求如下：

- **GRPO：** `messages` 和由 reward 函数使用的字符串 `answer`；
- **DPO：** 共享的 `messages`，以及 `chosen`、`rejected` 回复；
- **SFT：** `text`、`messages`、`prompt`/`completion`，或预分词的 `input_ids`（可选 `labels`）；
- **OPD：** prompt 行，教师会对学生采样的 completion 打分。

字段校验规则和完整 JSONL 示例见 [训练器说明](triotune/trainer/README_CN.md)。

## Token 用量指标

[PyTRIO](https://pytrio.com/dashboard) 记录 training、prefill 和 sampler token。Triotune 在 SwanLab 中记录对应数量及累计值，采样优先使用 PyTRIO 返回的计数，训练按实际提交的计算请求累计。

本地回退计数、失败或重试请求可能造成偏差，最终以 PyTRIO 服务端记录为准。

## 文档

- [训练器说明](triotune/trainer/README_CN.md)
- [AgentRL 训练说明](docs/agenrl/README_CN.md)
- [PPO 算法原理](docs/algorithms/ppo_series_CN.md)
- [GRPO 算法原理](docs/algorithms/grpo_series_CN.md)

## 🗃 仓库结构

```text
triotune/
├── api/           # 公共配置与 client 契约
├── agentrl/       # 智能体执行、轨迹与训练循环
├── dataset/       # 共享数据校验与分词
├── trainer/       # 算法 trainer 与组件
└── utils/         # 公共运行时工具
```

## 🎯 当前范围

Triotune 聚焦于基于 PyTRIO 的纯文本训练。自定义模型模块需先适配 PyTRIO 公共客户端契约，再使用训练器接口。Web UI、Megatron 集成、多模态训练和通用插件注册系统不在当前支持范围内。
