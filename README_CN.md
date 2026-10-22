# Triotune

**基于 PyTRIO 的训练案例与训练器组件。**

Triotune 是一个构建在 PyTRIO 公共客户端 API 之上的轻量级、纯文本训练
cookbook。名称中的 **Trio** 来自 PyTRIO，**tune** 参考了 Torchtune。项目将
模型执行、分词、采样和优化交给 PyTRIO，负责数据准备、奖励与 advantage
计算、token 对齐以及训练循环编排。

<p align="center">
  <a href="README.md">English</a>
</p>

> **项目状态：** Triotune 仍处于早期阶段。目前提供的是内置的文本训练流程
> 和案例；任意自定义模型模块的通用注册接口尚未纳入项目。

## 🗂️ 目录

- [项目简介](#项目简介)
- [近期动态](#近期动态)
- [安装](#安装)
- [算法](#算法)
- [数据契约](#数据契约)
- [Token 用量指标](#token-用量指标)
- [Python API](#python-api)
- [文档](#文档)
- [仓库结构](#仓库结构)
- [当前范围](#当前范围)

## 👋 项目简介

Triotune 为若干基于 PyTRIO 的训练算法提供类似 Trainer 的使用方式。它只
负责客户端侧逻辑；模型权重、LoRA 训练客户端、tokenizer、采样、前向/反向、
优化器更新和 checkpoint 存储仍由 PyTRIO 提供。

当前包含以下训练器：

- **PPO**：一个异步 trainer 支持 PPO、GRPO、RLOO、GSPO、SAPO、
  CISPO、DAPO 等预设。
- **DPO**：chosen/rejected 偏好对、可选的 reference policy 打分，以及通过
  PyTRIO 提交的本地成对损失。
- **SFT**：纯文本监督微调，支持 prompt/completion mask，并提供异步训练路径。
- **OPD**：在线蒸馏，由学生模型采样 completion，再由固定教师模型提供 token
  级分数。

在底层 PyTRIO 客户端支持的前提下，PPO、DPO 和 OPD 提供异步训练路径。
公共工具统一处理文本、token ID 和聊天消息输入，以及 checkpoint 命名和
SwanLab 指标记录。

## 📰 近期动态

- **2026-10-19：** Triotune 0.1.1 统一了数据集预处理格式，并将发行包和源码
  目录重命名为 `triotune`，同时新增模块化 PPO、在策略蒸馏流程及基于 YAML 的
  训练配方配置。
- **2026-10-12：** Triotune 0.1.0（`pytrio-cookbook`）首次发布到
  [PyPI](https://pypi.org/project/pytrio-cookbook/)。

## 🚀 开始使用

### 📦 安装

- Python 3.12（当前包声明 `>=3.12,<3.13`）
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

## 🔧 支持矩阵

### 🧠 算法

| 算法 | 文档 | 案例 |论文 |
| --- | --- | --- | --- |
| **GRPO** | [📖 文档](./docs/algorithms/grpo_series.md) | [🔧 案例](./examples/grpo) | [📄 论文](https://arxiv.org/abs/2402.03300) |
| **PPO** | [📖 文档](./docs/algorithms/ppo_series_CN.md) | [🔧 案例](./examples/ppo) | [📄 论文](https://arxiv.org/abs/1707.06347) |
| **DPO** | - | - | [📄 论文](https://arxiv.org/abs/2305.18290) |
| **OPD** | - | - | [📄 论文](https://arxiv.org/abs/2601.18734) |
| **SFT** | - | - |  |

### 📖 模型

PyTRIO 当前支持以下模型：

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.8-27B`

## 🧩 数据契约

训练器有意只接受以下几种基础输入：

- 字符串；
- 非空 token ID 列表；
- 非空的 `{role, content}` 聊天消息列表。

各算法的字段要求如下：

- **GRPO：** `messages` 和由 reward 函数使用的字符串 `answer`；
- **DPO：** 共享的 `messages`，以及 `chosen`、`rejected` 回复；
- **SFT：** `text`、`messages`、`prompt`/`completion`，或预分词的
  `input_ids`（可选 `labels`）；
- **OPD：** prompt 行，教师会对学生采样的 completion 打分。

字段校验规则和完整 JSONL 示例见
[trainer 说明](triotune/trainer/README.md)。

## Token 用量指标

使用 trainer 的 SwanLab run 会通过 `usage/train_tokens` 记录本步提交给训练计算的
token 数，并通过 `usage/train_tokens_total` 记录累计值。包含采样的流程还会记录
`usage/sample_input_tokens`、`usage/sample_output_tokens` 及各自的 `_total`
累计值。训练用量按实际发生的远端计算逐次累加，因此 PPO 重复执行的 epoch、
minibatch 也会分别计入。采样用量优先采用 PyTRIO 响应中的计数；对于兼容的测试
客户端或旧响应格式，才根据已经 tokenized 的 prompt 和返回序列计算。

`usage/sample_input_tokens` 不会标成 prefill 或 cache token。前缀缓存是否命中是在
服务端完成请求路由后决定的，取决于请求被分配到哪个副本、该副本当时驻留的缓存
前缀、是否只命中部分前缀，以及缓存是否已被淘汰。对客户端而言，无论服务端重新
计算还是命中缓存，发送的 prompt token 完全相同，所以仅根据 prompt 长度无法还原
两者的实际拆分。PyTRIO 0.2.9 的采样响应只提供汇总的输入和输出 token 数，没有
cached-input 或 uncached-prefill 计数。因此 Triotune 不猜测这两个数，也不维护按
模型硬编码的单价表来计算费用。

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

## 🗃️ 仓库结构

```text
triotune/
├── api/           # 公共配置与 client 契约
├── dataset/       # 共享数据校验与分词
├── trainer/       # 算法 trainer 与组件
└── utils/         # 公共运行时工具
```

## 🎯 当前范围

Triotune 当前聚焦于纯文本、基于 PyTRIO 的实验。项目不宣称提供 Web UI、
Megatron 集成、多模态训练，也不提供通用的插件/注册表系统。如果需要使用
自定义模型模块，请先让它适配 PyTRIO 的公共客户端契约，再使用现有 trainer
接口；“任意自定义模块注册”是后续工作，不是 0.1.2 已支持的能力。
