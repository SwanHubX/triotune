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

## 目录

- [项目简介](#项目简介)
- [近期动态](#近期动态)
- [安装](#安装)
- [快速开始](#快速开始)
- [算法与案例](#算法与案例)
- [数据契约](#数据契约)
- [Python API](#python-api)
- [文档](#文档)
- [仓库结构](#仓库结构)
- [当前范围](#当前范围)

## 项目简介

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

## 近期动态

- **2026-10-12：** Triotune 0.1.0（`pytrio-cookbook`）首次发布到
  [PyPI](https://pypi.org/project/pytrio-cookbook/)。

## 安装

### 环境要求

- Python 3.12（当前包声明 `>=3.12,<3.13`）
- 可用的 PyTRIO 训练服务，以及该案例所使用模型的访问权限

推荐使用 [uv](https://docs.astral.sh/uv/)，仓库中的 `uv.lock` 会锁定环境依赖：

```bash
uv sync
uv run python -c "import trio_cookbook; print(trio_cookbook.__name__)"
```

使用 DPO 时安装可选的 PyTorch 依赖：

```bash
uv sync --extra dpo
```

也可以直接安装已发布的包：

```bash
pip install pytrio-cookbook==0.1.0
```

从源码目录使用 pip 安装：

```bash
pip install -r requirements.txt
pip install -e .
```

发行包名称是 `pytrio-cookbook`，Python 导入名称是 `trio_cookbook`。

## 快速开始

GSM8K 案例会准备 JSONL 数据集，然后使用 PPO trainer 的 GRPO 预设。准备阶段不会创建
PyTRIO 客户端，适合作为本地环境的第一步检查：

```bash
# 在仓库根目录执行
bash trio-cookbook/scripts/train_grpo_gms8k.sh --prepare-only

# 使用默认 YAML 配置启动训练
bash trio-cookbook/scripts/train_grpo_gms8k.sh
```

常用覆盖参数包括 `--steps`、`--batch-size`、`--group-size`、
`--max-examples` 和 `--help`。默认配置使用 Qwen 模型、LoRA、SwanLab 记录，
以及 GSM8K 的 boxed-answer reward。请根据实际环境修改
`trio-cookbook/configs/grpo_gms8k.yaml` 中的模型和训练配置。

## 算法与案例

| 算法 | 内置案例 | 数据 | 启动入口 |
| --- | --- | --- | --- |
| PPO / GRPO | [GSM8K](trio-cookbook/recipes/train_grpo.py) / [PPO](trio-cookbook/recipes/train_ppo.py) | prompt + 参考答案 | [`train_grpo_gms8k.sh`](trio-cookbook/scripts/train_grpo_gms8k.sh) / [`train_ppo_gms8k.sh`](trio-cookbook/scripts/train_ppo_gms8k.sh) |
| DPO | [HH-RLHF](trio-cookbook/recipes/train_dpo.py) | `messages`、`chosen`、`rejected` | [`train_dpo_rlhf.sh`](trio-cookbook/scripts/train_dpo_rlhf.sh) |
| SFT | [嬛嬛对话](trio-cookbook/recipes/train_sft.py) | 文本、messages 或 prompt/completion | [`train_sft_huanhuan.sh`](trio-cookbook/scripts/train_sft_huanhuan.sh) |
| OPD | [DeepMath-103K](trio-cookbook/recipes/train_opd.py) | 仅 prompt | [`train_opd_deepmath.sh`](trio-cookbook/scripts/train_opd_deepmath.sh) |

每个案例都有对应的 YAML 配置，位于
[`trio-cookbook/configs/`](trio-cookbook/configs/)。`recipes/` 中还保留了直接
调用 PyTRIO 的独立脚本，可作为易读的算法基线，与 trainer 版本进行对照。

## 数据契约

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
[trainer 说明](trio-cookbook/trainer/README.md) 与
[脚本说明](trio-cookbook/scripts/README.md)。

## Python API

公开入口按算法组织：

```python
import asyncio

from trio_cookbook.trainer import PPOConfig, PPOTrainer

rows = [
    {
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "answer": "4",
    },
]

trainer = PPOTrainer(
    model="your-pytrio-model-id",
    reward_fn=your_reward_function,
    args=PPOConfig.for_algorithm("grpo", num_generations=4, max_steps=100),
    train_dataset=rows,
)
asyncio.run(trainer.train())
```

`DPOTrainer`、`SFTTrainer` 和 `OPDTrainer` 使用相同的配置与数据集导向风格。
请将示例中的模型 ID 和 reward 函数替换为 PyTRIO 服务实际支持的内容。

## 文档

- [训练器内部实现与 Python 示例](trio-cookbook/trainer/README.md)
- [案例、配置与脚本命令](trio-cookbook/scripts/README.md)
- [包目录说明](trio-cookbook/README.md)
- [框架评测记录](docs/README.md)

## 仓库结构

```text
trio-cookbook/
├── configs/       # 内置流程的 YAML 配置
├── recipes/       # trainer 版本和独立 PyTRIO 案例
├── scripts/       # 启动脚本和数据准备说明
└── trainer/       # trainer 及公共配置、数据和 client 契约
```

## 当前范围

Triotune 当前聚焦于纯文本、基于 PyTRIO 的实验。项目不宣称提供 Web UI、
Megatron 集成、多模态训练，也不提供通用的插件/注册表系统。如果需要使用
自定义模型模块，请先让它适配 PyTRIO 的公共客户端契约，再使用现有 recipe
或 trainer 接口；“任意自定义模块注册”是后续工作，不是 0.1.0 已支持的能力。
