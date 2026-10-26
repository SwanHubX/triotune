# Agent RL

中文 | [English](README.md)

## 概述

triotune.agentrl 将使用工具的智能体接入现有 PPOTrainer，支持同步多轮执行、轨迹采集、奖励分配、训练更新和评估。模型执行、优化与 checkpoint 存储由 PyTRIO 负责。

## 工作原理

1. 为一批任务导出固定的策略快照。
2. 生成动作、解析工具调用、执行工具并追加观察，直到 episode 结束。
3. 保留原始输出 token 和行为策略的 log probability，构造带动作掩码和轮次 ID 的 Trajectory；观察和模板 token 提供上下文，不参与策略损失。
4. 分配结果奖励和可选的过程奖励，再调用 trainer.step() 更新策略。
5. 为下一批任务刷新快照，并通过相同的执行与评分逻辑进行评估。

可恢复的模型动作错误会成为观察或被评分的失败。基础设施故障不会成为零奖励训练样本。重试次数有上限，工具重试还需要显式声明重复执行安全。

## 模块

| 位置 | 职责 |
|---|---|
| config.py | Agent 限制、重试、评估设置和适配器组装 |
| contracts.py | 执行协议、episode 记录和数据校验 |
| runtime/ | 对话渲染、episode 执行、采样和工具 |
| rollout/ | 分组采集和原始 token 轨迹构造 |
| training/ | 训练循环、奖励组合和评估指标 |
| diagnostics/ | Episode 日志和 tokenizer 运行前检查 |

## 快速开始

在仓库根目录安装依赖，并配置 PyTRIO 凭据：

```bash
uv sync --locked --extra ppo
uv run python examples/agent/calculator/train.py --mode tokenizer
uv run python examples/agent/calculator/train.py --mode probe
uv run python examples/agent/calculator/train.py --mode train
```

计算器使用 [config.yaml](../../examples/agent/calculator/config.yaml)，默认模型为 Qwen/Qwen3.5-4B，执行两次更新。报告和 episode 日志保存在 output/calculator-agent/。使用 train-report.json 中的 checkpoint URI 进行独立评估：

```bash
uv run python examples/agent/calculator/train.py --mode evaluate --checkpoint "trio://YOUR_RUN/weights/YOUR_CHECKPOINT"
```

## 自定义智能体

创建带有训练和评估数据的 PPOTrainer、工具执行器和 episode 评分器。在 async 函数内执行以下组装代码，其中 trainer、tools 和 scorer 由应用提供：

```python
from triotune.agentrl import AgentConfig, AgentTrainingLoop

config = AgentConfig(
    limits={"max_turns": 5},
    retry={"max_attempts": 3, "retry_tools": True},
    log_path="output/agentrl/episodes.jsonl",
)
snapshot = await trainer.create_sampling_snapshot(stop=config.stop)
workflow = config.build_workflow(
    tokenizer=snapshot.tokenizer,
    tools=tools,
    scorer=scorer,
)
state = await AgentTrainingLoop(trainer, workflow, config=config).train()
```

默认输入字段为 messages。工具可使用 FunctionToolExecutor；评分器返回 EpisodeScore，包含终局奖励、成功标记和可选的 token 对齐过程奖励。CompositeScorer、ToolCostScorer 和 TurnRewardScorer 用于组合结果、成本与过程信号。

默认适配器使用 JSON 动作，例如 {"tool_calls":[{"name":"calculate","arguments":{"operation":"add","a":2,"b":3}}]} 和 {"final":"5"}。请在 system 消息中说明动作协议和工具。其他输入字段可通过 AgentConfig.prompt_key 设置，也可通过公开协议替换适配器。

## 当前范围

采样使用 temperature=1 和 top_p=1，以满足 PPO 的 log probability 契约。默认渲染器将上下文追加到原始采样 token，使用其他模型前需检查 tokenizer 兼容性。第三方 SDK 代理、分支轨迹、上下文压缩和采样训练重叠不在当前实现范围内。
