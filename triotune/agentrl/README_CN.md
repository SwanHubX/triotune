# AgentRL

[English](README.md) | [完整训练文档](../../docs/agenrl/README_CN.md)

triotune.agentrl 将使用工具的智能体接入现有 PPO 系列训练器。PyTRIO 负责模型采样、优化、指标和 checkpoint；本包负责 episode 协议、工具、评分以及精确 token 轨迹导出。

## 训练内容

一个候选会按以下循环执行：

1. PyTRIO 导出一个固定的采样策略快照。
2. 智能体生成动作、调用注册工具并读取观察，直到 episode 结束。
3. runner 保留模型原始输出 token 和行为策略 log probability。prompt 模板和工具观察只作为上下文，不参与策略损失。
4. scorer 分配终局奖励以及可选的过程奖励。
5. AgentTrainingLoop 将校验后的轨迹交给 trainer.step，再为下一批数据创建新的快照。

这样可以保留候选身份、奖励分组、策略版本和动作对齐关系，并把可恢复的模型错误与基础设施故障分开，避免服务故障变成误导性的零奖励样本。

## 快速开始

在仓库根目录执行：

```bash
uv sync --locked --extra ppo
uv run python examples/agent/calculator/train.py --mode tokenizer
uv run python examples/agent/calculator/train.py --mode probe
uv run python examples/agent/calculator/train.py --mode train
```

可运行的计算器 workflow 位于 [examples/agent/calculator](../../examples/agent/calculator)。完整训练步骤、配置选择和诊断方式见 [docs/agenrl/README_CN.md](../../docs/agenrl/README_CN.md)。

## 包边界

| 包目录 | 职责 |
| --- | --- |
| config.py | 限制、重试、渲染器、导出模式、评估和 workflow 组装 |
| runtime/ | 对话渲染、采样、episode 执行和工具 |
| rollout/ | 分组采集和精确 token 轨迹构造 |
| training/ | Agent 训练循环、奖励组合和评估指标 |
| diagnostics/ | episode 日志和 tokenizer 训练前检查 |
| integrations/openai/ | OpenAI 兼容执行和 PyTRIO 轨迹桥接 |

奖励提交、分支、上下文压缩、策略年龄和分组过滤见 [agentrl_training_CN.md](../../docs/agenrl/agentrl_training_CN.md)。OpenAI session 所有权、回放、恢复和第三方转发见 [agentrl_openai_CN.md](../../docs/agenrl/agentrl_openai_CN.md)。

采样 temperature 和 top-p 固定为 1，以满足 PPO 的 log probability 契约。先使用同步采集和固定快照；完成 tokenizer 与 probe 检查后，再开启有界异步采集。
