# 使用 PyTRIO 进行 AgentRL 训练

[English](README.md)

Triotune 的 AgentRL 用于训练需要决定动作、调用工具、读取工具结果并完成任务的模型。它把这类多轮执行接入现有 PPO 系列训练器，解决普通文本强化学习中“奖励属于完整工具 episode，但优化器需要每个候选的原始输出 token、行为策略 log probability、mask 和分组身份”的衔接问题。

## 本项目解决的问题

训练路径以 PyTRIO 作为模型采样、优化、指标和 checkpoint 的底层接口：

- 一个批次内的所有候选使用固定的 PyTRIO 策略快照。
- 工具调用和观察保留在对话上下文中，只有模型原始输出 token 参与策略损失。
- 终局奖励以及可选的轮次、交互或 token 奖励在进入 PPO 或 GRPO 前完成校验和对齐。
- 候选和分组 provenance 会随轨迹保留，避免相对奖励算法混合无关任务或不同策略版本。
- 可恢复的模型动作错误会成为观察或失败 episode；基础设施故障可以丢弃，不会静默变成零奖励样本。
- 同一个 workflow 可用于训练采集、评估、JSONL episode 日志和 PyTRIO 指标发布。

[AReaL agentic RL 教程](https://areal-ai.io/docs/en/tutorial/agentic_rl.html) 可以作为概念入门。本项目针对 PyTRIO 训练后端进一步规定精确 token 轨迹契约、快照生命周期、奖励对齐、分组过滤、重试边界和策略版本检查，保证采集结果可以安全进入优化器。

## 训练步骤

每个实验按以下顺序组织：

1. 安装 PPO 依赖，并准备 PyTRIO 凭据。
2. 构造包含 messages 的任务行；答案等只供评分器使用的字段不要放进 prompt。
3. 注册带 schema 和参数校验的工具。
4. 编写 episode scorer，返回终局分数；需要时追加过程奖励。
5. 用 AgentConfig 组装 workflow。workflow 负责渲染 prompt、通过 PyTRIO 采样、执行工具并导出对齐后的轨迹。
6. 创建 PPOTrainer 并运行 AgentTrainingLoop。循环会固定当前策略快照、采集完整分组、调用 trainer.step，再为下一次更新刷新快照。
7. 用同一个 workflow 和 scorer 评估，并使用 PyTRIO 报告的 checkpoint URI。

计算器示例完整覆盖这条路径：[train.py](../../examples/agent/calculator/train.py)、[agent.py](../../examples/agent/calculator/agent.py) 和 [config.yaml](../../examples/agent/calculator/config.yaml)。

## 可运行示例

先运行 tokenizer 和 probe，提前发现模板或 token 对齐问题，再开始远程更新：

```bash
uv sync --locked --extra ppo
uv run python examples/agent/calculator/train.py --mode tokenizer
uv run python examples/agent/calculator/train.py --mode probe
uv run python examples/agent/calculator/train.py --mode train
uv run python examples/agent/calculator/train.py --mode evaluate \
  --checkpoint "trio://YOUR_RUN/weights/YOUR_CHECKPOINT"
```

下面是应用侧最小组装。任务行、工具注册和 scorer 由应用负责，策略执行和优化由 PyTRIO 负责。

```python
import asyncio
import pytrio

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import AgentConfig, AgentTrainingLoop
from examples.agent.calculator.agent import calculator_tools, CalculatorScorer


async def train(rows, eval_rows):
    service = await asyncio.to_thread(pytrio.ServiceClient)
    trainer = PPOTrainer(
        args=PPOConfig.for_algorithm(
            "grpo",
            model={"name": "Qwen/Qwen3.5-4B", "lora_rank": 8},
            training={"batch_size": 2, "max_steps": 2, "seed": 42},
            rollout={
                "num_generations": 4,
                "max_tokens": 1024,
                "temperature": 1.0,
                "top_p": 1.0,
            },
        ),
        train_dataset=rows,
        eval_dataset=eval_rows,
        service_client=service,
    )
    agent = AgentConfig(
        limits={"max_turns": 5, "max_generated_tokens": 1024},
        retry={"max_attempts": 3, "retry_tools": True},
        export_mode="concat",
        log_path="output/agentrl/episodes.jsonl",
    )
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    workflow = agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=CalculatorScorer(),
    )
    loop = AgentTrainingLoop(trainer, workflow, config=agent)
    await loop.train()
```

新任务首先只替换任务行、工具和 scorer，保持采样 temperature 与 top-p 都为 1.0。PPO 契约使用固定 PyTRIO 快照记录的行为 log probability。答案等私有字段应保留在原始 row 中，由 scorer 读取。

## 数据、工具和奖励

任务行通常包含 messages 和任务标识。scorer 可以同时读取原始 row 与完成后的 episode，包括最终文本、工具结果、状态和交互记录。

```python
from triotune.agentrl import EpisodeScore


class AnswerScorer:
    async def score(self, row, episode):
        success = (
            episode.status == "completed"
            and episode.final_text == row["answer"]
        )
        return EpisodeScore(float(success), success)


rows = [
    {
        "task_id": "addition-1",
        "messages": [{"role": "user", "content": "What is 2 + 3?"}],
        "answer": "5",
    }
]
```

### 注册工具并配置失败处理

本地 Python 工具使用 FunctionTool 和 FunctionToolExecutor。下面复用计算器示例中的工具注册表，替换为自己的工具时保持同样的组装方式。snapshot 由 trainer.create_sampling_snapshot 返回，AnswerScorer 使用上一段定义。

```python
from examples.agent.calculator.agent import calculator_tools
from triotune.agentrl import AgentConfig

agent = AgentConfig(
    retry={"max_attempts": 3, "retry_tools": True},
    on_system_failure="drop_group",
)


def make_workflow(snapshot):
    return agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )
```

默认 JSON 执行器需要在 system 消息中说明工具和动作格式，例如调用工具时输出 {"tool_calls":[{"name":"calculate","arguments":{"operation":"add","a":2,"b":3}}]}，结束时输出 {"final":"5"}。直接使用计算器示例中的 calculator_rows 或 generated_rows，可以同时获得相应的 system 提示。

retry_tools 只有与工具自身的 retry_safe 声明同时开启才允许重试。不要为无法安全重复的副作用开启重试。drop_group 会丢弃不满足可用候选数要求的分组，故障记录仍留在 episode 日志中；使用 raise 可以在基础设施故障时停止训练。

### 添加过程奖励

在终局正确性奖励之外，可以按轮次添加工具成本。下面为每个有模型输出的轮次按已返回的工具结果数扣分；用 ToolCostScorer() 替换 workflow 中的 AnswerScorer() 即可生效。

```python
from triotune.agentrl import EpisodeScore, RewardAssignment


class ToolCostScorer(AnswerScorer):
    async def score(self, row, episode):
        outcome = await super().score(row, episode)
        costs = tuple(
            RewardAssignment(
                "turn",
                -0.01 * len(turn.tool_results),
                position=index,
                source="tool_cost",
            )
            for index, turn in enumerate(episode.turns)
            if turn.tool_results and turn.generation.tokens
        )
        return EpisodeScore(
            outcome.reward,
            outcome.success,
            assignments=costs,
        )
```

position 使用 episode.turns 中从 0 开始的索引。scorer 返回原始奖励，后续训练流程负责对齐、归一化和 advantage 计算。若使用 RewardLedger 提交延迟奖励，每个逻辑贡献只提交一次，重试复用 submission ID，避免重复计分。

## 轨迹和 rollout 设置

### 选择轨迹导出方式

根据优化目标选择导出模式，并用选中的 AgentConfig 创建 workflow：

- concat：一个完整 episode 对应一行 PPO 数据。
- individual：每个生成轮次对应一行，此前的模型输出和工具结果作为上下文；只有当前轮次参与策略损失。
- branch：按交互树导出分支路径，适用于从同一个状态比较不同后续动作。

```python
from triotune.agentrl import AgentConfig

export_configs = {
    "episode": AgentConfig(export_mode="concat"),
    "turn": AgentConfig(export_mode="individual"),
    "tree": AgentConfig(
        export_mode="branch",
        shared_prefix_policy="weighted",
    ),
}
agent = export_configs["turn"]


def make_workflow(snapshot):
    return agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )
```

示例选择 individual。它会改变奖励在时间上的建模方式：每行独立计算 GAE，效果不一定等同于 concat。

使用 branch 前，自定义执行器必须实际采集分支，并在 episode.branches 中填写 EpisodeBranch，包括分叉位置、交互 ID、分支奖励和完成状态。只修改 export_mode 不会自动生成反事实分支。

共享前缀默认使用 deduplicate，同一模型动作只计一次；repeat 让每条路径分别计入该动作；weighted 按各分支的 shared_prefix_weight 分配共享动作的贡献。使用 weighted 时应同时设置每个分支的权重。分支记录示例见 [分支轨迹](agentrl_training_CN.md#分支树)。

### 开启异步采集

先用默认同步模式验证任务、工具和奖励，再用下列函数替换原来的训练调用。trainer 是前面创建的 PPOTrainer；训练结束后，训练循环会关闭 workflow 并完成 trainer 的日志收尾。

```python
from triotune.agentrl import AgentConfig, AgentTrainingLoop


async def train_with_prefetch(trainer):
    agent = AgentConfig(
        export_mode="concat",
        max_concurrency=4,
        async_rollouts=True,
        rollout_prefetch_capacity=1,
        max_policy_staleness=1,
    )
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    workflow = agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )
    return await AgentTrainingLoop(trainer, workflow, config=agent).train()
```

预取容量表示提前采集的批次数，max_concurrency 限制候选并发量。max_policy_staleness 必须不小于 rollout_prefetch_capacity。每个 rollout 始终使用同一个快照；异步模式只是让后续采集与当前更新重叠。不要对同一个 trainer 并发调用 step。

### 缩短长对话上下文

工具结果较长时，可以在创建 workflow 前启用上下文压缩：

```python
agent = AgentConfig(
    context_compaction=True,
    compaction_keep_recent_interactions=2,
    compaction_summary_chars=2048,
    prefix_cache_size=256,
)
```

压缩保留初始提示、最近交互和工具错误交互，用摘要替换较旧内容。摘要会改变模型可见的上下文，因此需要重新评估任务成功率。压缩前采样的动作仍保留原始 token，并按片段导出；prefix_cache_size 只缓存模板分词结果，不缓存生成结果。更多配置见 [上下文压缩](agentrl_training_CN.md#上下文压缩与前缀复用)。

## OpenAI 接口

### 使用 OpenAI 接口的工具调用进行训练

如果希望用 function_tool 注册工具并以 OpenAI 风格运行 Agent，可以用 PyTRIOBackend 绑定当前训练快照，再通过 OpenAIEpisodeExecutor 接入同一个训练循环：

```bash
uv sync --locked --extra ppo --extra openai
```

下面复用前面定义的 AnswerScorer 和 rows。与默认 JSON 执行器不同，这里由 Agent 处理工具调用，不需要让模型输出 {"final":"5"}。

```python
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentRolloutWorkflow,
    AgentTrainingLoop,
    EpisodeLimits,
    OpenAIEpisodeExecutor,
    function_tool,
)
from triotune.agentrl.integrations.openai import (
    Agent,
    OpenAIClient,
    OpenAIWorkflow,
    PyTRIOBackend,
)

MODEL = "Qwen/Qwen3.5-4B"


@function_tool(retry_safe=True)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def make_openai_workflow(snapshot):
    client = OpenAIClient(
        backend=PyTRIOBackend(snapshot, model=MODEL, max_tokens=256),
    )
    return OpenAIWorkflow(Agent(
        name="calculator",
        instructions="Use the add tool. Answer with the number only.",
        tools=[add],
        model=MODEL,
        client=client,
    ))


async def train_openai(rows):
    runner = OpenAIEpisodeExecutor(
        make_openai_workflow,
        scorer=AnswerScorer(),
        limits=EpisodeLimits(
            max_turns=5,
            max_tokens_per_turn=256,
            max_generated_tokens=1024,
        ),
    )
    workflow = AgentRolloutWorkflow(runner, max_concurrency=2)
    trainer = PPOTrainer(
        MODEL,
        args=PPOConfig.for_algorithm(
            "grpo", batch_size=1, num_generations=4, max_steps=1,
        ),
        train_dataset=rows,
    )
    return await AgentTrainingLoop(trainer, workflow).train()
```

在 async 入口中调用 await train_openai(rows)。这条路径通过 PyTRIO 采样和训练，需要 PyTRIO 凭据，不需要 OpenAI API key。

make_openai_workflow 每次必须创建新的 workflow、Agent、client 和 backend，并绑定传入的 snapshot，不能在候选之间复用或切换共享 client 的 backend。桥接层会关闭这些候选资源；trainer 持有的采样客户端由 trainer 管理。

评分继续使用原始 row，答案字段不会传入模型。不要同时在 AnswerScorer 和 SDK reward 提交中重复计算同一个终局奖励。外部 OpenAI 服务的响应和 provider trace 不能直接交给 PPO，训练必须保留 PyTRIO 采样的原始 token、行为 log probability 和策略版本。

已有第三方异步执行函数时，可以通过 OpenAIEpisodeExecutor 的 execute 参数接入。session 绑定、标准 AsyncOpenAI client、取消恢复和本地 HTTP 转发的完整用法见 [OpenAI 工具与 session](agentrl_openai_CN.md)。需要第三方执行回调或 HTTP 转发时，按该文档安装 openai-proxy 依赖。

## 诊断和评估

### 训练前检查一组 rollout

先运行上面的 tokenizer 和 probe 命令。自定义应用也可以在调用 loop.train 之前，使用同一个 trainer、workflow 和 AgentConfig 检查一组任务；下面的函数只采集和评分，不调用 optimizer：

```python
async def inspect_rollout(trainer, workflow, agent, rows):
    snapshot = await trainer.create_sampling_snapshot(stop=agent.stop)
    batch = await workflow.collect(
        rows[:1],
        snapshot,
        n_samples=2,
        is_eval=True,
        strict=False,
    )
    print(batch.metrics(prefix="probe"))
    print(batch.groups)
    for episode in batch.episodes:
        print(
            episode.status,
            episode.snapshot_id,
            episode.policy_version,
            episode.group_id,
        )
    for trajectory in batch.trajectories:
        print("action_tokens", sum(trajectory.effective_action_weights))
    return batch
```

strict=False 让失败候选保留在返回的 batch 中，便于检查；它不是把失败轨迹加入训练的开关。查看状态定位格式错误、工具故障或超时，查看 batch.groups 判断候选是否完整及是否被接受。使用 GRPO 等组内相对奖励算法时，应确保采样结果存在奖励差异或有效过程信号；启用 dynamic_sampling 后，没有这些信号的分组会被过滤。

### 保存和读取 episode 日志

在创建 workflow 时传入启用了 log_path 的 AgentConfig，运行后调用 show_episode_log 查看每条 episode 的状态、策略来源、分组和奖励：

```python
import json
from pathlib import Path
from triotune.agentrl import AgentConfig

agent = AgentConfig(log_path="output/agentrl/episodes.jsonl")


def make_logged_workflow(snapshot):
    return agent.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=calculator_tools(),
        scorer=AnswerScorer(),
    )


def show_episode_log(path="output/agentrl/episodes.jsonl"):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            score = episode.get("score") or {}
            print(
                episode["status"],
                episode["snapshot_id"],
                episode["policy_version"],
                episode["group_id"],
                score.get("reward"),
            )
```

log_path 会在 agent.build_workflow 时接入日志器，仅修改配置不会改变已经创建的 workflow。手动创建 AgentRolloutWorkflow 时，通过 logger 参数传入 JsonlEpisodeLogger。不同进程使用不同日志路径，避免同时写入同一个文件。

### 单独评估 checkpoint

从训练报告中取得 checkpoint URI，使用同样的工具和 scorer 评估：

```bash
uv run python examples/agent/calculator/train.py --mode evaluate \
  --checkpoint "trio://YOUR_RUN/weights/YOUR_CHECKPOINT"
```

评估固定一份策略快照，只采集和评分，不更新权重。计算器示例将评估指标保存到 output/calculator-agent/evaluate-report.json；对比成功率、工具错误率与生成 token 数，判断策略是否完成任务以及成本是否合适。
