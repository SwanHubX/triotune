# Agent 执行与训练数据

[English](agentrl_training.md)

需要调整 episode 如何转换为 PPO/GRPO 训练数据时，可以参考本文。安装方法和完整训练步骤见 [README](README_CN.md)；OpenAI 风格的 Agent 执行方式见 [OpenAI 工具与 session](agentrl_openai_CN.md)。

## 采集一个 episode 批次

数据处理路径如下：

```text
任务行 → 固定 PyTRIO 快照 → 候选 episode → scorer
       → 轨迹导出 → 分组校验 → PPOTrainer.step
```

trainer 负责采样快照和优化，workflow 负责工具、episode 执行、评分和轨迹导出。下面的函数使用已有 trainer 和 workflow 完成一次更新；调用方负责管理两者的生命周期。

```python
async def collect_and_update(trainer, workflow, rows):
    snapshot = await trainer.create_sampling_snapshot()
    batch = await workflow.collect(
        rows,
        snapshot,
        n_samples=trainer.args.num_generations,
    )
    print(batch.metrics())
    return await trainer.step(batch.trajectories)
```

整个采集过程使用同一个快照。导出时必须保留原始采样 token ID 及其行为策略 log probability，不能通过重新编码显示文本来构造训练 token。

## 候选身份

collector 会在调用 episode executor 前创建 ExecutionContext。使用以下字段关联任务、候选和模型请求：

| 字段 | 含义 |
| --- | --- |
| task_id | 应用侧任务标识 |
| group_id | 同一个 prompt 的候选组，用于相对奖励比较 |
| rollout_id | 一个逻辑候选，即使导出了多行数据也保持不变 |
| episode_id | 该候选的一次 episode 执行 |
| sample_index / group_size | 候选位置和请求的候选数量 |
| snapshot_id / policy_version | 该候选使用的固定采样策略 |
| interaction_id / parent_interaction_id | 一次模型请求及其父请求 |

从执行上下文读取身份，不要让模型通过工具参数提供这些 ID：

```python
from triotune.agentrl import current_execution_context


def log_tool_context():
    context = current_execution_context()
    if context is not None:
        print(context.task_id, context.rollout_id, context.interaction_id)
```

episode scorer 使用根上下文；interaction scorer 使用对应请求的上下文，评分重试期间也保持一致：

```python
from triotune.agentrl import RewardAssignment, current_execution_context


class InteractionBonus:
    async def score(self, row, episode, interaction):
        assert current_execution_context() == interaction.context
        return RewardAssignment("interaction", 0.1, source="interaction_bonus")
```

将 InteractionBonus() 作为 interaction_scorer 参数传给 EpisodeRunner。runner 会为该奖励填写对应的生成轮次位置。

- asyncio 任务之间的上下文相互隔离，asyncio.to_thread 会复制当前上下文。
- 无论成功、失败还是取消，runner 都会恢复调用方的上下文。
- 进程 worker 需要显式传递并建立自己的上下文。
- 即使请求在生成 token 前失败，也会记录 Interaction。
- resource_owner 表示资源归属，不表示可以关闭共享 client。

Interaction 记录包含 token 区间 [start, end)、工具结果、状态、重试和以秒为单位的延迟。生成、工具和奖励事件记录执行步骤；时间戳使用进程内的单调时钟。

## 轨迹导出与分组策略

### 选择训练数据的行布局

| export_mode | 导出结果 | 适用场景 |
| --- | --- | --- |
| concat | 整个 episode 一行；上下文压缩后每个片段一行 | 优化完整 episode |
| individual | 每个生成轮次一行 | 分别优化各轮次 |
| branch | EpisodeBranch 声明的分支路径 | 比较同一状态下的不同后续动作 |

```python
from triotune.agentrl import AgentConfig

config = AgentConfig(
    export_mode="individual",
    min_usable_group_size=2,
    shared_prefix_policy="deduplicate",
)


def build_workflow(snapshot, tools, scorer):
    return config.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=tools,
        scorer=scorer,
    )
```

individual 会把此前的动作和工具观察保留为带 mask 的上下文。每行接收候选的终局奖励和自身的过程奖励，并独立计算 GAE。当 discount、lambda、过程奖励或 critic bootstrap 跨越轮次边界时，不能假定它与 concat 的优化目标相同。

### 决定是否接受不完整的候选组

min_usable_group_size=None 要求所有请求的候选都可用。设置为 2 时，至少两个候选可用就可以接受该组，导出的片段数量不影响计数。采集时请求的候选数不能小于此阈值。

```python
async def inspect_groups(workflow, rows, snapshot):
    batch = await workflow.collect(
        rows, snapshot, n_samples=4, strict=False,
    )
    print(batch.groups)
    print("dropped_groups", batch.dropped_groups)
    print("accepted_rows", len(batch.trajectories))
    return batch
```

strict=False 会返回失败尝试供检查；只有被接受的分组才进入 batch.trajectories，失败不会被替换成零奖励样本。strict=True 会在分组未达到接受阈值时抛出异常。评估和 pass@k 仍然计入失败尝试。

### 实现自定义 exporter

实现 export(episode)，返回 RolloutExport，并保留以下信息：

| 输出 | 必须保留的含义 |
| --- | --- |
| logical_rollout_id / group_id | 原始候选和分组身份 |
| fragment_map | 各 fragment ID 到对应 interaction ID 的映射 |
| action_keys | 每个有效 token 对应的 (interaction_id, original_output_offset) |
| loss_mask | 区分有效模型动作与插入或重复的上下文 |
| tokens / logprobs | 原始采样 token 及其行为策略概率 |

collector 会校验 token 身份、因果位置和条件前缀。上下文位置没有 action key，不含有效动作的行会被省略；初始 prompt 不同的行不能共用一个 PPO 组。完整实现可参考 [轨迹 exporter](../../triotune/agentrl/rollout/trajectory.py)。

## 结构化奖励与直接 advantage

### 选择奖励目标

| target | position | 效果 |
| --- | --- | --- |
| trajectory | None | episode 终局奖励，或广播到所有有效动作的 advantage |
| interaction | 生成轮次索引 | 作用于该次交互生成结果的信号 |
| turn | 生成轮次索引 | 作用于该轮次的信号 |
| token | (轮次索引, 输出 token 偏移) | 作用于一个原始输出 token 的信号 |

索引从 0 开始，指向原始生成输出，包括返回的停止 token。turn 和 interaction 的标量奖励放在最后一个输出 token 上。interaction 的 position 按生成轮次计数，不计入没有输出的失败请求。

下面的 scorer 同时计算答案正确性奖励和每轮的小额工具成本：

```python
from triotune.agentrl import EpisodeScore, RewardAssignment


class OutcomeWithToolCost:
    async def score(self, row, episode):
        success = (
            episode.status == "completed"
            and episode.final_text == row["answer"]
        )
        costs = tuple(
            RewardAssignment(
                "turn", -0.01 * len(turn.tool_results),
                position=index, source="tool_cost",
            )
            for index, turn in enumerate(episode.turns)
            if turn.tool_results and turn.generation.tokens
        )
        return EpisodeScore(float(success), success, assignments=costs)
```

使用旧式 process_rewards 向量时，每个向量的长度必须与该轮次输出 token 数完全一致。exporter 会拒绝未知位置和长度不匹配的数据。

### 在评分器重试时安全提交奖励

RewardLedger 按 rollout、source、mode、target 和 position 分组。先应用 discount，再按 sum、mean 或 last 聚合；不同 source 的贡献相加，同一个聚合组必须使用同一种聚合规则。

```python
from triotune.agentrl import RewardAssignment, RewardLedger

ledger = RewardLedger()
contributions = (
    RewardAssignment("trajectory", 2.0, aggregation="mean", source="judge"),
    RewardAssignment("trajectory", 4.0, aggregation="mean", source="judge"),
)
identity = {"logical_rollout_id": "candidate-0", "submission_id": "judge-0"}

assert ledger.submit(contributions, **identity) == contributions
assert ledger.submit(contributions, **identity) == ()  # 完全相同的重试。
outcome, process, direct = ledger.aggregate()
assert outcome == 3.0
```

- 同一 submission ID 携带不同内容会抛出 AgentContractError，ledger 不会被修改。新的贡献需要新的 ID。
- 没有显式 ID 时，同一 rollout 中同 source 的整组 trajectory 奖励会去重，过程奖励仍然累加。可能重试的提交应使用 ID。
- 将当前候选接受的 assignments 传给 EpisodeScore.assignments，不要把其他候选的 ledger 内容附加到该分数上。

### 添加不改变 critic 目标的 advantage

只有信号需要直接修改策略 advantage 时才使用 mode="advantage"。将以下 assignment 加到 scorer 返回的 EpisodeScore.assignments 中：

```python
from triotune.agentrl import EpisodeScore, RewardAssignment

score = EpisodeScore(
    reward=1.0,
    success=True,
    assignments=(
        RewardAssignment(
            "trajectory", 0.5, mode="advantage", source="prm",
        ),
    ),
)
```

| 信号 | 奖励归一化 / GAE | Critic returns | 最终 advantage 归一化 |
| --- | --- | --- | --- |
| 普通终局或过程奖励 | 按配置的奖励流程处理 | 参与计算 | 作用于计算得到的 advantage |
| 直接 advantage | 跳过奖励归一化和 GAE | 不变 | 仍然生效 |

trajectory advantage 会作用于每个有效的原始模型输出 token。工具观察和重复且被 mask 的上下文，其直接 advantage 为 0。共享动作仍受片段去重或分支权重控制。

RewardLedger.aggregate 返回 outcome、process 和 direct；trajectory advantage 表示为 direct[None] = (value,)。

### 读取 episode 日志中的奖励字段

| 日志字段 | PPO 处理前的含义 |
| --- | --- |
| outcome_reward | EpisodeScore.reward 加结构化 trajectory 奖励，附加到最后一个有生成输出的交互 |
| process_rewards | 旧式和结构化过程奖励完成聚合、折扣与对齐后的结果 |
| token_advantages | 独立记录的、只影响策略的 advantage |
| reward | outcome_reward + sum(process_rewards)，已经包含这两个部分 |

RewardEvent 描述这些数值，不要把事件中的数值再加到奖励上。它们尚未经过 shaping、归一化、GAE、KL 惩罚和损失加权。分支终局奖励记录在分支上，并在分支导出时替代 episode 终局奖励。

EpisodeRunner.submit_reward 只在评分和对齐成功后提交。完全相同的重试复用已提交结果，不会新增事件；提交不同内容会抛出 AgentContractError，导出后也一样。重新评分应使用独立记录。

## PPO 奖励插件接口

需要在 PPO 准备阶段处理奖励时，使用 PPOAlgorithmPlan。插件在本地执行，执行时间早于 PyTRIO Datum 构造。

| Hook | 用途 |
| --- | --- |
| reward_shaper.shape(trajectory, reward) | 转换候选的终局奖励 |
| reward_shaper.shape_process(trajectory, rewards) | 转换已对齐的过程奖励向量 |
| process_shaper | 显式覆盖过程奖励 shaping |
| normalizer.normalize(rows, group_ids, masks=None) | 在比较组之间归一化终局奖励行 |

下面的最小示例展示终局奖励和过程奖励各自的处理结果：

```python
from triotune.trainer.ppo import (
    PPOAlgorithmPlan, PPOConfig, Trajectory, prepare_advantages,
)


class Shaper:
    def shape(self, trajectory, reward):
        return reward

    def shape_process(self, trajectory, rewards):
        return [3.0 * value for value in rewards]


class Normalizer:
    def normalize(self, rows, group_ids, masks=None):
        return [[2.0 * value for value in row] for row in rows]


plan = PPOAlgorithmPlan(reward_shaper=Shaper(), normalizer=Normalizer())
row = Trajectory([1], [2], [-1.0], 1.0, "group", token_rewards=[1.0])
batch = prepare_advantages(
    [row], PPOConfig(use_critic=False, reward_norm=None, adv_norm=None), plan,
)
assert batch.returns == [[5.0]]  # 过程奖励 3 + 归一化终局奖励 2。
```

构造 PPOTrainer 时传入 plan=plan，即可在训练中使用该配置。

- normalizer 按逻辑候选计数，不按导出片段计数。终局奖励准备阶段 masks 为 None，它不是 token loss mask。
- normalizer 可以声明 config 参数来接收 NormConfig。旧式双参数和命名 config 的 hook 仍受支持；仅接受 **kwargs 的 hook 会收到 masks，不会隐式收到 config。
- 显式 process_shaper 会替代 reward_shaper.shape_process，两者不会串联。旧式过程奖励 shape 或 callable 形式仍受支持。
- 过程奖励输出必须有限且与 completion 对齐。hook 错误会向上传播；内部 TypeError 不会触发第二次插件调用。
- 插件对象只在本地使用；PyTRIO 原生更新和自定义更新接收相同的数值目标。

## 分支树

当执行器已经从同一个状态实际采样出不同后续路径时，使用分支导出。仅选择 export_mode="branch" 不会自动生成分支。

```text
共享请求
├── tool-a 请求 → final-a 请求 → 奖励 1.0
└── tool-b 请求 → final-b 请求 → 奖励 0.0
```

在执行器中完成交互采集后附加分支记录。传给下面函数的 ID 必须已经存在于 episode.interactions 中，且从 fork_id 开始组成父子关系连续的后缀路径：

```python
from triotune.agentrl import EpisodeBranch


def attach_branches(episode, fork_id, left_ids, right_ids):
    episode.branches = [
        EpisodeBranch(
            branch_id="tool-a",
            parent_interaction_id=fork_id,
            interaction_ids=tuple(left_ids),
            group_id="tool-choice",
            reward=1.0,
            shared_prefix_weight=0.5,
        ),
        EpisodeBranch(
            branch_id="tool-b",
            parent_interaction_id=fork_id,
            interaction_ids=tuple(right_ids),
            group_id="tool-choice",
            reward=0.0,
            shared_prefix_weight=0.5,
        ),
    ]
```

在 AgentConfig 中设置 export_mode="branch"，并选择 shared_prefix_policy：

| 策略 | 共享动作的贡献 |
| --- | --- |
| deduplicate | 第一次出现时参与损失，后续副本被 mask |
| repeat | 每条路径分别计入自己的副本 |
| weighted | 每个分支保留一次，并按声明的前缀权重归一化 |

分支独有动作的权重保持为 1。actor 和 critic 使用相同权重。由于原生 objective 没有 token 权重字段，weighted 行使用自定义 PyTRIO loss 接口。即使选择 repeat，冲突的 token、logprob 或前缀仍会被拒绝。

- 分支 reward 是完整的分支终局奖励，会替代 episode 终局奖励。
- 过程奖励只作用于该路径中的交互。trajectory advantage 作用于所有已完成路径，不改变分支终局奖励或 critic returns。
- 同级分支共用 rollout 内的 group_id；不同 rollout 的分支奖励基线不能合并。
- 只有完整的同级分支组参与训练。未完成分支应设置 completed=False、reward=None，不能用 0 代替未知奖励。
- 通过 batch.branch_groups 查看分支完整性，通过 batch.groups 查看候选完整性。RolloutExport.branch_map 会保留未完成的同级分支。
- PPO 在终局奖励归一化和序列归约时，每个分支只计一次，即使上下文压缩把它拆成了多个片段。

## 上下文压缩与前缀复用

旧工具交互导致请求过长时，在创建 workflow 前开启压缩：

```python
from triotune.agentrl import AgentConfig

config = AgentConfig(
    context_compaction=True,
    compaction_keep_recent_interactions=2,
    compaction_summary_chars=2048,
    prefix_cache_size=256,
)
```

| 设置或行为 | 作用 |
| --- | --- |
| compaction_keep_recent_interactions | 原样保留最近的交互块 |
| compaction_summary_chars | 限制默认摘要长度 |
| 初始指令和工具错误交互 | 始终保留 |
| 自定义保留 hook | 选择其他需要保留的交互 |
| 自定义 summarizer | 接收被省略的交互和已有摘要消息，以同步或异步方式返回文本或消息 |
| prefix_cache_size | 限制模板分词缓存数量，不缓存生成结果或 logprob |

也可以向 EpisodeRunner 传入自定义 ContextCompactor。

压缩会改变模型可见的上下文，启用后需要重新评估成功率。原始采样 token 不会被重新编码：每次上下文重置都使用精确的 prompt ID 开始一个新片段。各片段有独立的 prompt 组，但保留原候选的奖励组，过程信号仍附着在原始动作上。分支路径同样遵循这一规则。

Datum 批次中的本地 prefix_groups 附加信息描述共享输入前缀。目前 PyTRIO 接受普通 Datum 列表，没有通用的树注意力选项；模板缓存和分支动作加权不意味着共享模型前向计算。

## 异步 rollout 调度

通过 AgentTrainingLoop，让固定快照的采集与串行 PPO 更新有界重叠：

```python
from triotune.agentrl import AgentConfig, AgentTrainingLoop


async def train_with_overlap(trainer, tools, scorer):
    config = AgentConfig(
        async_rollouts=True,
        rollout_prefetch_capacity=1,
        max_policy_staleness=1,
        max_concurrency=4,
    )
    snapshot = await trainer.create_sampling_snapshot(stop=config.stop)
    workflow = config.build_workflow(
        tokenizer=snapshot.tokenizer, tools=tools, scorer=scorer,
    )
    return await AgentTrainingLoop(trainer, workflow, config=config).train()
```

| 设置 | 含义 |
| --- | --- |
| rollout_prefetch_capacity | 在更新前预取的批次数 |
| max_policy_staleness | 允许的最大数值策略年龄，必须覆盖预取容量 |
| max_concurrency | workflow 级候选并发上限，由重叠的采集调用共享 |

每个 rollout 始终使用一个快照。批次按启动顺序消费，同一 trainer 的更新保持串行。来自未来版本或超过年龄限制的策略数据，会在 reference 评分、critic 计算和优化之前被拒绝。

增量采集时，collect_iter 会逐个返回已完成的 prompt 组批次，max_pending_groups 单独限制已调度的分组任务数。恢复时仅处理从未启动的任务行，并继续使用原始快照：

```python
async def resume_unstarted(workflow, partial_batch, unstarted_rows, snapshot):
    return await workflow.resume(
        partial_batch,
        unstarted_rows,
        snapshot,
        n_samples=4,
    )
```

调用方负责跟踪哪些行已经启动；resume 不会自动识别或跳过它们。使用不同快照会被拒绝。正在执行的工具任务会被取消并等待收尾，不会自动重放，因为副作用可能无法安全重复。merge_batches 合并已完成批次时会拒绝重复 ID。

## PPO 数据来源与策略年龄

将自定义批次交给优化器前，检查策略版本：

```python
from triotune.trainer.ppo.trajectory import validate_policy_versions


async def update_checked(trainer, batch):
    validate_policy_versions(
        batch.trajectories,
        current_version=trainer.state.global_step,
        max_staleness=1,
    )
    return await trainer.step(batch.trajectories, max_staleness=1)
```

PPOTrainer.step 已经执行这项校验；如果需要在数据交给 trainer 前提前验证，可单独调用该函数。

| 版本数据 | 接受条件 |
| --- | --- |
| 整数 | 0 ≤ current_version − sampled_version ≤ max_staleness |
| 不透明标签 | 与当前标签完全相同 |
| 一个批次 | 全部使用同一个版本，不能混合已知和未知版本 |
| 完全没有版本信息的直接构造轨迹 | 为兼容旧调用而保留支持，但无法检查策略年龄 |

数值版本只属于导出它的 trainer。无关运行中相同的 step 数不代表相同策略。

需要跟踪底层请求时，batch_datums(prepared_batch) 会返回兼容 SDK 的 Datum 列表以及按顺序排列的数据来源附加信息。actor 和 critic 通过 datum_provenance 保留最近提交行的来源映射，需要时可与远端 request ID 一起保存。来源字符串不会放入数值 loss inputs。

critic 行保留原轨迹来源，token 归约使用有效动作权重。超长惩罚按每个逻辑 episode 或分支中的有效动作数计算，不重复计入片段上下文。
