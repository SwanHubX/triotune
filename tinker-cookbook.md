# Tinker RL 训练框架设计总结

> 本文总结 `tinker-cookbook/tinker_cookbook/rl/train.py` 的框架设计原理与执行步骤。

## 1. 框架定位

`train.py` 是面向通用 MDP（Markov Decision Process）的强化学习训练编排层。它不负责定义具体任务环境、奖励函数或底层 GPU 算子，而是把以下模块组织成完整训练生命周期：

- 通过 `RLDatasetBuilder`、`RLDataset` 和 `EnvGroupBuilder` 构造任务与环境组；
- 使用 `SamplingClient` 并发生成轨迹组；
- 将轨迹奖励转换为组内 advantage，再组装成 `tinker.Datum`；
- 通过 `TrainingClient` 提交前向、反向和优化请求；
- 在权重更新后创建新的采样客户端，保证后续 rollout 使用最新策略；
- 统一处理评估、KL 约束、检查点、断点续训、日志、追踪及 rollout 容错；
- 支持同步、同步流式 minibatch 和异步有限离策略三种执行模式。

因此，该文件的核心职责是“训练控制面”：CPU 侧负责数据流、任务并发和状态管理，重计算由 Tinker 服务执行。

## 2. 总体架构

```text
Config
  |
  v
main：初始化、恢复状态、构建客户端/数据集/评估器
  |
  +--> 同步模式 do_sync_training
  +--> 同步流式模式 do_sync_training_with_stream_minibatch
  +--> 异步模式 do_async_training
            |
            v
EnvGroupBuilder -> rollout -> TrajectoryGroup
            |
            v
prepare_minibatch：指标 -> advantage -> Datum -> 可选 KL 惩罚
            |
            v
forward_backward_async -> optim_step_async
            |
            v
KL/诊断指标 -> 保存权重/检查点 -> 新 SamplingClient
            |
            v
评估、日志、下一轮 rollout
```

## 3. 框架设计原理

### 3.1 配置与运行对象分离

框架使用 `chz` 配置类描述训练方案，再由 builder 创建运行时对象：

- `Config`：统一管理模型、数据集、损失函数、优化器、采样、评估、日志和执行模式；
- `KLReferenceConfig`：描述 KL 参考模型及可选检查点；
- `StreamMinibatchConfig`：描述流式训练的组数和 minibatch 划分；
- `AsyncConfig`：描述允许的最大离策略步数和每批最少轨迹组数。

这种设计把“实验声明”与“运行时资源”解耦，使相同训练引擎能够复用到不同 recipe、环境与评估器。

### 3.2 以轨迹组作为 RL 基本单位

训练数据不是孤立样本，而是 `TrajectoryGroup`。同一问题或环境实例生成多条轨迹，框架在组内根据相对奖励计算 advantage。这样可以进行组内基线消减，并支持过滤奖励完全相同、没有相对学习信号的组。

代码中的维度命名遵循：`P` 表示 problem/group，`G` 表示组内轨迹，`T` 表示 token，`D` 表示最终训练 datum。

### 3.3 采样、训练与评估客户端职责分离

- `ServiceClient`：创建训练及采样客户端；
- `TrainingClient`：持有可训练状态，执行 `forward_backward`、`optim_step` 和权重保存；
- `SamplingClient`：持有某个权重快照，只负责 rollout 或评估；
- KL 参考 `SamplingClient`：固定参考策略，用于计算和注入 KL 惩罚。

`SamplingClient` 对应的是权重快照，而不是自动跟随训练状态的动态视图。因此每次参数更新后，框架都会保存最新权重并创建新的采样客户端，防止继续从旧策略采样。

### 3.4 异步 API 与流水线优先

框架尽量先提交任务，再等待结果：

- 同一批中的多个 rollout 使用 `asyncio.gather` 并发执行；
- 多个 evaluator 作为独立 task 并行运行；
- `train_step` 提前提交下一子批的 `forward_backward_async` 和 `optim_step_async`；
- 流式模式累计够一个 minibatch 就提交前反向，不等待整批 rollout 完成；
- 异步模式让数据加载、rollout、训练和评估四类循环并发运行。

其目的不是改变 RL 数学定义，而是减少采样、网络请求与服务端计算之间的空闲时间。

### 3.5 统一的数据准备与更新闭环

不论选择哪种执行模式，核心更新链路保持一致：

```text
TrajectoryGroup
  -> trajectory metrics
  -> group-relative advantages
  -> tinker.Datum
  -> optional KL penalty
  -> forward/backward
  -> optimizer step
  -> sample/train KL metrics
  -> checkpoint or weight snapshot
  -> fresh SamplingClient
```

三种模式主要改变“何时取得足够轨迹”和“采样与训练如何重叠”，而不重复定义训练数学逻辑。

### 3.6 有界离策略，而不是无约束异步

异步模式在 `WrappedTrajectoryGroup` 中记录 `sampling_client_step`。训练时比较该步与当前训练步：

- 差值不超过 `max_steps_off_policy` 时允许训练；
- 样本过旧且数据加载仍在进行时，将原 `EnvGroupBuilder` 重新入队采样；
- 关闭阶段不再重入队，避免有界队列死锁。

这在吞吐量与策略新鲜度之间建立显式边界。

### 3.7 可恢复性与权重语义明确

框架区分三类启动方式：

- 从训练日志中的最近检查点恢复：同时恢复权重与优化器状态；
- 从用户指定检查点启动：只加载权重，使用新的优化器状态；
- 全新训练：从基础模型创建 LoRA 训练客户端。

检查点也分为周期检查点、轻量采样权重快照、滚动恢复检查点和最终永久检查点，以平衡恢复能力、采样刷新成本与存储成本。

### 3.8 横切能力集中处理

框架把与具体环境无关的能力集中在训练编排层：

- rollout 错误策略和错误计数；
- 终止原因对应的奖励调整策略；
- 轨迹摘要、完整 logtree、标量指标与 W&B 日志；
- trace event、迭代耗时和可选 Gantt 图；
- 文件名清洗、日志范围和训练/评估 split 管理。

环境只需关注交互与奖励，训练循环无需了解任务内部细节。

## 4. 主训练流程（有序步骤）

1. **接收配置与可选 rollout executor。** `main` 接收 `Config`；如传入进程池或其他 executor，则将 rollout 调度器切换到该 executor。多进程应采用 `spawn` 或 `forkserver`，避免 Tinker 客户端后台事件循环在线程 `fork` 后失效。

2. **初始化日志与追踪。** 创建机器学习日志器及 `TrainingRunStore`；若启用 trace，则初始化事件文件，并为主异步任务命名。

3. **查找断点。** 从 `log_path` 查找最近检查点，并据此确定 `start_batch`；没有断点时从第 0 批开始。

4. **创建服务客户端并写入元数据。** 建立 `ServiceClient`，附带 recipe、W&B 链接与 renderer 等元数据，同时检查 renderer 是否符合模型推荐设置。

5. **创建或恢复训练客户端。** 若为断点续训，则恢复权重和优化器；若指定 `load_checkpoint_path`，则只加载权重；否则为基础模型创建 LoRA 训练客户端。

6. **构建 tokenizer、数据集与 rollout 策略。** 获取 tokenizer，调用 `dataset_builder` 得到训练集及可选测试集，并根据模型默认值或显式配置建立 rollout 容错策略。

7. **构建评估器。** 实例化配置中的 evaluator；若数据集 builder 返回测试集，则额外包装为 `RLTestSetEvaluator`。

8. **确定训练范围。** 用数据集长度计算 `num_batches`，再由 `max_steps` 截断得到 `end_batch`。

9. **创建可选 KL 参考客户端。** 当 `kl_penalty_coef > 0` 时，校验 `kl_reference_config` 并创建固定参考策略；否则不执行 KL 惩罚链路。

10. **创建检查点管理器。** 注入训练客户端、保存周期、TTL、滚动检查点周期及日志存储对象。

11. **选择执行模式。** `async_config` 优先选择异步模式；否则若存在 `stream_minibatch_config`，选择同步流式模式；两者均未配置时选择普通同步模式。

12. **执行迭代训练。** 所选训练循环负责评估、rollout、数据准备、参数更新、采样客户端刷新、指标写入与滚动检查点。

13. **保存最终状态。** 实际执行过训练时保存不过期的最终检查点；若训练早已完成，则只完成检查点管理器的收尾工作。

14. **释放资源。** 关闭可选 executor、解除全局 rollout executor、关闭日志器，并报告训练完成。

## 5. 单次参数更新流程（有序步骤）

1. **统计轨迹。** `prepare_minibatch` 使用环境 builder 的 logging tags 计算轨迹级聚合指标，并打印少量代表性轨迹用于人工诊断。

2. **计算 advantage。** `compute_advantages` 对各轨迹组计算组内相对优势。

3. **组装训练数据。** `assemble_training_data` 把轨迹、采样 log-probability 和 advantage 转换为 `tinker.Datum` 列表。

4. **注入可选 KL 惩罚。** 若启用参考策略，查询参考 log-probability，并按系数及位置折扣把 KL 惩罚合并进训练信号，同时记录相关指标。

5. **划分 optimizer substep。** 普通更新由 `train_step` 将 datum 划分为最多 `num_substeps` 个子批；流式更新则依据 `groups_per_batch`、`num_substeps` 和 `num_minibatches` 计算每个 minibatch 所需轨迹组数。

6. **提交前向与反向。** 调用 `forward_backward_async`；发送前通过 `_remove_mask` 移除 `loss_fn_inputs` 中的 `mask` 字段，其他损失输入保持不变。

7. **提交优化步骤。** 使用固定的 Adam 参数 `beta1=0.9`、`beta2=0.95`、`eps=1e-8` 和配置学习率调用 `optim_step_async`。请求会尽量与前反向请求流水化提交。

8. **收集训练 log-probability 和优化指标。** 从服务响应中提取每个 datum 的训练 log-probability，并合并最终 optimizer metrics。

9. **计算采样/训练偏差。** 比较 rollout 时的采样 log-probability 与训练前向得到的 log-probability，形成 KL 等一致性诊断指标。

10. **保存新权重并刷新采样客户端。** 到达周期时保存完整检查点，否则保存轻量权重快照；两种情况都返回绑定最新权重的新 `SamplingClient`。

11. **计算可选 post-update KL。** 若 `compute_post_kl=True`，使用更新后的采样客户端额外计算参数更新前后策略差异。

12. **返回闭环结果。** 返回新采样客户端与完整指标，供下一轮 rollout 和日志系统使用。

## 6. 三种执行模式

### 6.1 普通同步模式

该模式最容易推理，每轮数据完全由同一个当前策略生成，属于严格的 on-policy 执行方式。

1. 为当前权重创建采样客户端。
2. 按配置周期，用当前采样客户端并行执行全部 evaluator。
3. 从数据集取得本轮 `EnvGroupBuilder` 列表。
4. 并发执行所有组的 rollout，并为有限数量的组开启详细 logtree。
5. 统计 rollout 错误，去除失败或完全无效的结果；若整批均失败，则跳过本轮更新。
6. 按需导出每条轨迹的 JSONL 摘要。
7. 若启用配置，则移除奖励恒定的轨迹组。
8. 对完整批次执行统一的数据准备和参数更新流程。
9. 使用更新后的权重创建新的采样客户端，并按需保存滚动检查点。
10. 写入轨迹、优化、KL、错误、耗时和 trace 指标，然后进入下一轮。

### 6.2 同步流式 minibatch 模式

该模式仍以迭代为边界保持 on-policy，但在同一迭代内让 rollout 与训练重叠。

1. 用本轮开始时的采样客户端启动该批所有 rollout task。
2. rollout task 完成后，把 `WrappedTrajectoryGroup` 放入异步队列；失败或被过滤的组以 `None` 表示。
3. 训练端持续从队列消费结果，累计到 `groups_per_minibatch` 后立即准备 datum。
4. 为当前 optimizer substep 的每个 minibatch 依次提交 `forward_backward_async`，暂不等待返回。
5. 当前 substep 的全部前反向请求提交完毕后，提交一次 `optim_step_async`。
6. 收集所有前反向结果与 optimizer 结果，再进入下一个 substep。
7. 汇总整批轨迹和采样客户端版本指标，计算 KL 诊断，保存新权重并创建新采样客户端。
8. 导出轨迹摘要、保存滚动检查点并记录指标。

此模式要求：`groups_per_batch` 能被 `num_substeps` 整除，且每个 substep 的组数能被 `num_minibatches` 整除。

### 6.3 异步有限离策略模式

该模式通过队列同时运行数据供给、轨迹生成、训练和评估，以更高吞吐换取可控范围内的 off-policy 数据。

1. 创建有界的环境 builder 队列和轨迹结果队列，并保存初始训练/采样状态。
2. dataloader 循环按数据集顺序把 `EnvGroupBuilder` 放入队列。
3. 启动 `groups_per_batch` 个 rollout worker；每个 worker 读取 builder，记录当前 `sampling_client_step`，生成轨迹后包装为 `WrappedTrajectoryGroup`。
4. 训练循环从结果队列读取轨迹，比较生成步与当前训练步，过滤超过 `max_steps_off_policy` 的陈旧数据。
5. 对过旧数据，在正常运行阶段重新入队对应 builder；数据源结束后直接丢弃，避免关闭时死锁。
6. 非流式异步模式累计至少 `groups_per_batch` 个有效组后更新；若同时配置流式 minibatch，则交由流式训练函数按 minibatch 消费。
7. 参数更新后替换共享采样客户端，递增 `sampling_client_step`，并通知 evaluation loop。
8. evaluation loop 在收到权重更新通知且命中 `eval_every` 时，用当时的客户端快照执行并行评估。
9. dataloader 耗尽后，为每个 worker 发送 `_Shutdown`；最后一个 worker 再通知训练循环；训练循环最终通知评估循环，形成级联关闭。
10. 所有循环通过 `asyncio.gather` 等待完成，保证任务有序收尾。

## 7. 评估、日志与可观测性

1. `run_evaluations_parallel` 为每个 evaluator 创建独立 task，结果汇总到同一个 metrics 字典。
2. `RLTestSetEvaluator` 可以额外输出逐轨迹摘要；其他 evaluator 只需实现采样客户端调用接口。
3. `_get_logtree_scope` 同时管理可读 HTML 和结构化 logtree JSON，便于人工查看完整模型交互。
4. `_maybe_export_rollout_summary_jsonl` 输出适合机器分析的逐轨迹 JSONL。
5. `ml_logger` 统一记录标量指标，并可连接 W&B。
6. `trace.trace_iteration` 记录分阶段耗时；可选输出 trace events 和 Gantt HTML，用于分析采样、训练与评估的重叠情况。

## 8. 关键扩展点

- **任务扩展**：实现 `RLDatasetBuilder`、`RLDataset` 与 `EnvGroupBuilder`，无需修改主训练循环。
- **环境与奖励扩展**：由环境产生 trajectory、reward 和自定义 metrics；训练器只消费标准类型。
- **rollout 策略扩展**：通过 `RolloutStrategy` 定义失败是立即终止、重试还是保留最小可用组。
- **评估扩展**：通过 `SamplingClientEvaluatorBuilder` 注入任意基于当前策略快照的评估逻辑。
- **损失扩展**：使用 `loss_fn` 与 `loss_fn_config` 选择 Tinker 服务支持的 importance sampling、PPO 等损失。
- **调度扩展**：通过 `rollout_executor` 把 rollout 放到独立进程或其他执行后端。

## 9. 实现中的关键约束

- 更新权重后必须创建新的 `SamplingClient`，旧客户端不会自动同步新权重。
- 异步样本必须携带生成它的策略步数，否则无法判断 off-policy 陈旧程度。
- streaming 的组数划分必须满足整除关系，否则 minibatch/substep 语义不完整。
- `kl_penalty_coef > 0` 时必须提供 `kl_reference_config`。
- rollout 失败或过滤后可能返回 `None`，训练循环必须先筛除无效组。
- 环境对象是单次使用的；需要重新采样时重新使用 builder 创建新环境，而不是重置旧环境。
- 恢复训练与加载预训练权重含义不同：前者恢复优化器，后者不恢复。
- 进程型 rollout executor 不应使用 Linux 默认 `fork` 启动方式。

## 10. 总结

`train.py` 的设计重点是用一套统一的训练闭环承载不同并发策略。同步模式优先保证简单、严格 on-policy；同步流式模式在不跨迭代复用旧样本的前提下重叠采样与训练；异步模式则通过策略版本标记和最大离策略步数控制吞吐与数据新鲜度之间的折中。配置对象、builder、标准轨迹类型、统一数据准备、客户端快照和集中式可观测性共同构成了这一框架的可复用基础。
