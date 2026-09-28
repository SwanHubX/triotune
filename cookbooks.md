# TRL GRPO 前端 API 与 Tinker 格式适配性

## Tinker Cookbook 代码步骤（精简）

1. 读取 `Config`，初始化日志、追踪和断点信息。
2. 创建 `ServiceClient`，再创建或恢复 `TrainingClient`。
3. 由 `RLDatasetBuilder` 构造数据集；每批返回多个 `EnvGroupBuilder`。
4. 用当前 `SamplingClient` 并发执行每个环境组的 rollout，得到 `TrajectoryGroup`。
5. 计算组内 advantage，调用 `assemble_training_data` 转为 `tinker.Datum`。
6. 按需用参考 `SamplingClient` 注入 KL 惩罚。
7. 调用 `forward_backward_async`，再调用 `optim_step_async` 更新参数。
8. 计算采样/训练 KL 和其他指标，保存检查点或采样权重。
9. 用最新权重创建新的 `SamplingClient`，继续下一批或并行评估。
10. 训练结束后保存最终状态并释放客户端、executor 和日志资源。

## TRL 前端 API 与适配结论

TRL 的入口是 `GRPOTrainer(model, reward_funcs, args, train_dataset, ... )`：数据行提供 `prompt`，生成结果提供 token 与 log-probability，`reward_funcs` 返回标量奖励，Trainer 在本地计算组内 advantage、比率/裁剪损失并执行 PyTorch 优化；`rollout_func` 和 `environment_factory` 是可选扩展点。

结论：GRPO 的算法流程可以映射，但该 API 格式不能完全直接用于 Tinker，至少需要适配以下部分：

1. **模型/客户端**：TRL 要求 Hugging Face `PreTrainedModel`、`generate`/`forward` 和本地 optimizer；Tinker 使用远程 `TrainingClient`/`SamplingClient`，不能直接传入 `model` 或复用 Trainer 的 Accelerate/vLLM 状态。
2. **数据格式**：TRL 使用 `Dataset` 行（至少含 `prompt`）；Tinker 使用 `RLDatasetBuilder -> EnvGroupBuilder -> TrajectoryGroup`，环境是单次创建的，不能把 dataset 行直接当作 rollout 输入。
3. **奖励接口**：TRL 的 `reward_funcs(prompts, completions, **kwargs) -> list[float]` 是批量回调；Tinker 奖励来自 `Env.step`、`compute_group_rewards` 和轨迹指标，需额外封装环境状态及组级奖励。
4. **rollout 返回值**：TRL `rollout_func` 必须返回 `prompt_ids`、`completion_ids`、`logprobs` 字典；Tinker 采样结果要组织成 `Transition/Trajectory/TrajectoryGroup`，再由 `assemble_training_data` 生成 `Datum`。
5. **损失输入**：TRL 直接对 Torch logits 计算 GRPO/IS/PPO；Tinker 的 `forward_backward_async` 接收 `Datum`（`model_input` + `loss_fn_inputs`，如 `target_tokens`、`logprobs`、`advantages`）和 `loss_fn`，且当前流程会移除 `mask` 字段，不能直接传 TRL 的 batch。
6. **文本处理与权重同步**：TRL 依赖 tokenizer/processor 和 chat template；Tinker 依赖 renderer/`ModelInput`。每次保存新权重后必须重新创建 `SamplingClient`，不能假设采样端自动同步。

设计 Tinker 前端 API 时，应明确区分环境与奖励、采样与训练客户端、轨迹与 `Datum`、同步与异步 future，并为组大小、策略版本、off-policy 上限、KL 参考策略及 loss-specific 输入提供显式字段。


# FireWorks 的 cookbook 代码复杂度以及框架完整度判断

## 小总结

没有完整的、相对通用的训练框架，主要用于推理。

然后训练的话做了和tinker一样的api，比如client和tinker一致，具体参考 `training/utils/client.py` 。

其训练 recipes 里的逻辑是：

```python
设置训练、推理客户端（single） -> 处理数据集（single） -> 训练loss（每个算法single，比如grpo.py） -> 奖励计算 ……
```

也就是每个算法，或者说案例都是单独的，没有完整的框架

## 1. `training/recipes` 案例的通用性

结论：这些案例采用“公共训练骨架 + 算法/任务专用部分”，不是每个案例都从零实现强化学习。

1. `async_rl_loop.py` 是最通用的 RL recipe。用户主要提供 `rollout_fn_factory`；数据读取、并发 rollout、分组、优势计算、KL/TIS、off-policy gate、流水线、权重同步、评估、检查点和日志由主循环处理。
2. `rl_loop.py` 是较易修改的同步 GRPO 模板，但目标函数、采样和奖励仍明确写死为 GRPO 流程，适合 fork 后改造。
3. `igpo_loop.py` 包含 IGPO 专用的逐轮信息增益评分和 advantage 处理，复用公共客户端、`PromptGroup`、rollout 和检查点设施，因此属于“通用骨架上的专用算法”。
4. `dpo_loop.py`、`sft_loop.py`、`orpo_loop.py` 等是其他训练范式，不能视为同一个 RL 算法模板。

因此，recipe 层的复杂度主要来自异步调度、服务生命周期、数据/权重一致性和可恢复性；任务奖励或 rollout 通常是明确的扩展点，而不是重复编写全部训练基础设施。

## 2. `training/utils/rl` 的算法框架完整度

结论：FireWorks 已有相对完整、模块化的 RL 框架，但还不是所有算法都能由一个统一配置无差别切换。

1. `grpo.py` 实现 token-level PPO/GRPO 裁剪、TIS（train-inference importance sampling）和可选 reference KL；`gspo.py` 实现 sequence-level ratio、裁剪和 TIS。二者都是传给 `forward_backward_custom` 的损失闭包，不负责 rollout、数据集或 checkpoint。
2. `common.py` 的 `run_loss_loop`、`PromptGroup`/`combine_prompt_groups`、log-probability 对齐和 mask 处理，提供了不同策略共享的损失输入与统计接口。
3. 同一目录还提供 DAPO、DRO、CISPO、DPPO、REINFORCE、score centering、IGPO、TIS、router replay、异步 coordinator、rollout assembler 和可观测性等组件；`async_rl_loop.Config.policy_loss` 可在 GRPO、GSPO、DAPO、DRO、CISPO、DPPO、score centering 间选择，并可走客户端自定义损失或服务端 PPO/GSPO kernel。
4. 仍有边界：同步 `rl_loop` 和 `igpo_loop` 含算法专用控制流；`reinforce.py`、部分 IGPO/score-centering 能力并未全部纳入同一个 `policy_loss` 选择器；异步 RL API 仍标注为 experimental，且不同算法有各自的 log-probability、mask、序列长度和归一化约束。

总体判断：FireWorks 已具备“可复用 RL 编排层 + 多种可插拔 policy-loss”的完整雏形，足以支撑 GRPO/GSPO 及其变体的工程训练；若目标是完全通用的算法平台，还需统一算法注册、数据契约、损失配置校验和同步/异步 recipe 的扩展接口。
