# 基于 PyTRIO 设计 RL 算法框架：思考顺序与逐步验证

## 1. 先确定边界：框架是什么，不是什么

PyTRIO 已经提供远端训练与采样执行面：模型加载、LoRA、前向/反向、优化器更新、采样、权重保存和恢复由服务处理。要设计的 RL 框架位于客户端，负责算法和编排：

- 选择数据并构造 prompt；
- 用明确版本的策略产生 rollout；
- 计算 reward、advantage 和评估指标；
- 严格对齐 token、旧策略 logprob 与 loss mask；
- 构造 `trio.Datum`，提交 `forward_backward` 与 `optim_step`；
- 发布新采样权重，管理策略版本、并发、checkpoint 和本地运行状态；
- 给 GRPO、PPO、CISPO、DRO、DPO 等算法保留稳定扩展点。

不要在客户端框架中复制 Trio-Train 的模型、并行、LoRA 或优化器实现，也不要依赖其
内部事件 payload。Trio-Train 源码适合用来理解服务端真实约束，但客户端只应依赖
PyTRIO 公开 API。

最重要的分层判断是：

```text
客户端 RL 框架                         PyTRIO / Trio-Train
──────────────────────────────────    ─────────────────────────────
任务、环境、prompt                     tokenizer / SamplingClient
rollout 编排、reward、advantage   ──▶  sample / compute_logprobs
token 对齐、Datum、loss 选择       ──▶  forward_backward
策略版本、更新顺序                 ──▶  optim_step
运行状态、评估、日志、恢复清单     ──▶  save_state / sampler weights
```

## 2. 不可颠倒的总体顺序

按以下顺序思考和实现；上一层没有可重复的证据时，不进入下一层：

1. 定义训练目标和验收指标。
2. 固化 PyTRIO 能力边界和算法语义。
3. 定义状态机、数据契约与策略版本。
4. 分别验证 tokenizer、采样、reward、advantage。
5. 验证 token/logprob/advantage 的逐位置对齐。
6. 验证 loss 数学与 `Datum` 构造。
7. 打通一次参数更新并证明权重真的变化。
8. 用同步串行闭环证明端到端可学习。
9. 加入评估、日志、checkpoint 和可恢复性。
10. 抽象算法插件，而不是先抽象所有东西。
11. 在同步结果作为 oracle 的前提下引入异步和流水线。
12. 最后验证规模、性能、故障和成本。

这条顺序的核心原则是：先证明“更新的是正确目标”，再证明“更新得快”。

## 3. 第 0 阶段：把成功写成可执行契约

### 3.1 先回答这些问题

- 任务是单轮生成、多轮 agent，还是偏好学习？
- reward 是规则、模型评审、环境回报，还是它们的组合？
- 算法是 on-policy、有限陈旧的 off-policy，还是固定离线数据？
- advantage 是 token 级、sequence 级，还是 group-relative？
- 是否需要 reference policy、value model、GAE 或多轮 credit assignment？
- 训练的是 LoRA 还是其他可训练状态？采样器多久同步一次？
- 主指标是什么，防作弊和回归指标是什么？
- 预算按 prompt、completion token、训练 token、时间还是金额限制？

### 3.2 第一版建议边界

先选一个可判定、无外部副作用的单轮任务，以 GRPO 最小闭环作为第一条竖切：

- 规则 reward；
- 一个 prompt 采样 `group_size >= 2` 条 completion；
- `advantage = reward - group_mean`；
- 使用 `importance_sampling`；
- 每一步都从当前训练权重创建 sampler；
- 同步串行执行，不做跨 step rollout 预取；
- 使用 LoRA、小模型、小 batch 和固定随机种子。

选择它不是因为框架只支持 GRPO，而是因为它不需要 value model，reward 和 advantage 都容易手算，最适合作为整个框架的正确性基线。

### 3.3 Gate 0：目标验收

必须先写出一份机器可读的实验配置和验收条件，例如：

```yaml
algorithm: grpo
base_model: Qwen/Qwen3.5-4B
group_size: 4
loss_fn: importance_sampling
max_steps: 20
seed: 42
acceptance:
  datum_alignment_errors: 0
  non_finite_values: 0
  checkpoint_resume_equivalent: true
  toy_eval_improves: true
```

通过条件：同一配置可唯一确定算法、数据、策略同步频率、优化器和评估方式。若“效果变好”仍无法转成一个具体指标，先不要写训练循环。

## 4. 第 1 阶段：核对 PyTRIO 的真实能力边界

### 4.1 客户端对象和职责

框架至少需要理解以下公开对象：

| 对象/API | 框架中的职责 |
| --- | --- |
| `trio.ServiceClient` | 创建或恢复训练、采样客户端 |
| `TrainingClient` | 前反向、优化、保存训练状态、发布采样权重 |
| `SamplingClient` | rollout 和已有序列重新打分 |
| `trio.ModelInput` | 模型输入的公开表示 |
| `trio.Datum` | `model_input` 与 loss 专用张量的边界 |
| `trio.SamplingParams` | 采样分布与输出长度配置 |
| `trio.AdamParams` | 每次 `optim_step` 的优化器配置 |

关键事实：

- `sample()` 返回生成 token 及其采样时 logprobs；这些 logprobs 是 policy loss 的行为策略概率。
- `compute_logprobs()` 对一段已确定的完整输入重新打分，不生成 token。
- `forward_backward()` 累积梯度；`optim_step()` 才应用更新并清理梯度。
- `save_weights_and_get_sampling_client()` 将当前训练权重发布成一个新的采样快照。
- `save_state()` 保存权重和优化器，适合断点续训；只保存 sampler 权重不能等价恢复训练。
- 内置 policy loss 包括 `importance_sampling`、`ppo`、`cispo` 和 `dro`；
  更特殊的目标可走自定义 loss，但应在内置 loss 无法表达时再使用。

### 4.2 从 Trio-Train 源码得到的约束

源码进一步说明了以下服务端不变量：

- `api/datum.py` 对 `target`、`weights`、`logprobs`、`advantages` 做严格等长和 finite 校验，不应依赖静默截断或补齐。
- `losses/batch/*` 统一执行 normalize、prepare、align 和 pack；样本边界及 token 对齐必须在客户端先正确表达。
- 真正的 backward 发生在 `losses/*`，事件 handler 只负责装配和清理。
- optimizer 在 adapter 初始化阶段建立，`optimizer_step` 负责梯度同步、finite 检查、step 和 zero-grad。
- 训练服务是事件驱动的：事件按顺序执行，服务端不会替客户端重新定义算法周期。
- checkpoint、owner/replicated LoRA 和并行拓扑有独立事务边界；客户端不应根据内部 rank 布局猜测保存语义。

### 4.3 Gate 1：API 冒烟

只做以下四件事，不训练：

1. 创建 `ServiceClient` 和一个 LoRA `TrainingClient`。
2. 获取 tokenizer，验证 encode/decode 和 chat template。
3. 创建基础模型 sampler，固定 prompt 采样一次。
4. 对固定完整序列调用 `compute_logprobs()`。

记录模型名、tokenizer 标识、prompt token 数、completion token 数、停止原因和 API
延迟。通过条件是返回结构、长度、dtype 和 finite 性满足断言，而不只是“请求没有
报错”。

## 5. 第 2 阶段：先画状态机，再设计类

RL 框架的核心不是一个巨大的 `Trainer` 类，而是一组有明确提交边界的状态转换：

```mermaid
flowchart LR
    A[读取 prompt batch] --> B[锁定 behavior policy 版本]
    B --> C[并发 rollout]
    C --> D[reward 与 advantage]
    D --> E[构造并校验 Datum]
    E --> F[forward_backward]
    F --> G[optim_step]
    G --> H[提交 learner step]
    H --> I[发布新 sampler 版本]
    I --> J[评估/保存/下一批]
```

### 5.1 每条 rollout 必须携带的身份

建议内部轨迹对象至少包含：

```python
@dataclass(frozen=True)
class Trajectory:
    prompt_id: str
    sample_id: str
    behavior_policy_version: int
    prompt_tokens: list[int]
    completion_tokens: list[int]
    behavior_logprobs: list[float]
    reward_components: dict[str, float]
    terminal_reason: str
```

不要只保存拼接后的文本。没有 `behavior_policy_version`，无法判断数据是否
on-policy；没有原始 token 和 logprobs，无法可靠重建 policy loss。

### 5.2 至少维护三类状态

| 状态 | 示例 | 保存位置 |
| --- | --- | --- |
| 远端训练状态 | LoRA 权重、optimizer state | `TrainingClient.save_state()` |
| 本地算法状态 | step、policy version、data cursor、scheduler | 本地 run manifest |
| 可复现状态 | Python/NumPy RNG、采样种子、配置、代码版本 | 本地 run manifest |

一次逻辑更新只有在 `optim_step` 成功且本地 manifest 提交后才算完成。sampler 权重
发布是另一条可见性边界，不要把“learner 已更新”和“rollout 已看到新权重”视为
同一件事。

### 5.3 Gate 2：纯状态机测试

用 fake backend 验证：

- `forward_backward` 必须先于对应的 `optim_step`；
- forward 失败时不得增加 learner step 或发布新策略；
- optimizer 失败时不得把 rollout policy version 前移；
- 同一 rollout 不能被重复提交，除非恢复协议明确允许幂等重放；
- 空训练信号 batch（例如 GRPO 全组同 reward）不执行空的 optimizer step；
- checkpoint manifest 不引用尚未成功保存的远端状态。

这一步完全不调用 PyTRIO，运行速度应达到普通单元测试级别。

## 6. 第 3 阶段：分别证明 rollout、reward 和 advantage

### 6.1 Rollout 验证

对每条采样结果立即断言：

```python
assert completion_tokens
assert len(completion_tokens) == len(behavior_logprobs)
assert all(math.isfinite(x) for x in behavior_logprobs)
```

还要记录：

- 空输出率；
- EOS、stop string 和 `max_tokens` 截断比例；
- completion 长度的均值、P95 和最大值；
- 每个 prompt 实际返回的样本数；
- timeout、重试和重复样本数。

大量输出撞到 `max_tokens` 时，reward 下降可能只是被截断，并不代表策略退化。

### 6.2 Reward 验证

reward 是最容易“程序正确运行、训练目标完全错误”的部分。每个 reward 组件都应具备：

- 至少一条明确应通过的样本；
- 至少一条明确应失败的样本；
- 格式错误、空输出、超长输出和异常输入；
- reward 上下界和 finite 检查；
- 原始分量与聚合结果，不能只记录总分；
- 防 reward hacking 的独立 holdout 测试。

模型评审 reward 还要固定 rubric、模型版本、采样参数和解析协议，并监控解析失败率与重试偏差。

### 6.3 Advantage 验证

以最小 GRPO 为例，先实现官方案例同款的均值中心化：

\[
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G}r_j
\]

逐组断言：

- 分组只能发生在同一个 `prompt_id` 内；
- group size 符合配置或有明确缺样处理；
- `sum(advantages)` 在浮点容差内为 0；
- 全组 reward 相同时 advantage 全为 0，并标记为 degenerate group；
- 不允许跨 prompt 做全局中心化来冒充 group-relative advantage。

以后如果加入标准差归一化、GAE、leave-one-out baseline 或多 reward whitening，
应作为独立 `AdvantageEstimator` 实现，并为每个公式保留手算 golden case。

### 6.4 Gate 3：黄金样本

建立一个很小的 JSON fixture，固定 prompt、四条 completion、旧 logprobs 和 reward。
测试应精确验证解析、奖励、组内 advantage 和轨迹身份。这个 fixture 会成为后续
Datum 与 loss 测试的输入真相源。

## 7. 第 4 阶段：证明 token 对齐，而不是凭感觉拼数组

这是整个框架最关键的正确性关口。

设 prompt token 为 `P`，长度为 `m`；completion token 为 `Y`，长度为 `n`。按照 PyTRIO 官方 GRPO 示例，构造：

```python
observation_len = len(prompt_tokens) - 1

model_input = prompt_tokens + completion_tokens[:-1]
target_tokens = [0] * observation_len + completion_tokens
old_logprobs = [0.0] * observation_len + completion_logprobs
advantages = [0.0] * observation_len + [sequence_advantage] * len(completion_tokens)
```

四者长度都应为 `m + n - 1`：

```text
位置             0 ........ m-2 | m-1 ........ m+n-2
model_input       prompt 内部输入 | 最后 prompt token + completion[:-1]
target_tokens     占位            | completion[0:]
old_logprobs      0               | 采样返回的 completion logprobs
advantages        0               | sequence advantage 广播到每个 completion token
```

prompt 位置不参与 policy loss，所以 `advantages` 为 0；completion 的第一个 token
由最后一个 prompt token 预测。不能把 `prompt + completion` 同时直接作为 input 和
target，否则会产生一位偏移。

### 7.1 强制的 Datum 校验器

在发请求前 fail-fast：

```python
def validate_policy_datum(datum):
    n = len(datum.model_input)
    fields = datum.loss_fn_inputs
    assert len(fields["target_tokens"]) == n
    assert len(fields["logprobs"]) == n
    assert len(fields["advantages"]) == n
    assert np.isfinite(fields["logprobs"]).all()
    assert np.isfinite(fields["advantages"]).all()
```

还应验证 tokenizer/chat template 与 sampler 使用同一模型约定，禁止把 decode 后文本
重新 tokenize 来替代采样器返回 token；这种 round-trip 不保证 token 序列保持不变。

### 7.2 Gate 4：逐 token 对齐测试

至少包含：

- prompt 长度 1、completion 长度 1 的边界；
- 多 token prompt/completion 的手算例子；
- 空 completion 被拒绝；
- token/logprob 少一个或多一个时被拒绝；
- NaN/Inf 被拒绝；
- prompt mask 全 0，completion 区恰好覆盖所有生成 token；
- pack 多条样本后仍可还原每条样本的 token 范围和 loss 输出。

通过标准不是“服务端接受 Datum”，而是客户端逐位置语义与手算结果一致。

## 8. 第 5 阶段：先验证 loss 数学，再调用远端 backward

PyTRIO 的内置 policy loss 都依赖：

- learner 当前计算的 `target_logprobs = log p_theta(x)`；
- rollout 保存的 `sampling_logprobs = log q(x)`；
- 客户端计算的 `advantages`。

### 8.1 算法选择表

| 算法/场景 | 第一选择 | 关键验证 |
| --- | --- | --- |
| 单步 on-policy GRPO | `importance_sampling` | group advantage、ratio 接近 1 |
| 同批 rollout 多轮更新 | `ppo` | 旧 logprob 冻结、clip fraction |
| 异步或有陈旧数据 | `cispo` | policy lag、ratio 分布、裁剪比例 |
| 离线/较强 off-policy | `dro` | 二次偏移惩罚、beta 敏感性 |
| DPO/GSPO 等特殊目标 | `forward_backward_custom` | 自定义梯度、归一化和性能 |

不要只凭算法名字选择 loss。例如 GRPO 定义的是 group-relative advantage，真正提交
给服务端的 policy objective 仍需明确是 IS、PPO 还是其他形式。

### 8.2 本地数学 oracle

在 NumPy/PyTorch 中用很小的固定数组复现目标，例如 IS：

\[
L = -\sum_t \exp(\log p_{\theta,t} - \log q_t) A_t
\]

测试至少覆盖：

- `log p == log q` 时 ratio 为 1；
- 正 advantage 提高目标 token 概率，负 advantage 降低它；
- prompt 的零 advantage 不产生梯度；
- PPO 正负 advantage 两侧的裁剪行为；
- 极端 log-ratio 不出现 silent overflow；
- reduction 是 token sum、token mean 还是 sequence mean，必须写进算法契约。

### 8.3 Gate 5：远端 forward 对齐

优先先调用无 backward 的 `TrainingClient.forward()` 或等价公开能力，取得 learner
logprobs 和 `loss_fn_outputs`，与本地 oracle 计算的 loss 对齐，再进入
`forward_backward()`。

第一批 rollout 来自刚发布的同一策略时，completion 区的 `learner_logprob - behavior_logprob` 应接近 0。若明显偏离，优先检查：

- sampler 是否真来自当前 learner 权重；
- tokenizer 和 chat template 是否一致；
- token 是否发生重新编码；
- logprob 是否错位；
- rollout 是否混入旧 policy version。

## 9. 第 6 阶段：证明一次更新真的更新了正确参数

最小更新顺序：

```python
fwd_future = training_client.forward_backward(
    data=datums,
    loss_fn="importance_sampling",
)
fwd_result = fwd_future.result()
optim_future = training_client.optim_step(adam_params)
optim_future.result()
```

注意：同步 API 也返回 future。当前客户端实现会先等待 forward/backward 成功，再提交
optimizer；这样 forward 失败时不会误触发带有旧梯度的 optimizer。只有在服务端提供
明确的失败传播和梯度清理保证后，才可以把 optimizer 提前提交以重叠等待。

### Gate 6：单步更新

对一个固定 prompt/target：

1. 更新前用当前策略记录目标 token 的 logprob。
2. 构造一个明确的正 advantage batch。
3. 执行一次 `forward_backward + optim_step`。
4. 发布新的 sampling client。
5. 对同一固定序列重新计算 logprob。

通过条件：目标方向变化符合预期、所有指标 finite、策略版本只增加一次。不要用“生成文本刚好变了”作为唯一证据，因为随机采样会产生假阳性。

还要加入三个反例：

- 全零 advantage 时几乎不应产生有效更新；
- 正负 advantage 对称样本应产生可解释的抵消；
- 故意注入 NaN 时训练必须失败，而不是记录成功 step。

## 10. 第 7 阶段：同步闭环是异步实现的 oracle

第一条完整训练循环应保持简单：

```text
for step:
    读取 prompt batch
    从 learner 当前权重创建 sampling client
    每个 prompt 采样 G 条 completion
    计算 reward 和 group-relative advantage
    丢弃无训练信号的退化 group
    构造并校验 Datum
    forward_backward
    optim_step
    记录指标
    定期评估与保存
```

### Gate 7：微型过拟合

选 8～32 个可判定样本，验证框架可以在固定预算内过拟合：

- 平均 reward 或 success rate 有明确提升；
- 行为策略与 learner 的 KL/ratio 没有异常爆炸；
- `frac_degenerate`、有效 completion 数和有效 token 数可解释；
- 独立 holdout 没有出现显然的 reward hacking；
- 同 seed 的短跑在合理容差内复现。

若微型任务都不能学习，不要先通过调大模型、batch 或并发来解决，应回到 reward、对齐、loss 和学习率逐层检查。

## 11. 第 8 阶段：可观测性必须覆盖因果链

每个训练 step 至少记录以下指标，并携带 `run_id`、`step`、`policy_version`：

### Rollout

- prompt 数、completion 数和生成 token 数；
- completion 长度均值/P95/最大值；
- `max_tokens` 截断率、空输出率、采样失败率；
- sampler policy version 与 learner version 的 lag。

### Reward/advantage

- 总 reward 与每个 reward component；
- reward 均值、标准差、分位数；
- advantage 均值/标准差；
- degenerate group 比例、有效 group 和有效 token 数。

### Policy/loss

- `loss_sum` 以及明确分母后的规范化 loss；
- learner/behavior logprob 差值和 importance ratio 分布；
- clip fraction（使用裁剪目标时）；
- 近似 KL、entropy proxy、非 finite 数量；
- optimizer 返回指标、学习率和 grad clip 配置。

### 系统与成本

- rollout、reward、forward/backward、optimizer、保存各阶段延迟；
- 请求并发、queue wait、重试、timeout；
- prefill、sample、train token；
- 每步和每个有效样本的时间/金额。

只记录一个总 loss 不足以诊断 RL。reward 不变可能来自采样截断、组退化、策略陈旧、ratio 裁剪或更新根本未生效。

## 12. 第 9 阶段：checkpoint 与恢复要做等价性测试

PyTRIO 的远端 checkpoint 与框架本地状态必须配对。建议 manifest：

```json
{
  "schema_version": 1,
  "completed_step": 100,
  "learner_policy_version": 100,
  "published_sampler_version": 100,
  "pytrio_state_path": "...",
  "data_cursor": 3200,
  "rng_state_paths": {"python": "...", "numpy": "..."},
  "config_hash": "...",
  "code_revision": "..."
}
```

保存顺序：

1. 停在一个明确的 step 提交边界；
2. 调用 `save_state()` 保存模型权重和 optimizer；
3. 确认远端结果成功；
4. 以临时文件写本地 manifest，再原子重命名；
5. 只有 manifest 提交后才宣布 checkpoint 可恢复。

恢复时使用包含 optimizer 的恢复接口；仅从权重创建训练客户端应视为“新优化器继续训练”，不是严格 resume。

### Gate 8：中断恢复等价

比较两条短跑：

```text
A: 连续训练 N + M 步
B: 训练 N 步 → 保存 → 新进程恢复 → 再训练 M 步
```

比较数据游标、策略版本、评估输入、关键 logprob、reward 曲线和最终权重行为。对于远端采样存在非确定性的情况，应预先定义数值/统计容差，而不是事后挑选标准。

还应模拟：保存远端成功但本地 manifest 失败、进程在 optim 后崩溃、过期 sampler 仍有在途 rollout、API timeout 后结果状态未知等情况。

## 13. 第 10 阶段：从已工作的竖切提取扩展接口

不要一开始就设计“万能 Trainer”。同步 GRPO 闭环通过后，再从真实变化点提取接口：

```python
class PromptSource(Protocol): ...
class RolloutEngine(Protocol): ...
class RewardFunction(Protocol): ...
class AdvantageEstimator(Protocol): ...
class DatumBuilder(Protocol): ...
class PolicyObjective(Protocol): ...
class Evaluator(Protocol): ...
class CheckpointManager(Protocol): ...
```

推荐模块边界：

```text
framework/
├── config/          # 强类型配置、交叉字段校验
├── data/            # 数据源、游标、prompt renderer
├── rollout/         # 采样并发、轨迹、策略版本
├── rewards/         # 可组合 reward 与解析器
├── advantages/      # GRPO/GAE/LOO 等
├── objectives/      # PyTRIO loss 选择及自定义 loss
├── datum/           # 唯一 token 对齐与 Datum 构造入口
├── orchestration/   # 同步/异步状态机
├── evaluation/      # 固定评测，不复用训练 reward 状态
├── checkpoint/      # 远端 state + 本地 manifest
└── observability/   # 指标、trace、样本审计
```

### 13.1 哪些必须统一，哪些不应强行统一

必须统一：

- 轨迹身份、policy version、token/logprob 契约；
- Datum 校验、提交顺序、checkpoint 和指标命名；
- retry、timeout、取消和错误分类；
- 配置快照和复现信息。

不应强行统一：

- GRPO group advantage 与 PPO/GAE 的时序 advantage；
- 单轮生成与多轮环境状态；
- online rollout 与 DPO 的离线 chosen/rejected pair；
- token-level、sequence-level 和 pairwise loss 的 reduction。

DPO 应复用数据、客户端、checkpoint、日志等基础设施，但不应伪装成“有 rollout
group 的 GRPO”。它应拥有自己的 batch producer、pairwise Datum/custom loss 和
验证集。

### Gate 9：第二算法验证抽象

只有加入第二种真正不同的算法后，才能判断扩展接口是否合理。建议顺序：

1. GRPO + `importance_sampling`；
2. 同一 rollout 数据改用 PPO，验证 objective 插件；
3. DPO 或多轮 PPO，验证控制流而非只验证 loss 名称切换。

如果加入第二算法需要绕过 Datum 校验、状态机或 checkpoint，则抽象边界放错了。

## 14. 第 11 阶段：最后引入异步、流水线和陈旧策略

PyTRIO 的训练异步 API 有两阶段语义：先 `await ..._async()` 提交并得到 future，
再等待 future 的远端结果。官方时钟周期允许先提交 forward/backward、再提交
optimizer 以重叠等待；但本框架当前优先保证失败安全，会在确认 forward 成功后再提交
optimizer：

```python
fwd_future = await training_client.forward_backward_async(datums, loss_fn=loss_fn)
fwd_result = await fwd_future
optim_future = await training_client.optim_step_async(adam_params)
await optim_future
```

采样的 `sample_async()` 则在 `await` 后直接得到采样结果。框架应封装这一区别，避免业务插件自行处理 future 语义。

### 14.1 分三步增加并发

1. **批内 rollout 并发**：同一 sampler 版本下，不同 prompt 并发；不改变算法语义。
2. **训练请求提前提交**：保持 forward 在前、optimizer 在后，等待移到提交之后。
3. **跨 step 流水线**：rollout 与 learner update 重叠；这会引入 policy lag，属于算法变化，不只是性能优化。

跨 step 流水线必须定义：

- 最大允许 `learner_version - behavior_version`；
- 超限 rollout 是丢弃、降权还是使用 CISPO/PPO 修正；
- 每个 rollout 的旧 logprob 是否仍与其 behavior snapshot 一致；
- sampler 发布和在途请求的切换规则；
- checkpoint 时是否等待、记录或取消在途工作。

### Gate 10：同步/异步差分测试

同一 golden batch、同一初始 checkpoint：

- 同步和异步版本产生相同的 Datum 顺序和样本集合；
- loss 和更新后固定序列 logprob 在容差内一致；
- 并发异常不会导致重复 optimizer step；
- 取消一个 rollout 不会错配其他 prompt 的结果；
- 吞吐提升来自等待重叠，而不是悄悄丢样本或增加 policy lag。

## 15. 第 12 阶段：规模、性能与生产故障验证

正确性 gate 全通过后，再逐轴扩大，且一次只改变一个轴：

1. completion 长度；
2. group size；
3. prompt batch size；
4. rollout 并发；
5. 在途 step 数；
6. 模型/LoRA rank；
7. 长时间运行和恢复次数。

每次扩展比较：吞吐、P95 延迟、有效训练 token、reward、policy lag、ratio/KL、
失败率和成本。PyTRIO 的 prefilling cache 只优化采样的共享 prompt 前缀；
`compute_logprobs` 和训练仍需处理完整序列，因此不能从采样 cache 命中率推导训练
成本。

### Gate 11：故障注入

至少覆盖：

- rollout timeout、部分 group 缺样和重复响应；
- reward 服务失败或返回 NaN；
- forward 成功但 optimizer 返回失败；
- 保存和恢复失败；
- 进程在任意阶段退出；
- API 限流、临时网络错误和长尾请求；
- 长输出导致 token 预算耗尽；
- 过期 policy rollout 超过 staleness 上限。

重试必须按操作语义分类。只读采样通常可以用同一 request identity 重试；optimizer 是否已经提交成功存在歧义时，不能盲目重放，否则可能双重更新。

## 16. 推荐的完整验证金字塔

| 层级 | 是否调用 PyTRIO | 验证内容 | 进入下一层的证据 |
| --- | ---: | --- | --- |
| L0 静态/配置 | 否 | schema、交叉字段、非法组合 | 非法配置全部 fail-fast |
| L1 纯函数 | 否 | reward、advantage、token 对齐、loss oracle | golden case 全通过 |
| L2 状态机 | fake | 顺序、幂等、失败提交边界 | 无重复更新/错误版本前移 |
| L3 API smoke | 是 | tokenizer、sample、logprobs、返回结构 | 长度/dtype/finite 正确 |
| L4 单步训练 | 是 | forward、backward、optimizer、权重变化 | 更新方向正确 |
| L5 微型闭环 | 是 | 小数据过拟合和独立评估 | 指标按预期改善 |
| L6 恢复/差分 | 是 | resume、同步/异步等价 | 容差内一致 |
| L7 压力/故障 | 是 | 并发、长跑、成本、错误恢复 | SLO 和预算满足 |

任何高层测试都不能替代低层测试。端到端 reward 上升不能证明 token 对齐正确；
单元测试通过也不能证明真实 sampler 权重同步和 optimizer 生命周期正确。

## 17. 实施里程碑

### M1：最小正确内核

- 强类型配置；
- `Trajectory`、reward、GRPO advantage；
- 唯一的 policy Datum builder 和 validator；
- 本地 loss oracle；
- 全部 L0/L1 测试。

### M2：同步 PyTRIO 竖切

- Service/Training/Sampling client adapter；
- 单步更新；
- 同步 GRPO 小闭环；
- 基础指标与固定评估；
- L3～L5 证据。

### M3：可恢复运行

- `save_state` + 本地 manifest；
- data cursor、policy version 和 RNG 恢复；
- 连续/中断恢复差分测试。

### M4：可扩展算法层

- objective、advantage、reward 插件；
- PPO 第二实现；
- DPO 或多轮任务验证控制流边界。

### M5：异步与生产化

- 批内 rollout 并发；
- PyTRIO future 封装；
- bounded queue、backpressure、timeout 和取消；
- policy staleness gate；
- 压测、故障注入和成本报告。

## 18. 每次新增算法的评审清单

- [ ] 算法目标、采样分布和 learner 分布是否写清？
- [ ] reward、baseline、advantage 和 reduction 是否有公式？
- [ ] behavior logprob 来源和冻结周期是否明确？
- [ ] prompt/completion 的 token 对齐是否有 golden test？
- [ ] 使用哪个 PyTRIO loss，`loss_fn_config` 是什么？
- [ ] 如果使用 custom loss，是否证明内置 loss 无法表达？
- [ ] 空输出、截断、退化 group 和非 finite 值怎样处理？
- [ ] 一次逻辑 step 包含多少次 backward 和 optimizer step？
- [ ] sampler 何时发布，允许多大的 policy lag？
- [ ] checkpoint 是否覆盖 optimizer、本地游标、版本和 RNG？
- [ ] 同步基线是否通过，异步实现是否做差分验证？
- [ ] 主指标、反作弊指标、SLO 和成本上限是否明确？

## 19. 最终判断标准

一个基于 PyTRIO 的 RL 框架达到“可用”，不应以代码目录齐全或支持很多算法名字为标准，而应同时满足：

1. 任意训练 token 都能追溯到 prompt、completion、behavior policy 和 reward。
2. Datum 的每个位置都能解释其 target、旧 logprob、advantage 和 mask。
3. 单步更新方向能由本地数学 oracle 和远端结果共同证明。
4. 同步小闭环可以稳定学习，异步实现与其在定义的容差内一致。
5. 任意一次崩溃都能判断最后一个已提交 step，并从匹配的远端/本地状态恢复。
6. 加入新算法时主要替换 rollout/reward/advantage/objective，而不绕过公共正确性和恢复机制。
7. 性能优化都有吞吐证据，也明确说明引入的 policy lag 和算法影响。

达到这些条件后，再谈更多模型、更多并行方式和更复杂的 agent 环境，框架的扩展才建立在可信基础上。

## 参考资料

### PyTRIO 官方文档

- [Training API 产品页](https://pytrio.com/home/training-api)
- [训练指南](https://docs.pytrio.com/docs/guide/train)
- [损失函数](https://docs.pytrio.com/docs/guide/loss_fn)
- [异步 API](https://docs.pytrio.com/docs/guide/async)
- [时钟周期](https://docs.pytrio.com/docs/clock-cycle)
- [Prefill、Cache、Sample 与 Train](https://docs.pytrio.com/docs/prefill-cache)
- [GRPO 案例](https://docs.pytrio.com/docs/example/grpo)
- [`TrainingClient` API](https://docs.pytrio.com/docs/api/TrainingClient)
- [`SamplingClient` API](https://docs.pytrio.com/docs/api/SamplingClient)
- [`Datum` API](https://docs.pytrio.com/docs/api/Datum)
- [获取 Logprobs](https://docs.pytrio.com/docs/advanced/compute_logprobs)
- [保存权重](https://docs.pytrio.com/docs/guide/save)与[继续训练](https://docs.pytrio.com/docs/guide/resume)

### Trio-Train 源码依据

以下链接固定到调研时的 commit；仓库若为私有，需要相应 GitHub 访问权限。

- [架构导航](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/docs/ARCHITECTURE.md)
- [`api/datum.py`](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/api/datum.py)
- [`api/loss_types.py`](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/api/loss_types.py)
- [`losses/importance_sampling.py`](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/losses/importance_sampling.py)
- [`losses/ppo.py`](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/losses/ppo.py)
- [`losses/batch/`](https://github.com/SwanHubX/Trio-Train/tree/68b78b791f485e84a154c187d06e497de084fedd/losses/batch)
- [`events/handlers/forward_backward.py`](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/events/handlers/forward_backward.py)
- [`events/handlers/optimizer_step.py`](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/events/handlers/optimizer_step.py)
- [测试矩阵](https://github.com/SwanHubX/Trio-Train/blob/68b78b791f485e84a154c187d06e497de084fedd/docs/testing/README.md)
