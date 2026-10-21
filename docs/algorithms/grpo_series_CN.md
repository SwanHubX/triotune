# PPO、GRPO 及相关算法

[English](grpo_series.md)

Triotune 使用同一个异步 `PPOTrainer` 运行 PPO 和 GRPO 算法族：选择配置预设后，
即可复用数据、奖励和训练接口。GRPO 通过同一道题的多个回答之间的相对奖励学习，无需 critic。
支持的预设见[算法配置矩阵](#算法配置矩阵)。

本文参考 [AReaL GRPO 系列文档](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/grpo_series.md)
的章节结构，以下命令和参数均以 Triotune 的实际接口为准。

## 运行示例

在仓库根目录执行以下命令。需要 Python 3.12，并已配置可访问的 PyTRIO 训练服务；
模型计算由 PyTRIO 远端执行。

```bash
uv sync --locked --extra ppo
```

运行训练命令前，先完成[准备 GSM8K 数据](#准备-gsm8k-数据)。

### 从命令行切换算法

示例入口**只接受 `--config`**，不支持 Hydra 风格的参数覆盖。
下面通过 `PPOConfig.for_algorithm` 选择预设，生成完整的临时 YAML，
然后启动两步 GRPO 冒烟实验。将第一条命令中的 `grpo` 换成矩阵中的其他预设即可切换算法。

```bash
uv run python - grpo /tmp/triotune-grpo.yaml <<'PY'
import sys
from dataclasses import asdict
from pathlib import Path
import yaml
from triotune.trainer import PPOConfig
from triotune.utils.config import load_yaml_config

settings = load_yaml_config("examples/ppo/gsm8k_ppo_args.yaml")
# 删除原配置中明确指定的 PPO 选项，让所选预设提供这些设置。
settings.pop("actor")
settings.pop("critic")
settings["training"].update(max_steps=2, batch_size=2)
settings["rollout"].update(num_generations=4, max_tokens=512)
settings["checkpoint"]["save_every"] = 0
settings["valid_dataset"] = None
settings["reporting"].update(
    report_to=None, workspace=None, run_name=f"gsm8k-{sys.argv[1]}-smoke",
    tags=[sys.argv[1], "gsm8k"], log_dir="output/grpo-series-smoke",
)
config = PPOConfig.for_algorithm(sys.argv[1], **settings)
Path(sys.argv[2]).write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
PY
uv run python examples/ppo/gsm8k_ppo.py --config /tmp/triotune-grpo.yaml
```

模型仍为 `Qwen/Qwen3.5-9B`，LoRA rank 为 32，学习率为 `4e-5`，seed 为 42。
每次更新使用两道题，每题四个回答，即八条轨迹。
使用仓库自带的 100 步 PPO 配置，开启验证和 SwanLab：

```bash
uv run swanlab login
uv run python examples/ppo/gsm8k_ppo.py --config examples/ppo/gsm8k_ppo_args.yaml
```

运行前按自己的账号设置 `reporting.project`、`workspace` 和 `run_name`。
如需更长的 GRPO 实验，修改生成 YAML 中的 `training.max_steps`、`valid_dataset`、
`evaluation` 和 `reporting`，继续使用同一个启动入口。

### 配置接口

| 接口 | 用途 |
| --- | --- |
| `PPOConfig.for_algorithm("grpo", **modules)` | 先应用算法预设，再应用显式覆盖；actor/critic 使用字典时可以与预设默认值合并。 |
| `PPOConfig.from_modules(settings)` | 读取模块化配置字典，支持顶层 `algorithm` 选择器，默认为 `grpo`。 |
| `PPOConfig(**settings)` | 直接读取完整、显式的配置；GSM8K 示例采用此方式，不接受 `algorithm` 字段。 |

例如 `PPOConfig.from_modules({"algorithm": "gspo", "rollout":
{"num_generations": 4}})` 会选择序列级重要性采样。
显式配置优先于预设：若保留 PPO 文件中的 `critic.enabled: true` 或
`actor.reward_norm: null`，就会改变预期的 GRPO 设置。
传入 `PPOActorConfig`/`PPOCriticConfig` 实例会替换整个模块；
希望只覆盖预设的一部分时，请传字典。

## 核心配置参数

表中的名称同时表示 YAML 层级路径和 Python 配置属性。
除算法预设明确覆盖的选项外，以下为库的默认值。

### 训练与 rollout

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `model.name`、`model.lora_rank` | `null`、`32` | PyTRIO 基座模型 ID 和 actor LoRA rank；训练前必须提供模型。 |
| `training.batch_size`、`max_steps`、`seed` | `1`、`null`、`null` | 每次更新的题目数、更新次数上限和客户端/采样随机种子。 |
| `optimizer.learning_rate` | `1e-5` | actor AdamW 学习率；还支持 `beta1`、`beta2`、`eps`、`weight_decay`、`grad_clip_norm`。 |
| `rollout.num_generations`、`max_tokens` | `4`、`512` | 每题生成的回答数，以及回答长度上限。 |
| `rollout.temperature`、`top_p` | `1.0`、`1.0` | 两者必须为 1；仅有目标 token 的训练 logprob 无法还原修改后的采样概率。 |
| `rollout.prompt_key`、`chat_template_kwargs` | `messages`、`{}` | prompt 字段名和 tokenizer 模板参数，例如 `enable_thinking: false`。 |
| `actor.update_epochs`、`ppo_n_minibatches` | `1`、`1` | 复用同一批冻结的 rollout 目标，以及按完整题目组划分 actor 更新。 |
| `actor.loss_backend` | `custom` | 通过 PyTRIO 使用完整的本地 Torch loss 回调；`native` 只支持受限的普通 token 级 PPO。 |

`max_steps` 统计 rollout 更新次数，不是 optimizer 次数。训练只遍历传入的数据一次，
数据耗尽会提前结束；`update_epochs` 表示重复使用冻结的 rollout 目标，不是数据集 epoch。
actor 的 minibatch 数不能超过实际题目组数。
数据路径、样本上限和 shuffle 由示例的数据加载代码负责；当前 GSM8K 入口会应用
`valid_dataset.max_examples`，但不会应用 `train_dataset.max_examples`，也不会打乱训练数据。

### 准备 GSM8K 数据

[GSM8K 示例](../../examples/ppo/gsm8k_ppo.py) 读取本地 JSONL，不会自动下载数据。
每行至少包含 `messages` 和 `answer`。以下命令准备官方训练集和测试集，
要求模型把最终数字答案放入 `\boxed{}`，已有文件会直接复用。

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)
for split in ("train", "test"):
    path = Path(f"data/gsm8k_{split}.jsonl")
    if path.exists():
        continue
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            item = {
                "messages": [{"role": "user", "content": row["question"]
                    + r" Provide a numerical answer without units, written inside \boxed{}."}],
                "answer": row["answer"].rsplit("####", 1)[1].strip().replace(",", ""),
                "question": row["question"],
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
PY
```

这段准备代码使用单条 user 消息；已有实验数据可能包含 few-shot 示例。
比较算法时请使用相同的数据文件。示例奖励函数检查最后一个 boxed 答案，正确为 1，否则为 0。

JSONL 文件每行保存一条记录，最小格式如下（此处展开显示以便阅读）：

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "answer": "2"
}
```

### 策略梯度损失聚合（`actor.loss_aggregation`）

| 取值 | 聚合方式 |
| --- | --- |
| `token_mean`（默认） | 对所有有效生成 token 求平均。 |
| `seq_mean` | 先计算每个回答的 token 均值，再对回答等权平均。 |
| `prompt_mean` | 先计算每道题整组回答的 token 均值，再对题目组等权平均。 |
| `constant` | 总和除以轨迹数 × `loss_aggregation_divisor`，需设置正数除数。 |

### 奖励与优势归一化（`actor.reward_norm`、`actor.adv_norm`）

两者均可设为 `null` 关闭，或传入 `NormConfig` 字典：

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: group
    mean_leave1out: false
    std_unbiased: true
    eps: 0.00001
  adv_norm:
    mean_level: batch
    std_level: batch
```

`mean_level` 和 `std_level` 可以分别选择 `group`、`batch`、`null`。
group 表示比较同一道题的多个回答；`mean_leave1out: true` 在计算均值时排除当前回答。
所有预设默认保留 batch/batch 优势归一化，除非显式覆盖。
直接构造 `PPOConfig()` 默认关闭奖励归一化，GRPO 预设则会开启。

### 裁剪策略（`actor.eps_clip*`）

`eps_clip: 0.2` 对应普通 ratio 范围 `[0.8, 1.2]`；
`eps_clip_higher: 0.28` 将上界改为 `1.28`。
`c_clip` 可为负优势启用 dual clipping。
`loss_type` 支持 `ppo`、`sapo`、`cispo`，三者的裁剪方式不同。

### 重要性采样层级（`actor.importance_sampling_level`）

默认 `token`；`sequence` 使用有效回答 token 上 ratio 的几何均值，GSPO 选择此设置。
`use_decoupled_loss` 区分 behavior 和 proximal 策略；`recompute_logprob` 在冻结目标前
重新计算 proximal 分数。可选的 `rejection_lower`/`rejection_upper` 根据
proximal 与 behavior 的概率比屏蔽 token，要求 `use_decoupled_loss: true`。

## 算法配置矩阵

以下列出的是**当前实现的预设**，不代表复现论文的全部设置。
未特别注明时：token 级重要性采样、PPO loss、token 均值聚合、batch/batch 优势归一化、
裁剪参数 0.2，且不使用 critic。

| 预设 | 一句话介绍 | 奖励均值 / 标准差 | 特有设置 |
| --- | --- | --- | --- |
| `ppo` | 使用独立训练的 critic 构造基于 value 的优势。 | 关闭 | `critic.enabled=true` |
| `grpo` | 在同一道题的回答组内比较奖励。 | group / group | 基础组相对预设 |
| `drgrpo` | 移除组内标准差缩放。 | group / 无 | 聚合仍为 `token_mean` |
| `liteppo` | 在组内中心化，并用整个 batch 的标准差缩放。 | group / batch | 混合归一化 |
| `rloo` | 将其他回答的平均奖励作为 baseline。 | 排除自身的 group / 无 | `mean_leave1out=true` |
| `reinforce` | 直接使用结果奖励，不减组内 baseline。 | 关闭 | 无 critic |
| `reinforce_baseline` | 减去组均值，不做标准差缩放。 | group / 无 | 当前预设与 `drgrpo` 相同 |
| `gspo` | 使用序列级重要性采样比率。 | group / group | `importance_sampling_level=sequence` |
| `sapo` | 使用平滑门控代替硬裁剪。 | group / group | `loss_type=sapo` |
| `cispo` | 在策略目标中裁剪不参与求导的重要性权重。 | group / group | `loss_type=cispo`，上下裁剪参数为 `1.0/4.0` |
| `dapo` | 结合非对称裁剪、有效题目组补采样和长度惩罚。 | group / group | 上裁剪 `0.28`、动态采样、`overlong_tokens=128` |

## 算法专用选项

### 标准 PPO

`critic.enabled: true` 会创建独立的 PyTRIO LoRA 训练客户端。
critic 的 `optimizer`、`lora_rank`、`eps_clip`、`update_epochs`、`n_minibatches`
与 actor 分开设置。value 来自两个标记 token 的 logprob：
`value_scale * (logp(positive) - logp(negative)) + value_bias`。
默认标记是 `"1"` 和 `"0"` 的单 token 编码，必要时显式设置 `critic.token_ids`。
这是可训练的 logprob critic，不是原生 value head。

### GRPO、Dr.GRPO、LitePPO 与 RLOO

通过 `for_algorithm` 或 `from_modules` 切换；每题至少生成两个回答，
才能进行有意义的组内比较和 leave-one-out baseline 计算。
如果对照实验需要不归一化的优势，请显式修改 `actor.adv_norm`。

### GSPO

预设设置 `actor.importance_sampling_level: sequence`。
如果还需要回答等权聚合，另行设置 `actor.loss_aggregation: seq_mean`。

### SAPO 与 CISPO

SAPO 提供 `sapo_tau_pos`（1.0）和 `sapo_tau_neg`（1.05），要求
`use_decoupled_loss: false`。CISPO 要求 token 级重要性采样且
`use_decoupled_loss: false`，预设 ratio 范围为 `[0, 5]`。

### DAPO

`dynamic_sampling: true` 会丢弃组内奖励完全相同的题目，继续读取题目补足 batch。
`max_empty_batches`（32）限制连续补采样无有效结果的次数；数据耗尽时，最终 batch 仍可能较小。
`overlong_tokens`（128）指定 `rollout.max_tokens` 末尾的惩罚区间，
惩罚随长度增加，由 `overlong_penalty_factor`（1.0）缩放；该区间不能大于回答长度上限。

## 核心概念与接口

### Trainer 生命周期

| 接口 | 功能与约定 |
| --- | --- |
| `PPOTrainer(args=config, train_dataset=rows, reward_fn=fn, eval_dataset=valid_rows)` | 延迟初始化；也可直接传入 `model`，或注入客户端、tokenizer、critic。 |
| `await trainer.initialize()` | 创建 actor、可选 critic、tokenizer 和日志器，只初始化一次。 |
| `state = await trainer.train()` | 完成采样、评分、更新、评估、最终保存和日志收尾。 |
| `metrics = await trainer.step(trajectories)` | 更新一批自定义冻结轨迹；同一 trainer 不可并发调用。 |
| `await trainer.evaluate()` | 对验证题目采样、评分，不计算 reference/critic，也不更新优化器。 |
| `await trainer.save_checkpoint()` | 保存 actor 和框架管理的 critic 状态，返回 actor URI。 |
| `await trainer.close()` | 为自定义生命周期结束日志记录。 |

通过 `asyncio.run(...)` 执行异步入口。`train()` 负责日志清理；
手动调用 `step()` 时，需要自行管理保存和 `close()`。
`PPOState` 包含 `global_step`、`optimizer_steps`、`critic_optimizer_steps`、
`metrics`、`checkpoint_path` 和 `critic_checkpoint_path`。

### 奖励接口

`reward_fn(*, row, completion, completion_ids)` 支持同步或异步函数，
返回有限标量，或含有限分项的 `{"reward": total, "correctness": score, ...}`。
分项记录为 `reward/<name>/mean` 和 `eval/reward_<name>_mean`。
GSM8K 示例提供奖励函数；trainer 本身不会解释 `answer`。
prompt 支持文本、token ID 列表或聊天消息列表。

### GAE 时间单位与自定义轨迹

`actor.discount` 和 `gae_lambda` 默认为 1.0；`gae_lambda` 为固定标量。
`gae_timestep_unit` 支持 `token` 和 `turn`。turn 模式需要自定义
`Trajectory.turn_ids`，自带 GSM8K rollout 不会生成此字段。

从 `triotune.trainer.ppo` 导入 `Trajectory`，将轨迹列表交给 `step()`。
必填字段为 `prompt_tokens`、`completion_tokens`、behavior `logprobs`、
标量 `reward` 和 `group_id`。可选的 `loss_mask`、`reference_logprobs`、
`token_rewards`、`turn_ids` 与回答 token 对齐；`values` 的长度为回答长度 + 1，
多出的元素用于 bootstrap。turn 模式下，有效 token 的整数 turn ID 必须单调不减，
被 mask 的 token 使用 `-1`。当前不支持动态 lambda 函数接口。

## Triotune 实现说明

- **执行方式：**题目间并发采样，每次更新先冻结目标，再由独立客户端并发优化 actor 和 critic。
  不同训练 step 之间的 rollout 与策略更新仍顺序执行。
- **KL：**`actor.kl_ctl` 默认 0；设置正数后，使用冻结 reference 评分
  （`reference.name`，未指定时使用 actor 基座模型），支持 `kl_estimator: k1|k2|k3`。
  token 模式将 KL 加入奖励后计算 GAE；turn 模式将其保留为 actor 优势中的 token 局部项。
- **评估：**`evaluation.steps`、`before_train`、`after_train`、`batch_size`、`max_batches`
  控制在可重复遍历的验证集上评估奖励；使用当前 rollout 采样配置，包括每题回答数。
- **日志：**`reporting.report_to: swanlab` 开启上传。actor 和 critic 共用一个实验，
  指标位于 `reward/*`、`train/*`、`policy/*`、`critic/*`、`eval/*`、`time/*`、`usage/*`。
  部分指标需要启用对应组件或选项。`reporting.log_dir` 还会写入独立 JSONL 历史；
  PPO 每步发布指标，不受 `reporting.logging_step` 控制。
- **保存：**`checkpoint.save_every` 按 rollout 更新计数，0 表示只在结束时保存。
  URI 记录为 `checkpoint/actor_path` 和 `checkpoint/critic_path`。
  trainer 当前没有内置的 resume/load 入口。
- **边界：**模型部署和分布式计算由 PyTRIO 管理；AReaL 的集群后端、workflow API、
  Hydra 配置不是 Triotune 的接口。

更多细节见[实现与测试报告](../ppo_pytrio-use_eval.md)、
[配置源码](../../triotune/api/cli_args.py)和[算法组件](../../triotune/trainer/ppo/)。
