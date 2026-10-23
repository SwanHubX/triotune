# OPD 案例——GSM8K

[English](opd.md)

OPD 用固定教师给学生生成的 GSM8K 解答提供评分，并据此训练学生。

## 运行示例

### 环境安装

在仓库根目录运行，使用 Python 3.12 和可用的 PyTRIO 训练服务。

```bash
git clone https://github.com/SwanHubX/triotune.git
cd triotune
uv python install 3.12
uv sync --locked --all-extras
uv run python -c "import triotune; print(triotune.__name__)"
```

### 快速启动

```bash
uv run python examples/opd/gsm8k_opd.py --config examples/opd/gsm8k_opd_args.yaml
```

运行前先完成[准备 GSM8K 数据](#准备-gsm8k-数据)。入口只接受 --config，超参数在 YAML 中修改。

### 配置接口

| 接口 | 用途 |
| --- | --- |
| OPDConfig(**settings) | 直接读取完整显式配置，案例采用此接口。 |
| OPDConfig.from_modules(settings) | 读取模块化配置字典，兼容部分旧式字段。 |

配置不含 algorithm 选择器或算法名包装；${model.name} 由 load_yaml_config 解析。完整配置见 [gsm8k_opd_args.yaml](../../examples/opd/gsm8k_opd_args.yaml)。

## 核心配置参数

### 训练与 rollout

下表列的是案例配置值，不是全部库默认值。

| 参数 | 案例值 | 作用 |
| --- | --- | --- |
| model.name, model.lora_rank | Qwen/Qwen3.5-4B, 32 | PyTRIO 可训练基座模型和 LoRA rank。 |
| training.batch_size, max_steps, seed | 2, 100, 42 | batch 大小、优化器更新次数上限和客户端 seed。 |
| optimizer.learning_rate | 4e-5 | 策略/学生的 AdamW 学习率。 |
| optimizer.beta1, beta2, eps | 0.9, 0.95, 1e-8 | Adam 动量与数值稳定项。 |
| optimizer.weight_decay, grad_clip_norm | 0.0, 1.0 | 权重衰减与梯度范数上限；clip norm 为 0 关闭裁剪。 |
| train_dataset.path, valid_dataset.path | 见 YAML | 准备好的 JSONL 路径，相对仓库根目录解析。 |
| train_dataset.split, valid_dataset.split | train, test | GSM8K 官方训练/测试文件对应的逻辑 split。 |
| train_dataset.max_examples, valid_dataset.max_examples | null, 100 | 加载/打乱后的样本数上限；null 保留全部记录。 |
| train_dataset.shuffle, valid_dataset.shuffle | false, false | 保持源数据顺序；true 表示打乱一次。 |
| train_dataset.seed, valid_dataset.seed | 42, 42 | shuffle 的种子；null 回退到 training.seed。 |
| train_dataset.loader_options, valid_dataset.loader_options | {}, {} | JSON loader 的额外参数；不要重复 data_files 或 split。 |

max_steps 统计优化器更新，training.batch_size 统计 prompt 数；每个 prompt 会先生成 rollout.num_generations 条学生回答，再进行一次更新。数据只遍历一次，因此可能在达到 max_steps 前因数据耗尽而结束；不会隐式循环或重复 epoch。

### 准备 GSM8K 数据

案例读取本地 JSONL，不自动下载。以下 Python 代码准备官方 train/test 两个 split，已有文件直接复用。

prompt 格式和数据文件与 GRPO/PPO 案例相同。教师对学生实际生成的 token 评分；answer 为统一格式而保留，但 OPD 训练和验证不读取该字段。

```python
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
```

JSONL 每行一条记录，最小格式如下：

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "answer": "2"
}
```

### Teacher 与蒸馏目标

| 参数 | 案例值 | 作用 |
| --- | --- | --- |
| teacher.name, model_path | Qwen/Qwen3.8-27B, null | 固定教师及可选 LoRA URI，师生词表映射必须完全相同。 |
| rollout.num_generations | 2 | 每题回答数；两道题每次更新生成四条 completion。 |
| rollout.max_completion_length, max_length | 512, 2048 | 回答与总 token 预算，生成长度受剩余空间限制。 |
| rollout.prompt_key, add_generation_prompt | messages, true | prompt 字段与 assistant 生成前缀。 |
| rollout.chat_template_kwargs, stop | {enable_thinking: false}, null | 模板参数与可选停止字符串，null 使用 EOS 行为。 |
| execution.max_concurrent_sampling, max_concurrent_scoring | 16, 32 | 独立的采样与教师评分请求并发上限。 |
| objective.kl_coef | 1.0 | 教师与 behavior logprob 差信号的正缩放系数。 |
| objective.discount_factor | 0.0 | 0 保留 token 局部信号，最大 1 可累积未来 token 信号。 |
| objective.importance_ratio_cap | 5.0 | CISPO 中不参与求导的 ratio 上界；null 使用无上限的重要性采样。 |

#### 从学生 Rollout 到 Token Target

设 $b$ 为采样 token $a_t$ 的行为策略，$q$ 为固定教师，$\lambda$ 为 objective.kl_coef。OPD 构造不参与求导的逐 token 信号

$$
\delta_t=\lambda\left[\log q(a_t\mid s_t)-\log b(a_t\mid s_t)\right],
\qquad
G_t=\delta_t+\gamma G_{t+1},
$$

其中 $\gamma$ 为 objective.discount_factor。案例值为 0 时，$G_t=\delta_t$，credit 保持在当前 token。当前学生策略 $\pi_\theta$ 通过重要性比率 $\rho_t=\pi_\theta(a_t\mid s_t)/b(a_t\mid s_t)$ 更新。CISPO 使用 $\operatorname{stopgrad}(\min(\rho_t,c))$ 作为策略梯度项的系数，其中 $c$ 为 objective.importance_ratio_cap；设为 null 时使用无上限比率。这个基于采样的 score-function 目标无需传输教师的完整词表分布，即可推动学生接近教师。

采样固定为 temperature=1、top_p=1、top_k=-1，这些不是 OPD YAML 字段。超长 prompt 会在采样前报错。并发用于批内采样/评分重叠，优化器各步仍顺序使用新 rollout。

## 核心概念与接口

### Trainer 生命周期

| 接口 | 功能与约定 |
| --- | --- |
| OPDTrainer(args=config, train_dataset=rows, eval_dataset=valid_rows) | 创建客户端；数据由案例加载，配置对象不负责打开文件。 |
| state = await trainer.train_async() | 训练、周期验证、最终保存和日志收尾。 |
| await trainer.step_async(rows) | 单批更新；自定义循环需自行管理最终保存和日志生命周期。 |
| await trainer.evaluate_async() | 对配置的验证记录评估，不进行优化器更新。 |

案例通过 asyncio.run 调用异步入口；同一 trainer 不应并发调用这些方法。
另有同步 train()、step()、evaluate()；异步应用使用上表方法。

## Triotune 实现说明

- **评估：**evaluation.steps: 20、before_train: true、after_train: true；batch_size: 2、max_batches: 4 限制每轮验证规模。设 valid_dataset: null 关闭验证。重新采样并由教师评分，记录 eval/reverse_kl_sampled 等蒸馏指标，不代表 GSM8K 答案准确率。
- **日志：**reporting.report_to: swanlab 开启上传，project、workspace、run_name、tags 设置实验信息；workspace 为 null 使用账号默认值。设 report_to 为 null 关闭上传。log_dir 相对工作目录，保存 SwanLab 产物和独立 JSONL 历史。logging_step 控制 DPO/OPD 聚合窗口；SFT 每步上报。
- **保存：**checkpoint.save_every: 20 按更新次数保存，0 仍保存最终状态；URI 记录在 checkpoint/path 和返回状态的 checkpoint_path，没有内置 resume/load 入口。

更多接口细节见[配置源码](../../triotune/api/cli_args.py)和[trainer 实现](../../triotune/trainer/opd_trainer.py)。
