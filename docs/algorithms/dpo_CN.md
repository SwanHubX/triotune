# DPO 案例——GSM8K

[English](dpo.md)

DPO 通过固定参考策略下的偏好对，学习提高 chosen 解答相对 rejected 解答的概率。

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
uv run python examples/dpo/gsm8k_dpo.py --config examples/dpo/gsm8k_dpo_args.yaml
```

运行前先完成[准备 GSM8K 数据](#准备-gsm8k-数据)。入口只接受 `--config`，超参数在 YAML 中修改。

### 配置接口

| 接口 | 用途 |
| --- | --- |
| `DPOConfig(**settings)` | 直接读取完整显式配置，案例采用此接口。 |
| `DPOConfig.from_modules(settings)` | 读取模块化配置字典，兼容部分旧式字段。 |

配置不含 `algorithm` 选择器或算法名包装；`${model.name}` 由 `load_yaml_config` 解析。完整配置见 [gsm8k_dpo_args.yaml](../../examples/dpo/gsm8k_dpo_args.yaml)。

## 核心配置参数

### 训练与 rollout

下表列的是案例配置值，不是全部库默认值。

| 参数 | 案例值 | 作用 |
| --- | --- | --- |
| `model.name, model.lora_rank` | `Qwen/Qwen3.5-4B`, `32` | PyTRIO 可训练基座模型和 LoRA rank。 |
| `training.batch_size, max_steps, seed` | `2`, `100`, `42` | batch 大小、优化器更新次数上限和客户端 seed。 |
| `optimizer.learning_rate` | `1e-5` | 策略/学生的 AdamW 学习率。 |
| `optimizer.beta1, beta2, eps` | `0.9`, `0.95`, `1e-8` | Adam 动量与数值稳定项。 |
| `optimizer.weight_decay, grad_clip_norm` | `0.0`, `1.0` | 权重衰减与梯度范数上限；clip norm 为 0 关闭裁剪。 |
| `train_dataset.path, valid_dataset.path` | 见 YAML | 准备好的 JSONL 路径，相对仓库根目录解析。 |
| `train_dataset.split, valid_dataset.split` | `train`, `test` | GSM8K 官方训练/测试文件对应的逻辑 split。 |
| `train_dataset.max_examples, valid_dataset.max_examples` | `null`, `100` | 加载/打乱后的样本数上限；null 保留全部记录。 |
| `train_dataset.shuffle, valid_dataset.shuffle` | `false`, `false` | 保持源数据顺序；true 表示打乱一次。 |
| `train_dataset.seed, valid_dataset.seed` | `42`, `42` | shuffle 的种子；null 回退到 training.seed。 |
| `train_dataset.loader_options, valid_dataset.loader_options` | `{}`, `{}` | JSON loader 的额外参数；不要重复 data_files 或 split。 |

max_steps 统计优化器更新；DPO 的 batch 单位是偏好对、OPD 是题目、SFT 是监督序列。
数据只遍历一次，耗尽会提前结束；不会隐式循环或重复 epoch。样本上限与 shuffle 由案例加载器执行。

### 准备 GSM8K 数据

案例读取本地 JSONL，不自动下载。以下 Python 代码准备官方 train/test 两个 split，已有文件直接复用。

GSM8K 原始数据没有偏好对。此案例将标准推理和正确答案作为 chosen，保持推理不变、将最终数字加 1 作为 rejected。这是用于演示的人工构造负例，不是人工偏好标注。

```python
import json
from decimal import Decimal
from pathlib import Path
from datasets import load_dataset

Path("data").mkdir(exist_ok=True)
for split in ("train", "test"):
    path = Path(f"data/gsm8k_dpo_{split}.jsonl")
    if path.exists():
        continue
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            reasoning, answer = row["answer"].rsplit("####", 1)
            answer = answer.strip().replace(",", "")
            wrong_answer = format(Decimal(answer) + 1, "f")
            item = {
                "messages": [{"role": "user", "content": row["question"]
                    + r" Provide a numerical answer without units, written inside \boxed{}."}],
                "chosen": reasoning.strip() + "\n" + rf"\boxed{{{answer}}}",
                "rejected": reasoning.strip() + "\n" + rf"\boxed{{{wrong_answer}}}",
            }
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
```

JSONL 每行一条记录，最小格式如下：

```json
{
  "messages": [{"role": "user", "content": "What is 1 + 1? Give the answer inside \\boxed{}."}],
  "chosen": "1 + 1 = 2.\n\\boxed{2}",
  "rejected": "1 + 1 = 2.\n\\boxed{3}"
}
```

### Reference 与偏好目标

| 参数 | 案例值 | 作用 |
| --- | --- | --- |
| `reference.name` | `${model.name}` | 固定参考模型，token 到 ID 映射必须与策略相同。 |
| `data.max_length` | `2048` | prompt 加任一回答的长度上限，超长偏好对报错；null 不设上限。 |
| `data.prompt_key` | `messages` | 共同 prompt 的字段名。 |
| `data.add_eos_token` | `true` | 回答缺少 EOS 时补上。 |
| `data.chat_template_kwargs` | `{enable_thinking: false}` | 渲染 prompt 的模板参数。 |
| `objective.beta` | `0.1` | 缩放策略与 reference 偏好差距的正系数。 |

DPO 对回答的 logprob 求和，不计 prompt token，reference 不更新。有限的 data.max_length 会拒绝而非截断超长偏好对；请用实际 tokenizer 同时过滤两个候选，或设为 null。

## 核心概念与接口

### Trainer 生命周期

| 接口 | 功能与约定 |
| --- | --- |
| `DPOTrainer(args=config, train_dataset=rows, eval_dataset=valid_rows)` | 创建客户端；数据由案例加载，配置对象不负责打开文件。 |
| `state = await trainer.train_async()` | 训练、周期验证、最终保存和日志收尾。 |
| `await trainer.step_async(rows)` | 单批更新；自定义循环需自行管理最终保存和日志生命周期。 |
| `await trainer.evaluate_async()` | 对配置的验证记录评估，不进行优化器更新。 |

案例通过 `asyncio.run` 调用异步入口；同一 trainer 不应并发调用这些方法。
另有同步 `train()`、`step()`、`evaluate()`；异步应用使用上表方法。

## Triotune 实现说明

- **评估：**`evaluation.steps: 20`、`before_train: true`、`after_train: true`；`batch_size: 2`、`max_batches: 4` 限制每轮验证规模。设 `valid_dataset: null` 关闭验证。用当前策略和固定 reference 评分，记录 eval/loss、eval/reward_accuracy、eval/reward_margin；偏好准确率不等于 GSM8K 解题准确率。
- **日志：**`reporting.report_to: swanlab` 开启上传，project、workspace、run_name、tags 设置实验信息；workspace 为 null 使用账号默认值。设 report_to 为 null 关闭上传。log_dir 相对工作目录，保存 SwanLab 产物和独立 JSONL 历史。logging_step 控制 DPO/OPD 聚合窗口；SFT 每步上报。
- **保存：**`checkpoint.save_every: 20` 按更新次数保存，0 仍保存最终状态；URI 记录在 checkpoint/path 和返回状态的 checkpoint_path，没有内置 resume/load 入口。

更多接口细节见[配置源码](../../triotune/api/cli_args.py)和[trainer 实现](../../triotune/trainer/dpo_trainer.py)。
