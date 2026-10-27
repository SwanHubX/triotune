# 保存进度与断点续训

[English](checkpoint_resume.md)

PyTRIO 将模型和优化器状态保存在云端。启用进度保存后，Triotune 会生成本地 JSON 清单，记录对应的云端 checkpoint 和训练进度。恢复训练时，将 checkpoint.resume_from 设置为这份清单的路径。

## 1. 准备环境和数据

在仓库根目录安装依赖并登录 PyTRIO：

```bash
uv sync --locked
uv run --no-sync trio login
```

首次训练和恢复训练使用相同的数据与顺序。可以使用 list、tuple 或固定顺序的 Hugging Face Dataset；训练和评估数据都需要能够按索引重复读取。

## 2. 配置进度保存

在实验 YAML 中设置：

```yaml
training:
  max_steps: 4
  seed: 42

checkpoint:
  save_every: 2
  state_dir: output/checkpoints/sft-demo
  resume_from: null
```

路径默认相对运行命令时的工作目录。calculator 的实验 YAML 中，路径相对该 YAML 所在目录。

## 3. 运行一个完整的 SFT 示例

将下面的脚本保存为仓库根目录的 train_resume.py。它使用 4 条固定样本，计划训练 4 步，并接受暂停和恢复参数。

```python
import argparse
import asyncio

from triotune.trainer import SFTConfig, SFTTrainer


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()

    questions = [
        ("What is 1 + 1?", "2"),
        ("What is 2 + 2?", "4"),
        ("What is 3 + 3?", "6"),
        ("What is 4 + 4?", "8"),
    ]
    rows = [
        {
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": answer}],
        }
        for question, answer in questions
    ]
    config = SFTConfig(
        model={"name": "Qwen/Qwen3.5-4B", "lora_rank": 8},
        training={"batch_size": 1, "max_steps": 4, "seed": 42},
        data={
            "max_length": 256,
            "completion_only_loss": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        reporting={"report_to": None},
        checkpoint={
            "save_every": 2,
            "state_dir": "output/checkpoints/sft-demo",
            "resume_from": args.resume_from,
        },
    )
    trainer = SFTTrainer(args=config, train_dataset=rows)
    state = await trainer.train(stop_after_step=args.stop_after_step)
    print(f"global_step={state.global_step}")
    print(f"resume_from={state.checkpoint_manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 先训练到第 2 步并退出

```bash
uv run --no-sync python train_resume.py --stop-after-step 2
```

脚本完成第 2 步后保存并退出，打印 global_step=2 和清单路径。清单位于 output/checkpoints/sft-demo/sft-step-00000002.json，也可以直接复制打印的完整路径。

stop_after_step 填写全局步数。暂停时会保存进度；按步数安排的评估照常进行，训练结束时才执行的评估和样本展示留到总目标完成后执行。

### 从第 2 步恢复，继续训练到第 4 步

在新的进程中运行：

```bash
uv run --no-sync python train_resume.py \
  --resume-from output/checkpoints/sft-demo/sft-step-00000002.json
```

脚本会加载云端模型和优化器状态，恢复数据位置与训练进度，继续完成第 3、4 步。完成后打印 global_step=4 和新的清单路径。

max_steps=4 始终表示总共完成 4 步，恢复后本次还需执行 2 步。希望继续训练到更大的总步数时，可以提高 max_steps。

## 4. 在其他训练入口中使用

PPO、DPO、OPD 和 agent 使用相同的 checkpoint 配置。恢复时重新创建对应的配置和 trainer，设置 checkpoint.resume_from，再调用训练方法。

| 训练入口 | 保存进度并暂停 | 恢复后继续训练 |
| --- | --- | --- |
| SFTTrainer | await trainer.train(stop_after_step=K) | await trainer.train() |
| PPOTrainer | await trainer.train(stop_after_step=K) | await trainer.train() |
| DPOTrainer 异步 | await trainer.train_async(stop_after_step=K) | await trainer.train_async() |
| DPOTrainer 同步 | trainer.train(stop_after_step=K) | trainer.train() |
| OPDTrainer 异步 | await trainer.train_async(stop_after_step=K) | await trainer.train_async() |
| OPDTrainer 同步 | trainer.train(stop_after_step=K) | trainer.train() |
| AgentTrainingLoop | await loop.train(stop_after_step=K) | await loop.train() |

agent 恢复时，使用相同的 agent 配置和 workflow 创建新的 AgentTrainingLoop。使用同步 rollout 和 trainer 默认的 snapshot factory；保持 async_rollouts=False。

## 5. 日常使用要点

- 意外中断后，使用最近一次成功生成的 JSON 清单恢复。
- 恢复时保持模型、LoRA rank、batch size、seed、学习率、采样与评估设置以及数据内容和顺序一致。
- stop_after_step 应大于当前已完成的步数，并且不超过 max_steps。预先安排暂停时，保持原来的总目标步数。
- 每个训练任务使用独立的 state_dir。恢复后可以继续写入原目录，也可以设置新的保存目录；只设置 resume_from 时，清单继续写入原清单所在目录。
- 换机器时，携带 JSON 清单并准备相同的数据和配置，使用能访问对应云端 checkpoint 的 PyTRIO 账号登录。
- 将 JSON 清单路径传给 resume_from；云端模型和优化器的 URI 已记录在清单中。
