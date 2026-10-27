# Saving progress and resuming training

[中文](checkpoint_resume_CN.md)

PyTRIO saves model and optimizer states in the cloud. When progress saving is enabled, Triotune creates a local JSON manifest that records the corresponding cloud checkpoint and training progress. To resume training, set checkpoint.resume_from to the path of this manifest.

## 1. Prepare the environment and data

Install the dependencies and log in to PyTRIO from the repository root:

```bash
uv sync --locked
uv run --no-sync trio login
```

Use the same data in the same order for the initial run and the resumed run. You can use a list, a tuple, or a Hugging Face Dataset with a fixed order; both training and evaluation data must support repeated reads by index.

## 2. Configure progress saving

Set the following fields in your experiment YAML:

```yaml
training:
  max_steps: 4
  seed: 42

checkpoint:
  save_every: 2
  state_dir: output/checkpoints/sft-demo
  resume_from: null
```

By default, paths are relative to the working directory where you run the command. In calculator experiment YAML files, paths are relative to the directory containing that YAML file.

## 3. Run a complete SFT example

Save the following script as train_resume.py in the repository root. It uses 4 fixed examples, targets 4 training steps, and accepts pause and resume arguments.

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

### Train through step 2 and exit

```bash
uv run --no-sync python train_resume.py --stop-after-step 2
```

After completing step 2, the script saves progress and exits, printing global_step=2 and the manifest path. The manifest is at output/checkpoints/sft-demo/sft-step-00000002.json; you can also copy the full path printed by the script.

stop_after_step takes a global step number. Pausing saves progress; evaluations scheduled at step intervals still run, while evaluations and sample displays scheduled for the end of training run after the total target is reached.

### Resume from step 2 and continue through step 4

Run the following in a new process:

```bash
uv run --no-sync python train_resume.py \
  --resume-from output/checkpoints/sft-demo/sft-step-00000002.json
```

The script loads the cloud model and optimizer states, restores the data position and training progress, and completes steps 3 and 4. It then prints global_step=4 and the new manifest path.

max_steps=4 always means 4 completed steps in total, so this resumed run performs 2 more steps. For constant LR or a schedule with scheduler.total_steps set explicitly from the first run, you can increase max_steps. Keep the original target when the decay horizon is inferred from max_steps.

## 4. Use other training entry points

PPO, DPO, OPD, and agents use the same checkpoint configuration. To resume, recreate the corresponding configuration and trainer, set checkpoint.resume_from, and call the training method.

| Training entry point | Save progress and pause | Continue after restoring |
| --- | --- | --- |
| SFTTrainer | await trainer.train(stop_after_step=K) | await trainer.train() |
| PPOTrainer | await trainer.train(stop_after_step=K) | await trainer.train() |
| DPOTrainer async | await trainer.train_async(stop_after_step=K) | await trainer.train_async() |
| DPOTrainer sync | trainer.train(stop_after_step=K) | trainer.train() |
| OPDTrainer async | await trainer.train_async(stop_after_step=K) | await trainer.train_async() |
| OPDTrainer sync | trainer.train(stop_after_step=K) | trainer.train() |
| AgentTrainingLoop | await loop.train(stop_after_step=K) | await loop.train() |

To resume an agent, create a new AgentTrainingLoop with the same agent configuration and workflow. Use synchronous rollouts and the trainer's default snapshot factory; keep async_rollouts=False.

## 5. Everyday usage

- After an unexpected interruption, resume from the most recently generated complete JSON manifest.
- Keep the model, LoRA rank, batch size, seed, learning rate and scheduler configuration, sampling and evaluation settings, and data content and order the same when resuming.
- Set stop_after_step above the current completed step count and no higher than max_steps. Keep the original total target when planning a pause.
- Give each training task its own state_dir. A resumed run can continue writing to the original directory or use a new directory; when only resume_from is set, manifests continue to be written beside the original manifest.
- When moving to another machine, bring the JSON manifest, prepare the same data and configuration, and log in with a PyTRIO account that can access the corresponding cloud checkpoint.
- Pass the JSON manifest path to resume_from; the cloud model and optimizer URIs are already recorded in that manifest.
