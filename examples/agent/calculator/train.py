"""Run tokenizer checks, live probing, training or checkpoint evaluation.

uv run python examples/agent/calculator/train.py --mode train
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytrio

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import AgentConfig, AgentTrainingLoop, CostConfig, probe_tokenizer
from triotune.api.client_api import call_async
from triotune.utils.config import load_yaml_config

if __package__:
    from .agent import calculator_rows, calculator_workflow, generated_rows
else:
    from agent import calculator_rows, calculator_workflow, generated_rows


@dataclass(frozen=True)
class DataConfig:
    train_tasks: int = 32
    eval_tasks: int = 8
    seed: int = 42

    def __post_init__(self):
        for name in ("train_tasks", "eval_tasks"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError(f"{name} must be an integer in [1, 10000].")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("data.seed must be a nonnegative integer.")


def load_experiment(path: Path, *, model=None, steps=None):
    settings = load_yaml_config(path)
    unknown = set(settings) - {"trainer", "agent", "reward", "data", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    values = dict(settings["trainer"])
    algorithm = values.pop("algorithm", "grpo")
    if model is not None:
        values["model"] = {**values.get("model", {}), "name": model}
    if steps is not None:
        values["training"] = {**values.get("training", {}), "max_steps": steps}
    reporting = dict(values.get("reporting", {}))
    if reporting.get("log_dir") and not Path(reporting["log_dir"]).is_absolute():
        reporting["log_dir"] = str(
            (path.resolve().parent / reporting["log_dir"]).resolve()
        )
    values["reporting"] = reporting
    trainer = PPOConfig.for_algorithm(algorithm, **values)
    agent = AgentConfig.from_dict(settings.get("agent", {}))
    costs = CostConfig(**settings.get("reward", {}))
    data = DataConfig(**settings.get("data", {}))
    output = Path(settings.get("output_dir", "output/calculator-agent"))
    # Resolve output paths relative to the supplied experiment file.
    if not output.is_absolute():
        output = (path.resolve().parent / output).resolve()
    if agent.log_path is None:
        agent = replace(agent, log_path=str(output / "episodes.jsonl"))
    elif not Path(agent.log_path).is_absolute():
        agent = replace(agent, log_path=str(path.resolve().parent / agent.log_path))
    return trainer, agent, costs, data, output


def gradient_check(batch, config) -> dict[str, Any]:
    """Exercise the actual PPO loss with recorded live trajectories on CPU."""
    import torch

    from triotune.trainer.ppo import prepare_advantages
    from triotune.trainer.ppo.datum import batch_datums
    from triotune.trainer.ppo.loss import make_loss_fn

    if not batch.trajectories:
        raise RuntimeError("No trajectories available for gradient validation.")
    prepared = prepare_advantages(batch.trajectories, config)
    tensors = []
    for trajectory in prepared.trajectories:
        tensors.append(
            torch.tensor(
                [0.0] * (len(trajectory.prompt_tokens) - 1) + trajectory.logprobs,
                requires_grad=True,
            )
        )
    loss, _ = make_loss_fn(prepared, config)(batch_datums(prepared), tensors)
    loss.backward()
    action_gradient, context_gradient = 0.0, 0.0
    for trajectory, tensor in zip(prepared.trajectories, tensors, strict=True):
        gradients = tensor.grad[len(trajectory.prompt_tokens) - 1 :].tolist()
        for gradient, active in zip(gradients, trajectory.loss_mask, strict=True):
            if active:
                action_gradient += abs(gradient)
            else:
                context_gradient += abs(gradient)
    if context_gradient != 0:
        raise RuntimeError("Inserted context received a policy gradient.")
    return {
        "action_gradient_l1": action_gradient,
        "context_gradient_l1": context_gradient,
    }


async def run(args) -> dict:
    config, agent_config, costs, data, output = load_experiment(
        args.config, model=args.model, steps=args.steps
    )
    if getattr(args, "trace", None) is not None:
        agent_config = replace(agent_config, log_path=str(args.trace.resolve()))
    if getattr(args, "template_kwargs", None) is not None:
        agent_config = replace(
            agent_config, template_kwargs=json.loads(args.template_kwargs)
        )
    report = {
        "mode": args.mode,
        "model": config.model.name,
        "agent": agent_config.to_dict(),
    }
    if (
        args.mode == "train"
        and config.max_steps is not None
        and data.train_tasks < config.max_steps * config.batch_size
    ):
        raise ValueError(
            "data.train_tasks must cover max_steps * batch_size (plus any refill rows)."
        )
    if args.mode == "tokenizer":
        from transformers import AutoTokenizer

        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, config.model.name
        )
        report["tokenizer_probe"] = probe_tokenizer(
            tokenizer,
            template_kwargs=agent_config.template_kwargs,
            renderer=agent_config.renderer,
        )
    else:
        service = await asyncio.to_thread(pytrio.ServiceClient)
        training_client = None
        if args.checkpoint:
            training_client = await call_async(
                service, "create_training_client_from_state", path=args.checkpoint
            )
        train_rows = generated_rows(data.train_tasks, split="train", seed=data.seed)
        eval_rows = generated_rows(data.eval_tasks, split="eval", seed=data.seed)
        trainer = PPOTrainer(
            args=config,
            service_client=service,
            training_client=training_client,
            train_dataset=train_rows,
            eval_dataset=eval_rows,
        )
        managed = False
        try:
            snapshot = await trainer.create_sampling_snapshot(stop=agent_config.stop)
            report["tokenizer_probe"] = probe_tokenizer(
                snapshot.tokenizer,
                template_kwargs=agent_config.template_kwargs,
                renderer=agent_config.renderer,
            )
            workflow = calculator_workflow(
                config=agent_config, tokenizer=snapshot.tokenizer, costs=costs
            )
            loop = AgentTrainingLoop(trainer, workflow, config=agent_config)
            if args.mode == "probe":
                batch = await workflow.collect(
                    calculator_rows()[:1],
                    snapshot,
                    n_samples=2,
                    seed=config.seed,
                    is_eval=True,
                    strict=False,
                )
                report["metrics"] = batch.metrics(prefix="probe")
                report["snapshot_id"] = snapshot.snapshot_id
                report["gradient_check"] = gradient_check(batch, config)
                if not any(
                    e.trainable and e.score.success and len(e.turns) >= 2
                    for e in batch.episodes
                ):
                    raise RuntimeError(
                        "Probe did not complete a successful tool episode; inspect episodes.jsonl."
                    )
            elif args.mode == "evaluate":
                report["metrics"] = await loop.evaluate()
                report["checkpoint"] = args.checkpoint
                report["optimizer_updates"] = trainer.state.optimizer_steps
                if trainer.state.optimizer_steps != 0:
                    raise RuntimeError(
                        "Evaluation unexpectedly changed optimizer state."
                    )
            else:
                managed = True
                state = await loop.train()
                report["step"] = state.global_step
                report["checkpoint"] = state.checkpoint_path
                report["updates"] = loop.update_history
                report["metrics"] = state.metrics
                if (
                    config.max_steps is not None
                    and state.global_step != config.max_steps
                ):
                    raise RuntimeError(
                        "Task rows were exhausted before the requested updates; increase data.train_tasks."
                    )
                report["gradient_check"] = gradient_check(loop.last_train_batch, config)
                after = await trainer.create_sampling_snapshot(stop=agent_config.stop)
                before = loop.last_train_snapshot
                trajectory = loop.last_train_batch.trajectories[0]
                full_ids = trajectory.prompt_tokens + trajectory.completion_tokens

                async def logprobs(snapshot):
                    result = await call_async(
                        snapshot.sampler.client,
                        "compute_logprobs",
                        prompt=pytrio.ModelInput.from_ints(full_ids),
                    )
                    if len(result) != len(full_ids):
                        raise RuntimeError(
                            "Verification logprobs do not align with full token IDs."
                        )
                    return [
                        float(v)
                        for v, active in zip(
                            result[len(trajectory.prompt_tokens) :],
                            trajectory.loss_mask,
                            strict=True,
                        )
                        if active
                    ]

                old_lp, new_lp = await asyncio.gather(logprobs(before), logprobs(after))
                change = max(abs(a - b) for a, b in zip(old_lp, new_lp, strict=True))
                if not math.isfinite(change) or change <= 0:
                    raise RuntimeError(
                        "Updated snapshot showed no finite action-logprob change."
                    )
                report["policy_change"] = {
                    "before_snapshot": before.snapshot_id,
                    "after_snapshot": after.snapshot_id,
                    "max_action_logprob_delta": change,
                }
                if (
                    not state.checkpoint_path
                    or not loop.update_history
                    or any(u["optimizer_steps"] <= 0 for u in loop.update_history)
                ):
                    raise RuntimeError(
                        "Live training did not complete optimizer updates and checkpoint saving."
                    )
                if any(len(u["snapshot_ids"]) != 1 for u in loop.update_history) or len(
                    {u["snapshot_ids"][0] for u in loop.update_history}
                ) != len(loop.update_history):
                    raise RuntimeError(
                        "Each update must use one distinct refreshed policy snapshot."
                    )
        except BaseException as exc:
            if not managed:
                try:
                    await trainer.close(state="crashed", error=str(exc))
                except Exception as cleanup_error:  # noqa: BLE001 -- preserve the original failure
                    exc.add_note(f"Reporter cleanup also failed: {cleanup_error}")
            report["error"] = str(exc)
            try:
                save_report(output, args.mode, report)
            except Exception as report_error:  # noqa: BLE001 -- preserve the original failure
                exc.add_note(f"Failure report could not be saved: {report_error}")
            raise
        else:
            if not managed:
                await trainer.close()
    report_path = save_report(output, args.mode, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "mode": args.mode,
                "model": config.model.name,
                **{
                    key: report[key]
                    for key in ("step", "checkpoint", "gradient_check", "policy_change")
                    if key in report
                },
                "success_rate": report.get("metrics", {}).get(
                    "eval/agent/success_rate",
                    report.get("metrics", {}).get("probe/success_rate"),
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def save_report(output, mode, report):
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{mode}-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument(
        "--mode", choices=("tokenizer", "probe", "train", "evaluate"), default="probe"
    )
    parser.add_argument("--model", help="Override the PyTRIO base-model ID")
    parser.add_argument("--steps", type=int, help="Override the number of updates")
    parser.add_argument("--checkpoint", help="Saved actor state to evaluate")
    parser.add_argument("--trace", type=Path, help="Override the episode JSONL path")
    parser.add_argument(
        "--template-kwargs", help="Override chat-template options as a JSON object"
    )
    args = parser.parse_args(argv)
    if args.checkpoint and args.mode != "evaluate":
        parser.error("--checkpoint is supported only in evaluate mode.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
