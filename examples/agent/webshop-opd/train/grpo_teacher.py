"""Train a WebShop-specialized Qwen3.5-4B teacher with AgentRL GRPO."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

from triotune.agentrl import AgentConfig
from triotune.trainer import PPOConfig, PPOTrainer
from triotune.utils.config import load_yaml_config
from webshop_env.data_prepare import DataConfig, cached_files, task_rows
from webshop_env.environment import SessionPool, WebShopBackend, build_workflow
from webshop_env.metrics import WebShopMetricsAgentTrainingLoop

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "grpo-teacher.yaml"


@dataclass(frozen=True)
class Experiment:
    trainer: PPOConfig
    agent: AgentConfig
    data: DataConfig
    source_dir: Path
    environment_seed: int
    exact_success_alpha: float
    output_dir: Path


def resolve_path(value: str | Path, root: Path) -> Path:
    target = Path(value).expanduser()
    return target.resolve() if target.is_absolute() else (root / target).resolve()


def load_experiment(path: Path, *, steps: int | None = None) -> Experiment:
    """Build the GRPO preset and validate local paths without remote calls."""

    values = load_yaml_config(path)
    unknown = set(values) - {
        "trainer",
        "agent",
        "data",
        "environment",
        "output_dir",
    }
    if unknown:
        raise ValueError(f"Unknown WebShop GRPO settings: {sorted(unknown)}")
    root = path.resolve().parent
    output = resolve_path(values["output_dir"], root)

    trainer_values = dict(values["trainer"])
    if steps is not None:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer.")
        trainer_values.setdefault("training", {})["max_steps"] = steps
    for section, names in (
        ("reporting", ("log_dir",)),
        ("checkpoint", ("state_dir", "resume_from")),
    ):
        settings = trainer_values.get(section, {})
        for name in names:
            if settings.get(name):
                settings[name] = str(resolve_path(settings[name], root))
    trainer = PPOConfig.for_algorithm("grpo", **trainer_values)
    if trainer.model is None:
        raise ValueError("GRPO teacher training requires a student model.")
    if trainer.use_critic:
        raise ValueError("GRPO must remain critic-free.")
    if trainer.num_generations < 2:
        raise ValueError("GRPO requires at least two trajectories per task group.")

    agent = AgentConfig.from_dict(values.get("agent", {}))
    if agent.async_rollouts:
        raise ValueError("This GRPO baseline requires synchronous fresh rollouts.")
    if agent.log_path is not None:
        agent = replace(agent, log_path=str(resolve_path(agent.log_path, root)))

    data_values = dict(values.get("data", {}))
    data_values["directory"] = resolve_path(data_values["directory"], root)
    data = DataConfig(**data_values)
    environment = dict(values["environment"])
    source = resolve_path(environment.pop("source_dir"), root)
    seed = environment.pop("seed", 42)
    # This example owns reward shaping because the mixture is specific to
    # WebShop's partial-credit task score.  The generic PPO/AgentRL modules
    # continue receiving an ordinary scalar reward and need no modification.
    exact_success_alpha = environment.pop("exact_success_alpha", 0.0)
    if (
        isinstance(exact_success_alpha, bool)
        or not isinstance(exact_success_alpha, (int, float))
        or not 0.0 <= float(exact_success_alpha) <= 1.0
    ):
        raise ValueError("environment.exact_success_alpha must be in [0,1].")
    if environment or type(seed) is not int or seed < 0:
        raise ValueError(
            "environment accepts source_dir, seed, and exact_success_alpha only."
        )
    return Experiment(
        trainer,
        agent,
        data,
        source,
        seed,
        float(exact_success_alpha),
        output,
    )


def write_report(output: Path, report: dict[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.json"
    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


async def run(experiment: Experiment, *, stop_after_step: int | None = None) -> dict:
    """Own the WebShop catalog around one complete GRPO training lifecycle."""

    started = time.perf_counter()
    run_dir = experiment.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-grpo"
    run_dir.mkdir(parents=True, exist_ok=False)
    # Training trajectories retain complete evidence. Evaluation summaries are
    # handled by AgentTrainingLoop metrics and do not change the GRPO objective.
    agent = replace(experiment.agent, log_path=str(run_dir / "episodes.jsonl"))
    report = {
        "status": "failed",
        "algorithm": "agent_grpo",
        "reward": "webshop_task_score_exact_success_mixture",
        "exact_success_alpha": experiment.exact_success_alpha,
        "output_dir": str(run_dir),
        "trainer": asdict(experiment.trainer),
        "agent": agent.to_dict(),
        "data": asdict(experiment.data),
        "source_dir": str(experiment.source_dir),
        "environment_seed": experiment.environment_seed,
    }
    pool = None
    trainer = None
    loop_owns_cleanup = False
    try:
        files = cached_files(experiment.data)
        backend = await asyncio.to_thread(
            WebShopBackend,
            experiment.source_dir,
            files,
            experiment.environment_seed,
        )
        pool = SessionPool(backend)
        train_rows = task_rows(backend.goals, experiment.data, "train")
        eval_rows = task_rows(backend.goals, experiment.data, "eval")
        required = (
            experiment.trainer.max_steps * experiment.trainer.batch_size
            if experiment.trainer.max_steps is not None
            else 0
        )
        if len(train_rows) < required:
            raise ValueError(
                "Training rows are fewer than max_steps * batch_size; "
                "increase data.train_epochs."
            )

        trainer = PPOTrainer(
            args=experiment.trainer,
            train_dataset=train_rows,
            eval_dataset=eval_rows,
        )
        await trainer.initialize()
        workflow = build_workflow(
            agent,
            trainer.tokenizer,
            pool,
            exact_success_alpha=experiment.exact_success_alpha,
        )
        loop = WebShopMetricsAgentTrainingLoop(trainer, workflow, config=agent)
        loop_owns_cleanup = True
        state = await loop.train(stop_after_step=stop_after_step)
        report.update(
            status="completed",
            checkpoint=state.checkpoint_path,
            global_step=state.global_step,
            optimizer_steps=state.optimizer_steps,
            metrics=state.metrics,
            updates=loop.update_history,
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if trainer is not None and not loop_owns_cleanup:
            await trainer.close(state="crashed", error=str(error))
        raise
    finally:
        try:
            if pool is not None:
                await pool.close()
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            path = write_report(run_dir, report)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "report": str(path),
                        "checkpoint": report.get("checkpoint"),
                    },
                    ensure_ascii=False,
                )
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, help="Override total training steps.")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration without opening WebShop or PyTRIO.",
    )
    args = parser.parse_args()
    experiment = load_experiment(args.config, steps=args.steps)
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    asyncio.run(run(experiment, stop_after_step=args.stop_after_step))


if __name__ == "__main__":
    main()
