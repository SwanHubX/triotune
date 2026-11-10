"""Run multi-turn WebShop OPD through the shared repository package."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

# Use the repository package for both training and evaluation.
# The example directory supplies only its application modules and environment.
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

from triotune.agentrl import AgentConfig, AgentOPDTrainingLoop
from triotune.trainer import OPDConfig, OPDTrainer
from triotune.utils.config import load_yaml_config
from webshop_env.data_prepare import DataConfig, cached_files, task_rows
from webshop_env.environment import (
    SessionPool,
    WebShopBackend,
    build_workflow,
)

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "vanilla-opd.yaml"


@dataclass(frozen=True)
class Experiment:
    trainer: OPDConfig
    agent: AgentConfig
    data: DataConfig
    source_dir: Path
    environment_seed: int
    output_dir: Path


def resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve one user path relative to the YAML file that declared it."""

    target = Path(value).expanduser()
    return target.resolve() if target.is_absolute() else (root / target).resolve()


def load_experiment(path: Path, *, steps: int | None = None) -> Experiment:
    """Validate all settings without creating model or environment clients."""

    return experiment_from_settings(
        load_yaml_config(path), root=path.resolve().parent, steps=steps
    )


def experiment_from_settings(
    values: dict[str, Any], *, root: Path, steps: int | None = None
) -> Experiment:
    """Share validated OPD/environment settings with application curricula."""

    unknown = set(values) - {
        "trainer",
        "agent",
        "data",
        "environment",
        "output_dir",
    }
    if unknown:
        raise ValueError(f"Unknown WebShop OPD settings: {sorted(unknown)}")
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
    trainer = OPDConfig(**trainer_values)
    if trainer.model is None or trainer.teacher is None:
        raise ValueError("Vanilla Agent OPD requires one student and one teacher.")
    if trainer.num_generations < 1:
        raise ValueError("Agent OPD requires at least one rollout per task.")

    agent = AgentConfig.from_dict(values.get("agent", {}))
    if agent.async_rollouts:
        raise ValueError("Vanilla Agent OPD requires synchronous fresh rollouts.")
    if agent.log_path is None:
        agent = replace(agent, log_path=str(output / "episodes.jsonl"))
    else:
        agent = replace(agent, log_path=str(resolve_path(agent.log_path, root)))

    data_values = dict(values.get("data", {}))
    data_values["directory"] = resolve_path(data_values["directory"], root)
    data = DataConfig(**data_values)
    environment = dict(values["environment"])
    source = resolve_path(environment.pop("source_dir"), root)
    seed = environment.pop("seed", 42)
    if environment or type(seed) is not int or seed < 0:
        raise ValueError(
            "environment accepts source_dir and a nonnegative integer seed only."
        )
    return Experiment(trainer, agent, data, source, seed, output)


def write_report(output: Path, report: dict[str, Any]) -> Path:
    """Persist one strict JSON report without overwriting a previous run."""

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


async def run(
    experiment: Experiment,
    *,
    stop_after_step: int | None = None,
    loop_class: type[AgentOPDTrainingLoop] = AgentOPDTrainingLoop,
    algorithm: str = "agent_vanilla_opd",
    run_label: str = "vanilla-opd",
    workflow_factory=None,
    row_preparer=None,
    report_extra: dict[str, Any] | None = None,
) -> dict:
    """Own environment resources around one complete Agent OPD training run."""

    started = time.perf_counter()
    run_dir = experiment.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # Episode evidence belongs to this immutable run directory.  A shared path
    # would append unrelated policies and make later OPD audits ambiguous.
    agent = replace(experiment.agent, log_path=str(run_dir / "episodes.jsonl"))
    report = {
        "status": "failed",
        "algorithm": algorithm,
        "output_dir": str(run_dir),
        "trainer": asdict(experiment.trainer),
        "agent": agent.to_dict(),
        "data": asdict(experiment.data),
        "source_dir": str(experiment.source_dir),
        "environment_seed": experiment.environment_seed,
    }
    if report_extra:
        report.update(report_extra)
    pool = None
    trainer = None
    loop_owns_cleanup = False
    try:
        workflow_factory = workflow_factory or build_workflow
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
        if row_preparer is not None:
            train_rows, eval_rows = row_preparer(train_rows, eval_rows)
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

        trainer = OPDTrainer(
            args=experiment.trainer,
            train_dataset=train_rows,
            eval_dataset=eval_rows,
            defer_checkpoint_loop_binding=True,
        )
        workflow = workflow_factory(
            agent,
            trainer.tokenizer,
            pool,
        )
        loop = loop_class(
            trainer,
            workflow,
            config=agent,
        )
        loop_owns_cleanup = True
        state = await loop.train(stop_after_step=stop_after_step)
        report.update(
            status="completed",
            checkpoint=state.checkpoint_path,
            global_step=state.global_step,
            optimizer_steps=state.optimizer_steps,
            behavior_policy_version=state.behavior_policy_version,
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
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Pause cleanly after this global step for checkpoint testing.",
    )
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
