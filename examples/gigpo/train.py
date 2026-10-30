"""Orchestrate WebShop configuration, checks, sampling, and training.

Configuration creates PPOConfig, AgentConfig, and GiGPOConfig objects directly.
The module coordinates execution order and resource lifecycles while environment,
data, and advantage calculations remain in their dedicated modules.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytrio

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentTrainingLoop,
    GiGPOAdvantageProvider,
    GiGPOConfig,
    probe_tokenizer,
)
from triotune.utils.config import load_yaml_config

from .data_prepare import (
    DataConfig,
    asset_manifest,
    cached_files,
    goal_fingerprint,
    goal_ranges,
    task_rows,
)
from .environment import RewardConfig, SessionPool, WebShopBackend, build_workflow

ALGORITHMS = ("gigpo", "grpo", "ppo")
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gigpo-webshop-configs.yaml"


@dataclass(frozen=True)
class ContinuationConfig:
    """Describe a continuation from a remote optimizer-only checkpoint."""

    checkpoint: str
    global_step: int

    def __post_init__(self):
        if not isinstance(self.checkpoint, str) or not self.checkpoint.startswith(
            "trio://"
        ):
            raise ValueError("continuation.checkpoint must be a trio:// URI.")
        if type(self.global_step) is not int or self.global_step < 1:
            raise ValueError("continuation.global_step must be a positive integer.")


@dataclass(frozen=True)
class Experiment:
    """Hold validated experiment settings with resolved absolute paths.

    The algorithm selects advantage behavior. Trainer, agent, and GiGPO settings
    configure optimization, interaction, and joint advantages. Data and reward
    settings control goal splits and environment scores.
    """

    algorithm: str
    trainer: PPOConfig
    agent: AgentConfig
    data: DataConfig
    gigpo: GiGPOConfig
    reward: RewardConfig
    source_dir: Path
    environment_seed: int
    output_dir: Path
    continuation: ContinuationConfig | None = None


def resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve a configured path relative to its YAML directory."""
    target = Path(value).expanduser()
    return target.resolve() if target.is_absolute() else (root / target).resolve()


def load_experiment(
    path: Path,
    *,
    model: str | None = None,
    steps: int | None = None,
    algorithm: str | None = None,
    step_advantage_weight: float | None = None,
    output_dir: Path | None = None,
) -> Experiment:
    """Build and validate an experiment from YAML without opening a service.

    Keyword arguments apply CLI overrides for model, update count, algorithm,
    state-advantage weight, and output directory. Configuration conflicts fail
    before any remote work starts.
    """
    values = load_yaml_config(path)
    unknown = set(values) - {
        "algorithm",
        "trainer",
        "agent",
        "data",
        "environment",
        "gigpo",
        "reward",
        "output_dir",
        "continuation",
    }
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    selected = algorithm or values.get("algorithm", "gigpo")
    if selected not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
    if step_advantage_weight is not None and selected != "gigpo":
        raise ValueError("--step-advantage-weight requires algorithm=gigpo.")
    root = path.resolve().parent
    trainer_values = values["trainer"]
    if model is not None:
        trainer_values.setdefault("model", {})["name"] = model
    if steps is not None:
        trainer_values.setdefault("training", {})["max_steps"] = steps
    for section, names in (
        ("reporting", ("log_dir",)),
        ("checkpoint", ("state_dir", "resume_from")),
    ):
        for name in names:
            if trainer_values.get(section, {}).get(name):
                trainer_values[section][name] = str(
                    resolve_path(trainer_values[section][name], root)
                )
    # Algorithm overrides use their own normalization defaults instead of the
    # disabled normalization required by GiGPO replacement advantages.
    configured_algorithm = values.get("algorithm", "gigpo")
    if configured_algorithm == "gigpo" and selected != "gigpo":
        actor = trainer_values.setdefault("actor", {})
        actor.pop("reward_norm", None)
        actor.pop("adv_norm", None)
    # GiGPO uses a critic-free actor while its provider supplies full advantages.
    if selected == "gigpo":
        actor = trainer_values.setdefault("actor", {})
        actor.setdefault("reward_norm", None)
        actor.setdefault("adv_norm", None)
    trainer = PPOConfig.for_algorithm(
        "grpo" if selected == "gigpo" else selected, **trainer_values
    )
    if trainer.use_critic != (selected == "ppo"):
        raise ValueError("PPO requires critic.enabled=true; GiGPO/GRPO require false.")
    if (
        selected == "grpo"
        and trainer.reward_norm != PPOConfig.for_algorithm("grpo").reward_norm
    ):
        raise ValueError("GRPO requires its preset group reward normalization.")
    if selected == "ppo" and trainer.reward_norm is not None:
        raise ValueError("PPO requires reward_norm=null for the GAE baseline.")
    if trainer.num_generations < 2:
        raise ValueError(
            "Algorithm comparisons require at least two candidates per goal."
        )
    if trainer.eval_max_batches is not None:
        raise ValueError(
            "Use data.eval_limit for explicit, reported evaluation subsets."
        )
    continuation = (
        ContinuationConfig(**values["continuation"])
        if values.get("continuation") is not None
        else None
    )
    if continuation is not None and trainer.checkpoint.resume_from:
        raise ValueError(
            "Use either legacy continuation or checkpoint.resume_from, not both."
        )
    if continuation is not None and (
        trainer.max_steps is None
        or continuation.global_step >= trainer.max_steps
        or not trainer.checkpoint.state_dir
    ):
        raise ValueError(
            "Legacy continuation requires max_steps above global_step and "
            "checkpoint.state_dir for complete future checkpoints."
        )
    output = resolve_path(values.get("output_dir", "../../output/webshop-gigpo"), root)
    if output_dir is not None:
        output = output_dir.expanduser().resolve()
        trainer = replace(
            trainer,
            reporting=replace(trainer.reporting, log_dir=str(output / "metrics")),
        )
    agent = AgentConfig.from_dict(values.get("agent", {}))
    agent = replace(
        agent,
        log_path=str(resolve_path(agent.log_path, root))
        if agent.log_path
        else str(output / "episodes.jsonl"),
    )
    if (
        agent.parser != "strict_json"
        or agent.prompt_key != "messages"
        or agent.renderer != "token_append"
        or agent.async_rollouts
        or agent.export_mode != "concat"
        or agent.context_compaction
    ):
        raise ValueError(
            "Require strict_json/messages/token_append/concat, synchronous rollouts, no compaction."
        )
    if agent.retry.retry_tools:
        raise ValueError("Stateful WebShop actions must not be retried.")
    gigpo_values = dict(values.get("gigpo", {}))
    if step_advantage_weight is not None:
        gigpo_values["step_advantage_weight"] = step_advantage_weight
    gigpo = GiGPOConfig(**gigpo_values)
    if gigpo.enable_similarity:
        raise ValueError(
            "Structured WebShop anchors require gigpo.enable_similarity=false."
        )
    if selected == "gigpo":
        # Validation needs only configuration and the algorithm plan.
        GiGPOAdvantageProvider(gigpo).validate_trainer(
            SimpleNamespace(args=trainer, algorithm_plan=trainer.algorithm_plan)
        )
    environment = dict(values["environment"])
    source = resolve_path(environment.pop("source_dir"), root)
    seed = environment.pop("seed", 42)
    if environment or type(seed) is not int or seed < 0:
        raise ValueError(
            "environment accepts source_dir and a nonnegative integer seed only."
        )
    return Experiment(
        selected,
        trainer,
        agent,
        DataConfig(**values.get("data", {})),
        gigpo,
        RewardConfig(**values.get("reward", {})),
        source,
        seed,
        output,
        continuation,
    )


def advantage_provider(experiment: Experiment):
    """Return a GiGPO provider or select trainer-native advantages."""
    return (
        GiGPOAdvantageProvider(experiment.gigpo)
        if experiment.algorithm == "gigpo"
        else None
    )


def save_report(output: Path, mode: str, split: str, report: dict) -> Path:
    """Write a finite JSON report for one execution mode and split."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{mode}-{split}-report.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


async def open_environment(experiment: Experiment, split: str, report: dict):
    """Load pinned assets, environment sessions, and task rows.

    The requested split selects evaluation or test rows. Setup failures close the
    session pool immediately. The caller owns the pool after successful return.
    """
    files = cached_files(experiment.data)
    report["assets"] = asset_manifest(files)
    backend = await asyncio.to_thread(
        WebShopBackend, experiment.source_dir, files, experiment.environment_seed
    )
    pool = SessionPool(backend)
    try:
        goals = backend.goals
        ranges = goal_ranges(len(goals), experiment.data.split_protocol)
        train_rows = task_rows(goals, experiment.data, "train")
        eval_rows = task_rows(goals, experiment.data, split)
        revision = await asyncio.to_thread(
            subprocess.check_output,
            ["git", "-C", str(experiment.source_dir), "rev-parse", "HEAD"],
            text=True,
        )
        report.update(
            goal_count=len(goals),
            goal_sha256=goal_fingerprint(goals),
            goal_ranges={
                name: [indices.start, indices.stop] for name, indices in ranges.items()
            },
            range_convention="[start, stop)",
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
            eval_goal_indices=[row["goal_index"] for row in eval_rows],
            source_revision=revision.strip(),
        )
        return pool, train_rows, eval_rows
    except BaseException:
        await pool.close()
        raise


async def check_environment(pool: SessionPool, row: dict) -> dict:
    """Reset one goal, execute its search query, and return visible states."""
    session = "environment-check"
    try:
        initial = await pool.open(session, row["goal_index"])
        query = pool.backend.goals[row["goal_index"]]["query"]
        searched = await pool.step(session, f"search[{query}]")
        if not searched["available_actions"].get("clickables"):
            raise RuntimeError("WebShop search returned no clickable results.")
        return {"initial": initial, "search": searched}
    finally:
        await pool.release(session)


async def probe_policy(
    experiment: Experiment, workflow, snapshot, rows: list[dict]
) -> dict:
    """Sample one complete candidate group without optimizer updates."""
    config = experiment.trainer
    batch = await workflow.collect(
        rows[:1],
        snapshot,
        n_samples=config.num_generations,
        seed=config.seed,
        is_eval=False,
        strict=False,
    )
    if not batch.trajectories or any(
        episode.score is None for episode in batch.episodes
    ):
        raise RuntimeError("Probe failed; inspect episode statuses in episodes.jsonl.")
    result = {"metrics": batch.metrics(prefix="probe")}
    provider = advantage_provider(experiment)
    if provider is not None:
        result["advantage_rows"] = len(provider.advantages(batch.trajectories).values)
    return result


def apply_legacy_continuation(experiment: Experiment, trainer, train_rows) -> dict:
    """Reconstruct deterministic local counters for optimizer-only state."""
    continuation = experiment.continuation
    if continuation is None:
        return {}
    config = experiment.trainer
    if config.max_steps is None or continuation.global_step >= config.max_steps:
        raise ValueError(
            "Legacy continuation requires finite max_steps above global_step."
        )
    data_cursor = continuation.global_step * config.batch_size
    if data_cursor > len(train_rows):
        raise ValueError("Continuation data cursor exceeds training rows.")
    seed_offset = (
        data_cursor
        * config.num_generations
        * experiment.agent.limits.max_turns
    )
    trainer.state.global_step = continuation.global_step
    trainer.state.optimizer_steps = continuation.global_step
    trainer.state.data_cursor = data_cursor
    trainer.state.checkpoint_path = continuation.checkpoint
    trainer._actor.optimizer_steps = continuation.global_step
    trainer._checkpoint.saved_step = continuation.global_step
    return {
        "source_checkpoint": continuation.checkpoint,
        "restored_global_step": trainer.state.global_step,
        "restored_optimizer_steps": trainer.state.optimizer_steps,
        "restored_data_cursor": trainer.state.data_cursor,
        "restored_seed_offset": seed_offset,
        "exact_resume": False,
        "missing_legacy_state": [
            "rng",
            "reporting_aggregates",
            "prior_update_history",
        ],
    }


async def run(args) -> dict:
    """Execute one CLI experiment and persist its report on every exit path.

    ``--checkpoint`` applies only to independent evaluation. A continuation
    restores remote optimizer state and reconstructs deterministic local
    progress. Cancellation and failures still release all owned resources.
    """
    from .evaluation import evaluate_policy, load_actor_config

    if args.checkpoint is not None and not args.checkpoint.strip():
        raise ValueError("--checkpoint must be a nonempty cloud actor LoRA URI.")
    if args.checkpoint and args.mode != "evaluate":
        raise ValueError("--checkpoint is only supported by evaluate mode.")
    if args.eval_split == "test" and args.mode != "evaluate":
        raise ValueError("Final test tasks require explicit evaluate mode.")
    experiment = load_experiment(
        args.config,
        model=args.model,
        steps=args.steps,
        algorithm=getattr(args, "algorithm", None),
        step_advantage_weight=getattr(args, "step_advantage_weight", None),
        output_dir=getattr(args, "output_dir", None),
    )
    config = experiment.trainer
    report = {
        "mode": args.mode,
        "algorithm": experiment.algorithm,
        "eval_split": args.eval_split,
        "model": config.model.name,
        "trainer": asdict(config),
        "checkpoint_input": args.checkpoint,
        "data": asdict(experiment.data),
        "environment_seed": experiment.environment_seed,
        "gigpo": asdict(experiment.gigpo),
        "reward": asdict(experiment.reward),
        "agent": experiment.agent.to_dict(),
        "continuation": (
            asdict(experiment.continuation) if experiment.continuation else None
        ),
        "evaluation_uses_test_tasks": experiment.data.split_protocol == "verl",
    }
    pool = workflow = trainer = None
    managed_training = False
    try:
        pool, train_rows, eval_rows = await open_environment(
            experiment, args.eval_split, report
        )
        if args.mode == "check":
            report["environment_check"] = await check_environment(pool, train_rows[0])
        else:
            if (
                args.mode == "train"
                and config.max_steps is not None
                and len(train_rows) < config.max_steps * config.batch_size
            ):
                raise ValueError(
                    "Not enough task rows for max_steps * batch_size; increase data.train_epochs."
                )
            service = await asyncio.to_thread(pytrio.ServiceClient)
            if experiment.continuation and args.mode not in {"train", "probe"}:
                raise ValueError(
                    "continuation is supported in train and probe modes only."
                )
            if experiment.continuation:
                if args.checkpoint:
                    raise ValueError(
                        "Use continuation.checkpoint for training; --checkpoint is evaluation-only."
                    )
                effective_config = experiment.trainer
                client = None
            else:
                effective_config, client = await load_actor_config(
                    experiment,
                    service,
                    args.checkpoint,
                    evaluation=args.mode == "evaluate",
                )
            trainer = PPOTrainer(
                args=effective_config,
                service_client=service,
                training_client=client,
                train_dataset=train_rows,
                eval_dataset=eval_rows,
            )
            if experiment.continuation:
                trainer._checkpoint.bootstrap_remote_optimizer(
                    experiment.continuation.checkpoint
                )
                await trainer.initialize()
                continuation_runtime = apply_legacy_continuation(
                    experiment, trainer, train_rows
                )
            snapshot = await trainer.create_sampling_snapshot(
                stop=experiment.agent.stop
            )
            report["effective_trainer"] = asdict(effective_config)
            report["tokenizer_snapshot_id"] = snapshot.snapshot_id
            report["tokenizer_probe"] = probe_tokenizer(
                snapshot.tokenizer,
                template_kwargs=experiment.agent.template_kwargs,
                renderer=experiment.agent.renderer,
            )
            workflow = build_workflow(
                experiment.agent, snapshot.tokenizer, pool, experiment.reward
            )
            provider = (
                None if args.mode == "evaluate" else advantage_provider(experiment)
            )
            loop = AgentTrainingLoop(
                trainer,
                workflow,
                config=experiment.agent,
                batch_advantage_provider=provider,
            )
            if experiment.continuation:
                loop._seed_offset = continuation_runtime["restored_seed_offset"]
                report["continuation_runtime"] = continuation_runtime
            if args.mode == "probe":
                report.update(
                    await probe_policy(experiment, workflow, snapshot, train_rows)
                )
            elif args.mode == "evaluate":
                report.update(await evaluate_policy(loop, checkpoint=args.checkpoint))
            else:
                # The loop collects complete batches, computes advantages,
                # updates the policy, refreshes snapshots, evaluates, and saves.
                managed_training = True
                state = await loop.train()
                report.update(
                    step=state.global_step,
                    checkpoint=state.checkpoint_path,
                    metrics=state.metrics,
                    updates=loop.update_history,
                )
                if (
                    config.max_steps is not None
                    and state.global_step != config.max_steps
                ):
                    raise RuntimeError(
                        "Task rows exhausted before max_steps; increase data.train_epochs."
                    )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        if trainer is not None and not managed_training:
            await trainer.close(state="crashed", error=str(error))
        raise
    else:
        if trainer is not None and not managed_training:
            await trainer.close()
    finally:
        # Nested cleanup keeps closing resources and writing the report even if
        # an earlier cleanup operation fails.
        try:
            if workflow is not None:
                await workflow.close()
        finally:
            try:
                if pool is not None:
                    await pool.close()
            finally:
                path = save_report(
                    experiment.output_dir, args.mode, args.eval_split, report
                )
                print(
                    json.dumps(
                        {
                            "report": str(path),
                            "status": report.get("status"),
                            "checkpoint": report.get("checkpoint"),
                        },
                        ensure_ascii=False,
                    )
                )
    return report


def build_parser(*, evaluation: bool = False) -> argparse.ArgumentParser:
    """Build the shared training or evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="WebShop algorithm evaluation" if evaluation else __doc__
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--algorithm", choices=ALGORITHMS, help="Override the experiment algorithm"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Separate run directory, relative to the working directory",
    )
    parser.add_argument("--model", help="Override the PyTRIO base model ID")
    parser.add_argument(
        "--checkpoint", help="Cloud actor LoRA state URI returned by training"
    )
    parser.add_argument("--eval-split", choices=("eval", "test"), default="eval")
    if evaluation:
        parser.set_defaults(mode="evaluate", steps=None, step_advantage_weight=None)
    else:
        parser.add_argument(
            "--mode", choices=("check", "probe", "train", "evaluate"), default="train"
        )
        parser.add_argument("--steps", type=int, help="Override policy update count")
        parser.add_argument(
            "--step-advantage-weight",
            type=float,
            help="GiGPO state advantage weight; zero disables state credit",
        )
    return parser


def main(argv=None):
    """Parse optional arguments and run the experiment."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint and args.mode != "evaluate":
        parser.error("--checkpoint is only supported by --mode evaluate.")
    if args.eval_split == "test" and args.mode != "evaluate":
        parser.error("Final test tasks must be selected explicitly in evaluate mode.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
