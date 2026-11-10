"""Resume a trained WebShop GRPO policy for purchase-precision fine-tuning."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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

from triotune.agentrl import AgentConfig, EpisodeScore
from triotune.trainer import PPOConfig, PPOTrainer
from triotune.utils.config import load_yaml_config
from webshop_env.data_prepare import DataConfig, cached_files, task_rows
from webshop_env.environment import (
    SessionPool,
    WebShopBackend,
    WebShopScorer,
    build_workflow,
    current_rollout_id,
)
from webshop_env.metrics import WebShopMetricsAgentTrainingLoop

from train.grpo_teacher import resolve_path, write_report


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "resume-grpo-teacher.yaml"


@dataclass(frozen=True)
class PrecisionRewardConfig:
    """Make an exact, protocol-complete purchase the only positive outcome.

    An incorrect purchase receives ``incorrect_purchase_base`` minus a penalty
    proportional to the unmatched fraction reported by WebShop. The base must
    be below the no-purchase reward, so an uncertain policy is encouraged to
    keep checking instead of committing to a known-wrong item.
    """

    exact_purchase_reward: float = 1.0
    incorrect_purchase_base: float = -0.3
    mismatch_penalty: float = 0.4
    no_purchase_reward: float = -0.2
    invalid_action_reward: float = -0.4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"reward.{name} must be finite.")
        if self.exact_purchase_reward <= 0:
            raise ValueError("reward.exact_purchase_reward must be positive.")
        if self.mismatch_penalty < 0:
            raise ValueError("reward.mismatch_penalty must be nonnegative.")
        if not (
            self.incorrect_purchase_base
            < self.no_purchase_reward
            < self.exact_purchase_reward
        ):
            raise ValueError(
                "Reward ordering must be incorrect purchase < no purchase < exact purchase."
            )
        if self.invalid_action_reward > self.no_purchase_reward:
            raise ValueError(
                "reward.invalid_action_reward must not exceed no_purchase_reward."
            )

    def training_reward(
        self,
        *,
        task_score: float,
        purchased: bool,
        success: bool,
        status: str,
    ) -> float:
        """Return one auditable terminal reward for the precision phase."""

        if success:
            return float(self.exact_purchase_reward)
        if status == "invalid_action":
            return float(self.invalid_action_reward)
        if purchased:
            return float(
                self.incorrect_purchase_base
                - self.mismatch_penalty * (1.0 - task_score)
            )
        return float(self.no_purchase_reward)


class PrecisionWebShopScorer(WebShopScorer):
    """Use strict purchase rewards for training and native scores for evaluation."""

    def __init__(self, pool, config: PrecisionRewardConfig):
        super().__init__(pool)
        self.precision_config = config

    async def score(self, row, episode) -> EpisodeScore:
        done, task_score = await self.pool.outcome(current_rollout_id())
        success = episode.status == "completed" and done and task_score == 1.0
        if episode.is_eval:
            reward = task_score if done else 0.0
        else:
            reward = self.precision_config.training_reward(
                task_score=task_score,
                purchased=bool(done),
                success=success,
                status=episode.status,
            )
        return EpisodeScore(
            reward,
            success,
            {
                "task_score": task_score,
                "purchased": float(done),
                "exact_purchase": float(success),
                "incorrect_purchase": float(done and not success),
            },
        )


@dataclass(frozen=True)
class Experiment:
    trainer: PPOConfig
    agent: AgentConfig
    data: DataConfig
    source_dir: Path
    environment_seed: int
    reward_mode: str
    reward: PrecisionRewardConfig | None
    output_dir: Path
    checkpoint: str
    resume_step: int


@dataclass(frozen=True)
class ContinuationConfig:
    """Identify a remote Train checkpoint and its completed optimizer step."""

    checkpoint: str
    global_step: int
    additional_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checkpoint, str)
            or not self.checkpoint.startswith("trio://")
        ):
            raise ValueError("continuation.checkpoint must be a trio:// Train URI.")
        if type(self.global_step) is not int or self.global_step < 1:
            raise ValueError("continuation.global_step must be a positive integer.")
        if type(self.additional_steps) is not int or self.additional_steps < 1:
            raise ValueError("continuation.additional_steps must be a positive integer.")


def load_experiment(path: Path, *, steps: int | None = None) -> Experiment:
    """Load a strict-reward continuation and validate its remote checkpoint ID."""

    values = load_yaml_config(path)
    unknown = set(values) - {
        "trainer",
        "agent",
        "data",
        "environment",
        "reward",
        "continuation",
        "output_dir",
    }
    if unknown:
        raise ValueError(f"Unknown precision-resume settings: {sorted(unknown)}")
    root = path.resolve().parent
    output = resolve_path(values["output_dir"], root)
    continuation = ContinuationConfig(**values["continuation"])

    trainer_values = dict(values["trainer"])
    additional_steps = continuation.additional_steps if steps is None else steps
    if type(additional_steps) is not int or additional_steps < 1:
        raise ValueError("steps must be a positive number of additional updates.")
    trainer_values.setdefault("training", {})["max_steps"] = (
        continuation.global_step + additional_steps
    )
    for section, names in (
        ("reporting", ("log_dir",)),
        ("checkpoint", ("state_dir", "resume_from")),
    ):
        settings = trainer_values.get(section, {})
        for name in names:
            if settings.get(name):
                settings[name] = str(resolve_path(settings[name], root))
    trainer = PPOConfig.for_algorithm("grpo", **trainer_values)
    if trainer.model is None or trainer.use_critic or trainer.num_generations < 2:
        raise ValueError(
            "Precision continuation requires a critic-free GRPO model with grouped rollouts."
        )
    if trainer.checkpoint.resume_from:
        raise ValueError(
            "Use continuation.checkpoint for this remote-ID resume entry point."
        )
    resume_step = continuation.global_step

    agent = AgentConfig.from_dict(values.get("agent", {}))
    if agent.async_rollouts:
        raise ValueError("Precision continuation requires synchronous fresh rollouts.")
    if agent.log_path is not None:
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
    reward_values = dict(values.get("reward", {}))
    reward_mode = reward_values.pop("mode", "precision")
    if reward_mode not in {"native", "precision"}:
        raise ValueError("reward.mode must be native or precision.")
    if reward_mode == "native" and reward_values:
        raise ValueError("Native reward mode accepts only reward.mode.")
    reward = (
        PrecisionRewardConfig(**reward_values)
        if reward_mode == "precision"
        else None
    )
    return Experiment(
        trainer=trainer,
        agent=agent,
        data=data,
        source_dir=source,
        environment_seed=seed,
        reward_mode=reward_mode,
        reward=reward,
        output_dir=output,
        checkpoint=continuation.checkpoint,
        resume_step=resume_step,
    )


def build_resume_scorer(experiment: Experiment, pool):
    """Select native task score or strict purchase-precision reward."""

    if experiment.reward_mode == "native":
        return WebShopScorer(pool)
    if experiment.reward is None:
        raise RuntimeError("Precision reward configuration is missing.")
    return PrecisionWebShopScorer(pool, experiment.reward)


def apply_remote_continuation(
    experiment: Experiment, trainer: PPOTrainer, train_rows
) -> dict[str, Any]:
    """Rebuild deterministic local progress around a remote optimizer restore."""

    data_cursor = experiment.resume_step * experiment.trainer.batch_size
    if data_cursor > len(train_rows):
        raise ValueError("Continuation data cursor exceeds the training rows.")
    seed_offset = (
        data_cursor
        * experiment.trainer.num_generations
        * experiment.agent.limits.max_turns
    )
    trainer.state.global_step = experiment.resume_step
    trainer.state.optimizer_steps = experiment.resume_step
    trainer.state.data_cursor = data_cursor
    trainer.state.checkpoint_path = experiment.checkpoint
    trainer._actor.optimizer_steps = experiment.resume_step
    trainer._checkpoint.saved_step = experiment.resume_step
    return {
        "source_checkpoint": experiment.checkpoint,
        "restored_global_step": experiment.resume_step,
        "restored_optimizer_steps": experiment.resume_step,
        "restored_data_cursor": data_cursor,
        "restored_seed_offset": seed_offset,
        "exact_resume": False,
        "missing_remote_state": [
            "rng",
            "reporting_aggregates",
            "prior_update_history",
            "prior_token_accounting",
        ],
    }


async def run(experiment: Experiment, *, stop_after_step: int | None = None) -> dict:
    """Resume optimizer/runtime state with the configured WebShop reward."""

    started = time.perf_counter()
    run_dir = experiment.output_dir / (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{experiment.reward_mode}-grpo"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    agent = replace(experiment.agent, log_path=str(run_dir / "episodes.jsonl"))
    report: dict[str, Any] = {
        "status": "failed",
        "algorithm": f"agent_grpo_{experiment.reward_mode}_resume",
        "reward": {
            "mode": experiment.reward_mode,
            **(asdict(experiment.reward) if experiment.reward is not None else {}),
        },
        "checkpoint_input": experiment.checkpoint,
        "resume_step": experiment.resume_step,
        "target_step": experiment.trainer.max_steps,
        "output_dir": str(run_dir),
        "trainer": asdict(experiment.trainer),
        "agent": agent.to_dict(),
        "data": asdict(experiment.data),
        "source_dir": str(experiment.source_dir),
        "environment_seed": experiment.environment_seed,
    }
    pool = None
    trainer = None
    workflow = None
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
        required = experiment.trainer.max_steps * experiment.trainer.batch_size
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
        trainer._checkpoint.bootstrap_remote_optimizer(experiment.checkpoint)
        await trainer.initialize()
        continuation = apply_remote_continuation(experiment, trainer, train_rows)
        report["continuation"] = continuation
        scorer = build_resume_scorer(experiment, pool)
        workflow = build_workflow(
            agent,
            trainer.tokenizer,
            pool,
            scorer=scorer,
        )
        loop = WebShopMetricsAgentTrainingLoop(trainer, workflow, config=agent)
        loop._seed_offset = continuation["restored_seed_offset"]
        # Re-evaluate the restored teacher with the new purchase metrics before
        # any precision update. train() will not repeat step 200 because this
        # call advances the restored evaluation boundary to the current step.
        report["resume_metrics"] = await loop.evaluate()
        loop_owns_cleanup = True
        state = await loop.train(stop_after_step=stop_after_step)
        report.update(
            status="completed",
            checkpoint=state.checkpoint_path,
            sampler_checkpoint=state.sampler_checkpoint_path,
            checkpoint_manifest=state.checkpoint_manifest_path,
            global_step=state.global_step,
            optimizer_steps=state.optimizer_steps,
            metrics=state.metrics,
            updates=loop.update_history,
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if workflow is not None and not loop_owns_cleanup:
            await workflow.close()
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
                        "sampler_checkpoint": report.get("sampler_checkpoint"),
                    },
                    ensure_ascii=False,
                )
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--steps",
        type=int,
        help="Override continuation.additional_steps.",
    )
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config and checkpoint-ID syntax without remote calls.",
    )
    args = parser.parse_args()
    experiment = load_experiment(args.config, steps=args.steps)
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    asyncio.run(run(experiment, stop_after_step=args.stop_after_step))


if __name__ == "__main__":
    main()
