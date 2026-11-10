"""Train WebShop TCOD-B2F from cached successful teacher trajectories."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from itertools import cycle, islice
from pathlib import Path
from statistics import mean
from typing import Any

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

from triotune.agentrl import (
    AgentContractError,
    AgentOPDTrainingLoop,
    EpisodeCollectionError,
)
from triotune.utils.config import load_yaml_config
from webshop_env.environment import (
    INSTRUCTIONS,
    SessionPool,
    ShoppingEpisodeExecutor,
    build_workflow,
)

from train.tcod_f2b import (
    CurriculumConfig,
    TCODAgentConfig,
    TCODF2BTrainingLoop,
)
from train.vanilla_opd import (
    Experiment,
    experiment_from_settings,
    resolve_path,
    run as run_opd,
)

DEFAULT_CONFIG = (
    Path(__file__).with_name("configs") / "tcod-b2f-qwen3.8-27b-teacher.yaml"
)
ACTION_PATTERN = re.compile(r"^(search|click)\[[^\[\]\n]+\]$")


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """Locate successful trajectories and configure their teacher collection."""

    path: Path
    pass_k: int = 10
    temperature: float = 1.0
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        for name in ("pass_k", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"experts.{name} must be a positive integer.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("experts.seed must be a nonnegative integer.")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("experts.temperature must be finite and positive.")


@dataclass(frozen=True)
class B2FExperiment(Experiment):
    experts: ExpertConfig
    curriculum: CurriculumConfig


def load_experiment(
    path: Path,
    *,
    steps: int | None = None,
    resume_from: Path | None = None,
) -> B2FExperiment:
    """Validate B2F, OPD, and expert-cache settings without remote calls."""

    values = load_yaml_config(path)
    root = path.resolve().parent
    curriculum_values = values.pop("curriculum", None)
    expert_values = values.pop("experts", None)
    if not isinstance(curriculum_values, dict):
        raise ValueError("curriculum must declare k_start and growth_interval.")
    if not isinstance(expert_values, dict) or "path" not in expert_values:
        raise ValueError("experts must declare a trajectory-cache path.")
    expert_values = dict(expert_values)
    expert_values["path"] = resolve_path(expert_values["path"], root)
    curriculum = CurriculumConfig(**curriculum_values)
    experts = ExpertConfig(**expert_values)
    base = experiment_from_settings(values, root=root, steps=steps)
    agent = TCODAgentConfig(**asdict(base.agent), curriculum=curriculum)
    trainer = base.trainer
    if resume_from is not None:
        manifest = resume_from.expanduser()
        if not manifest.is_absolute():
            manifest = (Path.cwd() / manifest).resolve()
        if not manifest.is_file():
            raise ValueError(f"Resume checkpoint manifest does not exist: {manifest}")
        trainer = replace(
            trainer,
            checkpoint=replace(trainer.checkpoint, resume_from=str(manifest)),
        )
    return B2FExperiment(
        trainer=trainer,
        agent=agent,
        data=base.data,
        source_dir=base.source_dir,
        environment_seed=base.environment_seed,
        output_dir=base.output_dir,
        experts=experts,
        curriculum=curriculum,
    )


def load_expert_actions(path: Path) -> dict[int, tuple[str, ...]]:
    """Load one successful, nonterminal-prefix-safe action list per goal."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing TCOD-B2F expert cache: {path}. Run the collect script first."
        )
    experts: dict[int, tuple[str, ...]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid expert JSONL line {line_number}: {error}") from error
        goal_index = record.get("goal_index")
        actions = record.get("actions")
        if (
            type(goal_index) is not int
            or goal_index < 0
            or not isinstance(actions, list)
            or not actions
            or any(not isinstance(action, str) or not ACTION_PATTERN.fullmatch(action) for action in actions)
            or record.get("success") is not True
        ):
            raise ValueError(f"Invalid successful expert record at line {line_number}.")
        if goal_index in experts:
            raise ValueError(f"Duplicate expert goal_index {goal_index}.")
        experts[goal_index] = tuple(actions)
    if not experts:
        raise ValueError("TCOD-B2F expert cache contains no successful trajectories.")
    return experts


def prepare_training_rows(
    train_rows: list[dict],
    eval_rows: list[dict],
    experiment: B2FExperiment,
) -> tuple[list[dict], list[dict]]:
    """Retain teacher-success tasks and deterministically fill the update plan."""

    experts = load_expert_actions(experiment.experts.path)
    eligible = [
        {**row, "expert_actions": list(experts[row["goal_index"]])}
        for row in train_rows
        if row["goal_index"] in experts
    ]
    if len(eligible) < experiment.trainer.batch_size:
        raise ValueError(
            "TCOD-B2F needs at least one full batch of successful expert tasks; "
            f"found {len(eligible)}, need {experiment.trainer.batch_size}."
        )
    required = (
        experiment.trainer.max_steps * experiment.trainer.batch_size
        if experiment.trainer.max_steps is not None
        else len(eligible)
    )
    planned = [
        {**row, "expert_cycle": index // len(eligible)}
        for index, row in enumerate(islice(cycle(eligible), required))
    ]
    return planned, eval_rows


class B2FShoppingEpisodeExecutor:
    """Replay a successful teacher prefix, then hand its state to the student."""

    def __init__(
        self,
        executor: ShoppingEpisodeExecutor,
        curriculum: CurriculumConfig,
    ) -> None:
        if not isinstance(executor, ShoppingEpisodeExecutor):
            raise TypeError("B2F requires the WebShop episode executor.")
        self.executor = executor
        self.runner = executor.runner
        self.pool = executor.pool
        self.curriculum = curriculum
        self.completed_steps = 0
        self.limits = executor.limits
        self.trajectory_builder = executor.trajectory_builder
        self.resource_owner = executor.resource_owner
        self.anchor_observer = executor.anchor_observer

    async def start(self):
        await self.executor.start()

    async def close(self):
        await self.executor.close()

    def progress(self, completed_steps: int) -> None:
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("completed_steps must be a nonnegative integer.")
        self.completed_steps = completed_steps

    async def run(self, row, snapshot, **kwargs):
        if kwargs.get("is_eval", False):
            return await self.executor.run(row, snapshot, **kwargs)
        context = kwargs["context"]
        actions = row.get("expert_actions")
        if not isinstance(actions, list) or not actions:
            raise AgentContractError("B2F training row is missing expert_actions.")
        k = self.curriculum.horizon(self.completed_steps, self.limits.max_turns)
        prefix_length = max(0, len(actions) - k)
        # Never replay the purchase itself. With k>=1 this also follows directly
        # from L-k for a successful trajectory whose last action terminates.
        prefix_length = min(prefix_length, len(actions) - 1)
        try:
            state = await self.pool.open(context.rollout_id, row["goal_index"])
            history = []
            for action in actions[:prefix_length]:
                history.append(
                    {
                        "observation": state["observation"],
                        "action": action,
                    }
                )
                state = await self.pool.step(context.rollout_id, action)
                if state["done"]:
                    raise AgentContractError(
                        "Expert prefix terminated before the student takeover."
                    )
            state = {
                **state,
                # Match the reference workflow's two-step visible history
                # without converting teacher actions into student generations.
                "teacher_prefix_steps": prefix_length,
                "recent_teacher_history": history[-2:],
            }
            task = dict(row)
            task.pop("expert_actions", None)
            task["messages"] = [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ]
            # Prefix lengths vary by task. A shallow runner copy makes the
            # remaining total interaction budget episode-local while sharing
            # immutable renderers, tools, scorers, and runtime resources.
            runner = copy.copy(self.runner)
            runner.limits = replace(
                self.runner.limits,
                max_turns=max(1, self.limits.max_turns - prefix_length),
            )
            episode = await runner.run(task, snapshot, **kwargs)
            episode.export_metadata.update(
                {
                    "tcod_variant": "b2f",
                    "curriculum_k": k,
                    "teacher_prefix_length": prefix_length,
                    "expert_length": len(actions),
                    "student_max_turns": runner.limits.max_turns,
                }
            )
            return episode
        finally:
            await asyncio.shield(self.pool.release(context.rollout_id))


def build_b2f_workflow(config, tokenizer, pool, curriculum):
    workflow = build_workflow(config, tokenizer, pool)
    workflow.runner = B2FShoppingEpisodeExecutor(workflow.runner, curriculum)
    return workflow


class TCODB2FTrainingLoop(AgentOPDTrainingLoop):
    """Run OPD after application-owned teacher-prefix environment replay."""

    def __init__(self, trainer, workflow, *, config: TCODAgentConfig, **kwargs):
        if not isinstance(workflow.runner, B2FShoppingEpisodeExecutor):
            raise TypeError("TCOD-B2F requires a B2F WebShop executor.")
        self.b2f_executor = workflow.runner
        self.curriculum = config.curriculum
        super().__init__(trainer, workflow, config=config, **kwargs)

    async def collect_and_update(self, rows):
        completed_steps = self.trainer.state.global_step
        k = self.curriculum.horizon(
            completed_steps,
            self.config.limits.max_turns,
        )
        self.b2f_executor.progress(completed_steps)
        seed_offset = self._seed_offset
        update = None
        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                update = await super().collect_and_update(rows)
                break
            except EpisodeCollectionError as error:
                self._seed_offset = seed_offset
                retryable = TCODF2BTrainingLoop._sampling_timeouts(error)
                if not retryable or attempt == self.config.retry.max_attempts:
                    raise
                await asyncio.sleep(self.config.retry.delay(attempt))
            except ExceptionGroup as error:
                self._seed_offset = seed_offset
                retryable = TCODF2BTrainingLoop._teacher_unavailable(error)
                if not retryable or attempt == self.config.retry.max_attempts:
                    raise
                await asyncio.sleep(self.config.retry.delay(attempt))
        if update is None:
            raise RuntimeError("TCOD-B2F collection retry produced no update.")
        metadata = [episode.export_metadata for episode in update.batch.episodes]
        prefix_lengths = [int(item["teacher_prefix_length"]) for item in metadata]
        expert_lengths = [int(item["expert_length"]) for item in metadata]
        metrics = {
            "curriculum/k": float(k),
            "curriculum/teacher_prefix_mean": mean(prefix_lengths),
            "curriculum/teacher_prefix_min": float(min(prefix_lengths)),
            "curriculum/teacher_prefix_max": float(max(prefix_lengths)),
            "curriculum/expert_length_mean": mean(expert_lengths),
            "curriculum/collection_attempts": float(attempt),
        }
        self.update_history[-1]["curriculum"] = {
            "completed_steps": completed_steps,
            "k": k,
            "teacher_prefix_mean": mean(prefix_lengths),
            "teacher_prefix_min": min(prefix_lengths),
            "teacher_prefix_max": max(prefix_lengths),
            "collection_attempts": attempt,
        }
        self.update_history[-1]["metrics"].update(metrics)
        await self.trainer.publish_metrics(metrics)
        self.trainer._checkpoint.publish(self.trainer)
        return update


async def run(
    experiment: B2FExperiment,
    *,
    stop_after_step: int | None = None,
) -> dict:
    """Run B2F with cached experts and the shared WebShop resource lifecycle."""

    # Fail before allocating WebShop or PyTRIO resources when the required
    # pre-collected curriculum data is absent or malformed.
    load_expert_actions(experiment.experts.path)
    return await run_opd(
        experiment,
        stop_after_step=stop_after_step,
        loop_class=TCODB2FTrainingLoop,
        algorithm="agent_tcod_b2f",
        run_label="tcod-b2f",
        workflow_factory=lambda config, tokenizer, pool: build_b2f_workflow(
            config,
            tokenizer,
            pool,
            experiment.curriculum,
        ),
        row_preparer=lambda train_rows, eval_rows: prepare_training_rows(
            train_rows,
            eval_rows,
            experiment,
        ),
        report_extra={"experts": asdict(experiment.experts)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, help="Override total training updates.")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    experiment = load_experiment(
        args.config,
        steps=args.steps,
        resume_from=args.resume_from,
    )
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    asyncio.run(run(experiment, stop_after_step=args.stop_after_step))


if __name__ == "__main__":
    main()
