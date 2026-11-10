"""Train WebShop TCOD-F2B using the shared repository OPD API."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from pytrio.errors.actor import ActorAPIError

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

from triotune.agentrl import (
    AgentConfig,
    AgentOPDTrainingLoop,
    EpisodeCollectionError,
)
from triotune.utils.config import load_yaml_config
from webshop_env.environment import ShoppingEpisodeExecutor
from train.vanilla_opd import Experiment, experiment_from_settings, run as run_opd

DEFAULT_CONFIG = (
    Path(__file__).with_name("configs") / "tcod-f2b-grpo-teacher.yaml"
)


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Grow the rollout prefix after each interval of completed updates.

    The first update has completed_steps=0, matching the reference WebShop
    implementation's batch-index convention. The schedule is independent of
    max_steps, so extending or resuming a run never restarts its curriculum.
    """

    k_start: int = 1
    growth_interval: int = 4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"curriculum.{name} must be a positive integer.")

    def horizon(self, completed_steps: int, max_turns: int) -> int:
        if type(completed_steps) is not int or completed_steps < 0:
            raise ValueError("completed_steps must be a nonnegative integer.")
        if type(max_turns) is not int or max_turns < self.k_start:
            raise ValueError("max_turns must be an integer >= curriculum.k_start.")
        return min(self.k_start + completed_steps // self.growth_interval, max_turns)


@dataclass(frozen=True, slots=True)
class TCODAgentConfig(AgentConfig):
    """Include application-owned curriculum settings in checkpoint identity.

    AgentConfig.to_dict() serializes dataclass fields, so the existing Agent
    OPD checkpoint binding also validates this schedule on resume. The fixed
    limits describe evaluation and the maximum training horizon, never the
    temporary prefix of an individual update.
    """

    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def __post_init__(self) -> None:
        # Explicit base call also works with slots=True dataclass replacement.
        AgentConfig.__post_init__(self)
        if isinstance(self.curriculum, dict):
            object.__setattr__(self, "curriculum", CurriculumConfig(**self.curriculum))
        if not isinstance(self.curriculum, CurriculumConfig):
            raise TypeError("curriculum must be a CurriculumConfig or mapping.")
        self.curriculum.horizon(0, self.limits.max_turns)
        if self.async_rollouts:
            raise ValueError("TCOD-F2B requires synchronous fresh rollouts.")


def load_experiment(
    path: Path,
    *,
    steps: int | None = None,
    resume_from: Path | None = None,
) -> Experiment:
    """Validate the schedule and existing OPD settings without remote calls."""

    values = load_yaml_config(path)
    curriculum_values = values.pop("curriculum", None)
    if not isinstance(curriculum_values, dict):
        raise ValueError("curriculum must declare k_start and growth_interval.")
    curriculum = CurriculumConfig(**curriculum_values)
    experiment = experiment_from_settings(
        values, root=path.resolve().parent, steps=steps
    )
    agent = TCODAgentConfig(**asdict(experiment.agent), curriculum=curriculum)
    if resume_from is not None:
        manifest = resume_from.expanduser()
        if not manifest.is_absolute():
            manifest = (Path.cwd() / manifest).resolve()
        if not manifest.is_file():
            raise ValueError(f"Resume checkpoint manifest does not exist: {manifest}")
        trainer = replace(
            experiment.trainer,
            checkpoint=replace(
                experiment.trainer.checkpoint,
                resume_from=str(manifest),
            ),
        )
        experiment = replace(experiment, trainer=trainer)
    return replace(experiment, agent=agent)


class TCODF2BTrainingLoop(AgentOPDTrainingLoop):
    """Limit collection to an expanding prefix; reuse the existing OPD loss.

    The inherited lifecycle serializes collection, optimization, and evaluation.
    Temporary runner limits are restored even after teacher/tool failures or
    cancellation, so evaluation always receives the original full horizon.
    """

    def __init__(self, trainer, workflow, *, config: TCODAgentConfig, **kwargs):
        if not isinstance(config, TCODAgentConfig):
            raise TypeError("TCOD-F2B requires a TCODAgentConfig.")
        self.curriculum = config.curriculum
        self.max_turns = config.limits.max_turns
        # WebShop's wrapper copies the inner runner's limits. Both must change:
        # the collector reserves seed ranges from the wrapper while the inner
        # runner enforces the actual interaction budget.
        self._runners = [workflow.runner]
        if isinstance(workflow.runner, ShoppingEpisodeExecutor):
            self._runners.append(workflow.runner.runner)
        if any(runner.limits != config.limits for runner in self._runners):
            raise ValueError("TCOD runner limits must match the fixed agent limits.")
        super().__init__(trainer, workflow, config=config, **kwargs)

    @contextmanager
    def _training_horizon(self, horizon: int):
        previous = [runner.limits for runner in self._runners]
        try:
            for runner, limits in zip(self._runners, previous, strict=True):
                runner.limits = replace(limits, max_turns=horizon)
            yield
        finally:
            for runner, limits in zip(self._runners, previous, strict=True):
                runner.limits = limits

    @staticmethod
    def _sampling_timeouts(error: EpisodeCollectionError) -> bool:
        """Return whether every rejected episode exhausted student sampling.

        A strict collection can also contain usable episodes, but no optimizer
        or teacher request occurs before the collector raises. Retrying the
        whole transaction is therefore safe when every rejection came from
        timeout retries around the read-only student sampling request. Tool,
        scoring, contract, and model-action failures keep their existing
        fail-fast behavior.
        """

        rejected = [episode for episode in error.batch.episodes if not episode.trainable]
        return bool(rejected) and all(
            episode.status == "timeout"
            and episode.retries
            and all(retry.stage == "sampling" for retry in episode.retries)
            for episode in rejected
        )

    @staticmethod
    def _teacher_unavailable(error: ExceptionGroup) -> bool:
        """Return whether all task-group leaves are teacher HTTP 503 errors."""

        leaves: list[BaseException] = []

        def flatten(current: BaseException) -> None:
            if isinstance(current, BaseExceptionGroup):
                for child in current.exceptions:
                    flatten(child)
            else:
                leaves.append(current)

        flatten(error)
        return bool(leaves) and all(
            isinstance(leaf, ActorAPIError) and leaf.status_code == 503
            for leaf in leaves
        )

    async def collect_and_update(self, rows):
        completed_steps = self.trainer.state.global_step
        horizon = self.curriculum.horizon(completed_steps, self.max_turns)
        seed_offset = self._seed_offset
        update = None
        # Stop generation at k itself; no late actions or observations are
        # collected and discarded. Unfinished, zero-reward prefixes remain
        # eligible for the same dense teacher-scoring objective as vanilla OPD.
        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                with self._training_horizon(horizon):
                    update = await super().collect_and_update(rows)
                break
            except EpisodeCollectionError as error:
                # The failed transaction performed no teacher scoring or
                # optimizer update. Reuse its deterministic seed range and ask
                # the trainer for a fresh sampler snapshot on the next attempt.
                self._seed_offset = seed_offset
                retryable = self._sampling_timeouts(error)
                if not retryable or attempt == self.config.retry.max_attempts:
                    raise
                await asyncio.sleep(self.config.retry.delay(attempt))
            except ExceptionGroup as error:
                # Teacher scoring materializes every target before optimization.
                # A pure Actor HTTP 503 group therefore leaves model/optimizer
                # state untouched and is safe to replay with the same seeds.
                self._seed_offset = seed_offset
                retryable = self._teacher_unavailable(error)
                if not retryable or attempt == self.config.retry.max_attempts:
                    raise
                await asyncio.sleep(self.config.retry.delay(attempt))
        if update is None:  # Defensive: the loop either succeeds or raises.
            raise RuntimeError("TCOD-F2B collection retry produced no update.")
        metrics = {
            "curriculum/horizon": float(horizon),
            "curriculum/max_turns": float(self.max_turns),
            "curriculum/completed_steps": float(completed_steps),
            "curriculum/collection_attempts": float(attempt),
        }
        self.update_history[-1]["curriculum"] = {
            "completed_steps": completed_steps,
            "horizon": horizon,
            "max_turns": self.max_turns,
            "collection_attempts": attempt,
        }
        self.update_history[-1]["metrics"].update(metrics)
        await self.trainer.publish_metrics(metrics)
        # The parent commits its seed cursor and update evidence. Republish
        # after adding application metrics so a saved manifest also carries
        # the horizon audit. Schedule correctness depends only on global_step
        # and the immutable config, including if this extra publish fails.
        self.trainer._checkpoint.publish(self.trainer)
        return update


async def run(experiment: Experiment, *, stop_after_step: int | None = None) -> dict:
    """Use the shared WebShop resource/report lifecycle with the F2B loop."""

    if not isinstance(experiment.agent, TCODAgentConfig):
        raise TypeError("TCOD-F2B requires a curriculum experiment.")
    return await run_opd(
        experiment,
        stop_after_step=stop_after_step,
        loop_class=TCODF2BTrainingLoop,
        algorithm="agent_tcod_f2b",
        run_label="tcod-f2b",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, help="Override total training updates.")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from a complete local checkpoint manifest.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config and curriculum without opening WebShop or PyTRIO.",
    )
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
