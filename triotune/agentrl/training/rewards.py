"""Compose outcome scorers, verifiable costs, turn rewards and token PRMs.

Only the outcome scorer determines task success. Auxiliary scorers contribute
weighted terminal/process rewards; diagnostic components never implicitly add
to reward. All process vectors refer to original generated tokens, including
returned stop tokens. Turn scalars are placed on each turn's last output token.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol

from ..types import AgentContractError, EpisodeScore, EpisodeScorer


RewardTarget = Literal["interaction", "turn", "token", "trajectory"]
RewardMode = Literal["reward", "advantage"]
RewardAggregation = Literal["sum", "mean", "last"]


class RewardSubmitter(Protocol):
    """Accept delayed structured assignments for one logical rollout."""

    def submit(
        self,
        assignments,
        *,
        logical_rollout_id: str | None = None,
    ) -> tuple["RewardAssignment", ...]: ...


class RewardAggregator(Protocol):
    """Fold accepted assignments into outcome, process and advantage signals."""

    def aggregate(self): ...


@dataclass(frozen=True, slots=True)
class RewardAssignment:
    """One explicit contribution submitted by a scorer.

    ``target`` describes the temporal object receiving the contribution and
    ``mode`` determines whether it enters the return recurrence or is added
    directly to the policy advantage.  ``position`` is optional because a
    trajectory-level assignment needs no index; interaction, turn and token
    assignments use it to retain their original location.  ``source`` is a
    stable diagnostic name and is also the grouping key for aggregation.
    """

    target: RewardTarget
    value: float
    mode: RewardMode = "reward"
    aggregation: RewardAggregation = "sum"
    discount: float = 1.0
    source: str = "scorer"
    position: int | tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.target not in {"interaction", "turn", "token", "trajectory"}:
            raise ValueError("RewardAssignment.target is invalid.")
        if self.mode not in {"reward", "advantage"}:
            raise ValueError("RewardAssignment.mode is invalid.")
        if self.aggregation not in {"sum", "mean", "last"}:
            raise ValueError("RewardAssignment.aggregation is invalid.")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("RewardAssignment.value must be numeric.")
        if not math.isfinite(float(self.value)):
            raise ValueError("RewardAssignment.value must be finite.")
        if isinstance(self.discount, bool) or not isinstance(self.discount, Real):
            raise TypeError("RewardAssignment.discount must be numeric.")
        if not math.isfinite(float(self.discount)) or not 0 <= self.discount <= 1:
            raise ValueError("RewardAssignment.discount must be in [0, 1].")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("RewardAssignment.source must be nonempty.")
        if self.target == "trajectory" and self.position is not None:
            raise ValueError("Trajectory assignments do not have a position.")
        if self.target in {"interaction", "turn"} and self.position is not None and (
            type(self.position) is not int or self.position < 0
        ):
            raise ValueError("Interaction and turn positions must be nonnegative integers.")
        if self.target == "token" and self.position is not None and (
            not isinstance(self.position, tuple)
            or len(self.position) != 2
            or any(type(value) is not int or value < 0 for value in self.position)
        ):
            raise ValueError("Token positions must be (turn, token) integer pairs.")


def aggregate_assignments(
    assignments: Sequence[RewardAssignment],
) -> tuple[float, dict[int, tuple[float, ...]], dict[int, tuple[float, ...]]]:
    """Fold submitted assignments into outcome, process and direct-advantage data.

    Assignments are grouped by source, mode, target and position.  This makes
    ``mean`` and ``last`` explicit and prevents a scorer retry from silently
    adding the same logical submission twice when callers reuse the list.
    """

    groups: dict[tuple, list[RewardAssignment]] = {}
    for assignment in assignments:
        key = (
            assignment.mode,
            assignment.target,
            assignment.source,
            assignment.position,
        )
        groups.setdefault(key, []).append(assignment)

    outcome = 0.0
    process: dict[int, dict[int, float]] = {}
    direct: dict[int, dict[int, float]] = {}
    for (mode, target, _source, position), values in groups.items():
        weighted = [float(item.value) * float(item.discount) for item in values]
        if values[0].aggregation == "mean":
            value = sum(weighted) / len(weighted)
        elif values[0].aggregation == "last":
            value = weighted[-1]
        else:
            value = sum(weighted)
        if target == "trajectory":
            if mode == "reward":
                outcome += value
            continue
        if target == "token":
            if not isinstance(position, tuple):
                raise AgentContractError("Token assignments require a (turn, token) position.")
            turn, token = position
        else:
            if type(position) is not int:
                raise AgentContractError(f"{target.title()} assignments require a position.")
            turn, token = position, -1
        destination = direct if mode == "advantage" else process
        row = destination.setdefault(turn, {})
        row[token] = row.get(token, 0.0) + value

    def vectors(source: dict[int, dict[int, float]]) -> dict[int, tuple[float, ...]]:
        return {
            turn: tuple(values[index] for index in sorted(values))
            for turn, values in source.items()
        }

    return outcome, vectors(process), vectors(direct)


class RewardLedger:
    """Idempotent reward submission buffer for one logical rollout session.

    A ledger accepts delayed scorer results and suppresses repeated trajectory
    outcome submissions for the same logical rollout. Process and advantage
    assignments remain position keyed, so independently scored interactions
    can still contribute without multiplying the terminal outcome.
    """

    def __init__(self):
        self._assignments: list[RewardAssignment] = []
        self._trajectory_submissions: set[tuple[str, str, str]] = set()

    def submit(
        self,
        assignments: RewardAssignment | Sequence[RewardAssignment],
        *,
        logical_rollout_id: str | None = None,
    ) -> tuple[RewardAssignment, ...]:
        values = (assignments,) if isinstance(assignments, RewardAssignment) else tuple(assignments)
        accepted = []
        for assignment in values:
            if not isinstance(assignment, RewardAssignment):
                raise TypeError("RewardLedger accepts RewardAssignment objects.")
            if logical_rollout_id is not None and assignment.target == "trajectory":
                key = (logical_rollout_id, assignment.source, assignment.mode)
                if key in self._trajectory_submissions:
                    continue
                self._trajectory_submissions.add(key)
            self._assignments.append(assignment)
            accepted.append(assignment)
        return tuple(accepted)

    commit = submit

    @property
    def assignments(self) -> tuple[RewardAssignment, ...]:
        return tuple(self._assignments)

    def aggregate(self):
        return aggregate_assignments(self._assignments)


def validate_process_rewards(episode, rewards) -> None:
    for index, values in rewards.items():
        if type(index) is not int or not 0 <= index < len(episode.turns):
            raise AgentContractError("Process reward refers to an unknown turn.")
        if len(values) != len(episode.turns[index].generation.tokens):
            raise AgentContractError(
                "Process rewards must align with original output tokens."
            )


@dataclass(frozen=True, slots=True)
class RewardTerm:
    name: str
    scorer: EpisodeScorer
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or "/" in self.name:
            raise ValueError("Reward term name must be nonempty and contain no slash.")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
        ):
            raise ValueError("Reward weight must be finite.")


class CompositeScorer:
    """Run outcome and auxiliary scorers in order, preserving outcome success.

    Scorers must be pure/idempotent: transient failures can cause the runner to
    retry the whole scoring operation. No partial reward is committed on error.
    """

    def __init__(self, outcome: EpisodeScorer, terms: Sequence[RewardTerm] = ()):
        self.outcome, self.terms = outcome, tuple(terms)
        names = [term.name for term in terms]
        if len(names) != len(set(names)) or "outcome" in names:
            raise ValueError("Reward term names must be unique and cannot be outcome.")

    async def score(self, row, episode) -> EpisodeScore:
        base = await self.outcome.score(row, episode)
        base.__post_init__()
        validate_process_rewards(episode, base.process_rewards)
        reward = base.reward
        process = {i: list(values) for i, values in base.process_rewards.items()}
        assignments = list(base.assignments)
        components = {
            f"outcome/{name}": value for name, value in base.components.items()
        }
        components["outcome/reward"] = base.reward
        for term in self.terms:
            score = await term.scorer.score(row, episode)
            score.__post_init__()
            validate_process_rewards(episode, score.process_rewards)
            reward += term.weight * score.reward
            for i, values in score.process_rewards.items():
                target = process.setdefault(i, [0.0] * len(values))
                for j, value in enumerate(values):
                    target[j] += term.weight * value
            components.update(
                {
                    f"{term.name}/{name}": value
                    for name, value in score.components.items()
                }
            )
            components[f"{term.name}/reward"] = term.weight * (
                score.reward + sum(map(sum, score.process_rewards.values()))
            )
            # Keep the source and target semantics supplied by the scorer while
            # applying the composite term's numeric weight exactly once.
            assignments.extend(
                RewardAssignment(
                    target=item.target,
                    value=float(item.value) * term.weight,
                    mode=item.mode,
                    aggregation=item.aggregation,
                    discount=item.discount,
                    source=f"{term.name}/{item.source}",
                    position=item.position,
                )
                for item in score.assignments
            )
        return EpisodeScore(
            reward,
            base.success,
            components,
            {i: tuple(v) for i, v in process.items()},
            tuple(assignments),
        )


@dataclass(frozen=True, slots=True)
class CostConfig:
    tool_call: float = 0.0
    invalid_call: float = 0.0
    generated_token: float = 0.0
    format_reward: float = 0.0
    turn_discount: float = 1.0

    def __post_init__(self):
        for name in (
            "tool_call",
            "invalid_call",
            "generated_token",
            "format_reward",
            "turn_discount",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0 < self.turn_discount <= 1:
            raise ValueError("turn_discount must be in (0, 1].")


class ToolCostScorer:
    """Charge requested calls (even when execution failed), errors and tokens.

    Costs land on the action that incurred them. format_reward is awarded once
    on a completed final action. discount weights auxiliary turn rewards by
    discount**turn; PPO's actor.discount controls credit propagation separately.
    """

    def __init__(self, config: CostConfig | None = None):
        self.config = config or CostConfig()

    async def score(self, row, episode) -> EpisodeScore:
        config = self.config
        process, calls, errors = {}, 0, 0
        for i, turn in enumerate(episode.turns):
            n_calls = len(turn.action.tool_calls) if turn.action else 0
            n_errors = sum(r.is_error for r in turn.tool_results)
            calls += n_calls
            errors += n_errors
            values = [-config.generated_token] * len(turn.generation.tokens)
            values[-1] -= config.tool_call * n_calls + config.invalid_call * n_errors
            if i == len(episode.turns) - 1 and episode.status == "completed":
                values[-1] += config.format_reward
            process[i] = tuple(value * config.turn_discount**i for value in values)
        return EpisodeScore(
            0,
            False,
            {
                "requested_calls": float(calls),
                "invalid_calls": float(errors),
                "cost_penalty": float(sum(map(sum, process.values()))),
            },
            process,
        )


class TurnRewardScorer:
    """Adapt sync/async rule scorers returning {turn_index: scalar_reward}.

    A token PRM can instead implement EpisodeScorer directly and return
    EpisodeScore.process_rewards. CompositeScorer performs the same validation.
    """

    def __init__(self, function: Callable, *, discount: float = 1.0):
        if (
            isinstance(discount, bool)
            or not isinstance(discount, (int, float))
            or not 0 < discount <= 1
        ):
            raise ValueError("discount must be in (0, 1].")
        self.function, self.discount = function, discount

    async def score(self, row, episode) -> EpisodeScore:
        if inspect.iscoroutinefunction(self.function):
            values = await self.function(row, episode)
        else:
            values = await asyncio.to_thread(self.function, row, episode)
            if inspect.isawaitable(values):
                values = await values
        if not isinstance(values, Mapping):
            raise AgentContractError("Turn scorer must return a mapping.")
        rewards = {}
        for index, value in values.items():
            if type(index) is not int or not 0 <= index < len(episode.turns):
                raise AgentContractError("Turn reward refers to an unknown turn.")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise AgentContractError("Turn rewards must be finite scalars.")
            tokens = [0.0] * len(episode.turns[index].generation.tokens)
            tokens[-1] = value * self.discount**index
            rewards[index] = tuple(tokens)
        return EpisodeScore(0, False, process_rewards=rewards)
