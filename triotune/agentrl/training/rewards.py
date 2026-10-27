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
        submission_id: str | None = None,
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
    A trajectory advantage is broadcast to every active model-output token;
    it bypasses reward shaping and GAE and never changes critic returns.
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
        if (
            self.target in {"interaction", "turn"}
            and self.position is not None
            and (type(self.position) is not int or self.position < 0)
        ):
            raise ValueError(
                "Interaction and turn positions must be nonnegative integers."
            )
        if (
            self.target == "token"
            and self.position is not None
            and (
                not isinstance(self.position, tuple)
                or len(self.position) != 2
                or any(type(value) is not int or value < 0 for value in self.position)
            )
        ):
            raise ValueError("Token positions must be (turn, token) integer pairs.")


def _aggregate_assignment_data(assignments, rollout_ids=None):
    """Reduce signals by scope/source/target without losing target positions."""
    groups: dict[tuple, list[RewardAssignment]] = {}
    scopes = [None] * len(assignments) if rollout_ids is None else rollout_ids
    for scope, assignment in zip(scopes, assignments, strict=True):
        if not isinstance(assignment, RewardAssignment):
            raise TypeError("Expected RewardAssignment objects.")
        assignment.__post_init__()
        key = (
            scope,
            assignment.mode,
            assignment.target,
            assignment.source,
            assignment.position,
        )
        if key in groups and groups[key][0].aggregation != assignment.aggregation:
            raise AgentContractError(
                "Assignments for one source and target must share aggregation."
            )
        groups.setdefault(key, []).append(assignment)

    outcome = 0.0
    process, direct = {}, {}
    for (_scope, mode, target, _source, position), values in groups.items():
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
        elif position is None:
            raise AgentContractError(
                f"{target.title()} assignments require a position."
            )
        destination = direct if mode == "advantage" else process
        key = (target, position)
        destination[key] = destination.get(key, 0.0) + value
    if not all(
        math.isfinite(value) for value in (outcome, *process.values(), *direct.values())
    ):
        raise AgentContractError("Aggregated rewards and advantages must be finite.")
    return outcome, process, direct


def _project_score(score, turns):
    """Project one score onto original output tokens for both export and trace.

    Interaction/turn scalars belong on the last generated token, token signals
    retain their exact offsets, and trajectory advantages broadcast over model
    output only. Return the outcome separately: PPO places it after candidate
    normalization, while trace attaches it to the last generated interaction.
    Neither consumer may treat a direct advantage as a critic reward.
    """
    outcome, process, direct = _aggregate_assignment_data(score.assignments)
    if any(index >= len(turns) for index in score.process_rewards):
        raise AgentContractError("Process rewards refer to an unknown turn.")
    for target, position in (*process, *direct):
        if target == "trajectory":
            continue
        index, offset = position if target == "token" else (position, None)
        if index >= len(turns) or (
            offset is not None and offset >= len(turns[index].generation.tokens)
        ):
            raise AgentContractError(
                "Reward assignment refers to an unknown output token or turn."
            )

    rewards, advantages = [], []
    for index, turn in enumerate(turns):
        count = len(turn.generation.tokens)
        row = list(score.process_rewards.get(index, (0.0,) * count))
        if len(row) != count:
            raise AgentContractError(
                "Process rewards must align with original output tokens."
            )
        row[-1] += sum(
            process.get((target, index), 0.0) for target in ("turn", "interaction")
        )
        advantage = [direct.get(("trajectory", None), 0.0)] * count
        advantage[-1] += sum(
            direct.get((target, index), 0.0) for target in ("turn", "interaction")
        )
        for offset in range(count):
            row[offset] += process.get(("token", (index, offset)), 0.0)
            advantage[offset] += direct.get(("token", (index, offset)), 0.0)
        rewards.append(tuple(row))
        advantages.append(tuple(advantage))
    outcome += score.reward
    # Validate before a ledger or event list is changed. A finite submission
    # can still overflow when combined with another source or legacy vector.
    if not all(
        math.isfinite(value)
        for value in (
            outcome,
            *(value for row in rewards + advantages for value in row),
            *(sum(row) for row in rewards),
            outcome + sum(rewards[-1]) if rewards else outcome,
        )
    ):
        raise AgentContractError("Projected rewards and advantages must be finite.")
    return outcome, rewards, advantages


def _assignment_vectors(outcome, process, direct):
    """Keep legacy per-turn vectors and expose trajectory advantage at None."""

    def vectors(source):
        turns = {}
        for (target, position), value in source.items():
            if target == "trajectory":
                turns[None] = {0: value}
                continue
            turn, token = position if target == "token" else (position, -1)
            row = turns.setdefault(turn, {})
            row[token] = row.get(token, 0.0) + value
        return {
            turn: tuple(values[index] for index in sorted(values))
            for turn, values in turns.items()
        }

    return outcome, vectors(process), vectors(direct)


def aggregate_assignments(
    assignments: Sequence[RewardAssignment],
) -> tuple[float, dict[int, tuple[float, ...]], dict[int | None, tuple[float, ...]]]:
    """Fold contributions into outcome, process and direct-advantage vectors.

    Group by source, mode, target and position before applying sum/mean/last.
    This function does not deduplicate submissions; RewardLedger owns that
    boundary. A trajectory advantage is returned as ``direct[None] = (value,)``
    because it has no turn position. The exporter uses the same reduction with
    exact target positions to broadcast that scalar only onto model actions.
    """
    return _assignment_vectors(*_aggregate_assignment_data(assignments))


class RewardLedger:
    """Idempotent reward submission buffer with independent rollout scopes.

    Explicit submission IDs identify whole atomic payloads within a rollout.
    Exact retries are ignored; conflicting payloads raise before any mutation.
    Different IDs can contribute to the same signal's sum/mean/last aggregation.
    Without an ID, legacy rollout/source/mode deduplication applies to whole
    trajectory bundles, never to individual values inside a submit call.
    Process signals without explicit submission IDs remain additive.
    """

    def __init__(self):
        self._assignments: list[RewardAssignment] = []
        self._rollout_ids: list[str | None] = []
        self._trajectory_submissions: dict[tuple, tuple[RewardAssignment, ...]] = {}
        self._submissions: dict[tuple, tuple[RewardAssignment, ...]] = {}
        self._aggregations: dict[tuple, str] = {}

    def submit(
        self,
        assignments: RewardAssignment | Sequence[RewardAssignment],
        *,
        logical_rollout_id: str | None = None,
        submission_id: str | None = None,
    ) -> tuple[RewardAssignment, ...]:
        """Commit a validated payload, returning only newly accepted contributions."""
        for name, identity in (
            ("logical_rollout_id", logical_rollout_id),
            ("submission_id", submission_id),
        ):
            if identity is not None and (
                not isinstance(identity, str) or not identity.strip()
            ):
                raise AgentContractError(f"{name} must be a nonempty string or None.")
        values = (
            (assignments,)
            if isinstance(assignments, RewardAssignment)
            else tuple(assignments)
        )
        # Validate the entire payload before claiming an ID or appending values.
        _aggregate_assignment_data(values)
        submission_key = (logical_rollout_id, submission_id)
        if submission_id is not None and submission_key in self._submissions:
            if self._submissions[submission_key] != values:
                raise AgentContractError(
                    "submission_id conflicts with its previous reward payload."
                )
            return ()

        pending_bundles = {}
        if submission_id is None and logical_rollout_id is not None:
            for assignment in values:
                if assignment.target == "trajectory":
                    key = (logical_rollout_id, assignment.source, assignment.mode)
                    pending_bundles.setdefault(key, []).append(assignment)
        repeated = set()
        for key, bundle in pending_bundles.items():
            if key in self._trajectory_submissions:
                if self._trajectory_submissions[key] != tuple(bundle):
                    raise AgentContractError(
                        "Trajectory submission conflicts; use distinct submission_id values for new contributions."
                    )
                repeated.add(key)

        accepted, aggregations = [], dict()
        for assignment in values:
            if (
                assignment.target == "trajectory"
                and (logical_rollout_id, assignment.source, assignment.mode) in repeated
            ):
                continue
            key = (
                logical_rollout_id,
                assignment.mode,
                assignment.target,
                assignment.source,
                assignment.position,
            )
            previous = aggregations.get(
                key, self._aggregations.get(key, assignment.aggregation)
            )
            if previous != assignment.aggregation:
                raise AgentContractError(
                    "Assignments for one source and target must share aggregation."
                )
            aggregations[key] = assignment.aggregation
            accepted.append(assignment)

        self._assignments.extend(accepted)
        self._rollout_ids.extend([logical_rollout_id] * len(accepted))
        self._aggregations.update(aggregations)
        if submission_id is not None:
            self._submissions[submission_key] = values
        for key, bundle in pending_bundles.items():
            self._trajectory_submissions[key] = tuple(bundle)
        return tuple(accepted)

    commit = submit

    @property
    def assignments(self) -> tuple[RewardAssignment, ...]:
        return tuple(self._assignments)

    def aggregate(self):
        # Normalize each rollout's aggregation population independently, then
        # sum its signals; one runner can own a ledger across many candidates.
        return _assignment_vectors(
            *_aggregate_assignment_data(self._assignments, self._rollout_ids)
        )


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
