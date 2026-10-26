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

from ..types import AgentContractError, EpisodeScore, EpisodeScorer


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
        return EpisodeScore(
            reward, base.success, components, {i: tuple(v) for i, v in process.items()}
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
            {"requested_calls": float(calls), "invalid_calls": float(errors)},
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
