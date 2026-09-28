"""Reward function adapters compatible with the useful part of TRL's API."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any

from .prompts import as_prompt_text
from .types import GRPOBatch

RewardFunc = Callable[..., Sequence[float | None]]


def _reward_name(function: RewardFunc, index: int) -> str:
    """Return a stable metric name chosen by the reward author."""

    name = getattr(function, "reward_name", None) or getattr(function, "__name__", None)
    if not isinstance(name, str) or not name.strip() or name == "<lambda>":
        wrapped = getattr(function, "func", None)
        name = getattr(wrapped, "__name__", None)
    if not isinstance(name, str) or not name.strip() or name == "<lambda>":
        name = type(function).__name__
    name = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in name
    )
    return name.strip("_") or f"reward_{index}"


def _reward_names(functions: Sequence[RewardFunc]) -> list[str]:
    names: list[str] = []
    counts: dict[str, int] = {}
    for index, function in enumerate(functions):
        base_name = _reward_name(function, index)
        counts[base_name] = counts.get(base_name, 0) + 1
        occurrence = counts[base_name]
        names.append(base_name if occurrence == 1 else f"{base_name}_{occurrence}")
    return names


async def _call_reward(function: RewardFunc, kwargs: dict[str, Any]) -> Sequence[float | None]:
    result = function(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class RewardEvaluator:
    """Evaluate and aggregate custom rewards without loading local reward models.

    TRL can load a reward model by string.  That path is intentionally omitted:
    PyTRIO has no local ``AutoModelForSequenceClassification`` contract.  Use a
    callable backed by a remote service when a model-based reward is needed.
    """

    def __init__(
        self,
        reward_funcs: RewardFunc | Sequence[RewardFunc],
        weights: Sequence[float] | None = None,
        *,
        prompt_key: str = "messages",
    ) -> None:
        self.functions = [reward_funcs] if callable(reward_funcs) else list(reward_funcs)
        if not self.functions:
            raise ValueError("At least one reward function is required.")
        self.function_names = _reward_names(self.functions)
        if not isinstance(prompt_key, str) or not prompt_key.strip():
            raise ValueError("prompt_key must be a non-empty string.")
        self.prompt_key = prompt_key
        if weights is None:
            self.weights = [1.0] * len(self.functions)
        else:
            self.weights = [float(weight) for weight in weights]
            if len(self.weights) != len(self.functions):
                raise ValueError("reward_weights must match reward_funcs.")
        if any(not math.isfinite(weight) for weight in self.weights):
            raise ValueError("reward weights must be finite.")

    def _kwargs(self, groups: Sequence[GRPOBatch]) -> dict[str, Any]:
        prompts = [as_prompt_text(group.prompt) for group in groups for _ in group.samples]
        completions = [
            sample.text if sample.text is not None else sample.tokens
            for group in groups
            for sample in group.samples
        ]
        completion_ids = [sample.tokens for group in groups for sample in group.samples]
        columns: dict[str, list[Any]] = {}
        for group in groups:
            for key, value in group.row.items():
                if key != self.prompt_key:
                    columns.setdefault(key, []).extend([value] * len(group.samples))
        columns.update(
            {
                "prompts": prompts,
                "completions": completions,
                "completion_ids": completion_ids,
            }
        )
        return columns

    async def score_async(self, groups: Sequence[GRPOBatch]) -> tuple[list[list[float]], dict[str, float]]:
        kwargs = self._kwargs(groups)
        outputs = await asyncio.gather(*(_call_reward(function, kwargs) for function in self.functions))
        expected = sum(len(group.samples) for group in groups)
        per_function: list[list[float | None]] = []
        for index, output in enumerate(outputs):
            if len(output) != expected:
                raise ValueError(f"Reward function {index} returned {len(output)} values; expected {expected}.")
            per_function.append([None if value is None else float(value) for value in output])
        aggregate: list[float] = []
        for row in zip(*per_function):
            applicable = [weight * value for weight, value in zip(self.weights, row) if value is not None]
            if not applicable:
                raise ValueError("Every reward function returned None for one completion.")
            if any(not math.isfinite(value) for value in applicable):
                raise ValueError("Reward functions must return finite values.")
            aggregate.append(sum(applicable))
        metrics = {
            f"reward/{name}/mean": (
                sum(value for value in output if value is not None)
                / max(1, sum(value is not None for value in output))
            )
            for name, output in zip(self.function_names, per_function, strict=True)
        }
        return _split(aggregate, groups), metrics

    def score(self, groups: Sequence[GRPOBatch]) -> tuple[list[list[float]], dict[str, float]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.score_async(groups))
        raise RuntimeError("score() cannot run inside an active event loop; use score_async().")


def _split(values: list[float], groups: Sequence[GRPOBatch]) -> list[list[float]]:
    result: list[list[float]] = []
    cursor = 0
    for group in groups:
        result.append(values[cursor : cursor + len(group.samples)])
        cursor += len(group.samples)
    return result
