"""PyTRIO-native GRPO trainer.

The public shape resembles ``trl.GRPOTrainer`` where it is useful, but the
execution contract is intentionally different: ``training_client`` and
``sampling_client`` are remote clients, not local Hugging Face models.  This
module currently implements only the first GRPO algorithm (group-centered
advantages + PyTRIO ``importance_sampling``).
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..utils.client import call, call_async, optimizer_params, submit_async, wait, wait_async
from .advantages import group_advantages
from .config import GRPOConfig
from .datum import build_aligned_datum, to_client_datum
from .rewards import RewardEvaluator, RewardFunc
from .rollout import SamplingAdapter
from .types import GRPOBatch, GRPOResult, SequenceSample, as_prompt_text


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    behavior_policy_version: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class GRPOTrainer:
    """Train a text-generation policy through public PyTRIO client methods.

    ``model`` is accepted as a compatibility alias only when it is already a
    training client.  A string/path or a local Transformers model is rejected:
    loading and optimizer ownership belong to ``ServiceClient``/PyTRIO.
    """

    def __init__(
        self,
        training_client: Any = None,
        reward_funcs: RewardFunc | Sequence[RewardFunc] | None = None,
        args: GRPOConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        model: Any = None,
        sampling_client: Any = None,
        processing_class: Any = None,
        prompt_tokenizer: Any = None,
        prompt_builder: Callable[[Any], Any] | None = None,
        model_input_builder: Callable[[list[int]], Any] | None = None,
        datum_factory: Callable[..., Any] | None = None,
        sampling_params_factory: Any = None,
        optimizer_params_factory: Any = None,
        rollout_func: Callable[..., Any] | None = None,
    ):
        if training_client is None:
            training_client = model
        if training_client is None:
            raise TypeError("Pass a PyTRIO training_client; local model loading is not supported.")
        if isinstance(training_client, (str, bytes)):
            raise TypeError("GRPOTrainer expects a PyTRIO client, not a model name/path.")
        if processing_class is not None and not callable(getattr(processing_class, "encode", None)):
            # TRL's ProcessorMixin can carry text and image preprocessing.  A
            # PyTRIO text-only trainer ignores the image processor and keeps
            # only its tokenizer when one is available.
            processing_class = getattr(processing_class, "tokenizer", None)
            if processing_class is not None and not callable(getattr(processing_class, "encode", None)):
                processing_class = None
        if reward_funcs is None:
            raise ValueError("reward_funcs is required for client-side GRPO.")
        self.training_client = training_client
        self.config = args or GRPOConfig()
        self.train_dataset = train_dataset
        self.prompt_tokenizer = prompt_tokenizer or processing_class
        self.prompt_builder = prompt_builder
        self.model_input_builder = model_input_builder
        self.datum_factory = datum_factory
        self.optimizer_params_factory = optimizer_params_factory
        self.reward_evaluator = RewardEvaluator(reward_funcs, self.config.reward_weights)
        self.rollout_func = rollout_func
        self.state = TrainerState()
        self._sampling_client = sampling_client
        self._sampling_params_factory = sampling_params_factory
        self._decode = self._make_decode(self.prompt_tokenizer)

    @staticmethod
    def _make_decode(tokenizer: Any) -> Callable[[list[int]], str] | None:
        decode = getattr(tokenizer, "decode", None)
        return decode if callable(decode) else None

    def _prompt_tokens(self, prompt: Any) -> list[int]:
        if isinstance(prompt, (list, tuple)) and all(isinstance(token, int) for token in prompt):
            return list(prompt)
        if self.prompt_tokenizer is None:
            raise TypeError("Provide prompt_tokenizer or tokenized prompts for PyTRIO Datum alignment.")
        encoded = self.prompt_tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [int(token) for token in encoded]

    def _prompt_input(self, prompt: Any) -> Any:
        return self.prompt_builder(prompt) if self.prompt_builder else prompt

    def _new_sampling_client(self) -> Any:
        if self._sampling_client is not None:
            return self._sampling_client
        return call(self.training_client, "save_weights_and_get_sampling_client")

    async def _new_sampling_client_async(self) -> Any:
        if self._sampling_client is not None:
            return self._sampling_client
        return await call_async(self.training_client, "save_weights_and_get_sampling_client")

    def _groups(self, rows: Sequence[Mapping[str, Any]]) -> list[GRPOBatch]:
        groups = []
        for index, row in enumerate(rows):
            if self.config.prompt_key not in row:
                raise KeyError(f"Dataset row {index} is missing {self.config.prompt_key!r}.")
            prompt = row[self.config.prompt_key]
            if not (isinstance(prompt, (list, tuple)) and all(isinstance(token, int) for token in prompt)):
                prompt = as_prompt_text(prompt)
            groups.append(
                GRPOBatch(
                    str(index),
                    prompt,
                    self._prompt_tokens(prompt),
                    [],
                    dict(row),
                    self.state.behavior_policy_version,
                    self._prompt_input(prompt),
                )
            )
        return groups

    def _sample_groups(self, groups: list[GRPOBatch], client: Any) -> list[GRPOBatch]:
        adapter = SamplingAdapter(client, params_factory=self._sampling_params_factory, decode=self._decode)
        if self.rollout_func is None:
            return [adapter.sample(group, self.config) for group in groups]
        result = self.rollout_func([group.prompt for group in groups], self)
        if inspect.isawaitable(result):
            raise RuntimeError("rollout_func returned an awaitable; use train_async().")
        return self._replace_rollouts(groups, result)

    async def _sample_groups_async(self, groups: list[GRPOBatch], client: Any) -> list[GRPOBatch]:
        adapter = SamplingAdapter(client, params_factory=self._sampling_params_factory, decode=self._decode)
        if self.rollout_func is None:
            return list(
                await asyncio.gather(
                    *(adapter.sample_async(group, self.config) for group in groups)
                )
            )
        result = self.rollout_func([group.prompt for group in groups], self)
        result = await result if inspect.isawaitable(result) else result
        return self._replace_rollouts(groups, result)

    @staticmethod
    def _replace_rollouts(groups: list[GRPOBatch], result: Any) -> list[GRPOBatch]:
        if isinstance(result, Mapping) and "groups" in result:
            result = result["groups"]
        if len(result) != len(groups):
            raise ValueError("rollout_func must return one group per prompt.")
        return list(result)

    def _build_datums(self, groups: Sequence[GRPOBatch], reward_groups: Sequence[Sequence[float]]) -> tuple[list[Any], int, dict[str, float]]:
        datums: list[Any] = []
        skipped = 0
        metrics: dict[str, float] = {}
        batch_denominator = 1.0
        if self.config.scale_rewards == "batch":
            all_rewards = [float(value) for rewards in reward_groups for value in rewards]
            mean = sum(all_rewards) / max(1, len(all_rewards))
            variance = sum((value - mean) ** 2 for value in all_rewards) / max(1, len(all_rewards))
            batch_denominator = math.sqrt(variance) + self.config.reward_epsilon
        for group, rewards in zip(groups, reward_groups):
            advantages, group_metrics = group_advantages(
                rewards,
                scale="none" if self.config.scale_rewards == "batch" else self.config.scale_rewards,
                epsilon=self.config.reward_epsilon,
            )
            if self.config.scale_rewards == "batch":
                advantages = [value / batch_denominator for value in advantages]
            metrics.update({f"{group.prompt_id}/{key}": value for key, value in group_metrics.items()})
            if self.config.skip_zero_advantage and all(value == 0.0 for value in advantages):
                skipped += 1
                continue
            for sample_id, (sample, advantage) in enumerate(zip(group.samples, advantages)):
                datum = build_aligned_datum(group.prompt_tokens, sample, advantage, prompt_id=group.prompt_id, sample_id=sample_id, model_input_builder=self.model_input_builder)
                datums.append(to_client_datum(datum, self.datum_factory))
        return datums, skipped, metrics

    def _train_datums(self, datums: list[Any]) -> dict[str, float]:
        if not datums:
            return {"train/skipped": 1.0}
        # Submit both operations before waiting.  This is the PyTRIO clock
        # cycle; waiting for forward_backward first needlessly serializes the
        # service and differs from the local TRL optimizer loop.
        forward_method = getattr(self.training_client, "forward_backward", None)
        optim_method = getattr(self.training_client, "optim_step", None)
        if forward_method is None or optim_method is None:
            raise AttributeError("PyTRIO training_client needs forward_backward() and optim_step().")
        forward_future = forward_method(datums, loss_fn=self.config.loss_fn, loss_fn_config=self.config.loss_fn_config)
        optim_future = optim_method(optimizer_params(self.config, self.optimizer_params_factory))
        forward, optim = wait(forward_future), wait(optim_future)
        metrics: dict[str, float] = {}
        for value in (forward, optim):
            raw = getattr(value, "metrics", None) if value is not None else None
            if isinstance(raw, Mapping):
                metrics.update({str(key): float(number) for key, number in raw.items() if isinstance(number, (int, float)) and math.isfinite(float(number))})
        return metrics

    async def _train_datums_async(self, datums: list[Any]) -> dict[str, float]:
        if not datums:
            return {"train/skipped": 1.0}
        forward = await submit_async(self.training_client, "forward_backward", datums, loss_fn=self.config.loss_fn, loss_fn_config=self.config.loss_fn_config)
        optim = await submit_async(self.training_client, "optim_step", optimizer_params(self.config, self.optimizer_params_factory))
        forward_result, optim_result = await wait_async(forward), await wait_async(optim)
        metrics: dict[str, float] = {}
        for value in (forward_result, optim_result):
            raw = getattr(value, "metrics", None) if value is not None else None
            if isinstance(raw, Mapping):
                metrics.update({str(key): float(number) for key, number in raw.items() if isinstance(number, (int, float)) and math.isfinite(float(number))})
        return metrics

    def step(self, rows: Sequence[Mapping[str, Any]]) -> GRPOResult:
        groups = self._groups(rows)
        sampled = self._sample_groups(groups, self._new_sampling_client())
        reward_groups, reward_metrics = self.reward_evaluator.score(sampled)
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        metrics.update(reward_metrics)
        metrics.update(self._train_datums(datums))
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self.state.metrics = metrics
        self._sampling_client = None  # Force a fresh sampler snapshot after each update.
        return GRPOResult(self.state.global_step, metrics=metrics, num_datums=len(datums), skipped_groups=skipped)

    async def step_async(self, rows: Sequence[Mapping[str, Any]]) -> GRPOResult:
        groups = self._groups(rows)
        sampled = await self._sample_groups_async(groups, await self._new_sampling_client_async())
        reward_groups, reward_metrics = await self.reward_evaluator.score_async(sampled)
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        metrics.update(reward_metrics)
        metrics.update(await self._train_datums_async(datums))
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self.state.metrics = metrics
        self._sampling_client = None
        return GRPOResult(self.state.global_step, metrics=metrics, num_datums=len(datums), skipped_groups=skipped)

    def train(self) -> TrainerState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        iterator = iter(self.train_dataset)
        while self.config.max_steps is None or self.state.global_step < self.config.max_steps:
            rows = list(itertools.islice(iterator, self.config.batch_size))
            if not rows:
                break
            self.step(rows)
        return self.state

    async def train_async(self) -> TrainerState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train_async().")
        iterator = iter(self.train_dataset)
        while self.config.max_steps is None or self.state.global_step < self.config.max_steps:
            rows = list(itertools.islice(iterator, self.config.batch_size))
            if not rows:
                break
            await self.step_async(rows)
        return self.state
