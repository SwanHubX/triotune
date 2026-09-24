"""PyTRIO-native GRPO trainer.

The public shape resembles ``trl.GRPOTrainer`` where it is useful.  Callers
provide a PyTRIO-supported model name, and the trainer owns the service,
training, and sampling clients needed by the GRPO loop.  This module currently
implements group-centered advantages with PyTRIO ``importance_sampling``.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytrio as trio

from ..utils.client import call, call_async, optimizer_params, submit_async, wait, wait_async
from .advantages import group_advantages
from .config import GRPOConfig
from .datum import build_aligned_datum, to_client_datum
from .rewards import RewardEvaluator, RewardFunc
from .rollout import SamplingAdapter
from .types import GRPOBatch, GRPOResult, RolloutResult, SequenceSample, as_prompt_text


@dataclass(slots=True)
class TrainerState:
    """Mutable local state tracked across GRPO training steps."""

    # Number of completed trainer steps.
    global_step: int = 0

    # Version of the behavior policy used to generate the next rollout batch.
    behavior_policy_version: int = 0

    # Metrics collected during the most recently completed trainer step.
    metrics: dict[str, float] = field(default_factory=dict)

    # Metadata returned by the most recently completed rollout.
    metadata: dict[str, Any] | None = None


class GRPOTrainer:
    """Train a PyTRIO-supported text-generation model with GRPO.

    The trainer creates and owns its PyTRIO clients.  ``model_name`` is passed
    to ``ServiceClient.create_lora_training_client`` as the base model.
    """

    def __init__(
        self,
        model_name: str,
        reward_funcs: RewardFunc | Sequence[RewardFunc] | None = None,
        args: GRPOConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
        rollout_func: Callable[..., Any] | None = None,
    ):
        ## Model
        # TRL Model -> pytrio client
        if not isinstance(model_name, str) or not model_name.strip():
            raise TypeError(
                "model_name must be a non-empty PyTRIO model name. For the "
                "supported models, please refer to ServiceClient.get_supported_models()."
            )
        if reward_funcs is None:
            # 这个要保留，最好写一个默认reward奖励函数，怎么撰写，等我检查后续的代码，思考具体逻辑后再说（任务2:添加默认奖励函数）
            raise ValueError("reward_funcs is required for client-side GRPO.")

        self.model_name = model_name.strip()
        self.config = args or GRPOConfig()

        # 训练模型：先只写lora的吧，后续再增加全量的（暂定lora的意思，不要修改该注释）
        self._service_client = trio.ServiceClient()
        # 该训练模型后续会随着训练不断更新
        self._training_client = self._service_client.create_lora_training_client(
            base_model=self.model_name,
            rank=lora_rank,
            seed=self.config.seed,
        )
        # 可能有格式要求？AI写的代码，后续检查格式具体形式（任务1:检查数据集格式）
        self.train_dataset = train_dataset
        self._tokenizer = self._training_client.get_tokenizer()
        if not callable(getattr(self._tokenizer, "encode", None)):
            raise TypeError("PyTRIO TrainingClient tokenizer must provide a callable encode() method.")
        # 多个reward结果按照权重相加，而不是直接全部固定1倍率
        self.reward_evaluator = RewardEvaluator(reward_funcs, self.config.reward_weights)
        self.rollout_func = rollout_func
        self.state = TrainerState()
        self._decode = self._make_decode(self._tokenizer)

    @staticmethod
    def _make_decode(tokenizer: Any) -> Callable[[list[int]], str]:
        # tokenizer.decode
        return lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)

    def _prompt_tokens(self, prompt: Any) -> list[int]:
        # tokenizer.encode
        if isinstance(prompt, (list, tuple)) and all(isinstance(token, int) for token in prompt):
            return list(prompt)
        encoded = self._tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [int(token) for token in encoded]

    def _new_sampling_client(self) -> Any:
        return call(self._training_client, "save_weights_and_get_sampling_client")

    async def _new_sampling_client_async(self) -> Any:
        return await call_async(self._training_client, "save_weights_and_get_sampling_client")

    def _groups(self, rows: Sequence[Mapping[str, Any]]) -> list[GRPOBatch]:
        groups = []
        for index, row in enumerate(rows):
            # 默认通常要求数据格式为：{"prompt": "问题文本"}
            if self.config.prompt_key not in row:
                raise KeyError(f"Dataset row {index} is missing {self.config.prompt_key!r}.")
            prompt = row[self.config.prompt_key]
            if not (isinstance(prompt, (list, tuple)) and all(isinstance(token, int) for token in prompt)):
                prompt = as_prompt_text(prompt) # 规范化文本，可以提取content、messages、prompt包含这些字段的数据
            prompt_tokens = self._prompt_tokens(prompt)  # encode转换文本成tokens
            groups.append(
                GRPOBatch(
                    str(index),
                    prompt,
                    prompt_tokens,
                    [],
                    dict(row),
                    self.state.behavior_policy_version,
                    trio.ModelInput.from_ints(prompt_tokens),
                )
            )
        return groups

    def _sample_groups(self, groups: list[GRPOBatch], client: Any) -> RolloutResult:
        # 同步生成一批 prompt 的 rollout，也就是为每个 prompt 采样多个回答
        adapter = SamplingAdapter(client, params_factory=trio.SamplingParams, decode=self._decode)
        if self.rollout_func is None:
            result = {
                "groups": [adapter.sample(group, self.config) for group in groups],
            }
            return self._normalize_rollout_result(groups, result)
        result = self.rollout_func([group.prompt for group in groups], self)
        if inspect.isawaitable(result):
            raise RuntimeError("rollout_func returned an awaitable; use train_async().")
        return self._normalize_rollout_result(groups, result)

    async def _sample_groups_async(self, groups: list[GRPOBatch], client: Any) -> RolloutResult:
        adapter = SamplingAdapter(client, params_factory=trio.SamplingParams, decode=self._decode)
        if self.rollout_func is None:
            result = {
                "groups": list(
                    await asyncio.gather(
                        *(adapter.sample_async(group, self.config) for group in groups)
                    )
                ),
            }
        else:
            result = self.rollout_func([group.prompt for group in groups], self)
            result = await result if inspect.isawaitable(result) else result
        return self._normalize_rollout_result(groups, result)

    @staticmethod
    def _normalize_rollout_result(groups: list[GRPOBatch], result: Any) -> RolloutResult:
        if not isinstance(result, Mapping):
            raise TypeError("rollout_func must return a mapping containing groups.")
        required = {"groups"}
        allowed = required | {"metrics", "metadata"}
        missing = required.difference(result)
        if missing:
            raise ValueError(f"rollout_func result is missing fields: {sorted(missing)}.")
        unexpected = set(result).difference(allowed)
        if unexpected:
            raise ValueError(
                "rollout_func result has unexpected fields: "
                f"{sorted(str(key) for key in unexpected)}."
            )

        rollout_groups = result["groups"]
        if not isinstance(rollout_groups, Sequence) or isinstance(rollout_groups, (str, bytes)):
            raise TypeError("rollout_func result['groups'] must be a sequence of GRPOBatch objects.")
        rollout_groups = list(rollout_groups)
        if len(rollout_groups) != len(groups):
            raise ValueError("rollout_func must return one group per prompt.")
        if any(not isinstance(group, GRPOBatch) for group in rollout_groups):
            raise TypeError("rollout_func result['groups'] must contain only GRPOBatch objects.")

        normalized: RolloutResult = {"groups": rollout_groups}
        if "metrics" in result:
            raw_metrics = result["metrics"]
            if not isinstance(raw_metrics, Mapping):
                raise TypeError("rollout_func result['metrics'] must be a mapping.")
            metrics: dict[str, float] = {}
            for key, value in raw_metrics.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    raise TypeError("rollout metrics must have string keys and numeric values.")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("rollout metrics must be finite.")
                metrics[key] = number
            normalized["metrics"] = metrics

        if "metadata" in result:
            raw_metadata = result["metadata"]
            if not isinstance(raw_metadata, Mapping):
                raise TypeError("rollout_func result['metadata'] must be a mapping.")
            if any(not isinstance(key, str) for key in raw_metadata):
                raise TypeError("rollout metadata must have string keys.")
            normalized["metadata"] = dict(raw_metadata)
        return normalized

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
                datum = build_aligned_datum(
                    group.prompt_tokens,
                    sample,
                    advantage,
                    prompt_id=group.prompt_id,
                    sample_id=sample_id,
                    model_input_builder=trio.ModelInput.from_ints,
                )
                datums.append(to_client_datum(datum, trio.Datum))
        return datums, skipped, metrics

    def _train_datums(self, datums: list[Any]) -> dict[str, float]:
        if not datums:
            return {"train/skipped": 1.0}
        # Submit both operations before waiting.  This is the PyTRIO clock
        # cycle; waiting for forward_backward first needlessly serializes the
        # service and differs from the local TRL optimizer loop.
        forward_method = getattr(self._training_client, "forward_backward", None)
        optim_method = getattr(self._training_client, "optim_step", None)
        if forward_method is None or optim_method is None:
            raise AttributeError("PyTRIO training_client needs forward_backward() and optim_step().")
        forward_future = forward_method(datums, loss_fn=self.config.loss_fn, loss_fn_config=self.config.loss_fn_config)
        optim_future = optim_method(optimizer_params(self.config, trio.AdamParams))
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
        forward = await submit_async(self._training_client, "forward_backward", datums, loss_fn=self.config.loss_fn, loss_fn_config=self.config.loss_fn_config)
        optim = await submit_async(self._training_client, "optim_step", optimizer_params(self.config, trio.AdamParams))
        forward_result, optim_result = await wait_async(forward), await wait_async(optim)
        metrics: dict[str, float] = {}
        for value in (forward_result, optim_result):
            raw = getattr(value, "metrics", None) if value is not None else None
            if isinstance(raw, Mapping):
                metrics.update({str(key): float(number) for key, number in raw.items() if isinstance(number, (int, float)) and math.isfinite(float(number))})
        return metrics

    def step(self, rows: Sequence[Mapping[str, Any]]) -> GRPOResult:
        groups = self._groups(rows)
        rollout = self._sample_groups(groups, self._new_sampling_client())
        sampled = rollout["groups"]
        reward_groups, reward_metrics = self.reward_evaluator.score(sampled)
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        if "metrics" in rollout:
            metrics.update(rollout["metrics"])
        metrics.update(reward_metrics)
        metrics.update(self._train_datums(datums))
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self.state.metrics = metrics
        self.state.metadata = rollout.get("metadata")
        return GRPOResult(
            self.state.global_step,
            metrics=metrics,
            num_datums=len(datums),
            skipped_groups=skipped,
            metadata=rollout.get("metadata"),
        )

    async def step_async(self, rows: Sequence[Mapping[str, Any]]) -> GRPOResult:
        groups = self._groups(rows)
        rollout = await self._sample_groups_async(groups, await self._new_sampling_client_async())
        sampled = rollout["groups"]
        reward_groups, reward_metrics = await self.reward_evaluator.score_async(sampled)
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        if "metrics" in rollout:
            metrics.update(rollout["metrics"])
        metrics.update(reward_metrics)
        metrics.update(await self._train_datums_async(datums))
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self.state.metrics = metrics
        self.state.metadata = rollout.get("metadata")
        return GRPOResult(
            self.state.global_step,
            metrics=metrics,
            num_datums=len(datums),
            skipped_groups=skipped,
            metadata=rollout.get("metadata"),
        )

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
