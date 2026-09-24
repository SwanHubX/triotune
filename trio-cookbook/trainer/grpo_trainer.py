"""PyTRIO-native GRPO trainer.

The public shape resembles ``trl.GRPOTrainer`` where it is useful.  Callers
provide a PyTRIO-supported model name, and the trainer owns the service,
training, and sampling clients needed by the GRPO loop.  This module currently
implements group-centered advantages with PyTRIO ``importance_sampling``.
"""

from __future__ import annotations

import asyncio
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pytrio as trio
import swanlab

from ..utils.client import call, call_async, optimizer_params, submit_async, wait, wait_async
from .advantages import group_advantages
from .config import GRPOConfig
from .datum import build_aligned_datum, to_client_datum
from .rewards import RewardEvaluator, RewardFunc
from .rollout import SamplingAdapter
from .prompts import render_prompt
from .types import GRPOBatch, GRPOResult


@dataclass(slots=True)
class TrainerState:
    """Mutable local state tracked across GRPO training steps."""
    # Number of completed trainer steps.
    global_step: int = 0
    # Version of the behavior policy used to generate the next rollout batch.
    behavior_policy_version: int = 0
    # Metrics collected during the most recently completed trainer step.
    metrics: dict[str, float] = field(default_factory=dict)
    # URI of the most recently saved weights and optimizer state.
    checkpoint_path: str | None = None


class GRPOTrainer:
    """Train a PyTRIO-supported text-generation model with GRPO.

    The trainer creates and owns its PyTRIO clients.  ``model_name`` is passed
    to ``ServiceClient.create_lora_training_client`` as the base model. Dataset
    prompts may be plain text, pre-tokenized integer IDs, or text-only chat
    messages. Chat messages are rendered with the tokenizer chat template.
    """

    def __init__(
        self,
        model_name: str,
        reward_funcs: RewardFunc | Sequence[RewardFunc] | None = None,
        args: GRPOConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
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
        self.reward_evaluator = RewardEvaluator(
            reward_funcs,
            self.config.reward_weights,
            prompt_key=self.config.prompt_key,
        )
        self.state = TrainerState()
        self._decode = self._make_decode(self._tokenizer)
        self._swanlab_run = self._init_swanlab_run(lora_rank)

    def _init_swanlab_run(self, lora_rank: int) -> Any | None:
        if self.config.mode == "disabled":
            return None
        model_slug = self.model_name.rsplit("/", 1)[-1].replace("/", "-")
        return swanlab.init(
            mode=self.config.mode,
            project=self.config.project,
            name=self.config.run_name or f"{model_slug}-grpo",
            log_dir=self.config.log_dir,
            config={
                "model_name": self.model_name,
                "lora_rank": lora_rank,
                **asdict(self.config),
            },
        )

    def _log_swanlab(self, result: GRPOResult) -> None:
        if self._swanlab_run is None:
            return
        payload: dict[str, float] = {
            str(key): float(value)
            for key, value in result.metrics.items()
            if math.isfinite(float(value))
        }
        payload.update(
            {
                "trainer/step": float(result.step),
                "trainer/num_datums": float(result.num_datums),
                "trainer/skipped_groups": float(result.skipped_groups),
                "checkpoint/saved": float(result.checkpoint_path is not None),
            }
        )
        self._swanlab_run.log(payload, step=result.step)

    def _finish_swanlab(self, state: str = "success", error: str | None = None) -> None:
        if self._swanlab_run is None:
            return
        self._swanlab_run.finish(state=state, error=error)
        self._swanlab_run = None

    @staticmethod
    def _make_decode(tokenizer: Any) -> Callable[[list[int]], str]:
        # tokenizer.decode
        return lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)

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
            rendered = render_prompt(
                row[self.config.prompt_key],
                self._tokenizer,
                add_generation_prompt=self.config.add_generation_prompt,
                chat_template_kwargs=self.config.chat_template_kwargs,
            )
            groups.append(
                GRPOBatch(
                    str(index),
                    rendered.text,
                    rendered.tokens,
                    [],
                    dict(row),
                    self.state.behavior_policy_version,
                    trio.ModelInput.from_ints(rendered.tokens),
                )
            )
        return groups

    def _sample_groups(self, groups: list[GRPOBatch], client: Any) -> list[GRPOBatch]:
        # 同步生成一批 prompt 的 rollout，也就是为每个 prompt 采样多个回答
        adapter = SamplingAdapter(client, params_factory=trio.SamplingParams, decode=self._decode)
        return [adapter.sample(group, self.config) for group in groups]

    async def _sample_groups_async(self, groups: list[GRPOBatch], client: Any) -> list[GRPOBatch]:
        adapter = SamplingAdapter(client, params_factory=trio.SamplingParams, decode=self._decode)
        return list(
            await asyncio.gather(
                *(adapter.sample_async(group, self.config) for group in groups)
            )
        )

    def _build_datums(self, groups: Sequence[GRPOBatch], reward_groups: Sequence[Sequence[float]]) -> tuple[list[Any], int, dict[str, float]]:
        # 将输入转换成Datum格式
        datums: list[Any] = []
        skipped = 0
        metrics: dict[str, float] = {}
        batch_denominator = 1.0
        all_rewards = [float(value) for rewards in reward_groups for value in rewards]
        if all_rewards:
            mean = sum(all_rewards) / max(1, len(all_rewards))
            variance = sum((value - mean) ** 2 for value in all_rewards) / max(1, len(all_rewards))
            metrics["reward/mean"] = mean
            metrics["reward/std"] = math.sqrt(variance)
            if self.config.scale_rewards == "batch":
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
        # Wait for forward/backward before submitting the optimizer. If the
        # forward request fails, no optimizer request must be sent: otherwise
        # PyTRIO may apply stale or partial gradients from a previous batch.
        forward_method = getattr(self._training_client, "forward_backward", None)
        optim_method = getattr(self._training_client, "optim_step", None)
        if forward_method is None or optim_method is None:
            raise AttributeError("PyTRIO training_client needs forward_backward() and optim_step().")
        forward_future = forward_method(datums, loss_fn=self.config.loss_fn, loss_fn_config=self.config.loss_fn_config)
        forward = wait(forward_future)
        optim = wait(optim_method(optimizer_params(self.config, trio.AdamParams)))
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
        # Do not submit optim_step until forward/backward has completed
        # successfully; this keeps the optimizer boundary failure-safe.
        forward_result = await wait_async(forward)
        optim = await submit_async(self._training_client, "optim_step", optimizer_params(self.config, trio.AdamParams))
        optim_result = await wait_async(optim)
        metrics: dict[str, float] = {}
        for value in (forward_result, optim_result):
            raw = getattr(value, "metrics", None) if value is not None else None
            if isinstance(raw, Mapping):
                metrics.update({str(key): float(number) for key, number in raw.items() if isinstance(number, (int, float)) and math.isfinite(float(number))})
        return metrics

    def _save_state(self) -> str | None:
        if self.config.save_every == 0 or self.state.global_step % self.config.save_every != 0:
            return None
        result = call(
            self._training_client,
            "save_state",
            name=f"grpo-step-{self.state.global_step:08d}",
        )
        path = getattr(result, "path", None)
        if not isinstance(path, str) or not path:
            raise TypeError("PyTRIO save_state() returned no checkpoint path.")
        self.state.checkpoint_path = path
        return path

    async def _save_state_async(self) -> str | None:
        if self.config.save_every == 0 or self.state.global_step % self.config.save_every != 0:
            return None
        result = await call_async(
            self._training_client,
            "save_state",
            name=f"grpo-step-{self.state.global_step:08d}",
        )
        path = getattr(result, "path", None)
        if not isinstance(path, str) or not path:
            raise TypeError("PyTRIO save_state_async() returned no checkpoint path.")
        self.state.checkpoint_path = path
        return path

    def step(self, rows: Sequence[Mapping[str, Any]]) -> GRPOResult:
        # 执行一次完整的GRPO训练step
        groups = self._groups(rows)
        sampled = self._sample_groups(groups, self._new_sampling_client())
        reward_groups, reward_metrics = self.reward_evaluator.score(sampled)
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        metrics.update(reward_metrics)
        metrics.update(self._train_datums(datums))
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self.state.metrics = metrics
        checkpoint_path = self._save_state()
        result = GRPOResult(
            self.state.global_step,
            metrics=metrics,
            num_datums=len(datums),
            skipped_groups=skipped,
            checkpoint_path=checkpoint_path,
        )
        self._log_swanlab(result)
        return result

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
        checkpoint_path = await self._save_state_async()
        result = GRPOResult(
            self.state.global_step,
            metrics=metrics,
            num_datums=len(datums),
            skipped_groups=skipped,
            checkpoint_path=checkpoint_path,
        )
        self._log_swanlab(result)
        return result

    def train(self) -> TrainerState:
        try:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for train().")
            iterator = iter(self.train_dataset)
            while self.config.max_steps is None or self.state.global_step < self.config.max_steps:
                rows = list(itertools.islice(iterator, self.config.batch_size))
                if not rows:
                    break
                self.step(rows)
        except BaseException as error:
            self._finish_swanlab("crashed", str(error))
            raise
        else:
            self._finish_swanlab()
        return self.state

    async def train_async(self) -> TrainerState:
        try:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for train_async().")
            iterator = iter(self.train_dataset)
            while self.config.max_steps is None or self.state.global_step < self.config.max_steps:
                rows = list(itertools.islice(iterator, self.config.batch_size))
                if not rows:
                    break
                await self.step_async(rows)
        except BaseException as error:
            self._finish_swanlab("crashed", str(error))
            raise
        else:
            self._finish_swanlab()
        return self.state
