"""PyTRIO-native GRPO trainer.

Callers provide a PyTRIO-supported model name, and the trainer owns the service,
training, and sampling clients needed by the GRPO loop.  This module currently
implements group-centered advantages with PyTRIO ``importance_sampling``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pytrio as trio
import swanlab
from tqdm import tqdm

from .common.client import (
    call,
    call_async,
    optimizer_params,
    sampling_params,
    submit_async,
    wait,
    wait_async,
)
from .common.checkpoint import save_checkpoint, save_checkpoint_async
from .common.config import GRPOConfig
from .common.logging import accumulate_metrics, averaged_metrics, finish_run, start_run
from .dataset import prepare_prompt
from .grpo.advantages import group_advantages
from .grpo.datum import build_aligned_datum, to_client_datum
from .grpo.metrics import (
    FINAL_SUMMARY_KEYS,
    _is_checkpoint_metric,
    _is_per_prompt_metric,
    _rollout_metrics,
    _sample_train_kl,
    _swanlab_metric_name,
)
from .grpo.rewards import RewardEvaluator, RewardFunc
from .grpo.rollout import SamplingAdapter
from .grpo.types import GRPOBatch, GRPOResult, TrainerState


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
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
    ) -> None:
        """Create the training client, tokenizer, reward adapter and run state."""

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
        self.valid_dataset = valid_dataset

        # 训练模型：先只写lora的吧，后续再增加全量的（暂定lora的意思，不要修改该注释）
        self._service_client = trio.ServiceClient()
        # 该训练模型后续会随着训练不断更新
        self._training_client = self._service_client.create_lora_training_client(
            base_model=self.model_name,
            rank=lora_rank,
            seed=self.config.seed,
        )
        self.train_dataset = train_dataset
        self._tokenizer = self._training_client.get_tokenizer()
        if not callable(getattr(self._tokenizer, "encode", None)):
            raise TypeError(
                "PyTRIO TrainingClient tokenizer must provide a callable encode() method."
            )
        # Multiple rewards are combined with their configured coefficients.
        self.reward_evaluator = RewardEvaluator(
            reward_funcs,
            self.config.reward_weights,
            prompt_key=self.config.prompt_key,
        )
        self.state = TrainerState()
        self._last_eval_step: int | None = None
        self._total_input_tokens = 0
        self._metric_window: dict[str, tuple[float, int]] = {}
        self._decode = self._make_decode(self._tokenizer)
        self._swanlab_run = self._init_swanlab_run(lora_rank)

    # ------------------------------------------------------------------
    # Run reporting and final qualitative samples
    # ------------------------------------------------------------------

    def _init_swanlab_run(self, lora_rank: int) -> Any | None:
        """Create the optional run with model and LoRA metadata."""

        if self.config.report_to is None:
            return None
        model_slug = self.model_name.rsplit("/", 1)[-1].replace("/", "-")
        return start_run(
            self.config,
            default_name=f"{model_slug}-grpo",
            metadata={"model_name": self.model_name, "lora_rank": lora_rank},
        )

    def _log_swanlab(self, result: GRPOResult) -> None:
        """Accumulate batch metrics while omitting prompt-local diagnostics."""

        if self._swanlab_run is None:
            return
        payload: dict[str, float] = {}
        for key, value in result.metrics.items():
            name = str(key)
            if _is_per_prompt_metric(name) or _is_checkpoint_metric(name):
                continue
            number = float(value)
            if math.isfinite(number):
                payload[_swanlab_metric_name(name)] = number
        payload.update(
            {
                "train/num_datums": float(result.num_datums),
                "train/skipped_groups": float(result.skipped_groups),
            }
        )
        if not hasattr(self, "_metric_window"):
            self._metric_window = {}
        accumulate_metrics(self._metric_window, payload)
        if result.step % self.config.logging_step == 0:
            self._flush_swanlab(result.step)

    def _flush_swanlab(self, step: int) -> None:
        """Publish the current metric window and clear its accumulators."""

        if self._swanlab_run is None or not hasattr(self, "_metric_window"):
            return
        payload = averaged_metrics(self._metric_window)
        self._metric_window.clear()
        if payload:
            self._swanlab_run.log({**payload, "train/step": float(step)}, step=step)

    def _log_final_checkpoint(self) -> None:
        """Publish the final checkpoint URI when output logging is enabled."""

        if (
            self.config.output_dir
            and self._swanlab_run is not None
            and self.state.checkpoint_path is not None
        ):
            self._swanlab_run.log_text(
                key="checkpoint/path",
                data=self.state.checkpoint_path,
                step=self.state.global_step,
            )

    def _report_final_state(self) -> None:
        """Write a compact terminal and SwanLab summary of the final state."""

        summary = {
            key: self.state.metrics[key]
            for key in FINAL_SUMMARY_KEYS
            if key in self.state.metrics
        }
        lines = [
            f"Training finished at step {self.state.global_step}",
            "Final metrics: " + json.dumps(summary, sort_keys=True),
        ]
        for line in lines:
            tqdm.write(line)
        if self._swanlab_run is not None:
            self._swanlab_run.log_text(
                key="train/final_summary",
                data="\n".join(lines),
                step=self.state.global_step,
            )

    def _log_training_summary(
        self, started: float, initial_step: int, initial_tokens: int
    ) -> None:
        """Record throughput for work completed by this train invocation."""

        completed_steps = self.state.global_step - initial_step
        if completed_steps == 0:
            return
        elapsed = time.perf_counter() - started
        input_tokens = self._total_input_tokens - initial_tokens
        summary = {
            "time/train_runtime": elapsed,
            "time/train_steps_per_second": completed_steps / elapsed,
            "time/train_tokens_per_second": input_tokens / elapsed,
            "train/total_input_tokens": float(input_tokens),
        }
        self.state.metrics.update(summary)
        if self._swanlab_run is not None:
            self._swanlab_run.log(summary, step=self.state.global_step)

    @staticmethod
    def _sample_text(response: Any, decode: Callable[[list[int]], str]) -> str:
        """Return sampler text, decoding tokens only when text is absent."""

        sequences = response.sequences
        if not sequences:
            raise ValueError("Final inference returned no sequences.")
        sequence = sequences[0]
        return sequence.text if sequence.text is not None else decode(sequence.tokens)

    def _write_final_samples(
        self, groups: Sequence[GRPOBatch], responses: Sequence[Any]
    ) -> None:
        """Decode and publish one qualitative completion per prompt."""

        texts = []
        for group, response in zip(groups, responses, strict=True):
            question = group.row.get("question")
            title = question if isinstance(question, str) and question else group.prompt
            texts.append(
                swanlab.Text(
                    self._sample_text(response, self._decode),
                    caption=title,
                )
            )
        self._swanlab_run.log_text(
            key="sample/text",
            data=texts,
            step=self.state.global_step,
        )

    def _log_final_samples(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Generate a small synchronous sample table from final weights."""

        if self._swanlab_run is None or not rows:
            return
        client = self._new_sampling_client()
        params = sampling_params(
            self.config,
            trio.SamplingParams,
            stop=self._stop_sequences(),
        )
        groups = self._groups(rows[:4])
        responses = [
            call(
                client,
                "sample",
                prompt=group.sampling_prompt,
                num_samples=1,
                sampling_params=params,
            )
            for group in groups
        ]
        self._write_final_samples(groups, responses)

    async def _log_final_samples_async(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Generate final qualitative samples without blocking the event loop."""

        if self._swanlab_run is None or not rows:
            return
        client = await self._new_sampling_client_async()
        params = sampling_params(
            self.config,
            trio.SamplingParams,
            stop=self._stop_sequences(),
        )
        groups = self._groups(rows[:4])
        responses = await asyncio.gather(
            *(
                call_async(
                    client,
                    "sample",
                    prompt=group.sampling_prompt,
                    num_samples=1,
                    sampling_params=params,
                )
                for group in groups
            )
        )
        self._write_final_samples(groups, responses)

    def _finish_swanlab(
        self, state: str = "success", error: str | None = None
    ) -> None:
        """Finish the owned run once and release its handle."""

        if self._swanlab_run is None:
            return
        finish_run(self._swanlab_run, state=state, error=error)
        self._swanlab_run = None

    @staticmethod
    def _make_decode(tokenizer: Any) -> Callable[[list[int]], str]:
        """Capture the tokenizer's standard special-token-skipping decode."""

        return lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Sampling-client lifecycle and rollout preparation
    # ------------------------------------------------------------------

    def _new_sampling_client(self) -> Any:
        """Snapshot current training weights into a fresh behavior policy."""

        return call(self._training_client, "save_weights_and_get_sampling_client")

    async def _new_sampling_client_async(self) -> Any:
        """Asynchronously snapshot current weights into a behavior policy."""

        return await call_async(
            self._training_client,
            "save_weights_and_get_sampling_client",
        )

    def _stop_sequences(self) -> list[str] | None:
        """Resolve model EOS markers without emitting duplicates."""

        if not self.config.stop:
            return None
        return list(
            dict.fromkeys(
                value
                for value in (getattr(self._tokenizer, "eos_token", None), "<|im_end|>")
                if isinstance(value, str) and value
            )
        )

    def _groups(self, rows: Sequence[Mapping[str, Any]]) -> list[GRPOBatch]:
        """Render input rows into prompt groups before remote sampling."""

        groups = []
        for index, row in enumerate(rows):
            # The dataset layer accepts text, token IDs, or chat messages and
            # renders each prompt exactly once for rewards and model sampling.
            rendered = prepare_prompt(
                row,
                self._tokenizer,
                prompt_key=self.config.prompt_key,
                add_generation_prompt=self.config.add_generation_prompt,
                chat_template_kwargs=self.config.chat_template_kwargs,
                row_index=index,
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

    def _sample_groups(
        self, groups: list[GRPOBatch], client: Any
    ) -> list[GRPOBatch]:
        """Sample every group with one immutable behavior-policy client."""

        # 同步生成一批 prompt 的 rollout，也就是为每个 prompt 采样多个回答
        adapter = SamplingAdapter(
            client,
            params_factory=trio.SamplingParams,
            decode=self._decode,
            stop=self._stop_sequences(),
        )
        return [adapter.sample(group, self.config) for group in groups]

    async def _sample_groups_async(
        self, groups: list[GRPOBatch], client: Any
    ) -> list[GRPOBatch]:
        """Asynchronously sample every group with one behavior-policy client."""

        adapter = SamplingAdapter(
            client,
            params_factory=trio.SamplingParams,
            decode=self._decode,
            stop=self._stop_sequences(),
        )
        return list(
            await asyncio.gather(
                *(adapter.sample_async(group, self.config) for group in groups)
            )
        )

    # ------------------------------------------------------------------
    # Advantage alignment and remote policy update
    # ------------------------------------------------------------------

    def _build_datums(
        self,
        groups: Sequence[GRPOBatch],
        reward_groups: Sequence[Sequence[float]],
    ) -> tuple[list[Any], int, dict[str, float]]:
        """Center rewards and align accepted completions to PyTRIO datums."""

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
            metrics.update(
                {
                    f"{group.prompt_id}/{key}": value
                    for key, value in group_metrics.items()
                }
            )
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

    def _training_metrics(
        self,
        datums: Sequence[Any],
        forward_result: Any,
        optimizer_result: Any,
    ) -> dict[str, float]:
        """Normalize backend metrics and estimate sampled train-policy drift."""

        metrics: dict[str, float] = {"optim/lr": self.config.learning_rate}
        for value in (forward_result, optimizer_result):
            raw = getattr(value, "metrics", None) if value is not None else None
            if isinstance(raw, Mapping):
                metrics.update(
                    {
                        str(key): float(number)
                        for key, number in raw.items()
                        if isinstance(number, (int, float)) and math.isfinite(float(number))
                    }
                )

        for loss_key in ("loss_mean", "loss"):
            if loss_key in metrics:
                metrics["train/loss"] = metrics[loss_key]
                break
        for grad_norm_key in ("optim/grad_norm", "grad_norm", "gradient_norm"):
            if grad_norm_key in metrics:
                metrics["optim/grad_norm"] = metrics[grad_norm_key]
                break
        kl = _sample_train_kl(datums, forward_result)
        if kl is not None and math.isfinite(kl):
            metrics["policy/kl_sample_train"] = kl
        return metrics

    def _train_datums(self, datums: list[Any]) -> dict[str, float]:
        """Run failure-safe synchronous forward/backward then optimizer."""

        if not datums:
            return {"optim/lr": self.config.learning_rate, "train/skipped": 1.0}
        # Wait for forward/backward before submitting the optimizer. If the
        # forward request fails, no optimizer request must be sent: otherwise
        # PyTRIO may apply stale or partial gradients from a previous batch.
        forward_method = getattr(self._training_client, "forward_backward", None)
        optim_method = getattr(self._training_client, "optim_step", None)
        if forward_method is None or optim_method is None:
            raise AttributeError(
                "PyTRIO training_client needs forward_backward() and optim_step()."
            )
        forward_started = time.perf_counter()
        forward_future = forward_method(
            datums,
            loss_fn=self.config.loss_fn,
            loss_fn_config=self.config.loss_fn_config,
        )
        forward = wait(forward_future)
        forward_time = time.perf_counter() - forward_started
        optim_started = time.perf_counter()
        optim = wait(optim_method(optimizer_params(self.config, trio.AdamParams)))
        optim_time = time.perf_counter() - optim_started
        metrics = self._training_metrics(datums, forward, optim)
        metrics["time/forward_backward"] = forward_time
        metrics["time/optim"] = optim_time
        return metrics

    async def _train_datums_async(self, datums: list[Any]) -> dict[str, float]:
        """Run failure-safe asynchronous forward/backward then optimizer."""

        if not datums:
            return {"optim/lr": self.config.learning_rate, "train/skipped": 1.0}
        forward_started = time.perf_counter()
        forward = await submit_async(
            self._training_client,
            "forward_backward",
            datums,
            loss_fn=self.config.loss_fn,
            loss_fn_config=self.config.loss_fn_config,
        )
        # Do not submit optim_step until forward/backward has completed
        # successfully; this keeps the optimizer boundary failure-safe.
        forward_result = await wait_async(forward)
        forward_time = time.perf_counter() - forward_started
        optim_started = time.perf_counter()
        optim = await submit_async(
            self._training_client,
            "optim_step",
            optimizer_params(self.config, trio.AdamParams),
        )
        optim_result = await wait_async(optim)
        optim_time = time.perf_counter() - optim_started
        metrics = self._training_metrics(datums, forward_result, optim_result)
        metrics["time/forward_backward"] = forward_time
        metrics["time/optim"] = optim_time
        return metrics

    @staticmethod
    def _progress_postfix(result: GRPOResult) -> dict[str, str]:
        """Select compact scalars for the terminal progress bar."""

        keys = {
            "loss": "train/loss",
            "reward": "reward/mean",
            "entropy": "policy/entropy",
            "kl": "policy/kl_sample_train",
            "grad_norm": "optim/grad_norm",
        }
        return {
            label: f"{result.metrics[metric]:.4f}"
            for label, metric in keys.items()
            if metric in result.metrics
        }

    def _save_state(self, *, final: bool = False) -> str | None:
        """Save at the configured interval or when finalization requests it."""

        if self.config.save_every == 0 or (
            not final and self.state.global_step % self.config.save_every != 0
        ):
            return None
        path = save_checkpoint(self._training_client, "grpo", self.state.global_step)
        self.state.checkpoint_path = path
        return path

    async def _save_state_async(self, *, final: bool = False) -> str | None:
        """Asynchronous counterpart to :meth:`_save_state`."""

        if self.config.save_every == 0 or (
            not final and self.state.global_step % self.config.save_every != 0
        ):
            return None
        path = await save_checkpoint_async(
            self._training_client,
            "grpo",
            self.state.global_step,
        )
        self.state.checkpoint_path = path
        return path

    def step(self, rows: Sequence[Mapping[str, Any]]) -> GRPOResult:
        """Run rollout, reward, advantage alignment and one optimizer update."""

        step_started = time.perf_counter()
        groups = self._groups(rows)
        prompt_prepare_time = time.perf_counter() - step_started

        rollout_started = time.perf_counter()
        sampled = self._sample_groups(groups, self._new_sampling_client())
        rollout_time = time.perf_counter() - rollout_started

        reward_started = time.perf_counter()
        reward_groups, reward_metrics = self.reward_evaluator.score(sampled)
        reward_time = time.perf_counter() - reward_started

        train_started = time.perf_counter()
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        datum_prepare_time = time.perf_counter() - train_started
        input_tokens = sum(datum.model_input.length for datum in datums)
        metrics["train/input_tokens"] = float(input_tokens)
        metrics.update(_rollout_metrics(sampled))
        metrics.update(reward_metrics)
        metrics.update(self._train_datums(datums))
        train_time = time.perf_counter() - train_started
        self.state.global_step += 1
        self.state.behavior_policy_version += 1

        checkpoint_started = time.perf_counter()
        checkpoint_path = self._save_state()
        checkpoint_time = time.perf_counter() - checkpoint_started
        step_time = time.perf_counter() - step_started
        metrics.update(
            {
                "time/prepare": prompt_prepare_time + datum_prepare_time,
                "time/rollout": rollout_time,
                "time/reward": reward_time,
                "time/train": train_time,
                "time/checkpoint": checkpoint_time,
                "time/step": step_time,
                "time/tokens_per_second": input_tokens / step_time,
            }
        )
        self._total_input_tokens += input_tokens
        self.state.metrics = metrics
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
        """Asynchronous counterpart to :meth:`step`."""

        step_started = time.perf_counter()
        groups = self._groups(rows)
        prompt_prepare_time = time.perf_counter() - step_started

        rollout_started = time.perf_counter()
        sampled = await self._sample_groups_async(
            groups,
            await self._new_sampling_client_async(),
        )
        rollout_time = time.perf_counter() - rollout_started

        reward_started = time.perf_counter()
        reward_groups, reward_metrics = await self.reward_evaluator.score_async(sampled)
        reward_time = time.perf_counter() - reward_started

        train_started = time.perf_counter()
        datums, skipped, metrics = self._build_datums(sampled, reward_groups)
        datum_prepare_time = time.perf_counter() - train_started
        input_tokens = sum(datum.model_input.length for datum in datums)
        metrics["train/input_tokens"] = float(input_tokens)
        metrics.update(_rollout_metrics(sampled))
        metrics.update(reward_metrics)
        metrics.update(await self._train_datums_async(datums))
        train_time = time.perf_counter() - train_started
        self.state.global_step += 1
        self.state.behavior_policy_version += 1

        checkpoint_started = time.perf_counter()
        checkpoint_path = await self._save_state_async()
        checkpoint_time = time.perf_counter() - checkpoint_started
        step_time = time.perf_counter() - step_started
        metrics.update(
            {
                "time/prepare": prompt_prepare_time + datum_prepare_time,
                "time/rollout": rollout_time,
                "time/reward": reward_time,
                "time/train": train_time,
                "time/checkpoint": checkpoint_time,
                "time/step": step_time,
                "time/tokens_per_second": input_tokens / step_time,
            }
        )
        self._total_input_tokens += input_tokens
        self.state.metrics = metrics
        result = GRPOResult(
            self.state.global_step,
            metrics=metrics,
            num_datums=len(datums),
            skipped_groups=skipped,
            checkpoint_path=checkpoint_path,
        )
        self._log_swanlab(result)
        return result

    # ------------------------------------------------------------------
    # Validation: rollout and reward only, never backward
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluation_metrics(
        groups: Sequence[GRPOBatch],
        reward_groups: Sequence[Sequence[float]],
        reward_metrics: Mapping[str, float],
    ) -> dict[str, float]:
        """Reduce rollout rewards and lengths without constructing training datums."""

        rewards = [reward for values in reward_groups for reward in values]
        samples = [sample for group in groups for sample in group.samples]
        if not rewards or len(rewards) != len(samples):
            raise ValueError("GRPO validation rewards must align with sampled completions.")
        lengths = [len(sample.tokens) for sample in samples]
        logprobs = [value for sample in samples for value in sample.logprobs]
        metrics = {
            "eval/reward": sum(rewards) / len(rewards),
            "eval/prompts": float(len(groups)),
            "eval/completions": float(len(samples)),
            "eval/completion_length": sum(lengths) / len(lengths),
        }
        if logprobs:
            metrics["eval/entropy"] = -sum(logprobs) / len(logprobs)
        metrics.update(
            {
                "eval/" + name: value
                for name, value in reward_metrics.items()
            }
        )
        return metrics

    def _evaluation_batches(
        self, dataset: Iterable[Mapping[str, Any]]
    ) -> Iterable[list[Mapping[str, Any]]]:
        """Yield prompt batches using GRPO evaluation limits."""

        config = self.config.evaluation
        iterator = iter(dataset)
        batch_size = config.batch_size or self.config.batch_size
        produced = 0
        while config.max_batches is None or produced < config.max_batches:
            rows = list(itertools.islice(iterator, batch_size))
            if not rows:
                break
            produced += 1
            yield rows

    def _publish_evaluation(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Store and report GRPO rollout metrics at the current policy step."""

        values = {str(name): float(value) for name, value in metrics.items()}
        self._last_eval_step = self.state.global_step
        self.state.metrics.update(values)
        if self._swanlab_run is not None:
            self._swanlab_run.log(
                {**values, "train/step": float(self.state.global_step)},
                step=self.state.global_step,
            )
        return values

    def evaluate(
        self,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Sample the current policy and score validation prompts without training."""

        dataset = valid_dataset if valid_dataset is not None else self.valid_dataset
        if dataset is None:
            raise ValueError("valid_dataset is required for evaluate().")
        client = self._new_sampling_client()
        totals: dict[str, float] = {}
        prompts_total = 0
        for rows in self._evaluation_batches(dataset):
            groups = self._sample_groups(self._groups(rows), client)
            rewards, reward_metrics = self.reward_evaluator.score(groups)
            metrics = self._evaluation_metrics(groups, rewards, reward_metrics)
            count = len(groups)
            for name, value in metrics.items():
                if name not in {"eval/prompts", "eval/completions"}:
                    totals[name] = totals.get(name, 0.0) + value * count
            totals["eval/completions"] = totals.get("eval/completions", 0.0) + metrics[
                "eval/completions"
            ]
            prompts_total += count
        if prompts_total == 0:
            raise ValueError("valid_dataset must contain at least one prompt.")
        metrics = {
            name: value / prompts_total
            for name, value in totals.items()
            if name != "eval/completions"
        }
        metrics["eval/prompts"] = float(prompts_total)
        metrics["eval/completions"] = totals["eval/completions"]
        return self._publish_evaluation(metrics)

    async def evaluate_async(
        self,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Asynchronous counterpart to :meth:`evaluate`."""

        dataset = valid_dataset if valid_dataset is not None else self.valid_dataset
        if dataset is None:
            raise ValueError("valid_dataset is required for evaluate_async().")
        client = await self._new_sampling_client_async()
        totals: dict[str, float] = {}
        prompts_total = 0
        for rows in self._evaluation_batches(dataset):
            groups = await self._sample_groups_async(self._groups(rows), client)
            rewards, reward_metrics = await self.reward_evaluator.score_async(groups)
            metrics = self._evaluation_metrics(groups, rewards, reward_metrics)
            count = len(groups)
            for name, value in metrics.items():
                if name not in {"eval/prompts", "eval/completions"}:
                    totals[name] = totals.get(name, 0.0) + value * count
            totals["eval/completions"] = totals.get("eval/completions", 0.0) + metrics[
                "eval/completions"
            ]
            prompts_total += count
        if prompts_total == 0:
            raise ValueError("valid_dataset must contain at least one prompt.")
        metrics = {
            name: value / prompts_total
            for name, value in totals.items()
            if name != "eval/completions"
        }
        metrics["eval/prompts"] = float(prompts_total)
        metrics["eval/completions"] = totals["eval/completions"]
        return self._publish_evaluation(metrics)

    # ------------------------------------------------------------------
    # Full dataset training
    # ------------------------------------------------------------------

    def train(self) -> TrainerState:
        """Consume one dataset pass through the synchronous training path."""

        started = time.perf_counter()
        initial_step = self.state.global_step
        initial_tokens = self._total_input_tokens
        valid_dataset = getattr(self, "valid_dataset", None)
        progress = tqdm(
            total=self.config.max_steps,
            desc="GRPO train",
            unit="step",
            ncols=160,
        )
        try:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for train().")
            if (
                valid_dataset is not None
                and self.config.evaluation.before_train
                and getattr(self, "_last_eval_step", None) is None
            ):
                self.evaluate()
            iterator = iter(self.train_dataset)
            last_rows: list[Mapping[str, Any]] = []
            while (
                self.config.max_steps is None
                or self.state.global_step < self.config.max_steps
            ):
                rows = list(itertools.islice(iterator, self.config.batch_size))
                if not rows:
                    break
                result = self.step(rows)
                last_rows = rows
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
                if (
                    valid_dataset is not None
                    and self.config.evaluation.every_steps > 0
                    and self.state.global_step
                    % self.config.evaluation.every_steps
                    == 0
                    and getattr(self, "_last_eval_step", None)
                    != self.state.global_step
                ):
                    self.evaluate()
            if (
                last_rows
                and self.config.save_every
                and self.state.global_step % self.config.save_every
            ):
                self._save_state(final=True)
            if (
                valid_dataset is not None
                and self.config.evaluation.after_train
                and getattr(self, "_last_eval_step", None) != self.state.global_step
            ):
                self.evaluate()
            self._flush_swanlab(self.state.global_step)
            self._log_training_summary(started, initial_step, initial_tokens)
            self._report_final_state()
            self._log_final_checkpoint()
            self._log_final_samples(last_rows)
        except BaseException as error:
            self._finish_swanlab("crashed", str(error))
            raise
        else:
            self._finish_swanlab()
        finally:
            progress.close()
        return self.state

    async def train_async(self) -> TrainerState:
        """Consume one dataset pass through the asynchronous training path."""

        started = time.perf_counter()
        initial_step = self.state.global_step
        initial_tokens = self._total_input_tokens
        valid_dataset = getattr(self, "valid_dataset", None)
        progress = tqdm(
            total=self.config.max_steps,
            desc="GRPO train",
            unit="step",
            ncols=160,
        )
        try:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for train_async().")
            if (
                valid_dataset is not None
                and self.config.evaluation.before_train
                and getattr(self, "_last_eval_step", None) is None
            ):
                await self.evaluate_async()
            iterator = iter(self.train_dataset)
            last_rows: list[Mapping[str, Any]] = []
            while (
                self.config.max_steps is None
                or self.state.global_step < self.config.max_steps
            ):
                rows = list(itertools.islice(iterator, self.config.batch_size))
                if not rows:
                    break
                result = await self.step_async(rows)
                last_rows = rows
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
                if (
                    valid_dataset is not None
                    and self.config.evaluation.every_steps > 0
                    and self.state.global_step
                    % self.config.evaluation.every_steps
                    == 0
                    and getattr(self, "_last_eval_step", None)
                    != self.state.global_step
                ):
                    await self.evaluate_async()
            if (
                last_rows
                and self.config.save_every
                and self.state.global_step % self.config.save_every
            ):
                await self._save_state_async(final=True)
            if (
                valid_dataset is not None
                and self.config.evaluation.after_train
                and getattr(self, "_last_eval_step", None) != self.state.global_step
            ):
                await self.evaluate_async()
            self._flush_swanlab(self.state.global_step)
            self._log_training_summary(started, initial_step, initial_tokens)
            self._report_final_state()
            self._log_final_checkpoint()
            await self._log_final_samples_async(last_rows)
        except BaseException as error:
            self._finish_swanlab("crashed", str(error))
            raise
        else:
            self._finish_swanlab()
        finally:
            progress.close()
        return self.state
