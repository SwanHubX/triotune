"""DPO training with PyTRIO clients for model execution.

Preference pairs are scored against a fixed reference sampler. PyTRIO's
``forward_backward_custom`` evaluates policy logprobs remotely, computes the
pairwise loss locally, and sends its gradients back to the training service.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytrio as trio
import swanlab
from tqdm import tqdm

from ...utils.client import call, call_async, optimizer_params
from ..common.checkpoint import save_checkpoint, save_checkpoint_async
from ..common.logging import accumulate_metrics, averaged_metrics, finish_run, start_run
from ..common.prompts import RenderedPrompt, render_prompt
from .config import DPOConfig
from .data import (
    Preference as _Preference,
    build_datum as _datum,
    prepare_preferences,
    reference_logprob as _reference_logprob,
)
from .loss import make_loss_fn as _loss_fn
from .metrics import swanlab_metrics as _swanlab_metrics
from .types import DPOResult, DPOState


class DPOTrainer:
    """Train a PyTRIO LoRA policy from ``messages``, ``chosen``, ``rejected`` rows.

    ``model`` and optional ``ref_model`` are PyTRIO base-model IDs. With no
    ``ref_model``, the fixed reference is the unmodified base model. A custom
    reference must use the same tokenizer vocabulary as the trained model.
    """

    def __init__(
        self,
        model: str,
        ref_model: str | None = None,
        args: DPOConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        if ref_model is not None and (not isinstance(ref_model, str) or not ref_model.strip()):
            raise ValueError("ref_model must be a PyTRIO base-model ID when provided.")
        self.model = model.strip()
        self.ref_model = ref_model.strip() if ref_model is not None else self.model
        self.args = args or DPOConfig()
        self.train_dataset = train_dataset
        self._service_client = trio.ServiceClient()
        self._training_client = self._service_client.create_lora_training_client(
            base_model=self.model, rank=lora_rank, seed=self.args.seed
        )
        self._reference_client = self._service_client.create_sampling_client(base_model=self.ref_model)
        self._tokenizer = self._training_client.get_tokenizer()
        self.state = DPOState()
        self._total_tokens = 0
        self._total_pairs = 0
        self._metric_window: dict[str, tuple[float, int]] = {}
        self._run = None
        if self.args.report_to == "swanlab":
            self._run = start_run(
                self.args,
                default_name=f"{self.model.rsplit('/', 1)[-1]}-dpo",
                metadata={"model": self.model, "ref_model": self.ref_model, "lora_rank": lora_rank},
            )

    def _preferences(self, rows: Sequence[Mapping[str, Any]]) -> list[_Preference]:
        return prepare_preferences(rows, self._tokenizer, self.args)

    def _datums(self, pairs: Sequence[_Preference]) -> list[Any]:
        return [
            _datum(pair.prompt, completion)
            for pair in pairs
            for completion in (pair.chosen, pair.rejected)
        ]

    def _reference_scores(self, pairs: Sequence[_Preference]) -> list[float]:
        scores = []
        for pair in pairs:
            for completion in (pair.chosen, pair.rejected):
                tokens = pair.prompt + completion
                values = call(self._reference_client, "compute_logprobs", trio.ModelInput.from_ints(tokens))
                scores.append(_reference_logprob(values, len(pair.prompt), len(tokens)))
        return scores

    async def _reference_scores_async(self, pairs: Sequence[_Preference]) -> list[float]:
        requests = [
            (pair.prompt, completion)
            for pair in pairs
            for completion in (pair.chosen, pair.rejected)
        ]
        values = await asyncio.gather(
            *(
                call_async(
                    self._reference_client,
                    "compute_logprobs",
                    trio.ModelInput.from_ints(prompt + completion),
                )
                for prompt, completion in requests
            )
        )
        return [
            _reference_logprob(result, len(prompt), len(prompt) + len(completion))
            for (prompt, completion), result in zip(requests, values, strict=True)
        ]

    def _metrics(self, forward: Any, optim: Any) -> dict[str, float]:
        metrics = {"optim/lr": self.args.learning_rate}
        for response in (forward, optim):
            for name, value in getattr(response, "metrics", {}).items():
                # PyTRIO's custom backward reports a linear surrogate loss here,
                # while train/loss is the actual pairwise DPO objective.
                if name in {"loss_sum", "loss_mean", "scalar/loss_sum", "scalar/loss_mean"}:
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[name] = float(value)
        if "grad_norm" in metrics:
            metrics["optim/grad_norm"] = metrics["grad_norm"]
        return metrics

    def _finish_run(self, state: str, error: str | None = None) -> None:
        if self._run is not None:
            finish_run(self._run, state=state, error=error)
            self._run = None

    def _log_swanlab(self, metrics: dict[str, float], count: int) -> None:
        if self._run is None:
            return
        accumulate_metrics(self._metric_window, {**_swanlab_metrics(metrics), "train/pairs": count})
        if self.state.global_step % self.args.logging_step == 0:
            self._flush_swanlab()

    def _flush_swanlab(self) -> None:
        if self._run is None:
            return
        payload = averaged_metrics(self._metric_window)
        self._metric_window.clear()
        if payload:
            self._run.log({**payload, "train/step": self.state.global_step}, step=self.state.global_step)

    def _finish_step(self, metrics: dict[str, float], count: int, started: float) -> DPOResult:
        self.state.global_step += 1
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            save_started = time.perf_counter()
            self._save_state()
            metrics["time/save"] = time.perf_counter() - save_started
        metrics["time/step"] = time.perf_counter() - started
        metrics["time/tokens_per_second"] = metrics["data/tokens"] / metrics["time/step"]
        metrics["time/pairs_per_second"] = count / metrics["time/step"]
        self._total_tokens += int(metrics["data/tokens"])
        self._total_pairs += count
        self.state.metrics = metrics
        result = DPOResult(self.state.global_step, metrics, count, self.state.checkpoint_path)
        self._log_swanlab(metrics, count)
        return result

    def _save_state(self) -> str:
        path = save_checkpoint(self._training_client, "dpo", self.state.global_step)
        self.state.checkpoint_path = path
        return path

    async def _save_state_async(self) -> str:
        path = await save_checkpoint_async(self._training_client, "dpo", self.state.global_step)
        self.state.checkpoint_path = path
        return path

    def step(self, rows: Sequence[Mapping[str, Any]]) -> DPOResult:
        started = time.perf_counter()
        pairs = self._preferences(rows)
        datums = self._datums(pairs)
        prepared = time.perf_counter()
        reference = self._reference_scores(pairs)
        referenced = time.perf_counter()
        forward = call(
            self._training_client,
            "forward_backward_custom",
            datums,
            _loss_fn(reference, self.args.beta),
        )
        forwarded = time.perf_counter()
        optim = call(self._training_client, "optim_step", optimizer_params(self.args, trio.AdamParams))
        optimized = time.perf_counter()
        metrics = self._metrics(forward, optim)
        metrics.update({
            "data/tokens": sum(datum.model_input.length for datum in datums),
            "time/prepare": prepared - started,
            "time/reference": referenced - prepared,
            "time/forward_backward": forwarded - referenced,
            "time/optim": optimized - forwarded,
        })
        return self._finish_step(metrics, len(pairs), started)

    async def step_async(self, rows: Sequence[Mapping[str, Any]]) -> DPOResult:
        started = time.perf_counter()
        pairs = self._preferences(rows)
        datums = self._datums(pairs)
        prepared = time.perf_counter()
        reference = await self._reference_scores_async(pairs)
        referenced = time.perf_counter()
        forward = await call_async(
            self._training_client,
            "forward_backward_custom",
            datums,
            _loss_fn(reference, self.args.beta),
        )
        forwarded = time.perf_counter()
        optim = await call_async(
            self._training_client, "optim_step", optimizer_params(self.args, trio.AdamParams)
        )
        optimized = time.perf_counter()
        metrics = self._metrics(forward, optim)
        metrics.update({
            "data/tokens": sum(datum.model_input.length for datum in datums),
            "time/prepare": prepared - started,
            "time/reference": referenced - prepared,
            "time/forward_backward": forwarded - referenced,
            "time/optim": optimized - forwarded,
        })
        self.state.global_step += 1
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            save_started = time.perf_counter()
            await self._save_state_async()
            metrics["time/save"] = time.perf_counter() - save_started
        metrics["time/step"] = time.perf_counter() - started
        metrics["time/tokens_per_second"] = metrics["data/tokens"] / metrics["time/step"]
        metrics["time/pairs_per_second"] = len(pairs) / metrics["time/step"]
        self._total_tokens += int(metrics["data/tokens"])
        self._total_pairs += len(pairs)
        self.state.metrics = metrics
        result = DPOResult(self.state.global_step, metrics, len(pairs), self.state.checkpoint_path)
        self._log_swanlab(metrics, len(pairs))
        return result

    @staticmethod
    def _progress_postfix(result: DPOResult) -> dict[str, str]:
        return {
            key: f"{value:.4f}"
            for key, value in result.metrics.items()
            if key in {"train/loss", "rewards/accuracies"}
        }

    def _log_final_checkpoint(self) -> None:
        if self._run is not None and self.state.checkpoint_path is not None:
            self._run.log_text(
                key="checkpoint/path",
                data=self.state.checkpoint_path,
                step=self.state.global_step,
            )

    def _final_sample_prompts(self, rows: Sequence[Mapping[str, Any]]) -> list[RenderedPrompt]:
        return [
            render_prompt(
                row[self.args.prompt_key],
                self._tokenizer,
                chat_template_kwargs=self.args.chat_template_kwargs,
            )
            for row in rows[:4]
        ]

    def _final_sampling_params(self, prompt_length: int) -> Any:
        max_tokens = 128
        if self.args.max_length is not None:
            max_tokens = min(max_tokens, self.args.max_length - prompt_length)
        return trio.SamplingParams(max_tokens=max_tokens, seed=self.args.seed)

    def _write_final_samples(self, prompts: Sequence[RenderedPrompt], responses: Sequence[Any]) -> None:
        texts = []
        for prompt, response in zip(prompts, responses, strict=True):
            if not response.sequences:
                raise ValueError("Final inference returned no sequences.")
            sequence = response.sequences[0]
            completion = sequence.text
            if completion is None:
                completion = self._tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            texts.append(swanlab.Text(completion, caption=prompt.text))
        self._run.log_text(key="sample/text", data=texts, step=self.state.global_step)

    def _log_final_samples(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._run is None or not rows:
            return
        client = call(self._training_client, "save_weights_and_get_sampling_client")
        prompts = self._final_sample_prompts(rows)
        responses = [
            call(
                client,
                "sample",
                prompt=trio.ModelInput.from_ints(prompt.tokens),
                num_samples=1,
                sampling_params=self._final_sampling_params(len(prompt.tokens)),
            )
            for prompt in prompts
        ]
        self._write_final_samples(prompts, responses)

    async def _log_final_samples_async(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._run is None or not rows:
            return
        client = await call_async(self._training_client, "save_weights_and_get_sampling_client")
        prompts = self._final_sample_prompts(rows)
        responses = await asyncio.gather(
            *(
                call_async(
                    client,
                    "sample",
                    prompt=trio.ModelInput.from_ints(prompt.tokens),
                    num_samples=1,
                    sampling_params=self._final_sampling_params(len(prompt.tokens)),
                )
                for prompt in prompts
            )
        )
        self._write_final_samples(prompts, responses)

    def _log_training_summary(self, started: float) -> None:
        if not self.state.global_step:
            return
        elapsed = time.perf_counter() - started
        summary = {
            "time/train_runtime": elapsed,
            "time/train_steps_per_second": self.state.global_step / elapsed,
            "time/train_tokens_per_second": self._total_tokens / elapsed,
            "time/train_pairs_per_second": self._total_pairs / elapsed,
            "data/total_tokens": self._total_tokens,
            "data/total_pairs": self._total_pairs,
        }
        self.state.metrics.update(summary)
        if self._run is not None:
            self._run.log(_swanlab_metrics(summary), step=self.state.global_step)

    def train(self) -> DPOState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        started = time.perf_counter()
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="DPO train", unit="step")
        last_rows: list[Mapping[str, Any]] = []
        try:
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = self.step(rows)
                last_rows = rows
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                self._save_state()
            self._flush_swanlab()
            self._log_final_checkpoint()
            self._log_training_summary(started)
            self._log_final_samples(last_rows)
        except BaseException as error:
            self._finish_run("crashed", str(error))
            raise
        else:
            self._finish_run("success")
        finally:
            progress.close()
        return self.state

    async def train_async(self) -> DPOState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train_async().")
        started = time.perf_counter()
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="DPO train", unit="step")
        last_rows: list[Mapping[str, Any]] = []
        try:
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = await self.step_async(rows)
                last_rows = rows
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                await self._save_state_async()
            self._flush_swanlab()
            self._log_final_checkpoint()
            self._log_training_summary(started)
            await self._log_final_samples_async(last_rows)
        except BaseException as error:
            self._finish_run("crashed", str(error))
            raise
        else:
            self._finish_run("success")
        finally:
            progress.close()
        return self.state
