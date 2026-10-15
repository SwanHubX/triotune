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

from ..api.cli_args import DPOConfig
from ..api.client_api import call, call_async, optimizer_params
from ..dataset import (
    IGNORE_INDEX,
    PreferenceExample,
    PromptExample,
    SupervisedExample,
    prepare_preference,
    prepare_prompt,
)
from ..utils.checkpoint import save_checkpoint, save_checkpoint_async
from ..utils.metrics import MetricsAPI
from .dpo.loss import make_loss_fn as _loss_fn
from .dpo.state import DPOResult, DPOState


def _datum(prompt: list[int], completion: list[int]) -> Any:
    """Build the masked, next-token-aligned PyTRIO sequence used by DPO."""

    example = SupervisedExample(
        input_ids=prompt + completion,
        labels=[IGNORE_INDEX] * len(prompt) + completion,
    )
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(example.model_input),
        loss_fn_inputs={"target_tokens": example.target_tokens},
    )


def _reference_logprob(
    values: Sequence[float | None], prompt_length: int, total_length: int
) -> float:
    """Reduce reference token scores over the completion only."""

    if len(values) != total_length:
        raise ValueError("Reference sampler returned the wrong number of token logprobs.")
    completion_values = values[prompt_length:]
    if not completion_values or any(
        value is None or not math.isfinite(value) for value in completion_values
    ):
        raise ValueError("Reference sampler returned missing or non-finite completion logprobs.")
    return sum(float(value) for value in completion_values if value is not None)


def _softplus(value: float) -> float:
    """Numerically stable ``log(1 + exp(value))`` for DPO evaluation."""

    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


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
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
    ) -> None:
        """Create policy/reference clients and validate their token contract."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        if ref_model is not None and (
            not isinstance(ref_model, str) or not ref_model.strip()
        ):
            raise ValueError("ref_model must be a PyTRIO base-model ID when provided.")
        self.model = model.strip()
        self.ref_model = ref_model.strip() if ref_model is not None else self.model
        self.args = args or DPOConfig()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self._service_client = trio.ServiceClient()
        self._training_client = self._service_client.create_lora_training_client(
            base_model=self.model, rank=lora_rank, seed=self.args.seed
        )
        self._reference_client = self._service_client.create_sampling_client(
            base_model=self.ref_model
        )
        self._tokenizer = self._training_client.get_tokenizer()
        reference_tokenizer = getattr(
            self._reference_client, "get_tokenizer", lambda: None
        )()
        policy_vocab = getattr(self._tokenizer, "get_vocab", None)
        reference_vocab = getattr(reference_tokenizer, "get_vocab", None)
        if (
            callable(policy_vocab)
            and callable(reference_vocab)
            and policy_vocab() != reference_vocab()
        ):
            raise ValueError(
                "DPO policy and reference models must use identical token-to-ID vocabularies."
            )
        self.state = DPOState()
        self._last_eval_step: int | None = None
        self._total_tokens = 0
        self._total_pairs = 0
        self._metrics_api = MetricsAPI.from_config(
            self.args,
            default_name=f"{self.model.rsplit('/', 1)[-1]}-dpo",
            metadata={
                "model": self.model,
                "ref_model": self.ref_model,
                "lora_rank": lora_rank,
            },
            logging_step=self.args.logging_step,
        )

    # ------------------------------------------------------------------
    # Preference preparation and model scoring
    # ------------------------------------------------------------------

    def _preferences(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[PreferenceExample]:
        """Validate and tokenize ordered chosen/rejected dataset rows."""

        if not rows:
            raise ValueError("A DPO step requires at least one preference pair.")
        return [
            prepare_preference(
                row,
                self._tokenizer,
                prompt_key=self.args.prompt_key,
                add_eos_token=self.args.add_eos_token,
                chat_template_kwargs=self.args.chat_template_kwargs,
                max_length=self.args.max_length,
                row_index=index,
            )
            for index, row in enumerate(rows)
        ]

    def _datums(self, pairs: Sequence[PreferenceExample]) -> list[Any]:
        """Expand every pair to chosen then rejected PyTRIO datums."""

        return [
            _datum(pair.prompt, completion)
            for pair in pairs
            for completion in (pair.chosen, pair.rejected)
        ]

    def _scores(self, client: Any, pairs: Sequence[PreferenceExample]) -> list[float]:
        """Return completion-only sequence logprobs in pair order."""

        scores = []
        for pair in pairs:
            for completion in (pair.chosen, pair.rejected):
                tokens = pair.prompt + completion
                values = call(
                    client,
                    "compute_logprobs",
                    trio.ModelInput.from_ints(tokens),
                )
                scores.append(_reference_logprob(values, len(pair.prompt), len(tokens)))
        return scores

    async def _scores_async(
        self, client: Any, pairs: Sequence[PreferenceExample]
    ) -> list[float]:
        """Asynchronously score all chosen/rejected sequences in one batch."""

        requests = [
            (pair.prompt, completion)
            for pair in pairs
            for completion in (pair.chosen, pair.rejected)
        ]
        values = await asyncio.gather(
            *(
                call_async(
                    client,
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

    def _reference_scores(self, pairs: Sequence[PreferenceExample]) -> list[float]:
        """Score preference completions under the fixed reference policy."""

        return self._scores(self._reference_client, pairs)

    async def _reference_scores_async(
        self, pairs: Sequence[PreferenceExample]
    ) -> list[float]:
        """Asynchronously score completions under the fixed reference policy."""

        return await self._scores_async(self._reference_client, pairs)

    # ------------------------------------------------------------------
    # Validation: current policy versus the fixed reference
    # ------------------------------------------------------------------

    def _evaluation_metrics(
        self,
        policy_scores: Sequence[float],
        reference_scores: Sequence[float],
    ) -> dict[str, float]:
        """Compute the actual pairwise DPO objective without accumulating gradients."""

        if len(policy_scores) != len(reference_scores) or len(policy_scores) % 2:
            raise ValueError("DPO evaluation scores must contain aligned preference pairs.")
        losses = []
        margins = []
        chosen_rewards = []
        rejected_rewards = []
        for index in range(0, len(policy_scores), 2):
            chosen_reward = self.args.beta * (
                policy_scores[index] - reference_scores[index]
            )
            rejected_reward = self.args.beta * (
                policy_scores[index + 1] - reference_scores[index + 1]
            )
            margin = chosen_reward - rejected_reward
            losses.append(_softplus(-margin))
            margins.append(margin)
            chosen_rewards.append(chosen_reward)
            rejected_rewards.append(rejected_reward)
        count = len(losses)
        return {
            "eval/loss": sum(losses) / count,
            "eval/reward_accuracy": sum(margin > 0 for margin in margins) / count,
            "eval/reward_margin": sum(margins) / count,
            "eval/reward_chosen": sum(chosen_rewards) / count,
            "eval/reward_rejected": sum(rejected_rewards) / count,
            "eval/pairs": float(count),
        }

    def _evaluation_batches(
        self, dataset: Iterable[Mapping[str, Any]]
    ) -> Iterable[list[Mapping[str, Any]]]:
        """Yield preference batches using DPO evaluation limits."""

        config = self.args.evaluation
        iterator = iter(dataset)
        batch_size = config.batch_size or self.args.batch_size
        produced = 0
        while config.max_batches is None or produced < config.max_batches:
            rows = list(itertools.islice(iterator, batch_size))
            if not rows:
                break
            produced += 1
            yield rows

    def _publish_evaluation(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Store DPO validation metrics without changing optimizer state."""

        values = {str(name): float(value) for name, value in metrics.items()}
        self._last_eval_step = self.state.global_step
        self.state.metrics.update(values)
        self._metrics_api.publish(values, step=self.state.global_step)
        return values

    def evaluate(
        self,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Evaluate current-policy preference accuracy and DPO loss."""

        dataset = valid_dataset if valid_dataset is not None else self.valid_dataset
        if dataset is None:
            raise ValueError("valid_dataset is required for evaluate().")
        policy_client = call(
            self._training_client, "save_weights_and_get_sampling_client"
        )
        totals: dict[str, float] = {}
        pairs_total = 0
        for rows in self._evaluation_batches(dataset):
            pairs = self._preferences(rows)
            metrics = self._evaluation_metrics(
                self._scores(policy_client, pairs), self._reference_scores(pairs)
            )
            count = len(pairs)
            for name, value in metrics.items():
                if name != "eval/pairs":
                    totals[name] = totals.get(name, 0.0) + value * count
            pairs_total += count
        if pairs_total == 0:
            raise ValueError("valid_dataset must contain at least one preference pair.")
        metrics = {
            name: value / pairs_total for name, value in totals.items()
        }
        metrics["eval/pairs"] = float(pairs_total)
        return self._publish_evaluation(metrics)

    async def evaluate_async(
        self,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Asynchronous counterpart to :meth:`evaluate`."""

        dataset = valid_dataset if valid_dataset is not None else self.valid_dataset
        if dataset is None:
            raise ValueError("valid_dataset is required for evaluate_async().")
        policy_client = await call_async(
            self._training_client, "save_weights_and_get_sampling_client"
        )
        totals: dict[str, float] = {}
        pairs_total = 0
        for rows in self._evaluation_batches(dataset):
            pairs = self._preferences(rows)
            policy, reference = await asyncio.gather(
                self._scores_async(policy_client, pairs),
                self._reference_scores_async(pairs),
            )
            metrics = self._evaluation_metrics(policy, reference)
            count = len(pairs)
            for name, value in metrics.items():
                if name != "eval/pairs":
                    totals[name] = totals.get(name, 0.0) + value * count
            pairs_total += count
        if pairs_total == 0:
            raise ValueError("valid_dataset must contain at least one preference pair.")
        metrics = {
            name: value / pairs_total for name, value in totals.items()
        }
        metrics["eval/pairs"] = float(pairs_total)
        return self._publish_evaluation(metrics)

    # ------------------------------------------------------------------
    # One optimizer update and training metrics
    # ------------------------------------------------------------------

    def _metrics(self, forward: Any, optim: Any) -> dict[str, float]:
        """Normalize remote metrics while excluding the custom-loss surrogate."""

        metrics = {"optim/lr": self.args.learning_rate}
        for response in (forward, optim):
            for name, value in getattr(response, "metrics", {}).items():
                # PyTRIO's custom backward reports a linear surrogate loss here,
                # while train/loss is the actual pairwise DPO objective.
                if name in {
                    "loss_sum",
                    "loss_mean",
                    "scalar/loss_sum",
                    "scalar/loss_mean",
                }:
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[name] = float(value)
        if "grad_norm" in metrics:
            metrics["optim/grad_norm"] = metrics["grad_norm"]
        return metrics

    def _finish_run(self, state: str, error: str | None = None) -> None:
        """Close SwanLab once and clear the owned run handle."""

        self._metrics_api.finish(state=state, error=error)

    def _log_swanlab(self, metrics: dict[str, float], count: int) -> None:
        """Accumulate step metrics and flush at the configured interval."""

        self._metrics_api.record(
            {**metrics, "train/pairs": count},
            step=self.state.global_step,
        )

    def _flush_swanlab(self) -> None:
        """Publish and clear the current DPO metric window."""

        self._metrics_api.flush(step=self.state.global_step)

    def _finish_step(
        self, metrics: dict[str, float], count: int, started: float
    ) -> DPOResult:
        """Publish one successful synchronous update and optional checkpoint."""

        self.state.global_step += 1
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            save_started = time.perf_counter()
            self._save_state()
            metrics["time/save"] = time.perf_counter() - save_started
        metrics["time/step"] = time.perf_counter() - started
        metrics["time/tokens_per_second"] = (
            metrics["data/tokens"] / metrics["time/step"]
        )
        metrics["time/pairs_per_second"] = count / metrics["time/step"]
        self._total_tokens += int(metrics["data/tokens"])
        self._total_pairs += count
        self.state.metrics = metrics
        result = DPOResult(
            self.state.global_step,
            metrics,
            count,
            self.state.checkpoint_path,
        )
        self._log_swanlab(metrics, count)
        return result

    def _save_state(self) -> str:
        """Save policy weights and optimizer state for the current step."""

        path = save_checkpoint(self._training_client, "dpo", self.state.global_step)
        self.state.checkpoint_path = path
        return path

    async def _save_state_async(self) -> str:
        """Asynchronous counterpart to :meth:`_save_state`."""

        path = await save_checkpoint_async(self._training_client, "dpo", self.state.global_step)
        self.state.checkpoint_path = path
        return path

    def step(self, rows: Sequence[Mapping[str, Any]]) -> DPOResult:
        """Score one preference batch, backpropagate DPO loss, and optimize."""

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
        optim = call(
            self._training_client,
            "optim_step",
            optimizer_params(self.args, trio.AdamParams),
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
        return self._finish_step(metrics, len(pairs), started)

    async def step_async(self, rows: Sequence[Mapping[str, Any]]) -> DPOResult:
        """Asynchronous counterpart to :meth:`step`."""

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
        metrics["time/tokens_per_second"] = (
            metrics["data/tokens"] / metrics["time/step"]
        )
        metrics["time/pairs_per_second"] = len(pairs) / metrics["time/step"]
        self._total_tokens += int(metrics["data/tokens"])
        self._total_pairs += len(pairs)
        self.state.metrics = metrics
        result = DPOResult(
            self.state.global_step,
            metrics,
            len(pairs),
            self.state.checkpoint_path,
        )
        self._log_swanlab(metrics, len(pairs))
        return result

    @staticmethod
    def _progress_postfix(result: DPOResult) -> dict[str, str]:
        """Select compact scalars for the terminal progress bar."""

        return {
            key: f"{value:.4f}"
            for key, value in result.metrics.items()
            if key in {"train/loss", "reward/accuracy"}
        }

    def _log_final_checkpoint(self) -> None:
        """Publish the final checkpoint URI when reporting is enabled."""

        if self.state.checkpoint_path is not None:
            self._metrics_api.log_text(
                key="checkpoint/path",
                data=self.state.checkpoint_path,
                step=self.state.global_step,
            )

    def _final_sample_prompts(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[PromptExample]:
        """Prepare at most four prompts for final qualitative sampling."""

        return [
            prepare_prompt(
                row,
                self._tokenizer,
                prompt_key=self.args.prompt_key,
                chat_template_kwargs=self.args.chat_template_kwargs,
                max_length=self.args.max_length,
                min_completion_tokens=1,
                row_index=index,
            )
            for index, row in enumerate(rows[:4])
        ]

    # ------------------------------------------------------------------
    # Final qualitative samples and run summary
    # ------------------------------------------------------------------

    def _final_sampling_params(self, prompt_length: int) -> Any:
        """Fit final generation inside the configured total token budget."""

        max_tokens = 128
        if self.args.max_length is not None:
            max_tokens = min(max_tokens, self.args.max_length - prompt_length)
        return trio.SamplingParams(max_tokens=max_tokens, seed=self.args.seed)

    def _write_final_samples(
        self, prompts: Sequence[PromptExample], responses: Sequence[Any]
    ) -> None:
        """Decode and publish current-policy responses to validation prompts."""

        texts = []
        for prompt, response in zip(prompts, responses, strict=True):
            if not response.sequences:
                raise ValueError("Final inference returned no sequences.")
            sequence = response.sequences[0]
            completion = sequence.text
            if completion is None:
                completion = self._tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            texts.append(swanlab.Text(completion, caption=prompt.text))
        self._metrics_api.log_text(
            key="sample/text",
            data=texts,
            step=self.state.global_step,
        )

    def _log_final_samples(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Snapshot current weights and synchronously generate final samples."""

        if not self._metrics_api.enabled or not rows:
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
        """Asynchronously generate final samples from a current-policy snapshot."""

        if not self._metrics_api.enabled or not rows:
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
        """Record whole-run throughput after at least one successful update."""

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
        self._metrics_api.publish(summary, step=self.state.global_step)

    # ------------------------------------------------------------------
    # Full dataset training
    # ------------------------------------------------------------------

    def train(self) -> DPOState:
        """Consume one dataset pass with synchronous PyTRIO calls."""

        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        started = time.perf_counter()
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="DPO train", unit="step")
        last_rows: list[Mapping[str, Any]] = []
        try:
            if (
                self.valid_dataset is not None
                and self.args.evaluation.before_train
                and self._last_eval_step is None
            ):
                self.evaluate()
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = self.step(rows)
                last_rows = rows
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
                if (
                    self.valid_dataset is not None
                    and self.args.evaluation.every_steps > 0
                    and self.state.global_step % self.args.evaluation.every_steps == 0
                    and self._last_eval_step != self.state.global_step
                ):
                    self.evaluate()
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                self._save_state()
            if (
                self.valid_dataset is not None
                and self.args.evaluation.after_train
                and self._last_eval_step != self.state.global_step
            ):
                self.evaluate()
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
        """Consume one dataset pass with asynchronous PyTRIO calls."""

        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train_async().")
        started = time.perf_counter()
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="DPO train", unit="step")
        last_rows: list[Mapping[str, Any]] = []
        try:
            if (
                self.valid_dataset is not None
                and self.args.evaluation.before_train
                and self._last_eval_step is None
            ):
                await self.evaluate_async()
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = await self.step_async(rows)
                last_rows = rows
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
                if (
                    self.valid_dataset is not None
                    and self.args.evaluation.every_steps > 0
                    and self.state.global_step % self.args.evaluation.every_steps == 0
                    and self._last_eval_step != self.state.global_step
                ):
                    await self.evaluate_async()
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                await self._save_state_async()
            if (
                self.valid_dataset is not None
                and self.args.evaluation.after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self.evaluate_async()
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
