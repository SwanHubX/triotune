"""PyTRIO OPD: student rollouts, fixed teacher scores, remote policy updates.

Only sampled-token logprobs are transferred. This implements the local
reverse-KL policy-gradient surrogate, not full-vocabulary KL distillation.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytrio as trio
from tqdm import tqdm

from ..api.cli_args import OPDConfig
from ..api.client_api import call_async, optimizer_params
from ..dataset import PromptExample, prepare_prompt
from ..utils.checkpoint import save_checkpoint_async
from ..utils.metrics import MetricsAPI
from .opd.distillation import (
    _ScoredCompletion,
    build_opd_datum,
    distillation_advantages,
    teacher_completion_logprobs,
    validate_logprobs,
    validate_tokens,
)
from .opd.state import OPDResult, OPDState


class OPDTrainer:
    """Distill a fixed teacher into a PyTRIO LoRA student using prompt-only rows.

    ``model`` and ``teacher_model`` are PyTRIO base-model IDs. Optional
    ``teacher_model_path`` loads a teacher LoRA checkpoint on that base model.
    Token-to-ID vocabularies must match; prompts are rendered using the student
    template and the exact same tokens are scored by the teacher.

    Async methods are the primary implementation; sync methods run them with
    asyncio.run. Client construction is synchronous (use asyncio.to_thread in
    an async application). Calls on one trainer must not overlap.
    """

    def __init__(
        self,
        model: str,
        teacher_model: str,
        args: OPDConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
        teacher_model_path: str | None = None,
    ) -> None:
        """Create student/teacher clients and verify token-ID compatibility."""

        for name, value in (("model", model), ("teacher_model", teacher_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a PyTRIO base-model ID.")
        if (
            isinstance(lora_rank, bool)
            or not isinstance(lora_rank, int)
            or lora_rank < 1
        ):
            raise ValueError("lora_rank must be a positive integer.")
        if teacher_model_path is not None and (
            not isinstance(teacher_model_path, str) or not teacher_model_path.strip()
        ):
            raise ValueError("teacher_model_path must be a non-empty checkpoint URI.")
        self.model = model.strip()
        self.teacher_model = teacher_model.strip()
        self.teacher_model_path = teacher_model_path
        self.args = args or OPDConfig()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.state = OPDState()
        self._last_eval_step: int | None = None
        self._service_client = trio.ServiceClient()
        self._training_client = self._service_client.create_lora_training_client(
            base_model=self.model, rank=lora_rank, seed=self.args.seed,
        )
        self._teacher_client = self._service_client.create_sampling_client(
            base_model=self.teacher_model, model_path=self.teacher_model_path,
        )
        self._tokenizer = self._training_client.get_tokenizer()
        teacher_tokenizer = self._teacher_client.get_tokenizer()
        if self._tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError(
                "OPD requires identical student and teacher token-to-ID vocabularies."
            )
        self._student_sampler: Any = None
        self._busy = False
        self._requires_restart = False
        self._sampling_seed_offset = 0
        self._total_tokens = 0
        self._metrics_api = MetricsAPI.from_config(
            self.args,
            default_name=f"{self.model.rsplit('/', 1)[-1]}-opd",
            metadata={
                "model": self.model,
                "teacher_model": self.teacher_model,
                "teacher_model_path": teacher_model_path,
                "lora_rank": lora_rank,
                "sampling_temperature": 1.0,
                "sampling_top_p": 1.0,
                "sampling_top_k": -1,
            },
            logging_step=self.args.logging_step,
        )

    # ------------------------------------------------------------------
    # Re-entrancy and failure-state guard
    # ------------------------------------------------------------------

    def _enter(self) -> None:
        """Reserve the trainer for one operation and reject unsafe reuse."""

        if self._busy:
            raise RuntimeError("Calls on one OPDTrainer must not overlap.")
        if self._requires_restart:
            raise RuntimeError(
                "An update failed or was cancelled; recreate the trainer from "
                "a valid checkpoint."
            )
        self._busy = True

    # ------------------------------------------------------------------
    # Prompt preparation, student rollout, and teacher scoring
    # ------------------------------------------------------------------

    def _prompts(self, rows: Sequence[Mapping[str, Any]]) -> list[PromptExample]:
        """Render prompt-only rows and reserve at least one completion token."""

        if not rows:
            raise ValueError("An OPD step requires at least one prompt.")
        prompts = []
        for index, row in enumerate(rows):
            try:
                prompt = prepare_prompt(
                    row,
                    self._tokenizer,
                    prompt_key=self.args.prompt_key,
                    add_generation_prompt=self.args.add_generation_prompt,
                    chat_template_kwargs=self.args.chat_template_kwargs,
                    max_length=self.args.max_length,
                    min_completion_tokens=1,
                    row_index=index,
                )
            except KeyError as error:
                # OPD historically treats every malformed prompt row as a
                # value error; retain that trainer contract at this boundary.
                raise ValueError(str(error)) from error
            prompts.append(prompt)
        return prompts

    def _sampling_params(self, prompt: PromptExample, index: int) -> Any:
        """Build unbiased student sampling parameters within the token budget."""

        max_tokens = self.args.max_completion_length
        if self.args.max_length is not None:
            max_tokens = min(max_tokens, self.args.max_length - len(prompt.tokens))
        seed = self.args.seed
        if seed is not None:
            seed += self._sampling_seed_offset + index
        return trio.SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            seed=seed,
            stop=self.args.stop,
        )

    async def _score_completion(
        self,
        prompt: PromptExample,
        tokens: list[int],
        student: list[float],
        scoring_slots: asyncio.Semaphore,
    ) -> _ScoredCompletion:
        """Score one sampled completion and build its fixed-advantage Datum."""

        async with scoring_slots:
            values = await call_async(
                self._teacher_client,
                "compute_logprobs",
                trio.ModelInput.from_ints(prompt.tokens + tokens),
            )
        teacher = teacher_completion_logprobs(values, len(prompt.tokens), len(tokens))
        advantages = distillation_advantages(
            student, teacher, coef=self.args.kl_coef, discount=self.args.discount_factor,
        )
        return _ScoredCompletion(
            build_opd_datum(prompt.tokens, tokens, student, advantages),
            len(tokens),
            sum(student),
            sum(teacher),
            sum(advantages),
        )

    async def _sample_and_score(
        self,
        prompt: PromptExample,
        index: int,
        sampler: Any,
        sampling_slots: asyncio.Semaphore,
        scoring_slots: asyncio.Semaphore,
    ) -> list[_ScoredCompletion]:
        """Sample every completion for one prompt, then score them as a unit."""

        params = self._sampling_params(prompt, index)
        async with sampling_slots:
            response = await call_async(
                sampler,
                "sample",
                prompt=trio.ModelInput.from_ints(prompt.tokens),
                num_samples=self.args.num_generations,
                sampling_params=params,
            )
        sequences = response.sequences
        if len(sequences) != self.args.num_generations:
            raise ValueError(
                "Student sampler returned an unexpected number of completions."
            )
        # Validate the entire response before any of its teacher requests.
        samples = []
        for sequence in sequences:
            tokens = validate_tokens(sequence.tokens, source="Student completion")
            if len(tokens) > params.max_tokens:
                raise ValueError("Student completion exceeds the requested token budget.")
            if sequence.logprobs is None:
                raise ValueError("Student sampling must return completion logprobs.")
            student = validate_logprobs(sequence.logprobs, source="Student completion")
            if len(student) != len(tokens):
                raise ValueError("Student completion tokens and logprobs must align.")
            samples.append((tokens, student))
        # Teacher scoring starts as soon as THIS prompt is ready. TaskGroup
        # cancels and drains sibling requests on error or caller cancellation.
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(
                    self._score_completion(prompt, tokens, student, scoring_slots)
                )
                for tokens, student in samples
            ]
        return [task.result() for task in tasks]

    async def _rollout_and_score(
        self, prompts: list[PromptExample], sampler: Any
    ) -> list[_ScoredCompletion]:
        """Bound prompt rollout and teacher scoring across the complete batch."""

        sampling_slots = asyncio.Semaphore(self.args.max_concurrent_sampling)
        scoring_slots = asyncio.Semaphore(self.args.max_concurrent_scoring)
        pending = iter(enumerate(prompts))
        results: list[list[_ScoredCompletion] | None] = [None] * len(prompts)

        async def worker() -> None:
            """Consume prompt indices from a lock-free local iterator."""

            # next() contains no await, so cooperative tasks cannot race here.
            for index, prompt in pending:
                results[index] = await self._sample_and_score(
                    prompt, index, sampler, sampling_slots, scoring_slots,
                )

        # Bound local tasks and the number of rollouts waiting for the teacher,
        # even when batch_size is much larger than either concurrency limit.
        workers = min(
            len(prompts),
            self.args.max_concurrent_sampling + self.args.max_concurrent_scoring,
        )
        async with asyncio.TaskGroup() as group:
            for _ in range(workers):
                group.create_task(worker())
        if any(result is None for result in results):
            raise RuntimeError("OPD pipeline did not finish every prompt.")
        return [sample for result in results if result is not None for sample in result]

    # ------------------------------------------------------------------
    # Backend metrics, checkpointing, and reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _remote_metrics(*responses: Any) -> dict[str, float]:
        """Normalize finite backend loss and gradient metrics."""

        metrics = {}
        for response in responses:
            raw = getattr(response, "metrics", None)
            if not isinstance(raw, Mapping):
                continue
            for key, value in raw.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    metrics[str(key)] = float(value)
        for key in ("loss_mean", "loss", "loss:mean", "scalar/loss_mean"):
            if key in metrics:
                metrics["train/loss"] = metrics[key]
                break
        for key in ("grad_norm", "gradient_norm"):
            if key in metrics:
                metrics["optim/grad_norm"] = metrics[key]
                break
        return metrics

    async def _save_state_async(self) -> str:
        """Save student weights and optimizer state at the current step."""

        path = await save_checkpoint_async(
            self._training_client,
            "opd",
            self.state.global_step,
            error_method="save_state",
        )
        self.state.checkpoint_path = path
        return path

    def _log(self, metrics: dict[str, float]) -> None:
        """Accumulate OPD metrics and flush at the configured interval."""

        self._metrics_api.record(metrics, step=self.state.global_step)

    def _flush_log(self) -> None:
        """Publish and clear the current OPD metric window."""

        self._metrics_api.flush(step=self.state.global_step)

    def _finish_run(self, state: str, error: str | None = None) -> None:
        """Close SwanLab once and clear the owned run handle."""

        self._metrics_api.finish(state=state, error=error)

    async def _step_async(self, rows: Sequence[Mapping[str, Any]]) -> OPDResult:
        """Execute one fresh-rollout OPD update without taking the public lock."""

        started = time.perf_counter()
        prompts = await asyncio.to_thread(self._prompts, rows)
        prepared = time.perf_counter()
        if self._student_sampler is None:
            self._student_sampler = await call_async(
                self._training_client,
                "save_weights_and_get_sampling_client",
            )
        snapshotted = time.perf_counter()
        version = self.state.behavior_policy_version
        scored = await self._rollout_and_score(prompts, self._student_sampler)
        referenced = time.perf_counter()
        datums = [sample.datum for sample in scored]
        # A failed/cancelled backward can leave partial remote gradients. Keep
        # the trainer unusable until both backward and optimizer have succeeded.
        self._requires_restart = True
        forward = await call_async(
            self._training_client,
            "forward_backward",
            datums,
            loss_fn="importance_sampling",
            auto_shift=False,
        )
        forwarded = time.perf_counter()
        # This dependency is deliberate: optim must never follow failed backward.
        optim = await call_async(
            self._training_client,
            "optim_step",
            optimizer_params(self.args, trio.AdamParams),
        )
        optimized = time.perf_counter()
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self._sampling_seed_offset += len(prompts)
        self._student_sampler = None
        self._requires_restart = False
        input_tokens = sum(datum.model_input.length for datum in datums)
        completion_tokens = sum(sample.tokens for sample in scored)
        metrics = self._remote_metrics(forward, optim)
        metrics.update(
            {
                "optim/lr": self.args.learning_rate,
                "data/prompts": float(len(prompts)),
                "data/completions": float(len(datums)),
                "data/input_tokens": float(input_tokens),
                "data/completion_tokens": float(completion_tokens),
                "distill/reverse_kl_sampled": sum(
                    sample.student_sum - sample.teacher_sum for sample in scored
                )
                / completion_tokens,
                "distill/advantage_mean": sum(
                    sample.advantage_sum for sample in scored
                )
                / completion_tokens,
                "distill/student_logprob": sum(
                    sample.student_sum for sample in scored
                )
                / completion_tokens,
                "distill/teacher_logprob": sum(
                    sample.teacher_sum for sample in scored
                )
                / completion_tokens,
                "policy/behavior_version": float(version),
                "time/prepare": prepared - started,
                "time/snapshot": snapshotted - prepared,
                "time/rollout_teacher": referenced - snapshotted,
                "time/forward_backward": forwarded - referenced,
                "time/optim": optimized - forwarded,
            }
        )
        checkpoint_path = None
        save_started = time.perf_counter()
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            checkpoint_path = await self._save_state_async()
        metrics["time/checkpoint"] = time.perf_counter() - save_started
        metrics["time/step"] = time.perf_counter() - started
        metrics["time/tokens_per_second"] = input_tokens / metrics["time/step"]
        self._total_tokens += input_tokens
        self.state.metrics = metrics
        self._log(metrics)
        return OPDResult(self.state.global_step, metrics, len(datums), version, checkpoint_path)

    # ------------------------------------------------------------------
    # Public one-step update
    # ------------------------------------------------------------------

    async def step_async(self, rows: Sequence[Mapping[str, Any]]) -> OPDResult:
        """Generate fresh student completions and perform one optimizer update."""
        self._enter()
        try:
            return await self._step_async(rows)
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # Validation: sampled student versus fixed teacher
    # ------------------------------------------------------------------

    def _evaluation_batches(
        self, dataset: Iterable[Mapping[str, Any]]
    ) -> Iterable[list[Mapping[str, Any]]]:
        """Yield prompt batches using OPD evaluation limits."""

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
        """Store sampled distillation metrics without changing policy version."""

        values = {str(name): float(value) for name, value in metrics.items()}
        self._last_eval_step = self.state.global_step
        self.state.metrics.update(values)
        self._metrics_api.publish(values, step=self.state.global_step)
        return values

    async def _evaluate_async(
        self,
        dataset: Iterable[Mapping[str, Any]],
    ) -> dict[str, float]:
        """Measure sampled student/teacher divergence without an optimizer update."""

        sampler = await call_async(
            self._training_client, "save_weights_and_get_sampling_client"
        )
        prompt_count = 0
        completion_count = 0
        token_count = 0
        student_sum = 0.0
        teacher_sum = 0.0
        advantage_sum = 0.0
        for rows in self._evaluation_batches(dataset):
            prompts = await asyncio.to_thread(self._prompts, rows)
            scored = await self._rollout_and_score(prompts, sampler)
            prompt_count += len(prompts)
            completion_count += len(scored)
            token_count += sum(sample.tokens for sample in scored)
            student_sum += sum(sample.student_sum for sample in scored)
            teacher_sum += sum(sample.teacher_sum for sample in scored)
            advantage_sum += sum(sample.advantage_sum for sample in scored)
        if prompt_count == 0 or token_count == 0:
            raise ValueError("valid_dataset must contain trainable OPD prompts.")
        metrics = {
            "eval/reverse_kl_sampled": (student_sum - teacher_sum) / token_count,
            "eval/student_logprob": student_sum / token_count,
            "eval/teacher_logprob": teacher_sum / token_count,
            "eval/advantage_mean": advantage_sum / token_count,
            "eval/prompts": float(prompt_count),
            "eval/completions": float(completion_count),
            "eval/tokens": float(token_count),
        }
        return self._publish_evaluation(metrics)

    async def evaluate_async(
        self,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Evaluate the current student against the fixed teacher."""

        dataset = valid_dataset if valid_dataset is not None else self.valid_dataset
        if dataset is None:
            raise ValueError("valid_dataset is required for evaluate_async().")
        self._enter()
        try:
            return await self._evaluate_async(dataset)
        finally:
            self._busy = False

    def evaluate(
        self,
        valid_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Synchronous entry point for :meth:`evaluate_async`."""

        return self._run_sync(self.evaluate_async, valid_dataset)

    # ------------------------------------------------------------------
    # Full dataset training and synchronous facades
    # ------------------------------------------------------------------

    async def train_async(self) -> OPDState:
        """Consume one pass of the dataset, with strictly fresh rollouts per update."""
        self._enter()
        started = time.perf_counter()
        initial_step, initial_tokens = self.state.global_step, self._total_tokens
        progress = tqdm(
            total=self.args.max_steps,
            initial=initial_step,
            desc="OPD train",
            unit="step",
        )
        try:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for train_async().")
            if (
                self.valid_dataset is not None
                and self.args.evaluation.before_train
                and self._last_eval_step is None
            ):
                await self._evaluate_async(self.valid_dataset)
            iterator = iter(self.train_dataset)
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = await self._step_async(rows)
                progress.set_postfix(
                    kl=f"{result.metrics['distill/reverse_kl_sampled']:.4f}",
                    refresh=False,
                )
                progress.update(1)
                if (
                    self.valid_dataset is not None
                    and self.args.evaluation.every_steps > 0
                    and self.state.global_step % self.args.evaluation.every_steps == 0
                    and self._last_eval_step != self.state.global_step
                ):
                    await self._evaluate_async(self.valid_dataset)
            if self.state.global_step > initial_step:
                if not self.args.save_every or self.state.global_step % self.args.save_every:
                    await self._save_state_async()
                if self._metrics_api.enabled:
                    elapsed = time.perf_counter() - started
                    self._flush_log()
                    self._metrics_api.publish(
                        {
                            "time/train_runtime": elapsed,
                            "time/train_steps_per_second": (
                                self.state.global_step - initial_step
                            )
                            / elapsed,
                            "time/train_tokens_per_second": (
                                self._total_tokens - initial_tokens
                            )
                            / elapsed,
                        },
                        step=self.state.global_step,
                    )
                    self._metrics_api.log_text(
                        key="checkpoint/path",
                        data=self.state.checkpoint_path,
                        step=self.state.global_step,
                    )
            if (
                self.valid_dataset is not None
                and self.args.evaluation.after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self._evaluate_async(self.valid_dataset)
        except BaseException as error:
            self._finish_run("crashed", str(error))
            raise
        else:
            self._flush_log()
            self._finish_run("success")
        finally:
            self._busy = False
            progress.close()
        return self.state

    def _run_sync(self, method: Any, *args: Any) -> Any:
        """Run an async public entry point when no event loop is active."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(method(*args))
        raise RuntimeError(
            "Use await step_async()/train_async()/evaluate_async() inside an existing event loop."
        )

    def step(self, rows: Sequence[Mapping[str, Any]]) -> OPDResult:
        """Synchronous entry point for step_async()."""
        return self._run_sync(self.step_async, rows)

    def train(self) -> OPDState:
        """Synchronous entry point for train_async()."""
        return self._run_sync(self.train_async)
