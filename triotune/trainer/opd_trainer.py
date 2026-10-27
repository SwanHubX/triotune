"""PyTRIO OPD: student rollouts, routed teacher scores, remote policy updates.

Only sampled-token logprobs are transferred. This implements the local
reverse-KL policy-gradient surrogate, not full-vocabulary KL distillation.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pytrio as trio
from tqdm import tqdm

from ..api.cli_args import OPDConfig
from ..api.client_api import call_async, optimizer_params
from ..dataset import prepare_prompt
from ..utils.checkpoint import TrainingCheckpoint, save_checkpoint_async, training_limit
from ..utils.dataloader import iter_batches, num_batches
from ..utils.metrics import MetricsAPI
from ..utils.token_usage import compute_train_tokens
from ..utils.updates import run_update_async
from .opd.contracts import (
    OPDPrompt,
    OPDResult,
    OPDState,
    OPDTeacherGroup,
    OPDTeacherSpec,
    OPDTrainingSample,
)
from .opd.execution import OPDExecutionPlan


def _remote_metrics(*responses: Any) -> dict[str, float]:
    """Normalize finite backend loss and gradient metrics."""

    metrics: dict[str, float] = {}
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


def _importance_metrics(forward: Any, datums: Sequence[Any], cap: float | None):
    """Measure current/behavior ratios returned by the policy forward pass."""

    outputs = getattr(forward, "loss_fn_outputs", None)
    if not isinstance(outputs, list) or len(outputs) != len(datums):
        return {}
    ratios = []
    for datum, output in zip(datums, outputs, strict=True):
        if not isinstance(output, Mapping) or "logprobs" not in output:
            return {}
        fields = getattr(datum, "loss_fn_inputs", {})
        labels = getattr(fields.get("target_tokens"), "data", None)
        behavior = getattr(fields.get("logprobs"), "data", None)
        current = getattr(output["logprobs"], "data", None)
        if not all(isinstance(values, list) for values in (labels, behavior, current)):
            return {}
        if not len(labels) == len(behavior) == len(current):
            return {}
        ratios.extend(
            math.exp(min(700.0, float(now) - float(old)))
            for label, old, now in zip(labels, behavior, current, strict=True)
            if label != -100
        )
    if not ratios:
        return {}
    values = np.asarray(ratios, dtype=np.float64)
    metrics = {
        "policy/importance_ratio_mean": float(np.mean(values)),
        "policy/importance_ratio_std": float(np.std(values)),
    }
    if cap is not None:
        metrics["policy/cap_fraction"] = float(np.mean(values > cap))
    return metrics


@dataclass(slots=True)
class _OPDMetricTotals:
    """Streaming sufficient statistics for OPD trainer reporting."""

    prompt_count: int = 0
    completion_count: int = 0
    token_count: int = 0
    completion_length_min: int | None = None
    completion_length_max: int = 0
    truncated_count: int = 0
    student_logprob_sum: float = 0.0
    teacher_logprob_sum: float = 0.0
    advantage_sum: float = 0.0
    advantage_square_sum: float = 0.0
    teacher_weight_token_sum: float = 0.0
    weighted_reverse_kl_sum: float = 0.0
    weighted_reverse_kl_square_sum: float = 0.0

    def add(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prompt_count: int,
    ) -> None:
        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count < 0
        ):
            raise ValueError("prompt_count must be a nonnegative integer.")
        self.prompt_count += prompt_count
        self.completion_count += len(samples)
        self.token_count += sum(sample.tokens for sample in samples)
        lengths = [sample.tokens for sample in samples]
        if lengths:
            smallest = min(lengths)
            self.completion_length_min = (
                smallest
                if self.completion_length_min is None
                else min(self.completion_length_min, smallest)
            )
            self.completion_length_max = max(self.completion_length_max, max(lengths))
        self.truncated_count += sum(sample.rollout.truncated for sample in samples)
        self.student_logprob_sum += sum(sample.student_sum for sample in samples)
        self.teacher_logprob_sum += sum(sample.teacher_sum for sample in samples)
        self.advantage_sum += sum(sample.advantage_sum for sample in samples)
        self.advantage_square_sum += sum(
            value**2 for sample in samples for value in sample.advantages
        )
        self.teacher_weight_token_sum += sum(
            sample.teacher_weight_sum * sample.tokens for sample in samples
        )
        self.weighted_reverse_kl_sum += sum(
            sample.teacher_weight_sum * (sample.student_sum - sample.teacher_sum)
            for sample in samples
        )
        self.weighted_reverse_kl_square_sum += sum(
            (
                sample.teacher_weight_sum * (student - teacher)
            ) ** 2
            for sample in samples
            for student, teacher in zip(
                sample.rollout.behavior_logprobs,
                sample.teacher_logprobs,
                strict=True,
            )
        )

    def distillation(self, *, prefix: str) -> dict[str, float]:
        if self.token_count <= 0:
            raise ValueError("OPD metrics require at least one completion token.")
        reverse_kl = self.weighted_reverse_kl_sum / self.token_count
        return {
            f"{prefix}/reverse_kl_sampled": reverse_kl,
            f"{prefix}/reverse_kl_sampled_std": math.sqrt(
                max(
                    0.0,
                    self.weighted_reverse_kl_square_sum / self.token_count
                    - reverse_kl**2,
                )
            ),
            f"{prefix}/student_logprob": self.student_logprob_sum
            / self.token_count,
            f"{prefix}/teacher_logprob": self.teacher_logprob_sum
            / self.token_count,
            f"{prefix}/advantage_mean": self.advantage_sum / self.token_count,
            f"{prefix}/advantage_std": math.sqrt(
                max(
                    0.0,
                    self.advantage_square_sum / self.token_count
                    - (self.advantage_sum / self.token_count) ** 2,
                )
            ),
            f"{prefix}/teacher_weight_sum": self.teacher_weight_token_sum
            / self.token_count,
        }


@dataclass(frozen=True, slots=True)
class _OPDMetricReducer:
    """Build training and evaluation views from OPD sufficient statistics."""

    def training(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prompt_count: int,
        behavior_policy_version: int,
    ) -> dict[str, float]:
        totals = _OPDMetricTotals()
        totals.add(samples, prompt_count=prompt_count)
        metrics = totals.distillation(prefix="distill")
        metrics.update(
            {
                "data/prompts": float(totals.prompt_count),
                "data/completions": float(totals.completion_count),
                "data/completion_tokens": float(totals.token_count),
                "policy/behavior_version": float(behavior_policy_version),
                "policy/entropy_sampled": -totals.student_logprob_sum
                / totals.token_count,
                "rollout/completion_length_mean": totals.token_count
                / totals.completion_count,
                "rollout/completion_length_min": float(
                    totals.completion_length_min
                ),
                "rollout/completion_length_max": float(
                    totals.completion_length_max
                ),
                "rollout/truncated_fraction": totals.truncated_count
                / totals.completion_count,
            }
        )
        return metrics

    def evaluation(
        self,
        samples: Sequence[OPDTrainingSample],
        *,
        prompt_count: int,
    ) -> dict[str, float]:
        totals = _OPDMetricTotals()
        totals.add(samples, prompt_count=prompt_count)
        return self.evaluation_totals(totals)

    def evaluation_totals(self, totals: _OPDMetricTotals) -> dict[str, float]:
        metrics = totals.distillation(prefix="eval")
        metrics.update(
            {
                "eval/prompts": float(totals.prompt_count),
                "eval/completions": float(totals.completion_count),
                "eval/tokens": float(totals.token_count),
                "eval/completion_length_mean": totals.token_count
                / totals.completion_count,
                "eval/completion_length_min": float(totals.completion_length_min),
                "eval/completion_length_max": float(totals.completion_length_max),
                "eval/truncated_fraction": totals.truncated_count
                / totals.completion_count,
            }
        )
        return metrics


class OPDTrainer:
    """Distill one or more fixed teachers into a PyTRIO LoRA student.

    The legacy ``teacher_model``/``teacher_model_path`` pair configures one
    teacher. Multi-teacher runs pass ``teachers`` and optional ``teacher_groups``.
    Each row may select a group through ``teacher_group_key``; all teachers in
    that group score the exact student-sampled tokens. Student and teacher
    token-to-ID vocabularies must match.

    eval_dataset holds replayable prompts. OPDConfig.eval_steps schedules student
    rollouts and routed teacher scoring between updates. Evaluation does not
    advance the optimizer step or the behavior-policy version.

    Async methods are the primary implementation; sync methods run them with
    asyncio.run. Client construction is synchronous (use asyncio.to_thread in
    an async application). Calls on one trainer must not overlap.
    """

    def __init__(
        self,
        model: str | None = None,
        teacher_model: str | None = None,
        args: OPDConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int | None = None,
        teacher_model_path: str | None = None,
        teachers: Sequence[OPDTeacherSpec] | None = None,
        teacher_groups: Sequence[OPDTeacherGroup] | None = None,
        teacher_group_key: str = "teacher_group",
        execution_plan: OPDExecutionPlan | None = None,
    ) -> None:
        """Create clients and bind an immutable OPD algorithm execution plan."""

        self.args = args or OPDConfig()
        configured_model = self.args.model
        if model is None and configured_model is not None:
            model = configured_model.name
        if lora_rank is None:
            lora_rank = configured_model.lora_rank if configured_model is not None else 32
        if teachers is None and self.args.teacher is not None:
            if teacher_model is None:
                teacher_model = self.args.teacher.name
            if teacher_model_path is None:
                teacher_model_path = self.args.teacher.model_path
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        if (
            isinstance(lora_rank, bool)
            or not isinstance(lora_rank, int)
            or lora_rank < 1
        ):
            raise ValueError("lora_rank must be a positive integer.")
        self.model = model.strip()
        if not isinstance(teacher_group_key, str) or not teacher_group_key.strip():
            raise ValueError("teacher_group_key must be a non-empty string.")
        self.teacher_group_key = teacher_group_key.strip()

        # Keep the original single-teacher constructor fully compatible while
        # making the multi-teacher form explicit and unambiguous.
        if teachers is None:
            if not isinstance(teacher_model, str) or not teacher_model.strip():
                raise ValueError("teacher_model must be a PyTRIO base-model ID.")
            teacher_specs = (
                OPDTeacherSpec(
                    teacher_id="default",
                    model=teacher_model,
                    model_path=teacher_model_path,
                ),
            )
        else:
            if teacher_model is not None or teacher_model_path is not None:
                raise ValueError(
                    "Pass either teacher_model/teacher_model_path or teachers, "
                    "not both."
                )
            if (
                isinstance(teachers, (str, bytes))
                or not isinstance(teachers, Sequence)
                or not teachers
                or any(not isinstance(spec, OPDTeacherSpec) for spec in teachers)
            ):
                raise ValueError(
                    "teachers must be a non-empty sequence of OPDTeacherSpec."
                )
            teacher_specs = tuple(teachers)
        teacher_ids = [spec.teacher_id for spec in teacher_specs]
        if len(set(teacher_ids)) != len(teacher_ids):
            raise ValueError("Teacher IDs must be unique.")

        if teacher_groups is None:
            group_specs = (
                OPDTeacherGroup(
                    name="default",
                    teachers={teacher_id: 1.0 for teacher_id in teacher_ids},
                ),
            )
        else:
            if (
                isinstance(teacher_groups, (str, bytes))
                or not isinstance(teacher_groups, Sequence)
                or not teacher_groups
                or any(
                    not isinstance(group, OPDTeacherGroup)
                    for group in teacher_groups
                )
            ):
                raise ValueError(
                    "teacher_groups must be a non-empty sequence of "
                    "OPDTeacherGroup."
                )
            group_specs = tuple(teacher_groups)
        group_names = [group.name for group in group_specs]
        if len(set(group_names)) != len(group_names):
            raise ValueError("Teacher group names must be unique.")
        unknown_ids = {
            teacher_id
            for group in group_specs
            for teacher_id in group.teachers
            if teacher_id not in teacher_ids
        }
        if unknown_ids:
            names = ", ".join(sorted(unknown_ids))
            raise ValueError(f"Teacher groups reference unknown IDs: {names}.")

        self.teacher_specs = teacher_specs
        self.teacher_groups = {
            group.name: dict(group.teachers) for group in group_specs
        }
        if "default" in self.teacher_groups:
            self.default_teacher_group: str | None = "default"
        elif len(group_specs) == 1:
            self.default_teacher_group = group_specs[0].name
        else:
            self.default_teacher_group = None
        # Retain the old public attributes for single-teacher integrations.
        self.teacher_model = (
            teacher_specs[0].model if len(teacher_specs) == 1 else None
        )
        self.teacher_model_path = (
            teacher_specs[0].model_path if len(teacher_specs) == 1 else None
        )
        if execution_plan is not None and not isinstance(
            execution_plan, OPDExecutionPlan
        ):
            raise TypeError("execution_plan must be an OPDExecutionPlan.")
        self.execution_plan = execution_plan or OPDExecutionPlan.from_config(
            self.args
        )
        self._objective = self.execution_plan.objective
        self._metric_reducer = _OPDMetricReducer()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = OPDState()
        self._last_eval_step: int | None = None
        self._checkpoint = TrainingCheckpoint(
            self.args, "opd", train_dataset, eval_dataset,
            identity={"model": self.model, "lora_rank": lora_rank,
                      "teachers": [asdict(spec) for spec in teacher_specs],
                      "teacher_groups": self.teacher_groups, "teacher_group_key": teacher_group_key},
        )
        if self._checkpoint.enabled and execution_plan is not None:
            raise ValueError("Complete checkpoints require the built-in OPD execution plan.")
        self._checkpoint.bind_loop()
        self._service_client = trio.ServiceClient()
        self._training_client = self._checkpoint.create_client(self._service_client,
            base_model=self.model, rank=lora_rank, seed=self.args.seed,
        )
        self._tokenizer = self._training_client.get_tokenizer()
        student_vocab = self._tokenizer.get_vocab()
        self._teacher_clients: dict[str, Any] = {}
        for spec in self.teacher_specs:
            client = self._service_client.create_sampling_client(
                base_model=spec.model,
                model_path=spec.model_path,
            )
            if student_vocab != client.get_tokenizer().get_vocab():
                raise ValueError(
                    "OPD requires identical student and teacher token-to-ID "
                    f"vocabularies; teacher {spec.teacher_id!r} is incompatible."
                )
            self._teacher_clients[spec.teacher_id] = client
        # Preserve the private single-teacher handle used by older tests and
        # extensions; new execution always consumes the ID-to-client mapping.
        self._teacher_client = next(iter(self._teacher_clients.values()))
        self._student_sampler: Any = None
        self._busy = False
        self._requires_restart = False
        self._sampling_seed_offset = 0
        self._total_tokens = 0
        self._sample_input_tokens = 0
        self._sample_output_tokens = 0
        self._metrics_api = MetricsAPI.from_config(
            self.args,
            default_name=f"{self.model.rsplit('/', 1)[-1]}-opd",
            metadata={
                "model": self.model,
                "teacher_model": self.teacher_model,
                "teacher_model_path": self.teacher_model_path,
                "teachers": [
                    {
                        "teacher_id": spec.teacher_id,
                        "model": spec.model,
                        "model_path": spec.model_path,
                    }
                    for spec in self.teacher_specs
                ],
                "teacher_groups": self.teacher_groups,
                "teacher_group_key": self.teacher_group_key,
                "lora_rank": lora_rank,
                "sampling_temperature": 1.0,
                "sampling_top_p": 1.0,
                "sampling_top_k": -1,
            },
            logging_step=self.args.logging_step,
        )

        self._checkpoint.restore(self)

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

    def _prompts(self, rows: Sequence[Mapping[str, Any]]) -> list[OPDPrompt]:
        """Render prompt rows and attach immutable teacher-group routing."""

        if not rows:
            raise ValueError("An OPD step requires at least one prompt.")
        prompts = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"Row {index} must be a mapping.")
            route = row.get(self.teacher_group_key, self.default_teacher_group)
            if route is None:
                raise ValueError(
                    f"Row {index} must provide {self.teacher_group_key!r} because "
                    "multiple teacher groups are configured without a default."
                )
            if not isinstance(route, str) or not route.strip():
                raise ValueError(
                    f"Row {index} {self.teacher_group_key!r} must be a non-empty "
                    "string."
                )
            route = route.strip()
            if route not in self.teacher_groups:
                raise ValueError(f"Row {index} uses unknown teacher group {route!r}.")
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
            prompts.append(OPDPrompt(prompt.text, prompt.tokens, route))
        return prompts

    async def _rollout_and_score(
        self, prompts: list[OPDPrompt], sampler: Any
    ) -> list[OPDTrainingSample]:
        """Delegate the complete algorithm transaction to the bound plan."""

        return await self.execution_plan.materialize(
            prompts,
            sampler=sampler,
            teacher_clients=self._teacher_clients,
            teacher_groups=self.teacher_groups,
            config=self.args,
            behavior_policy_version=self.state.behavior_policy_version,
            seed_offset=self._sampling_seed_offset,
            sampling_params_factory=trio.SamplingParams,
            model_input_factory=trio.ModelInput.from_ints,
        )

    # ------------------------------------------------------------------
    # Backend metrics, checkpointing, and reporting
    # ------------------------------------------------------------------

    async def _save_state_async(self) -> str:
        """Save student weights and optimizer state at the current step."""

        path = await save_checkpoint_async(
            self._training_client,
            "opd",
            self.state.global_step,
            error_method="save_state",
        )
        self.state.checkpoint_path = path
        self._checkpoint.remote_saved(self, publish=False)
        return path

    def _log(self, metrics: dict[str, float]) -> None:
        """Accumulate OPD metrics and flush at the configured interval."""

        token_count = metrics.get("data/completion_tokens", 1.0)
        weights = {
            name: token_count
            for name in metrics
            if name.startswith(("distill/", "policy/"))
        }
        if "train/loss" in metrics:
            weights["train/loss"] = metrics.get("token_count", token_count)
        self._metrics_api.record(
            metrics,
            step=self.state.global_step,
            weights=weights,
        )

    def _flush_log(self) -> None:
        """Publish and clear the current OPD metric window."""

        self._metrics_api.flush(step=self.state.global_step)

    def _finish_run(self, state: str, error: str | None = None, *, flush: bool = True) -> None:
        """Close SwanLab once and clear the owned run handle."""

        self._metrics_api.finish(state=state, error=error, flush=flush)

    async def _step_async(self, rows: Sequence[Mapping[str, Any]]) -> OPDResult:
        """Execute one fresh-rollout OPD update without taking the public lock."""

        self._checkpoint.require_training_loop()
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
        update = await run_update_async(
            self._training_client,
            self._objective,
            datums,
            optimizer_params(self.args, trio.AdamParams),
        )
        self.state.global_step += 1
        self.state.optimizer_steps += 1
        self.state.behavior_policy_version += 1
        self._sampling_seed_offset += len(prompts)
        self._student_sampler = None
        self._requires_restart = False
        input_tokens = compute_train_tokens(datums)
        sample_input_tokens = sum(len(prompt.tokens) for prompt in prompts)
        sample_output_tokens = sum(sample.tokens for sample in scored)
        metrics = _remote_metrics(update.forward, update.optimizer)
        metrics.update(
            _importance_metrics(
                update.forward,
                datums,
                self.args.importance_ratio_cap,
            )
        )
        metrics.update(
            self._metric_reducer.training(
                scored,
                prompt_count=len(prompts),
                behavior_policy_version=version,
            )
        )
        metrics.update(
            {
                "optim/lr": self.args.learning_rate,
                "data/input_tokens": float(input_tokens),
                "time/prepare": prepared - started,
                "time/snapshot": snapshotted - prepared,
                "time/rollout_teacher": referenced - snapshotted,
                "time/forward_backward": update.forward_seconds,
                "time/optim": update.optimizer_seconds,
            }
        )
        if self.args.importance_ratio_cap is not None:
            metrics["policy/importance_ratio_cap"] = float(
                self.args.importance_ratio_cap
            )
        checkpoint_path = None
        self._total_tokens += input_tokens
        self._sample_input_tokens += sample_input_tokens
        self._sample_output_tokens += sample_output_tokens
        metrics.update(
            {
                "usage/train_tokens": float(input_tokens),
                "usage/train_tokens_total": float(self._total_tokens),
                "usage/sample_input_tokens": float(sample_input_tokens),
                "usage/sample_output_tokens": float(sample_output_tokens),
                "usage/sample_input_tokens_total": float(
                    self._sample_input_tokens
                ),
                "usage/sample_output_tokens_total": float(
                    self._sample_output_tokens
                ),
            }
        )
        self.state.metrics = metrics
        save_started = time.perf_counter()
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            checkpoint_path = await self._save_state_async()
        metrics["time/checkpoint"] = time.perf_counter() - save_started
        metrics["time/step"] = time.perf_counter() - started
        metrics["time/tokens_per_second"] = input_tokens / metrics["time/step"]
        self._log(metrics)
        self._checkpoint.publish(self)
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

        config = self.args
        batch_size = config.eval_batch_size or config.batch_size
        yield from iter_batches(dataset, batch_size, max_batches=config.eval_max_batches)

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
        totals = _OPDMetricTotals()
        batch_size = self.args.eval_batch_size or self.args.batch_size
        for rows in tqdm(
            self._evaluation_batches(dataset),
            total=num_batches(
                dataset,
                batch_size,
                max_batches=self.args.eval_max_batches,
            ),
            desc="OPD eval",
            unit="batch",
        ):
            prompts = await asyncio.to_thread(self._prompts, rows)
            scored = await self._rollout_and_score(prompts, sampler)
            totals.add(scored, prompt_count=len(prompts))
        if totals.prompt_count == 0 or totals.token_count == 0:
            raise ValueError("eval_dataset must contain trainable OPD prompts.")
        metrics = self._metric_reducer.evaluation_totals(totals)
        return self._publish_evaluation(metrics)

    async def evaluate_async(
        self,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Evaluate the current student against the fixed teacher."""

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("eval_dataset is required for evaluate_async().")
        self._enter()
        try:
            return await self._evaluate_async(dataset)
        finally:
            self._busy = False

    def evaluate(
        self,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Synchronous entry point for :meth:`evaluate_async`."""

        return self._run_sync(self.evaluate_async, eval_dataset)

    # ------------------------------------------------------------------
    # Full dataset training and synchronous facades
    # ------------------------------------------------------------------

    async def train_async(self, *, stop_after_step: int | None = None) -> OPDState:
        """Consume one pass of the dataset, with strictly fresh rollouts per update."""
        limit, paused = training_limit(self.args, self.state.global_step, stop_after_step)
        self._checkpoint.begin()
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
                self._checkpoint.evaluation_due(self, self._last_eval_step)
            ):
                await self._evaluate_async(self.eval_dataset)
            batches = iter_batches(self._checkpoint.rows(self), self.args.batch_size)
            while limit is None or self.state.global_step < limit:
                rows = next(batches, None)
                if not rows:
                    break
                result = await self._step_async(rows)
                progress.set_postfix(
                    {
                        label: f"{result.metrics[name]:.4f}"
                        for label, name in (
                            ("loss", "train/loss"),
                            ("reverse_kl", "distill/reverse_kl_sampled"),
                            ("grad_norm", "optim/grad_norm"),
                            ("tokens", "usage/train_tokens_total"),
                        )
                        if name in result.metrics
                    },
                    refresh=False,
                )
                progress.update(1)
                if (
                    self.eval_dataset is not None
                    and self.args.eval_steps > 0
                    and self.state.global_step % self.args.eval_steps == 0
                    and self._last_eval_step != self.state.global_step
                ):
                    await self._evaluate_async(self.eval_dataset)
                    self._checkpoint.publish(self)
            if self.state.global_step > initial_step:
                if not self.args.save_every or self.state.global_step % self.args.save_every:
                    await self._save_state_async()
                elapsed = time.perf_counter() - started
                summary = {
                    "time/train_runtime": elapsed,
                    "time/train_steps_per_second": (
                        self.state.global_step - initial_step
                    )
                    / elapsed,
                    "time/train_tokens_per_second": (
                        self._total_tokens - initial_tokens
                    )
                    / elapsed,
                    "usage/train_tokens_total": float(self._total_tokens),
                    "usage/sample_input_tokens_total": float(
                        self._sample_input_tokens
                    ),
                    "usage/sample_output_tokens_total": float(
                        self._sample_output_tokens
                    ),
                }
                self.state.metrics.update(summary)
                if not paused:
                    self._flush_log()
                self._metrics_api.publish(summary, step=self.state.global_step)
                self._metrics_api.log_text(
                    key="checkpoint/path",
                    data=self.state.checkpoint_path,
                    step=self.state.global_step,
                )
            if (
                self.eval_dataset is not None
                and not paused
                and self.args.eval_after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self._evaluate_async(self.eval_dataset)
            if self.state.checkpoint_path is not None:
                self._checkpoint.publish(self)
        except BaseException as error:
            self._finish_run("crashed", str(error))
            raise
        else:
            if not paused:
                self._flush_log()
            self._finish_run("success", flush=not paused)
        finally:
            self._busy = False
            self._checkpoint.end()
            progress.close()
        return self.state

    def _run_sync(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        """Run an async public entry point when no event loop is active."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(method(*args, **kwargs))
        raise RuntimeError(
            "Use await step_async()/train_async()/evaluate_async() inside an existing event loop."
        )

    def step(self, rows: Sequence[Mapping[str, Any]]) -> OPDResult:
        """Synchronous entry point for step_async()."""
        return self._run_sync(self.step_async, rows)

    def train(self, *, stop_after_step: int | None = None) -> OPDState:
        """Synchronous entry point for train_async()."""
        return self._run_sync(self.train_async, stop_after_step=stop_after_step)
