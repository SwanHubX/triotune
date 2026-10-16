"""PyTRIO OPD: student rollouts, routed teacher scores, remote policy updates.

Only sampled-token logprobs are transferred. This implements the local
reverse-KL policy-gradient surrogate, not full-vocabulary KL distillation.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytrio as trio
from tqdm import tqdm

from ..api.cli_args import OPDConfig
from ..api.client_api import call_async, optimizer_params
from ..dataset import prepare_prompt
from ..utils.checkpoint import save_checkpoint_async
from ..utils.dataloader import iter_batches
from ..utils.metrics import MetricsAPI
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
from .opd.metrics import OPDMetricTotals, remote_metrics


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
        model: str,
        teacher_model: str | None = None,
        args: OPDConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
        teacher_model_path: str | None = None,
        teachers: Sequence[OPDTeacherSpec] | None = None,
        teacher_groups: Sequence[OPDTeacherGroup] | None = None,
        teacher_group_key: str = "teacher_group",
        execution_plan: OPDExecutionPlan | None = None,
    ) -> None:
        """Create clients and bind an immutable OPD algorithm execution plan."""

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
        self.args = args or OPDConfig()
        if execution_plan is not None and not isinstance(
            execution_plan, OPDExecutionPlan
        ):
            raise TypeError("execution_plan must be an OPDExecutionPlan.")
        self.execution_plan = execution_plan or OPDExecutionPlan.from_config(
            self.args
        )
        self._objective = self.execution_plan.objective
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.state = OPDState()
        self._last_eval_step: int | None = None
        self._service_client = trio.ServiceClient()
        self._training_client = self._service_client.create_lora_training_client(
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

    @staticmethod
    def _remote_metrics(*responses: Any) -> dict[str, float]:
        """Normalize finite backend loss and gradient metrics."""

        return remote_metrics(*responses)

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
        update = await run_update_async(
            self._training_client,
            self._objective,
            datums,
            optimizer_params(self.args, trio.AdamParams),
        )
        self.state.global_step += 1
        self.state.behavior_policy_version += 1
        self._sampling_seed_offset += len(prompts)
        self._student_sampler = None
        self._requires_restart = False
        input_tokens = sum(datum.model_input.length for datum in datums)
        metrics = self._remote_metrics(update.forward, update.optimizer)
        metrics.update(
            self.execution_plan.metrics.training(
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
        totals = OPDMetricTotals()
        for rows in self._evaluation_batches(dataset):
            prompts = await asyncio.to_thread(self._prompts, rows)
            scored = await self._rollout_and_score(prompts, sampler)
            totals.add(scored, prompt_count=len(prompts))
        if totals.prompt_count == 0 or totals.token_count == 0:
            raise ValueError("eval_dataset must contain trainable OPD prompts.")
        metrics = self.execution_plan.metrics.evaluation_totals(totals)
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
                self.eval_dataset is not None
                and self.args.eval_before_train
                and self._last_eval_step is None
            ):
                await self._evaluate_async(self.eval_dataset)
            batches = iter_batches(self.train_dataset, self.args.batch_size)
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = next(batches, None)
                if not rows:
                    break
                result = await self._step_async(rows)
                progress.set_postfix(
                    kl=f"{result.metrics['distill/reverse_kl_sampled']:.4f}",
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
                self.eval_dataset is not None
                and self.args.eval_after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self._evaluate_async(self.eval_dataset)
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
