"""Minimal text-only SFT trainer using PyTRIO's remote cross-entropy loss."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pytrio as trio
from tqdm import tqdm

from ..api.cli_args import SFTConfig
from ..api.client_api import call_async, optimizer_params, submit_async, wait_async
from ..api.loss_api import BuiltinPyTrioObjective, LossSpec, SupervisedLossInput
from ..dataset import SupervisedExample, prepare_sft
from ..utils.dataloader import iter_batches
from ..utils.metrics import MetricsAPI
from ..utils.token_usage import compute_train_tokens
from .sft.state import SFTState


class SFTTrainer:
    """Train a PyTRIO LoRA model from supported supervised dataset rows.

    Prompt-completion rows train on the completion by default; other rows train
    on all next tokens. Input rows may contain text, messages, a
    prompt/completion pair, or pre-tokenized IDs. ``assistant_only_loss``
    requires a chat template that returns assistant token masks. Packing and
    multimodal inputs are unsupported. Public training methods are coroutines.

    eval_dataset contains held-out rows in the same format as train_dataset.
    It must be replayable for periodic evaluation. SFTConfig.eval_steps counts
    completed optimizer updates; evaluation drains outstanding training batches
    and runs forward only, leaving optimizer state and global_step unchanged.
    """

    def __init__(
        self,
        model: str,
        args: SFTConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
    ) -> None:
        """Create the remote training client and local pipeline state."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        self.model = model.strip()
        self.args = args or SFTConfig()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._client = trio.ServiceClient().create_lora_training_client(
            base_model=self.model, rank=lora_rank, seed=self.args.seed
        )
        self._tokenizer = self._client.get_tokenizer()
        self._objective = BuiltinPyTrioObjective(LossSpec("cross_entropy"))
        self.state = SFTState()
        self._last_eval_step: int | None = None
        self._train_tokens = 0
        self._metrics_api = MetricsAPI.from_config(
            self.args,
            default_name=f"{self.model.rsplit('/', 1)[-1]}-sft",
            metadata={"model": self.model, "lora_rank": lora_rank},
        )

    # ------------------------------------------------------------------
    # Dataset conversion
    # ------------------------------------------------------------------

    def _prepare_example(self, row: Mapping[str, Any]) -> SupervisedExample:
        """Convert one supported input row through the shared dataset contract."""

        return prepare_sft(
            row,
            self._tokenizer,
            text_field=self.args.dataset_text_field,
            max_length=self.args.max_length,
            completion_only_loss=self.args.completion_only_loss,
            assistant_only_loss=self.args.assistant_only_loss,
            add_eos_token=self.args.add_eos_token,
            chat_template_kwargs=self.args.chat_template_kwargs,
        )

    def _prepare_datums(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> tuple[list[SupervisedExample], list[Any]]:
        """Prepare canonical examples and their PyTRIO cross-entropy datums."""

        examples = [self._prepare_example(row) for row in rows]
        datums = [
            SupervisedLossInput(
                model_tokens=example.model_input,
                target_tokens=example.target_tokens,
            ).to_datum()
            for example in examples
        ]
        return examples, datums

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _evaluation_batches(
        self, dataset: Iterable[Mapping[str, Any]]
    ) -> Iterable[list[Mapping[str, Any]]]:
        """Yield SFT validation batches under this trainer's evaluation limits."""

        config = self.args
        batch_size = config.eval_batch_size or config.batch_size
        yield from iter_batches(dataset, batch_size, max_batches=config.eval_max_batches)

    def _publish_evaluation(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Store and log SFT validation metrics without advancing the step."""

        values = {str(name): float(value) for name, value in metrics.items()}
        self._last_eval_step = self.state.global_step
        self.state.metrics.update(values)
        self._metrics_api.publish(values, step=self.state.global_step)
        return values

    async def evaluate(
        self,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Measure token-weighted validation loss without backward or optimizer calls."""

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("eval_dataset is required for evaluate().")
        loss_sum = 0.0
        token_count = 0
        example_count = 0
        batch_count = 0
        for rows in self._evaluation_batches(dataset):
            examples, datums = self._prepare_datums(rows)
            response = await self._objective.forward_async(self._client, datums)
            raw = getattr(response, "metrics", {})
            active_tokens = int(
                raw.get(
                    "token_count",
                    sum(
                        target != -100
                        for example in examples
                        for target in example.target_tokens
                    ),
                )
            )
            mean_loss = float(raw.get("loss_mean", raw.get("loss", math.nan)))
            if active_tokens < 1 or not math.isfinite(mean_loss):
                raise ValueError("PyTRIO forward() returned invalid validation metrics.")
            loss_sum += float(raw.get("loss_sum", mean_loss * active_tokens))
            token_count += active_tokens
            example_count += len(rows)
            batch_count += 1
        if batch_count == 0:
            raise ValueError("eval_dataset must contain at least one row.")
        metrics = {
            "eval/loss": loss_sum / token_count,
            "eval/tokens": float(token_count),
            "eval/examples": float(example_count),
            "eval/batches": float(batch_count),
        }
        return self._publish_evaluation(metrics)

    # ------------------------------------------------------------------
    # One optimizer update
    # ------------------------------------------------------------------

    async def step(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Submit one batch and wait for it; train() pipelines multiple batches."""
        return await (await self._submit_step(rows, self.state.global_step + 1))

    async def _submit_step(
        self,
        rows: Sequence[Mapping[str, Any]],
        step: int,
        previous: asyncio.Task[dict[str, float]] | None = None,
        clock: list[float] | None = None,
    ) -> asyncio.Task[dict[str, float]]:
        """Submit FB, optimizer and optional save requests in SDK order.

        Submission is separated from result collection so ``train()`` can keep
        multiple batches in flight. The returned task publishes state only
        after the previous batch task, preserving monotonically ordered logs.
        """

        if not rows:
            raise ValueError("An SFT step requires at least one row.")
        started = time.perf_counter()
        examples, datums = self._prepare_datums(rows)
        train_tokens = compute_train_tokens(datums)

        # Await submission only; result resolution happens in collect(). The
        # PyTRIO client preserves FB -> optim -> save request ordering.
        forward_future = await self._objective.submit_async(self._client, datums)
        optim_future = await submit_async(
            self._client,
            "optim_step",
            optimizer_params(self.args, trio.AdamParams),
        )
        save_future = None
        if self.args.save_every and step % self.args.save_every == 0:
            save_future = await submit_async(
                self._client,
                "save_state",
                name=f"sft-step-{step:08d}",
            )
        token_count = sum(
            target != -100
            for example in examples
            for target in example.target_tokens
        )

        async def collect() -> dict[str, float]:
            """Resolve one queued update and publish it in dataset order."""

            forward = await wait_async(forward_future)
            optim = await wait_async(optim_future)
            latency = time.perf_counter() - started
            # Results may arrive out of order; publish state/logs in batch order.
            if previous is not None:
                await previous
            self._train_tokens += train_tokens
            metrics = {
                str(key): float(value)
                for result in (forward, optim)
                for key, value in getattr(result, "metrics", {}).items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
            if "loss_mean" in metrics:
                metrics["train/loss"] = metrics["loss_mean"]
            now = time.perf_counter()
            metrics.update(
                {
                    "train/tokens": float(token_count),
                    "train/learning_rate": self.args.learning_rate,
                    "usage/train_tokens": float(train_tokens),
                    "usage/train_tokens_total": float(self._train_tokens),
                    "time/request_latency": latency,
                    "time/step": now
                    - (clock[1] if clock is not None else started),
                }
            )
            if clock is not None:
                metrics["time/train"] = now - clock[0]
                clock[1] = now
            if save_future is not None:
                self.state.checkpoint_path = (await wait_async(save_future)).path
            self.state.global_step = step
            self.state.metrics = metrics
            self._metrics_api.record(metrics, step=step)
            return metrics

        return asyncio.create_task(collect())

    async def _save_state(self) -> None:
        """Save weights and optimizer state at the current completed step."""

        saved = await call_async(
            self._client,
            "save_state",
            name=f"sft-step-{self.state.global_step:08d}",
        )
        self.state.checkpoint_path = saved.path

    # ------------------------------------------------------------------
    # Full dataset training
    # ------------------------------------------------------------------

    async def train(self) -> SFTState:
        """Consume one dataset pass while bounding outstanding remote batches."""

        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        batches = iter_batches(self.train_dataset, self.args.batch_size)
        progress = tqdm(total=self.args.max_steps, desc="SFT train", unit="step")
        pending: list[asyncio.Task[dict[str, float]]] = []
        previous = None
        submitted = self.state.global_step
        clock = [time.perf_counter()] * 2

        def report(task: asyncio.Task[dict[str, float]]) -> None:
            """Advance terminal progress after a successful batch task."""

            if not task.cancelled() and task.exception() is None:
                progress.set_postfix({"loss": task.result().get("train/loss")}, refresh=False)
                progress.update(1)

        try:
            if (
                self.eval_dataset is not None
                and self.args.eval_before_train
                and self._last_eval_step is None
            ):
                await self.evaluate()
            while self.args.max_steps is None or submitted < self.args.max_steps:
                if len(pending) >= self.args.max_in_flight:
                    await pending.pop(0)
                while pending and pending[0].done():
                    pending.pop(0).result()
                rows = next(batches, None)
                if not rows:
                    break
                submitted += 1
                previous = await self._submit_step(rows, submitted, previous, clock)
                previous.add_done_callback(report)
                pending.append(previous)
                if (
                    self.eval_dataset is not None
                    and self.args.eval_steps > 0
                    and submitted % self.args.eval_steps == 0
                    and self._last_eval_step != submitted
                ):
                    # All updates through this boundary must finish before forward
                    # evaluation reads the model; do not enqueue the next batch yet.
                    await asyncio.gather(*pending)
                    pending.clear()
                    await self.evaluate()
            await asyncio.gather(*pending)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                await self._save_state()
            if (
                self.eval_dataset is not None
                and self.args.eval_after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self.evaluate()
            if self.state.checkpoint_path is not None:
                self._metrics_api.log_text(
                    key="checkpoint/path",
                    data=self.state.checkpoint_path,
                    step=self.state.global_step,
                )
            self._metrics_api.finish()
        except BaseException as error:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._metrics_api.finish(state="crashed", error=str(error))
            raise
        finally:
            progress.close()
        return self.state
