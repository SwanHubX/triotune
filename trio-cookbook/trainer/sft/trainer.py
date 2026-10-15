"""Minimal text-only SFT trainer using PyTRIO's remote cross-entropy loss."""

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

from ...utils.client import call_async, optimizer_params, submit_async, wait_async
from ..common.logging import finish_run, start_run
from ..dataset import SupervisedExample, prepare_sft
from .config import SFTConfig
from .types import SFTState


class SFTTrainer:
    """Train a PyTRIO LoRA model from ``text``, ``messages``, ``prompt``/``completion``, or ``input_ids`` rows.

    Prompt-completion rows train on the completion by default; other rows train
    on all next tokens. ``assistant_only_loss`` requires a chat template that
    returns assistant token masks. Packing and multimodal inputs are unsupported.
    ``step()`` and ``train()`` are coroutines; await them from an event loop.
    """

    def __init__(
        self,
        model: str,
        args: SFTConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        *,
        lora_rank: int = 32,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        self.model = model.strip()
        self.args = args or SFTConfig()
        self.train_dataset = train_dataset
        self._client = trio.ServiceClient().create_lora_training_client(
            base_model=self.model, rank=lora_rank, seed=self.args.seed
        )
        self._tokenizer = self._client.get_tokenizer()
        self.state = SFTState()
        self._run = None
        if self.args.report_to == "swanlab":
            self._run = start_run(
                self.args,
                default_name=f"{self.model.rsplit('/', 1)[-1]}-sft",
                metadata={"model": self.model, "lora_rank": lora_rank},
            )

    def _prepare_example(self, row: Mapping[str, Any]) -> SupervisedExample:
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

    async def step(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Submit one batch and wait for it; train() pipelines multiple batches."""
        return await (await self._submit_step(rows, self.state.global_step + 1))

    async def _submit_step(
        self, rows: Sequence[Mapping[str, Any]], step: int,
        previous: asyncio.Task[dict[str, float]] | None = None, clock: list[float] | None = None,
    ) -> asyncio.Task[dict[str, float]]:
        if not rows:
            raise ValueError("An SFT step requires at least one row.")
        started = time.perf_counter()
        examples = [self._prepare_example(row) for row in rows]
        datums = [
            trio.Datum(
                model_input=trio.ModelInput.from_ints(example.model_input),
                loss_fn_inputs={"target_tokens": example.target_tokens},
            )
            for example in examples
        ]
        # Await submission only. The SDK orders FB -> optim -> save -> next FB.
        forward_future = await submit_async(self._client, "forward_backward", datums, loss_fn="cross_entropy", auto_shift=False)
        optim_future = await submit_async(self._client, "optim_step", optimizer_params(self.args, trio.AdamParams))
        save_future = None
        if self.args.save_every and step % self.args.save_every == 0:
            save_future = await submit_async(self._client, "save_state", name=f"sft-step-{step:08d}")
        token_count = sum(
            target != -100
            for example in examples
            for target in example.target_tokens
        )

        async def collect() -> dict[str, float]:
            forward = await wait_async(forward_future)
            optim = await wait_async(optim_future)
            latency = time.perf_counter() - started
            # Results may arrive out of order; publish state/logs in batch order.
            if previous is not None:
                await previous
            metrics = {
                str(key): float(value)
                for result in (forward, optim)
                for key, value in getattr(result, "metrics", {}).items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
            if "loss_mean" in metrics:
                metrics["train/loss"] = metrics["loss_mean"]
            now = time.perf_counter()
            metrics.update({
                "train/tokens": float(token_count),
                "train/learning_rate": self.args.learning_rate,
                "time/request_latency": latency,
                "time/step": now - (clock[1] if clock is not None else started),
            })
            if clock is not None:
                metrics["time/train"] = now - clock[0]
                clock[1] = now
            if save_future is not None:
                self.state.checkpoint_path = (await wait_async(save_future)).path
            self.state.global_step = step
            self.state.metrics = metrics
            if self._run is not None:
                self._run.log({**metrics, "train/step": step}, step=step)
            return metrics

        return asyncio.create_task(collect())

    async def _save_state(self) -> None:
        saved = await call_async(self._client, "save_state", name=f"sft-step-{self.state.global_step:08d}")
        self.state.checkpoint_path = saved.path

    async def train(self) -> SFTState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="SFT train", unit="step")
        pending: list[asyncio.Task[dict[str, float]]] = []
        previous = None
        submitted = self.state.global_step
        clock = [time.perf_counter()] * 2

        def report(task: asyncio.Task[dict[str, float]]) -> None:
            if not task.cancelled() and task.exception() is None:
                progress.set_postfix({"loss": task.result().get("train/loss")}, refresh=False)
                progress.update(1)

        try:
            while self.args.max_steps is None or submitted < self.args.max_steps:
                if len(pending) >= self.args.max_in_flight:
                    await pending.pop(0)
                while pending and pending[0].done():
                    pending.pop(0).result()
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                submitted += 1
                previous = await self._submit_step(rows, submitted, previous, clock)
                previous.add_done_callback(report)
                pending.append(previous)
            await asyncio.gather(*pending)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                await self._save_state()
            if self._run is not None:
                if self.state.checkpoint_path is not None:
                    self._run.log_text(
                        key="checkpoint/path", data=self.state.checkpoint_path, step=self.state.global_step
                    )
                self._run.finish()
        except BaseException as error:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if self._run is not None:
                finish_run(self._run, state="crashed", error=str(error))
            raise
        finally:
            progress.close()
        return self.state
