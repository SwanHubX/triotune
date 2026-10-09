"""Minimal text-only SFT trainer using PyTRIO's remote cross-entropy loss."""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pytrio as trio
import swanlab
from tqdm import tqdm

from ..utils.client import call_async, optimizer_params, submit_async, wait_async
from .prompts import _token_list, render_prompt
from .sft_config import SFTConfig


@dataclass(slots=True)
class SFTState:
    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


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
            self._run = swanlab.init(
                project=self.args.project,
                workspace=self.args.workspace,
                tags=self.args.tags,
                name=self.args.run_name or f"{self.model.rsplit('/', 1)[-1]}-sft",
                log_dir=self.args.log_dir,
                config={"model": self.model, "lora_rank": lora_rank, **asdict(self.args)},
            )

    def _tokens_and_labels(self, row: Mapping[str, Any]) -> tuple[list[int], list[int]]:
        args, tokenizer = self.args, self._tokenizer
        conversational = False
        if "input_ids" in row:
            tokens = _token_list(row["input_ids"], source="input_ids")
            if args.completion_only_loss and "labels" not in row:
                raise ValueError("Tokenized completion-only rows require labels with prompt tokens set to -100.")
            labels = _token_list(row.get("labels", tokens), source="labels")
        else:
            paired = "prompt" in row and "completion" in row
            if paired:
                prompt, completion = row["prompt"], row["completion"]
                conversational = isinstance(prompt, (list, tuple)) and isinstance(completion, (list, tuple))
                if conversational:
                    prompt_tokens = render_prompt(
                        prompt, tokenizer, chat_template_kwargs=args.chat_template_kwargs
                    ).tokens
                    value = list(prompt) + list(completion)
                elif isinstance(prompt, str) and isinstance(completion, str):
                    prompt_tokens = render_prompt(prompt, tokenizer).tokens
                    value = prompt + completion
                else:
                    raise TypeError("prompt and completion must both be text or chat messages.")
            elif "messages" in row:
                conversational, value = True, row["messages"]
            elif args.dataset_text_field in row:
                value = row[args.dataset_text_field]
                if not isinstance(value, str):
                    raise TypeError("The dataset text field must contain a string.")
            else:
                raise KeyError("Row needs input_ids, messages, prompt/completion, or the dataset text field.")

            if conversational and args.assistant_only_loss:
                output = tokenizer.apply_chat_template(
                    value, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_assistant_tokens_mask=True,
                    **args.chat_template_kwargs,
                )
                tokens = _token_list(output, source="chat template")
                if not isinstance(output, Mapping):
                    raise TypeError("The chat template must return input_ids and assistant_masks.")
                assistant_mask = output.get("assistant_masks")
                if hasattr(assistant_mask, "tolist"):
                    assistant_mask = assistant_mask.tolist()
                if not isinstance(assistant_mask, (list, tuple)) or (
                    len(assistant_mask) != len(tokens)
                    or not all(bit in (0, 1) for bit in assistant_mask)
                    or not any(assistant_mask)
                ):
                    raise ValueError("The chat template must return a non-empty assistant token mask.")
            else:
                tokens = render_prompt(
                    value, tokenizer, add_generation_prompt=False,
                    chat_template_kwargs=args.chat_template_kwargs,
                ).tokens
                assistant_mask = None
            if paired and tokens[: len(prompt_tokens)] != prompt_tokens:
                raise ValueError("The tokenized prompt is not a prefix of the full example.")
            if not conversational and args.add_eos_token:
                eos = getattr(tokenizer, "eos_token_id", None)
                if not isinstance(eos, int) or isinstance(eos, bool):
                    raise ValueError("add_eos_token requires tokenizer.eos_token_id.")
                if tokens[-1] != eos:
                    tokens.append(eos)
            completion_only = args.completion_only_loss
            if completion_only is None:
                completion_only = paired
            if completion_only and not paired:
                raise ValueError("completion_only_loss requires prompt/completion rows.")
            labels = [
                token if (not completion_only or index >= len(prompt_tokens))
                and (assistant_mask is None or assistant_mask[index]) else -100
                for index, token in enumerate(tokens)
            ]

        if len(tokens) != len(labels):
            raise ValueError("input_ids and labels must have the same length.")
        if any(token < 0 for token in tokens) or any(label < 0 and label != -100 for label in labels):
            raise ValueError("Token IDs must be nonnegative; only -100 may mask a label.")
        if args.assistant_only_loss and not conversational:
            raise ValueError("assistant_only_loss requires conversational data.")
        if args.max_length is not None:
            tokens, labels = tokens[: args.max_length], labels[: args.max_length]
        if len(tokens) < 2 or all(label == -100 for label in labels[1:]):
            raise ValueError("The example has no next-token labels after masking/truncation.")
        return tokens, labels

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
        examples = [self._tokens_and_labels(row) for row in rows]
        datums = [
            trio.Datum(
                model_input=trio.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": labels[1:]},
            )
            for tokens, labels in examples
        ]
        # Await submission only. The SDK orders FB -> optim -> save -> next FB.
        forward_future = await submit_async(self._client, "forward_backward", datums, loss_fn="cross_entropy", auto_shift=False)
        optim_future = await submit_async(self._client, "optim_step", optimizer_params(self.args, trio.AdamParams))
        save_future = None
        if self.args.save_every and step % self.args.save_every == 0:
            save_future = await submit_async(self._client, "save_state", name=f"sft-step-{step:08d}")
        token_count = sum(label != -100 for _, labels in examples for label in labels[1:])

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
                self._run.finish(state="crashed", error=str(error))
            raise
        finally:
            progress.close()
        return self.state
