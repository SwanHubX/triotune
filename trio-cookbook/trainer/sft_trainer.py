"""Minimal text-only SFT trainer using PyTRIO's remote cross-entropy loss."""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pytrio as trio
import swanlab
from tqdm import tqdm

from ..utils.client import call, optimizer_params
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
                name=self.args.run_name or f"{self.model.rsplit('/', 1)[-1]}-sft",
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

    def step(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
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
        forward = call(self._client, "forward_backward", datums, loss_fn="cross_entropy", auto_shift=False)
        optim = call(self._client, "optim_step", optimizer_params(self.args, trio.AdamParams))
        metrics = {
            str(key): float(value)
            for result in (forward, optim)
            for key, value in getattr(result, "metrics", {}).items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        if "loss_mean" in metrics:
            metrics["train/loss"] = metrics["loss_mean"]
        metrics.update({
            "train/tokens": float(sum(label != -100 for _, labels in examples for label in labels[1:])),
            "train/learning_rate": self.args.learning_rate,
            "time/step": time.perf_counter() - started,
        })
        self.state.global_step += 1
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            self._save_state()
        self.state.metrics = metrics
        if self._run is not None:
            self._run.log({**metrics, "train/step": self.state.global_step}, step=self.state.global_step)
        return metrics

    def _save_state(self) -> None:
        saved = call(self._client, "save_state", name=f"sft-step-{self.state.global_step:08d}")
        self.state.checkpoint_path = saved.path

    def train(self) -> SFTState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="SFT train", unit="step")
        try:
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                metrics = self.step(rows)
                progress.set_postfix({"loss": metrics["train/loss"]} if "train/loss" in metrics else {}, refresh=False)
                progress.update(1)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                self._save_state()
            if self._run is not None:
                if self.state.checkpoint_path is not None:
                    self._run.log_text(
                        key="checkpoint/path", data=self.state.checkpoint_path, step=self.state.global_step
                    )
                self._run.finish()
        except BaseException as error:
            if self._run is not None:
                self._run.finish(state="crashed", error=str(error))
            raise
        finally:
            progress.close()
        return self.state
