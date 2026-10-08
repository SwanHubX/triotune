"""TRL-style DPO API using PyTRIO clients for model execution.

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
from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any

import pytrio as trio
import swanlab
from tqdm import tqdm

from ..utils.client import call, call_async, optimizer_params
from .dpo_config import DPOConfig
from .prompts import render_prompt


@dataclass(slots=True)
class DPOState:
    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(slots=True)
class DPOResult:
    step: int
    metrics: dict[str, float]
    num_pairs: int
    checkpoint_path: str | None = None


@dataclass(slots=True)
class _Preference:
    prompt: list[int]
    chosen: list[int]
    rejected: list[int]


def _completion_tokens(value: Any, prompt: Any, prompt_tokens: list[int], tokenizer: Any) -> list[int]:
    if isinstance(value, str):
        tokens = tokenizer.encode(value, add_special_tokens=False)
    elif isinstance(value, (list, tuple)) and value and all(
        isinstance(item, Integral) and not isinstance(item, bool) for item in value
    ):
        tokens = value
    elif (
        isinstance(prompt, (list, tuple))
        and isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, Mapping) for item in prompt)
        and all(isinstance(item, Mapping) for item in value)
    ):
        full = render_prompt(list(prompt) + list(value), tokenizer, add_generation_prompt=False)
        if full.tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError("Chat template does not preserve prompt tokens before the completion.")
        tokens = full.tokens[len(prompt_tokens) :]
    else:
        raise TypeError("chosen and rejected must be text, token IDs, or chat completion messages.")
    tokens = list(tokens)
    if not tokens or any(not isinstance(token, Integral) or isinstance(token, bool) for token in tokens):
        raise ValueError("A preference completion must contain integer token IDs.")
    return [int(token) for token in tokens]


def _datum(prompt: list[int], completion: list[int]) -> Any:
    """Mask prompt labels and align completion labels with next-token logits."""

    tokens = prompt + completion
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": [-100] * (len(prompt) - 1) + completion},
    )


def _reference_logprob(values: Sequence[float | None], prompt_length: int, total_length: int) -> float:
    if len(values) != total_length:
        raise ValueError("Reference sampler returned the wrong number of token logprobs.")
    completion_values = values[prompt_length:]
    if not completion_values or any(value is None or not math.isfinite(value) for value in completion_values):
        raise ValueError("Reference sampler returned missing or non-finite completion logprobs.")
    return sum(float(value) for value in completion_values if value is not None)


def _loss_fn(reference: Sequence[float], beta: float):
    """Return PyTRIO's local custom-loss callback for ordered chosen/rejected pairs."""

    def loss(datums: list[Any], logprobs: list[Any]) -> tuple[Any, dict[str, float]]:
        import torch
        import torch.nn.functional as F

        if len(logprobs) != len(datums) or len(logprobs) != len(reference) or len(logprobs) % 2:
            raise ValueError("DPO requires aligned chosen/rejected datums and reference scores.")
        scores = []
        for datum, token_logprobs in zip(datums, logprobs, strict=True):
            labels = datum.loss_fn_inputs["target_tokens"].data
            if len(labels) != token_logprobs.numel():
                raise ValueError("Policy logprobs and target tokens are not aligned.")
            mask = torch.tensor([token != -100 for token in labels], device=token_logprobs.device)
            scores.append(token_logprobs.reshape(-1)[mask].sum())
        policy_chosen = torch.stack(scores[::2])
        policy_rejected = torch.stack(scores[1::2])
        reference_chosen = policy_chosen.new_tensor(reference[::2])
        reference_rejected = policy_chosen.new_tensor(reference[1::2])
        chosen_rewards = beta * (policy_chosen - reference_chosen)
        rejected_rewards = beta * (policy_rejected - reference_rejected)
        margins = chosen_rewards - rejected_rewards
        objective = F.softplus(-margins).mean()
        metrics = {
            "train/loss": objective.item(),
            "rewards/chosen": chosen_rewards.detach().mean().item(),
            "rewards/rejected": rejected_rewards.detach().mean().item(),
            "rewards/margins": margins.detach().mean().item(),
            "rewards/accuracies": (margins.detach() > 0).float().mean().item(),
            "logps/chosen": policy_chosen.detach().mean().item(),
            "logps/rejected": policy_rejected.detach().mean().item(),
        }
        return objective, metrics

    return loss


class DPOTrainer:
    """Train a PyTRIO LoRA policy from ``prompt``, ``chosen``, ``rejected`` rows.

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
        self._run = None
        if self.args.mode != "disabled":
            self._run = swanlab.init(
                mode=self.args.mode,
                project=self.args.project,
                workspace=self.args.workspace,
                tags=self.args.tags,
                name=self.args.run_name or f"{self.model.rsplit('/', 1)[-1]}-dpo",
                log_dir=self.args.log_dir,
                config={
                    "model": self.model,
                    "ref_model": self.ref_model,
                    "lora_rank": lora_rank,
                    **asdict(self.args),
                },
            )

    def _preferences(self, rows: Sequence[Mapping[str, Any]]) -> list[_Preference]:
        if not rows:
            raise ValueError("A DPO step requires at least one preference pair.")
        pairs = []
        for index, row in enumerate(rows):
            if not all(key in row for key in ("prompt", "chosen", "rejected")):
                raise KeyError(f"Dataset row {index} needs prompt, chosen, and rejected.")
            prompt = render_prompt(row["prompt"], self._tokenizer)
            completions = []
            for key in ("chosen", "rejected"):
                tokens = _completion_tokens(row[key], row["prompt"], prompt.tokens, self._tokenizer)
                eos = getattr(self._tokenizer, "eos_token_id", None)
                if self.args.add_eos_token:
                    if not isinstance(eos, Integral) or isinstance(eos, bool):
                        raise ValueError("add_eos_token requires tokenizer.eos_token_id.")
                    if tokens[-1] != eos:
                        tokens.append(eos)
                if (
                    self.args.max_length is not None
                    and len(prompt.tokens) + len(tokens) > self.args.max_length
                ):
                    raise ValueError(f"Dataset row {index} {key} exceeds max_length={self.args.max_length}.")
                completions.append(tokens)
            pairs.append(_Preference(prompt.tokens, *completions))
        return pairs

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
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[name] = float(value)
        if "grad_norm" in metrics:
            metrics["optim/grad_norm"] = metrics["grad_norm"]
        return metrics

    def _finish_run(self, state: str, error: str | None = None) -> None:
        if self._run is not None:
            self._run.finish(state=state, error=error)
            self._run = None

    def _finish_step(self, metrics: dict[str, float], count: int, started: float) -> DPOResult:
        self.state.global_step += 1
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            self._save_state()
        metrics["time/step"] = time.perf_counter() - started
        self.state.metrics = metrics
        result = DPOResult(self.state.global_step, metrics, count, self.state.checkpoint_path)
        if self._run is not None:
            self._run.log(
                {**metrics, "trainer/step": self.state.global_step, "trainer/num_pairs": count},
                step=self.state.global_step,
            )
        return result

    def _save_state(self) -> str:
        saved = call(self._training_client, "save_state", name=f"dpo-step-{self.state.global_step:08d}")
        path = getattr(saved, "path", None)
        if not isinstance(path, str) or not path:
            raise TypeError("PyTRIO save_state() returned no checkpoint path.")
        self.state.checkpoint_path = path
        return path

    async def _save_state_async(self) -> str:
        saved = await call_async(
            self._training_client, "save_state", name=f"dpo-step-{self.state.global_step:08d}"
        )
        path = getattr(saved, "path", None)
        if not isinstance(path, str) or not path:
            raise TypeError("PyTRIO save_state_async() returned no checkpoint path.")
        self.state.checkpoint_path = path
        return path

    def step(self, rows: Sequence[Mapping[str, Any]]) -> DPOResult:
        started = time.perf_counter()
        pairs = self._preferences(rows)
        datums = self._datums(pairs)
        reference = self._reference_scores(pairs)
        forward = call(
            self._training_client,
            "forward_backward_custom",
            datums,
            _loss_fn(reference, self.args.beta),
        )
        optim = call(self._training_client, "optim_step", optimizer_params(self.args, trio.AdamParams))
        return self._finish_step(self._metrics(forward, optim), len(pairs), started)

    async def step_async(self, rows: Sequence[Mapping[str, Any]]) -> DPOResult:
        started = time.perf_counter()
        pairs = self._preferences(rows)
        datums = self._datums(pairs)
        reference = await self._reference_scores_async(pairs)
        forward = await call_async(
            self._training_client,
            "forward_backward_custom",
            datums,
            _loss_fn(reference, self.args.beta),
        )
        optim = await call_async(
            self._training_client, "optim_step", optimizer_params(self.args, trio.AdamParams)
        )
        self.state.global_step += 1
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            await self._save_state_async()
        metrics = self._metrics(forward, optim)
        metrics["time/step"] = time.perf_counter() - started
        self.state.metrics = metrics
        result = DPOResult(self.state.global_step, metrics, len(pairs), self.state.checkpoint_path)
        if self._run is not None:
            self._run.log(
                {**metrics, "trainer/step": self.state.global_step, "trainer/num_pairs": len(pairs)},
                step=self.state.global_step,
            )
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

    def train(self) -> DPOState:
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for train().")
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="DPO train", unit="step")
        try:
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = self.step(rows)
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                self._save_state()
            self._log_final_checkpoint()
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
        iterator = iter(self.train_dataset)
        progress = tqdm(total=self.args.max_steps, desc="DPO train", unit="step")
        try:
            while self.args.max_steps is None or self.state.global_step < self.args.max_steps:
                rows = list(itertools.islice(iterator, self.args.batch_size))
                if not rows:
                    break
                result = await self.step_async(rows)
                progress.set_postfix(self._progress_postfix(result), refresh=False)
                progress.update(1)
            if self.state.global_step and (
                not self.args.save_every or self.state.global_step % self.args.save_every
            ):
                await self._save_state_async()
            self._log_final_checkpoint()
        except BaseException as error:
            self._finish_run("crashed", str(error))
            raise
        else:
            self._finish_run("success")
        finally:
            progress.close()
        return self.state
