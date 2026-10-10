"""Preference validation, token alignment, and reference score reduction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import pytrio as trio

from ..common.prompts import render_prompt
from .config import DPOConfig


@dataclass(slots=True)
class Preference:
    prompt: list[int]
    chosen: list[int]
    rejected: list[int]


def completion_tokens(
    value: Any,
    prompt: Any,
    prompt_tokens: list[int],
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
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
        full = render_prompt(
            list(prompt) + list(value),
            tokenizer,
            add_generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs,
        )
        if full.tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError("Chat template does not preserve prompt tokens before the completion.")
        tokens = full.tokens[len(prompt_tokens) :]
    else:
        raise TypeError("chosen and rejected must be text, token IDs, or chat completion messages.")
    tokens = list(tokens)
    if not tokens or any(not isinstance(token, Integral) or isinstance(token, bool) for token in tokens):
        raise ValueError("A preference completion must contain integer token IDs.")
    return [int(token) for token in tokens]


def build_datum(prompt: list[int], completion: list[int]) -> Any:
    """Mask prompt labels and align completion labels with next-token logits."""

    tokens = prompt + completion
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": [-100] * (len(prompt) - 1) + completion},
    )


def reference_logprob(values: Sequence[float | None], prompt_length: int, total_length: int) -> float:
    if len(values) != total_length:
        raise ValueError("Reference sampler returned the wrong number of token logprobs.")
    completion_values = values[prompt_length:]
    if not completion_values or any(value is None or not math.isfinite(value) for value in completion_values):
        raise ValueError("Reference sampler returned missing or non-finite completion logprobs.")
    return sum(float(value) for value in completion_values if value is not None)


def prepare_preferences(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, args: DPOConfig,
) -> list[Preference]:
    """Validate and tokenize ordered chosen/rejected pairs."""
    if not rows:
        raise ValueError("A DPO step requires at least one preference pair.")
    pairs = []
    for index, row in enumerate(rows):
        prompt_key = args.prompt_key
        if not all(key in row for key in (prompt_key, "chosen", "rejected")):
            raise KeyError(f"Dataset row {index} needs {prompt_key}, chosen, and rejected.")
        prompt = render_prompt(
            row[prompt_key],
            tokenizer,
            chat_template_kwargs=args.chat_template_kwargs,
        )
        completions = []
        for key in ("chosen", "rejected"):
            tokens = completion_tokens(
                row[key],
                row[prompt_key],
                prompt.tokens,
                tokenizer,
                args.chat_template_kwargs,
            )
            eos = getattr(tokenizer, "eos_token_id", None)
            if args.add_eos_token:
                if not isinstance(eos, Integral) or isinstance(eos, bool):
                    raise ValueError("add_eos_token requires tokenizer.eos_token_id.")
                if tokens[-1] != eos:
                    tokens.append(eos)
            if (
                args.max_length is not None
                and len(prompt.tokens) + len(tokens) > args.max_length
            ):
                raise ValueError(f"Dataset row {index} {key} exceeds max_length={args.max_length}.")
            completions.append(tokens)
        pairs.append(Preference(prompt.tokens, *completions))
    return pairs
