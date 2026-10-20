"""DPO-specific causal alignment and PyTRIO Datum construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...dataset import IGNORE_INDEX, SupervisedExample
from ...utils.datums import causal_datum


def build_dpo_datum(prompt: Sequence[int], completion: Sequence[int]) -> Any:
    """Mask prompt targets and align one preference completion causally."""

    example = SupervisedExample(
        input_ids=list(prompt) + list(completion),
        labels=[IGNORE_INDEX] * len(prompt) + list(completion),
    )
    return causal_datum(example.input_ids, example.target_tokens)


__all__ = ["build_dpo_datum"]
