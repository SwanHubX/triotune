"""Small, algorithm-neutral dataset iteration helpers.

The dataset package defines and validates one sample at a time. This module
owns the next stage of the pipeline: consuming an iterable and grouping its
already prepared rows into finite batches. It deliberately does not tokenize,
pad, shuffle, or construct a PyTRIO ``Datum``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from numbers import Integral
from typing import TypeVar


RowT = TypeVar("RowT")


def _validate_batch_size(batch_size: int) -> int:
    """Validate and normalize a positive batch size."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    return int(batch_size)


def iter_batches(
    rows: Iterable[RowT],
    batch_size: int,
    *,
    max_batches: int | None = None,
) -> Iterator[list[RowT]]:
    """Yield successive batches from a one-pass or reusable dataset.

    The final batch may be smaller than ``batch_size``. ``max_batches`` limits
    how many batches are consumed and is useful for validation without adding
    a second dataset abstraction. The source iterable is never materialized.
    """

    size = _validate_batch_size(batch_size)
    if max_batches is not None and (
        isinstance(max_batches, bool)
        or not isinstance(max_batches, Integral)
        or max_batches < 0
    ):
        raise ValueError("max_batches must be a nonnegative integer or None.")
    iterator = iter(rows)
    produced = 0
    while max_batches is None or produced < int(max_batches):
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        produced += 1
        yield batch
