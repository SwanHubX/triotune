"""Small, algorithm-neutral dataset iteration helpers.

The dataset package defines and validates one sample at a time. This module
owns the next stage of the pipeline: consuming an iterable and grouping its
already prepared rows into finite batches. It deliberately does not tokenize,
pad, shuffle, or construct a PyTRIO ``Datum``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator
from numbers import Integral
from typing import TypeVar

RowT = TypeVar("RowT")


def dataset_identity(rows) -> dict | None:
    """Hash ordered, replayable rows without storing another dataset copy.

    Complete checkpoints support sized, random-access sources only. Iterators,
    generators, shuffled iterable datasets and external cursors cannot prove
    that skipping a saved row count yields the same next input. JSON-compatible
    rows cover the public text/chat/token contracts and Hugging Face Dataset.
    """
    if rows is None:
        return None
    if not hasattr(rows, "__len__") or not hasattr(rows, "__getitem__") or iter(rows) is rows:
        raise ValueError("Complete checkpoints require a sized, random-access dataset.")
    digest = hashlib.sha256()
    for index in range(len(rows)):
        try:
            encoded = json.dumps(rows[index], sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Resumable dataset rows must be finite JSON-compatible values.") from error
        digest.update(encoded.encode("utf-8") + b"\n")
    return {"rows": len(rows), "sha256": digest.hexdigest()}


def training_rows(rows, state):
    """Continue at the saved raw-row cursor, including filtered/refill rows.

    The cursor advances when a row is consumed. It becomes durable only after
    the corresponding complete optimizer update is checkpointed. A failed
    attempt must be discarded with its trainer and restarted from the previous
    manifest; speculative/prefetched work is never published as committed work.
    """
    if state.data_cursor > len(rows):
        raise ValueError("Saved data cursor exceeds the dataset length.")
    for index in range(state.data_cursor, len(rows)):
        row = rows[index]
        state.data_cursor = index + 1
        yield row


def _validate_batch_size(batch_size: int) -> int:
    """Validate and normalize a positive batch size."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    return int(batch_size)


def num_batches(
    rows: Iterable[RowT],
    batch_size: int,
    *,
    max_batches: int | None = None,
) -> int | None:
    """Return the visible batch total when an iterable exposes its length."""

    size = _validate_batch_size(batch_size)
    if max_batches is not None and (
        isinstance(max_batches, bool)
        or not isinstance(max_batches, Integral)
        or max_batches < 0
    ):
        raise ValueError("max_batches must be a nonnegative integer or None.")
    try:
        total = (len(rows) + size - 1) // size  # type: ignore[arg-type]
    except TypeError:
        return None
    return min(total, int(max_batches)) if max_batches is not None else total


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
