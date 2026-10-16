"""Read raw rows before format-specific single-example preparation.

Supported sources are a mapping, a JSONL path, or an existing iterable.
This module parses the source structure; prepare.py validates algorithm fields,
tokenizes content, and applies EOS, masking, and sequence-length policies.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def load_rows(source: Any) -> Iterable[Mapping[str, Any]]:
    """Expose a source as an iterable of raw dataset rows.

    Args:
        source: One mapping, a JSONL path, or a dataset iterable such as HF Dataset.

    Returns:
        A one-element tuple for a mapping, a lazy generator for a file, or the
        original iterable. Rows from an existing iterable are validated later
        by prepare. File contents are not collected into an in-memory list.

    Raises:
        ValueError: A JSONL line contains invalid JSON; includes path and line.
        TypeError: The source is unsupported or a JSONL row is not an object.

    Files are opened and checked during iteration. The returned file generator
    is single-pass; call this function again to reopen it. Existing iterators
    retain their original replay and consumption semantics.
    """

    if isinstance(source, Mapping):
        # Iterating a mapping yields keys, so wrap it to preserve the whole row.
        return (source,)
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        path = Path(source)

        def read_jsonl() -> Iterator[Mapping[str, Any]]:
            """Parse nonempty lines lazily and retain source locations in errors."""

            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSONL at {path}:{line_number}."
                        ) from error
                    if not isinstance(row, Mapping):
                        raise TypeError(
                            f"JSONL row at {path}:{line_number} must be an object."
                        )
                    yield row

        return read_jsonl()
    if isinstance(source, Iterable):
        # Preserve source semantics: lists are replayable; generators are not.
        return source
    raise TypeError("Dataset source must be a mapping, JSONL path, or iterable.")
