"""Persist finalized execution evidence independently of training metrics.

Each JSONL record contains one episode's identity, fixed-policy provenance,
original generations, parsed actions, tool results, terminal status, and score.
This preserves the evidence needed to inspect reward decisions and alignment
failures rather than relying on aggregated success-rate charts alone.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from ..types import EpisodeResult


class JsonlEpisodeLogger:
    """An opt-in local sink containing tokens, actions, observations and scores.

    One instance serializes writes from concurrent episodes in one process.
    Separate processes must use separate paths. Prompts and tool observations
    are recorded verbatim; callers choose the path and any redaction policy.
    """

    def __init__(self, path: str | Path):
        """Configure an append-only destination without creating it eagerly.

        The parent directory is created on the first write. This logger does
        not rotate files or synchronize separate processes.
        """
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def write(self, episode: EpisodeResult) -> None:
        """Serialize a complete record before submitting blocking file work.

        Dataclass conversion retains nested turn and reward fields. Strict JSON
        serialization catches non-finite or unsupported values before appending
        a malformed line; storage errors propagate to the collection caller.
        """
        line = json.dumps(asdict(episode), ensure_ascii=False, allow_nan=False) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        """Append one pre-serialized line while the instance lock is held.

        This runs in a worker thread. Closing the stream flushes its buffer,
        but no fsync or cross-process transaction guarantee is provided.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
