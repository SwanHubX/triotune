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


class JsonlEvaluationSummaryLogger:
    """Persist compact task outcomes without prompt or generation token arrays.

    Training evidence needs exact token provenance for policy updates. Task
    evaluation does not: retaining repeated prompts and interaction events can
    make a few hundred long-horizon episodes occupy hundreds of megabytes. This
    logger stores outcome metrics and parsed action summaries only.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = asyncio.Lock()

    @staticmethod
    def summarize(episode: EpisodeResult, *, step: int) -> dict:
        actions = []
        for index, turn in enumerate(episode.turns):
            action = turn.action
            if action is None:
                value = {"turn": index, "type": "unparsed"}
            elif action.final_text is not None:
                value = {
                    "turn": index,
                    "type": "final",
                    "text": action.final_text,
                }
            else:
                value = {
                    "turn": index,
                    "type": "tool_calls",
                    "calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in action.tool_calls
                    ],
                }
            errors = [
                {"name": result.name, "error": result.content}
                for result in turn.tool_results
                if result.is_error
            ]
            if errors:
                value["tool_errors"] = errors
            actions.append(value)
        score = episode.score
        return {
            "step": step,
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "rollout_id": episode.rollout_id,
            "sample_index": episode.sample_index,
            "snapshot_id": episode.snapshot_id,
            "policy_version": episode.policy_version,
            "status": episode.status,
            "error": episode.error,
            "success": bool(score and score.success),
            "reward": score.reward if score else None,
            "components": dict(score.components) if score else {},
            "turns": len(episode.turns),
            "input_tokens": sum(turn.generation.input_tokens for turn in episode.turns),
            "output_tokens": sum(
                turn.generation.output_tokens for turn in episode.turns
            ),
            "elapsed_seconds": episode.elapsed_seconds,
            "actions": actions,
        }

    async def write_batch(
        self,
        episodes: list[EpisodeResult],
        *,
        step: int,
    ) -> None:
        lines = "".join(
            json.dumps(
                self.summarize(episode, step=step),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for episode in episodes
        )
        async with self._lock:
            await asyncio.to_thread(self._append, lines)

    def _append(self, lines: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(lines)
