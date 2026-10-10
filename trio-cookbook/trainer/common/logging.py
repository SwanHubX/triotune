"""Shared SwanLab initialization and metric window arithmetic.

Algorithms own their metric names, reporting cadence, and final samples.
These helpers do not control the training loop or asynchronous task ordering.
"""

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import swanlab


def start_run(args: Any, *, default_name: str, metadata: Mapping[str, Any]) -> Any:
    """Initialize an enabled run with the algorithm's metadata and config."""
    return swanlab.init(
        project=args.project,
        workspace=args.workspace,
        tags=args.tags,
        name=args.run_name or default_name,
        log_dir=args.log_dir,
        config={**metadata, **asdict(args)},
    )


def accumulate_metrics(
    window: dict[str, tuple[float, int]], metrics: Mapping[str, float],
) -> None:
    """Accumulate each metric independently, including sparse metrics."""
    for name, value in metrics.items():
        total, count = window.get(name, (0.0, 0))
        window[name] = (total + value, count + 1)


def averaged_metrics(window: Mapping[str, tuple[float, int]]) -> dict[str, float]:
    """Compute the current window without clearing it or writing logs."""
    return {name: total / count for name, (total, count) in window.items()}


def finish_run(run: Any, *, state: str, error: str | None = None) -> None:
    """Finish an enabled run; callers clear their own run reference."""
    if run is not None:
        run.finish(state=state, error=error)
