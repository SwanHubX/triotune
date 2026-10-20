"""Cross-cutting metric collection and SwanLab reporting.

Trainers remain responsible for computing metrics with the correct algorithmic
meaning. ``MetricsAPI`` owns only the shared transport concerns: validating
numeric values, applying stable chart names, averaging sparse metric windows,
publishing at a configured step interval, and closing the reporting run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from numbers import Real
from typing import Any

import swanlab


MetricFilter = Callable[[str], bool]


_COMMON_NAMES = {
    "optim/lr": "train/learning_rate",
    "optim/grad_norm": "train/grad_norm",
    "grad_norm": "train/grad_norm",
    "gradient_norm": "train/grad_norm",
    "data/tokens": "train/tokens",
    "data/total_tokens": "train/total_tokens",
    "data/total_pairs": "train/total_pairs",
    "token_count": "train/backend_token_count",
}


def metric_name(name: str) -> str:
    """Return a stable SwanLab chart name without knowing the algorithm.

    Algorithms should emit meaningful namespaces such as ``reward/``,
    ``policy/``, ``distill/`` or ``eval/``. This function only normalizes SDK
    and optimizer spellings that are shared by every trainer.
    """

    name = str(name)
    if name in _COMMON_NAMES:
        return _COMMON_NAMES[name]
    if name.startswith(("trainer/", "scalar/")):
        return "train/" + name.partition("/")[2]
    if "/" not in name:
        return "train/" + name
    return name


def normalize_metrics(
    metrics: Mapping[str, Any],
    *,
    exclude: MetricFilter | None = None,
) -> dict[str, float]:
    """Copy finite real scalars into their reporting namespace.

    Boolean values are rejected even though ``bool`` is an ``int`` subclass.
    Non-numeric and non-finite backend metadata is ignored instead of being
    sent to SwanLab as an invalid scalar series.
    """

    normalized: dict[str, float] = {}
    for raw_name, value in metrics.items():
        name = str(raw_name)
        if exclude is not None and exclude(name):
            continue
        if not isinstance(value, Real) or isinstance(value, bool):
            continue
        number = float(value)
        if math.isfinite(number):
            normalized[metric_name(name)] = number
    return normalized


def accumulate_metrics(
    window: dict[str, tuple[float, int]], metrics: Mapping[str, float],
) -> None:
    """Accumulate sparse metrics using an independent count for each key."""

    for name, value in metrics.items():
        total, count = window.get(name, (0.0, 0))
        window[name] = (total + value, count + 1)


def averaged_metrics(
    window: Mapping[str, tuple[float, int]],
) -> dict[str, float]:
    """Return one mean for every metric currently present in the window."""

    return {name: total / count for name, (total, count) in window.items()}


def start_run(args: Any, *, default_name: str, metadata: Mapping[str, Any]) -> Any:
    """Initialize SwanLab when reporting is enabled by trainer configuration."""

    if args.report_to is None:
        return None
    return swanlab.init(
        project=args.project,
        workspace=args.workspace,
        tags=args.tags,
        name=args.run_name or default_name,
        log_dir=args.log_dir,
        # Runtime metadata describes the concrete client actually created by
        # the trainer and therefore overrides optional experiment fields that
        # may be omitted for legacy constructor-based configurations.
        config={**asdict(args), **metadata},
    )


def finish_run(run: Any, *, state: str, error: str | None = None) -> None:
    """Finish an enabled run."""

    if run is not None:
        run.finish(state=state, error=error)


class MetricsAPI:
    """Collect arbitrary trainer metrics and publish them through one API."""

    def __init__(self, run: Any = None, *, logging_step: int = 1) -> None:
        if (
            isinstance(logging_step, bool)
            or not isinstance(logging_step, int)
            or logging_step < 1
        ):
            raise ValueError("logging_step must be a positive integer.")
        self._run = run
        self._logging_step = logging_step
        self._window: dict[str, tuple[float, int]] = {}

    @classmethod
    def from_config(
        cls,
        args: Any,
        *,
        default_name: str,
        metadata: Mapping[str, Any],
        logging_step: int = 1,
    ) -> MetricsAPI:
        """Create an enabled or disabled reporter from a trainer config."""

        return cls(
            start_run(args, default_name=default_name, metadata=metadata),
            logging_step=logging_step,
        )

    @property
    def enabled(self) -> bool:
        """Return whether this instance owns an active SwanLab run."""

        return self._run is not None

    def record(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        exclude: MetricFilter | None = None,
    ) -> dict[str, float]:
        """Add one training step and flush when its interval is complete."""

        payload = normalize_metrics(metrics, exclude=exclude)
        if self._run is None:
            return payload
        accumulate_metrics(self._window, payload)
        if step % self._logging_step == 0:
            self.flush(step=step)
        return payload

    def publish(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        exclude: MetricFilter | None = None,
    ) -> dict[str, float]:
        """Publish evaluation or summary metrics without window averaging."""

        payload = normalize_metrics(metrics, exclude=exclude)
        if self._run is not None and payload:
            self._run.log(
                {**payload, "train/step": float(step)},
                step=step,
            )
        return payload

    def flush(self, *, step: int) -> dict[str, float]:
        """Publish and clear the current training metric window."""

        payload = averaged_metrics(self._window)
        self._window.clear()
        if self._run is not None and payload:
            self._run.log(
                {**payload, "train/step": float(step)},
                step=step,
            )
        return payload

    def log_text(self, *, key: str, data: Any, step: int) -> None:
        """Publish algorithm-owned text, tables, or qualitative samples."""

        if self._run is not None:
            self._run.log_text(key=key, data=data, step=step)

    def finish(self, *, state: str = "success", error: str | None = None) -> None:
        """Finish the active run once and make later calls no-ops."""

        if self._run is not None:
            finish_run(self._run, state=state, error=error)
            self._run = None
