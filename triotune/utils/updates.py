"""Remote parameter updates shared by trainer algorithms.

This module owns only the execution boundary for one optimizer update. It
finishes ``forward/backward`` before submitting ``optim_step``, preventing the
remote optimizer from consuming incomplete gradients after failure or
cancellation.

Algorithms remain responsible for constructing Datums, selecting an Objective,
building optimizer parameters, and handling metrics, checkpoints, and retries.
SFT keeps its ordered multi-batch pipeline in ``SFTTrainer`` because it submits
later requests before earlier results finish instead of using this serial path.

Execution order::

    Objective.forward_backward(...)
        -> success
        -> TrainingClient.optim_step(...)
        -> UpdateResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..api.client_api import call, call_async
from ..api.loss_api import Objective

# ---------------------------------------------------------------------------
# Shared result for one update
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """Store remote results and per-stage durations for one update.

    ``forward`` and ``optimizer`` retain the original PyTRIO results so each
    algorithm can extract its own metrics. The duration fields cover only their
    respective remote stages, excluding rollout, reward computation, Datum
    construction, and checkpointing.
    """

    forward: Any
    optimizer: Any
    forward_seconds: float
    optimizer_seconds: float


# ---------------------------------------------------------------------------
# Synchronous update path
# ---------------------------------------------------------------------------


def run_update(
    client: Any,
    objective: Objective,
    datums: list[Any],
    optimizer_params: Any,
) -> UpdateResult:
    """Run one failure-safe synchronous forward/backward and optimizer update.

    ``objective.forward_backward`` resolves the complete backward operation.
    If it raises, this function exits before reaching ``optim_step``.
    """

    # The Objective selects either a built-in loss or the custom DPO path.
    forward_started = time.perf_counter()
    forward = objective.forward_backward(client, datums)
    forward_seconds = time.perf_counter() - forward_started

    # Submit the optimizer only after backward has produced complete gradients.
    optimizer_started = time.perf_counter()
    optimizer = call(client, "optim_step", optimizer_params)
    optimizer_seconds = time.perf_counter() - optimizer_started
    return UpdateResult(
        forward=forward,
        optimizer=optimizer,
        forward_seconds=forward_seconds,
        optimizer_seconds=optimizer_seconds,
    )


# ---------------------------------------------------------------------------
# Asynchronous update path
# ---------------------------------------------------------------------------


async def run_update_async(
    client: Any,
    objective: Objective,
    datums: list[Any],
    optimizer_params: Any,
) -> UpdateResult:
    """Run an asynchronous update with the same failure boundary.

    The function awaits the final forward/backward result before calling
    ``optim_step``. It does not block the event loop or submit the two dependent
    stages concurrently.
    """

    # Await successful backward completion; propagate failures and cancellation.
    forward_started = time.perf_counter()
    forward = await objective.forward_backward_async(client, datums)
    forward_seconds = time.perf_counter() - forward_started

    # Allow the remote optimizer to consume only complete gradients.
    optimizer_started = time.perf_counter()
    optimizer = await call_async(client, "optim_step", optimizer_params)
    optimizer_seconds = time.perf_counter() - optimizer_started
    return UpdateResult(
        forward=forward,
        optimizer=optimizer,
        forward_seconds=forward_seconds,
        optimizer_seconds=optimizer_seconds,
    )


__all__ = ["UpdateResult", "run_update", "run_update_async"]
