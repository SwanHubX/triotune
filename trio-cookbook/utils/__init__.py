"""Cross-cutting implementations shared by trainer algorithms."""

import sys

from ..api import client_api as client
from ..api.client_api import (
    call,
    call_async,
    optimizer_params,
    sampling_params,
    sdk_object,
    submit_async,
    wait,
    wait_async,
)
from .checkpoint import checkpoint_path, save_checkpoint, save_checkpoint_async
from .dataloader import iter_batches
from .datums import causal_datum, materialize_datum
from .metrics import MetricsAPI, metric_name, normalize_metrics
from .updates import UpdateResult, run_update, run_update_async

# ``trio_cookbook.utils.client`` was public before client contracts moved into
# ``api``. Keep that import path without duplicating its implementation here.
sys.modules[f"{__name__}.client"] = client

__all__ = [
    "MetricsAPI",
    "UpdateResult",
    "call",
    "call_async",
    "causal_datum",
    "checkpoint_path",
    "client",
    "iter_batches",
    "materialize_datum",
    "metric_name",
    "normalize_metrics",
    "optimizer_params",
    "run_update",
    "run_update_async",
    "sampling_params",
    "save_checkpoint",
    "save_checkpoint_async",
    "sdk_object",
    "submit_async",
    "wait",
    "wait_async",
]
