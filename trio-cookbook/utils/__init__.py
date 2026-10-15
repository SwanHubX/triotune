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
from .metrics import MetricsAPI, metric_name, normalize_metrics

# ``trio_cookbook.utils.client`` was public before client contracts moved into
# ``api``. Keep that import path without duplicating its implementation here.
sys.modules[f"{__name__}.client"] = client

__all__ = [
    "MetricsAPI",
    "call",
    "call_async",
    "checkpoint_path",
    "client",
    "metric_name",
    "normalize_metrics",
    "optimizer_params",
    "sampling_params",
    "save_checkpoint",
    "save_checkpoint_async",
    "sdk_object",
    "submit_async",
    "wait",
    "wait_async",
]
