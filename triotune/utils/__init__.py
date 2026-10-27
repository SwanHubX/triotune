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
from .checkpoint import (
    checkpoint_path,
    load_training_checkpoint,
    save_checkpoint,
    save_checkpoint_async,
)
from .dataloader import iter_batches
from .datums import causal_datum, materialize_datum, materialize_model_input
from .metrics import MetricsAPI, metric_name, normalize_metrics
from .reporting import isolated_metrics
from .scoring import completion_logprobs, score_completion, score_completion_async
from .token_usage import SampleTokens, compute_sample_tokens, compute_train_tokens
from .updates import UpdateResult, run_update, run_update_async

# ``triotune.utils.client`` was public before client contracts moved into
# ``api``. Keep that import path without duplicating its implementation here.
sys.modules[f"{__name__}.client"] = client

__all__ = [
    "MetricsAPI",
    "SampleTokens",
    "UpdateResult",
    "call",
    "call_async",
    "causal_datum",
    "checkpoint_path",
    "load_training_checkpoint",
    "client",
    "completion_logprobs",
    "compute_sample_tokens",
    "compute_train_tokens",
    "iter_batches",
    "isolated_metrics",
    "materialize_datum",
    "materialize_model_input",
    "metric_name",
    "normalize_metrics",
    "optimizer_params",
    "run_update",
    "run_update_async",
    "sampling_params",
    "save_checkpoint",
    "save_checkpoint_async",
    "score_completion",
    "score_completion_async",
    "sdk_object",
    "submit_async",
    "wait",
    "wait_async",
]
