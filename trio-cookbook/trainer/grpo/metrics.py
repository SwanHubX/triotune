"""GRPO rollout statistics, policy drift estimates, and chart names."""

from collections.abc import Mapping, Sequence
from typing import Any

from .types import GRPOBatch


FINAL_SUMMARY_KEYS = (
    "train/loss",
    "reward/mean",
    "policy/entropy",
    "policy/kl_sample_train",
    "optim/lr",
    "optim/grad_norm",
    "time/step",
    "time/train_runtime",
    "time/train_steps_per_second",
    "time/train_tokens_per_second",
)


def _is_per_prompt_metric(name: str) -> bool:
    """Return whether a metric belongs to one prompt rather than the batch."""

    _, separator, metric_name = name.partition("/")
    return bool(separator) and metric_name.startswith(("reward/", "advantage/"))


def _swanlab_metric_name(name: str) -> str:
    """Map trainer metrics to the namespaces exposed in SwanLab."""

    if name == "optim/lr":
        return "train/learning_rate"
    if name.startswith(("trainer/", "scalar/")):
        return "train/" + name.partition("/")[2]
    if "/" not in name:
        return "train/" + name
    return name


def _is_checkpoint_metric(name: str) -> bool:
    """Return whether a metric describes checkpoint handling."""

    return name == "time/checkpoint" or name.startswith("checkpoint/")


def _tensor_values(value: Any) -> list[float] | None:
    data = getattr(value, "data", None)
    if not isinstance(data, list):
        return None
    if not all(isinstance(item, (int, float)) for item in data):
        return None
    return [float(item) for item in data]


def _rollout_metrics(groups: Sequence[GRPOBatch]) -> dict[str, float]:
    samples = [sample for group in groups for sample in group.samples]
    logprobs = [value for sample in samples for value in sample.logprobs]
    lengths = [len(sample.tokens) for sample in samples]
    metrics: dict[str, float] = {}
    if logprobs:
        metrics["policy/entropy"] = -sum(logprobs) / len(logprobs)
    if lengths:
        metrics.update(
            {
                "rollout/completion_length_mean": sum(lengths) / len(lengths),
                "rollout/completion_length_min": float(min(lengths)),
                "rollout/completion_length_max": float(max(lengths)),
            }
        )
    return metrics


def _sample_train_kl(datums: Sequence[Any], forward_result: Any) -> float | None:
    outputs = getattr(forward_result, "loss_fn_outputs", None)
    if not isinstance(outputs, list) or len(outputs) != len(datums):
        return None

    squared_logprob_differences: list[float] = []
    for datum, output in zip(datums, outputs, strict=True):
        if not isinstance(output, Mapping) or "logprobs" not in output:
            return None
        loss_inputs = getattr(datum, "loss_fn_inputs", None)
        if not isinstance(loss_inputs, Mapping):
            return None
        old_logprobs = _tensor_values(loss_inputs.get("logprobs"))
        advantages = _tensor_values(loss_inputs.get("advantages"))
        current_logprobs = _tensor_values(output["logprobs"])
        if old_logprobs is None or advantages is None or current_logprobs is None:
            return None
        if not len(old_logprobs) == len(advantages) == len(current_logprobs):
            return None
        squared_logprob_differences.extend(
            (old - current) ** 2
            for old, current, advantage in zip(
                old_logprobs,
                current_logprobs,
                advantages,
                strict=True,
            )
            if advantage != 0.0
        )
    if not squared_logprob_differences:
        return None
    return 0.5 * sum(squared_logprob_differences) / len(squared_logprob_differences)
