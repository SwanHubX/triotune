"""DPO metric names for SwanLab charts."""

from collections.abc import Mapping


def swanlab_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Group DPO charts without changing result metric keys."""
    names = {
        "optim/lr": "train/learning_rate",
        "optim/grad_norm": "train/grad_norm",
        "rewards/chosen": "reward/reward_chosen",
        "rewards/rejected": "reward/reward_rejected",
        "rewards/margins": "reward/reward_margin",
        "rewards/accuracies": "reward/reward_accuracy",
        "logps/chosen": "train/logp_chosen",
        "logps/rejected": "train/logp_rejected",
        "data/tokens": "train/tokens",
        "data/total_tokens": "train/total_tokens",
        "data/total_pairs": "train/total_pairs",
        "token_count": "train/backend_token_count",
    }
    payload = {}
    for name, value in metrics.items():
        if name == "grad_norm" and "optim/grad_norm" in metrics:
            continue
        key = names.get(name)
        if key is None:
            key = name if name.startswith(("train/", "time/")) else f"train/{name.replace('/', '_')}"
        payload[key] = value
    return payload
