"""Independent mean/std levels, using explicit prompt IDs rather than reshapes."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .config import NormConfig


def normalize(
    rows: Sequence[Sequence[float]],
    group_ids: Sequence[str],
    config: NormConfig | None,
    masks: Sequence[Sequence[bool]] | None = None,
) -> list[list[float]]:
    if len(rows) != len(group_ids) or (masks is not None and len(masks) != len(rows)):
        raise ValueError("Normalization rows, masks and prompt groups must align.")
    if not rows:
        return []
    data = [np.asarray(row, dtype=np.float64) for row in rows]
    valid = (
        [np.ones(x.shape, dtype=bool) for x in data]
        if masks is None
        else [np.asarray(m, dtype=bool) for m in masks]
    )
    if any(
        x.ndim != 1 or x.shape != m.shape or not np.isfinite(x).all()
        for x, m in zip(data, valid)
    ):
        raise ValueError("Normalization requires finite, aligned one-dimensional rows.")
    groups = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        groups[group_id].append(i)

    def pools(level):
        return [list(range(len(data)))] if level == "batch" else list(groups.values())

    means = [np.zeros_like(x) for x in data]
    if config is not None and config.mean_level is not None:
        for indices in pools(config.mean_level):
            values = np.concatenate([data[i][valid[i]] for i in indices])
            if values.size == 0:
                continue
            for i in indices:
                if config.mean_leave1out:
                    if config.mean_level == "group" and len(indices) == 1:
                        means[i] = data[i].copy()
                    elif values.size > 1:
                        means[i] = (values.sum() - data[i]) / (values.size - 1)
                else:
                    means[i].fill(values.mean())
    scales = [np.ones_like(x) for x in data]
    if config is not None and config.std_level is not None:
        for indices in pools(config.std_level):
            residuals = np.concatenate(
                [(data[i] - means[i])[valid[i]] for i in indices]
            )
            count = residuals.size
            singleton = (
                config.std_level == "group"
                and len(indices) == 1
                and config.std_unbiased
            )
            std = (
                np.sqrt(np.square(residuals).sum() / (count - int(config.std_unbiased)))
                if count > int(config.std_unbiased) and not singleton
                else 1.0
            )
            for i in indices:
                scales[i].fill(std + config.eps)
    return [
        np.where(m, (x - mean) / scale, 0).tolist()
        for x, mean, scale, m in zip(data, means, scales, valid)
    ]
