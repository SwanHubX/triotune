"""Pure learning-rate policies shared by every PyTRIO optimizer owner.

The durable cursor is the owner's acknowledged optimizer_steps counter. This
module does not advance counters or mutate the base optimizer configuration.
SFT may reserve several update indices while results are still in flight;
PPO actor and critic each supply their own independent indices and horizons.
"""

from __future__ import annotations

import math
from typing import Any

from .dataloader import num_batches


def learning_rate_for_step(
    optimizer: Any,
    step: int,
    *,
    total_steps: int | None = None,
) -> float:
    """Return the LR submitted for one real optimizer update (step starts at 1).

    With no warmup, a multi-step decay starts at the base LR on update 1.
    Warmup ends at the base LR on update W; decay then reaches min_lr on update
    T and stays there for later updates. A one-update decay uses min_lr because
    that update is also its endpoint. Constant schedules, including warmup,
    work without a finite horizon. The floor also bounds warmup values.
    """
    if type(step) is not int or step < 1:
        raise ValueError("optimizer step must be a positive, one-based integer.")
    schedule = optimizer.scheduler
    base = optimizer.learning_rate
    floor = schedule.min_lr
    if floor is None:
        floor = 0.0 if schedule.type == "constant" else max(base * 0.01, math.ulp(0.0))
    total = schedule.total_steps if schedule.total_steps is not None else total_steps
    if schedule.type != "constant":
        if type(total) is not int or total < 1:
            raise ValueError("A linear/cosine scheduler requires a known total_steps.")
        if schedule.warmup_steps >= total:
            raise ValueError("A decay scheduler requires warmup_steps < total_steps.")
    if schedule.warmup_steps and step <= schedule.warmup_steps:
        return max(floor, base * step / schedule.warmup_steps)
    if schedule.type == "constant":
        return base
    if step >= total:
        return floor
    start = max(1, schedule.warmup_steps)
    progress = (step - start) / (total - start)
    factor = (
        1 - progress
        if schedule.type == "linear"
        else (1 + math.cos(math.pi * progress)) / 2
    )
    return max(floor, min(base, floor + (base - floor) * factor))


def optimizer_horizons(config: Any, dataset: Any = None) -> dict[str, int | None]:
    """Resolve planned optimizer counts without consuming a training iterable.

    Explicit scheduler.total_steps already counts optimizer updates. Otherwise
    max_steps supplies the rollout/batch budget, or a sized dataset supplies
    its batch count. PPO multiplies that budget by the component's own epochs
    and minibatches. Dynamic sampling cannot infer accepted batches from raw
    dataset length, so it requires max_steps or an explicit decay horizon.
    Constant/warmup policies do not depend on a horizon and store None.
    """
    optimizers = {"actor": config.optimizer}
    multipliers = {"actor": 1}
    if hasattr(config, "actor"):
        multipliers["actor"] = config.update_epochs * config.ppo_n_minibatches
        if config.use_critic:
            optimizers["critic"] = config.critic.optimizer
            multipliers["critic"] = (
                config.critic_update_epochs * config.critic_n_minibatches
            )
    result = {}
    for name, optimizer in optimizers.items():
        if optimizer.scheduler.type == "constant":
            result[name] = None
            continue
        total = optimizer.scheduler.total_steps
        if total is None:
            batches = config.max_steps
            if (
                batches is None
                and dataset is not None
                and not getattr(config, "dynamic_sampling", False)
            ):
                batches = num_batches(dataset, config.batch_size)
            total = None if batches is None else batches * multipliers[name]
        # Validate the resolved horizon before a trainer creates remote clients.
        learning_rate_for_step(optimizer, 1, total_steps=total)
        result[name] = total
    return result


def scheduler_state(
    optimizer: Any, completed_steps: int, total_steps: int | None
) -> dict:
    """Describe the next-step contract using the existing checkpoint counters."""
    return {
        "total_steps": total_steps,
        "optimizer_steps": completed_steps,
        "last_lr": learning_rate_for_step(
            optimizer, completed_steps, total_steps=total_steps
        )
        if completed_steps
        else None,
    }
