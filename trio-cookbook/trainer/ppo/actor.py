"""Policy scoring, frozen rollout targets and sequential minibatch updates."""

from dataclasses import replace

import numpy as np

from ...api.client_api import call_async, optimizer_params
from .advantages import prepare_advantages
from .config import PPOConfig
from .contracts import PreparedBatch, Trajectory
from .datum import batch_datums, build_datum, completion_logprobs
from .loss import make_loss_fn


def minibatches(batch: PreparedBatch, count: int):
    """Keep prompt groups intact, including variable response counts.

    A minibatch is an optimizer update; advantages/proximal policy stay frozen
    across every minibatch and update epoch of the collected rollout batch.
    """
    groups = {}
    for i, t in enumerate(batch.trajectories):
        if any(batch.masks[i]):
            groups.setdefault(t.group_id, []).append(i)
    if count > len(groups):
        raise ValueError("ppo_n_minibatches must not exceed active prompt groups.")
    units = list(groups.values())
    for chunk in np.array_split(np.arange(len(units)), count):
        indices = [i for unit in chunk for i in units[unit]]
        chunk_batch = PreparedBatch(
            *[
                [getattr(batch, name)[i] for i in indices]
                for name in (
                    "trajectories",
                    "advantages",
                    "returns",
                    "masks",
                    "proximal_logprobs",
                )
            ]
        )
        chunk_batch.prompt_mean_scale = count * len(chunk) / len(groups)
        yield chunk_batch


class PPOActor:
    def __init__(self, client, config: PPOConfig):
        self.client, self.config = client, config
        self.optimizer_steps = 0

    async def prepare(self, trajectories: list[Trajectory]) -> PreparedBatch:
        if not trajectories:
            raise ValueError("An update requires nonempty trajectories.")
        config = self.config
        # Revalidate caller-owned arrays and prevent accidental cross-prompt
        # grouping before making any remote gradient/optimizer request.
        prompts = {}
        for t in trajectories:
            t.__post_init__()
            previous = prompts.setdefault(t.group_id, t.prompt_tokens)
            if previous != t.prompt_tokens:
                raise ValueError(
                    "One group_id must identify one identical tokenized prompt."
                )
        proximal = None
        if config.recompute_logprob or config.use_decoupled_loss:
            result = await call_async(
                self.client,
                "forward",
                [build_datum(t) for t in trajectories],
                loss_fn="cross_entropy",
                auto_shift=False,
            )
            proximal = completion_logprobs(result, trajectories)
        if proximal is not None and not config.use_decoupled_loss:
            # Match AReaL: ordinary recomputation replaces old logprobs for
            # both KL rewards and the frozen PPO ratio denominator.
            trajectories = [
                replace(t, logprobs=lp)
                for t, lp in zip(trajectories, proximal, strict=True)
            ]
        batch = prepare_advantages(trajectories, config)
        if proximal is not None:
            batch.proximal_logprobs = proximal
        return batch

    async def update(self, batch: PreparedBatch) -> dict[str, float]:
        config = self.config
        chunks = list(minibatches(batch, config.ppo_n_minibatches))
        metrics = []
        for _ in range(config.update_epochs):
            for chunk in chunks:
                if config.loss_backend == "native":
                    result = await call_async(
                        self.client,
                        "forward_backward",
                        batch_datums(chunk, native=True),
                        loss_fn="ppo",
                        loss_fn_config={
                            "clip_low_threshold": 1 - config.eps_clip,
                            "clip_high_threshold": 1
                            + (
                                config.eps_clip
                                if config.eps_clip_higher is None
                                else config.eps_clip_higher
                            ),
                        },
                        auto_shift=False,
                    )
                else:
                    result = await call_async(
                        self.client,
                        "forward_backward_custom",
                        batch_datums(chunk),
                        loss_fn=make_loss_fn(chunk, config),
                    )
                # The custom SDK path needs its current forward result to build
                # backward weights. Do not mutate policy weights between them.
                await call_async(self.client, "optim_step", optimizer_params(config))
                self.optimizer_steps += 1
                row = dict(result.metrics)
                if "train/loss" in row:
                    # The SDK may also return the CE surrogate's raw `loss`.
                    # MetricsAPI maps that name to train/loss, so exclude it
                    # rather than overwrite the actual nonlinear objective.
                    row.pop("loss", None)
                    row.pop("scalar/loss", None)
                    row.pop("trainer/loss", None)
                metrics.append(row)
        keys = {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        }
        return {
            key: float(np.mean([row[key] for row in metrics if key in row]))
            for key in keys
        }
