"""PPO actor algorithm: rewards, GAE, policy targets, and PyTRIO updates.

The actor owns all policy-side mathematics that happens before a remote
PyTRIO update: reward shaping, optional KL penalties, GAE, normalization,
minibatch construction, and the clipped policy objective.  Model execution
remains remote; this module only prepares tensors and loss callbacks for the
PyTRIO client.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ...api.cli_args import NormConfig, PPOConfig
from ...api.client_api import call_async, optimizer_params
from .contracts import PreparedBatch, Trajectory
from .datum import batch_datums, build_datum, completion_logprobs
from .loss import make_loss_fn


# ---------------------------------------------------------------------------
# Reward shaping, GAE, and normalization
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    mask: Sequence[bool],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    truncated: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute token-level GAE while skipping masked positions.

    ``values`` contains one extra bootstrap value.  A truncated rollout uses
    that value; a normally terminated rollout bootstraps from zero.  Prompt,
    tool, and padding positions are excluded through ``mask``.
    """

    if len(rewards) != len(mask) or len(values) != len(mask) + 1:
        raise ValueError("GAE needs T rewards/mask and T+1 values including bootstrap.")
    if any(not math.isfinite(x) for x in (*rewards, *values, discount, gae_lambda)):
        raise ValueError("GAE inputs must be finite.")
    if not 0 <= discount <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE discount and lambda must be in [0, 1].")
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_value, running = values[-1] if truncated else 0.0, 0.0
    for index in reversed(range(len(rewards))):
        if mask[index]:
            delta = rewards[index] + discount * next_value - values[index]
            running = delta + discount * gae_lambda * running
            advantages[index], returns[index] = running, running + values[index]
            next_value = values[index]
    return advantages, returns


def normalize(
    rows: Sequence[Sequence[float]],
    group_ids: Sequence[str],
    config: NormConfig | None,
    masks: Sequence[Sequence[bool]] | None = None,
) -> list[list[float]]:
    """Normalize rows at independent batch/group mean and standard-deviation levels."""

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
        for x, m in zip(data, valid, strict=True)
    ):
        raise ValueError("Normalization requires finite, aligned one-dimensional rows.")
    groups = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)

    def pools(level):
        return [list(range(len(data)))] if level == "batch" else list(groups.values())

    means = [np.zeros_like(row) for row in data]
    if config is not None and config.mean_level is not None:
        for indices in pools(config.mean_level):
            values = np.concatenate([data[i][valid[i]] for i in indices])
            if values.size == 0:
                continue
            for index in indices:
                if config.mean_leave1out:
                    if config.mean_level == "group" and len(indices) == 1:
                        means[index] = data[index].copy()
                    elif values.size > 1:
                        means[index] = (values.sum() - data[index]) / (values.size - 1)
                else:
                    means[index].fill(values.mean())
    scales = [np.ones_like(row) for row in data]
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
            for index in indices:
                scales[index].fill(std + config.eps)
    return [
        np.where(mask, (row - mean) / scale, 0).tolist()
        for row, mean, scale, mask in zip(data, means, scales, valid, strict=True)
    ]


def prepare_advantages(
    trajectories: Sequence[Trajectory], config: PPOConfig
) -> PreparedBatch:
    """Build frozen rewards, returns, and normalized advantages for one update."""

    if not trajectories:
        raise ValueError("An update requires at least one trajectory.")
    group_ids = [trajectory.group_id for trajectory in trajectories]
    rewards = []
    masks = []
    for trajectory in trajectories:
        reward = trajectory.reward
        if config.overlong_tokens:
            excess = max(
                0,
                len(trajectory.completion_tokens)
                - (config.max_tokens - config.overlong_tokens),
            )
            reward -= (
                min(1.0, excess / config.overlong_tokens)
                * config.overlong_penalty_factor
            )
        reward = (reward + config.reward_bias) * config.reward_scaling
        rewards.append([max(-config.reward_clip, min(config.reward_clip, reward))])
        masks.append(list(trajectory.loss_mask))

    outcome_scores = normalize(rewards, group_ids, config.reward_norm)
    advantages, returns = [], []
    for trajectory, score, mask in zip(
        trajectories, outcome_scores, masks, strict=True
    ):
        if config.use_critic and trajectory.values is None:
            raise ValueError(
                "Critic-based PPO requires T+1 value predictions per trajectory."
            )
        if config.kl_ctl and trajectory.reference_logprobs is None:
            raise ValueError("kl_ctl > 0 requires fixed-reference completion logprobs.")
        token_rewards = (
            list(trajectory.token_rewards)
            if trajectory.token_rewards is not None
            else [0.0] * len(mask)
        )
        suppress_reward = config.mask_no_eos_with_zero and trajectory.truncated
        if suppress_reward:
            token_rewards = [0.0] * len(mask)
        for index, active in enumerate(mask):
            if active and config.kl_ctl:
                log_ratio = (
                    trajectory.logprobs[index]
                    - trajectory.reference_logprobs[index]
                )
                kl = (
                    log_ratio
                    if config.kl_estimator == "k1"
                    else 0.5 * log_ratio**2
                    if config.kl_estimator == "k2"
                    else math.expm1(min(700.0, -log_ratio)) + log_ratio
                )
                token_rewards[index] -= config.kl_ctl * max(-10.0, min(10.0, kl))
        active_indices = [index for index, active in enumerate(mask) if active]
        if active_indices and not suppress_reward:
            token_rewards[active_indices[-1]] += score[0]
        adv, ret = compute_gae(
            token_rewards,
            trajectory.values
            if config.use_critic
            else [0.0] * (len(mask) + 1),
            mask,
            discount=config.discount,
            gae_lambda=config.gae_lambda,
            truncated=trajectory.truncated,
        )
        advantages.append(adv)
        returns.append(ret)
    if not any(any(mask) for mask in masks):
        raise ValueError("All trajectories were masked; no policy update can be performed.")
    advantages = normalize(advantages, group_ids, config.adv_norm, masks)
    return PreparedBatch(
        list(trajectories),
        advantages,
        returns,
        masks,
        [list(trajectory.logprobs) for trajectory in trajectories],
    )


# ---------------------------------------------------------------------------
# Prompt-group batching and remote actor optimization
# ---------------------------------------------------------------------------

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
    """Prepare frozen policy targets and execute actor updates through PyTRIO."""

    def __init__(self, client, config: PPOConfig):
        """Bind one remote training client and its immutable PPO configuration."""

        self.client, self.config = client, config
        self.optimizer_steps = 0

    async def prepare(self, trajectories: list[Trajectory]) -> PreparedBatch:
        """Recompute optional proximal scores and freeze one actor update batch."""

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
        """Run configured minibatches and epochs using PyTRIO loss callbacks."""

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
