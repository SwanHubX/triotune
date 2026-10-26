"""Reward shaping and advantage calculation for PPO-family algorithms."""

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ...api.cli_args import NormConfig, PPOConfig
from .trajectory import PreparedBatch, Trajectory
from .gae import compute_gae


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
                sum(trajectory.loss_mask)
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
        # Keep task/process rewards separate from policy-reference KL.  Token
        # GAE consumes both signals exactly as before.  Turn GAE follows
        # AReaL's macro-timestep design: task rewards participate in the turn
        # recurrence, while KL remains a token-local actor penalty and never
        # enters the critic's turn-level return target.
        task_rewards = (
            list(trajectory.token_rewards)
            if trajectory.token_rewards is not None
            else [0.0] * len(mask)
        )
        suppress_reward = config.mask_no_eos_with_zero and trajectory.truncated
        if suppress_reward:
            task_rewards = [0.0] * len(mask)
        kl_rewards = [0.0] * len(mask)
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
                kl_rewards[index] = -config.kl_ctl * max(-10.0, min(10.0, kl))
        active_indices = [index for index, active in enumerate(mask) if active]
        if active_indices and not suppress_reward:
            task_rewards[active_indices[-1]] += score[0]

        gae_rewards = (
            [task + kl for task, kl in zip(task_rewards, kl_rewards, strict=True)]
            if config.gae_timestep_unit == "token"
            else task_rewards
        )
        adv, ret = compute_gae(
            gae_rewards,
            trajectory.values
            if config.use_critic
            else [0.0] * (len(mask) + 1),
            mask,
            discount=config.discount,
            gae_lambda=config.gae_lambda,
            truncated=trajectory.truncated,
            timestep_unit=config.gae_timestep_unit,
            turn_ids=trajectory.turn_ids,
        )
        if config.gae_timestep_unit == "turn":
            adv = [value + kl for value, kl in zip(adv, kl_rewards, strict=True)]
        advantages.append(adv)
        returns.append(ret)
    if not any(any(mask) for mask in masks):
        raise ValueError(
            "All trajectories were masked; no policy update can be performed."
        )
    advantages = normalize(advantages, group_ids, config.adv_norm, masks)
    return PreparedBatch(
        list(trajectories),
        advantages,
        returns,
        masks,
        [list(trajectory.logprobs) for trajectory in trajectories],
    )
