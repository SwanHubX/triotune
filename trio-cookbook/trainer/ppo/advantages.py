"""Shared reward -> GAE -> advantage-normalization path for all presets."""

import math
from collections.abc import Sequence

from .config import PPOConfig
from .contracts import PreparedBatch, Trajectory
from .gae import compute_gae
from .normalization import normalize


def prepare_advantages(
    trajectories: Sequence[Trajectory], config: PPOConfig
) -> PreparedBatch:
    if not trajectories:
        raise ValueError("An update requires at least one trajectory.")
    ids = [t.group_id for t in trajectories]
    rewards = []
    masks = []
    for t in trajectories:
        reward = t.reward
        if config.overlong_tokens:
            excess = max(
                0,
                len(t.completion_tokens) - (config.max_tokens - config.overlong_tokens),
            )
            reward -= (
                min(1.0, excess / config.overlong_tokens)
                * config.overlong_penalty_factor
            )
        reward = (reward + config.reward_bias) * config.reward_scaling
        rewards.append([max(-config.reward_clip, min(config.reward_clip, reward))])
        masks.append(list(t.loss_mask))
    scores = normalize(rewards, ids, config.reward_norm)
    advantages, returns = [], []
    for t, score, mask in zip(trajectories, scores, masks, strict=True):
        if config.use_critic and t.values is None:
            raise ValueError(
                "Critic-based PPO requires T+1 value predictions per trajectory."
            )
        if config.kl_ctl and t.reference_logprobs is None:
            raise ValueError("kl_ctl > 0 requires fixed-reference completion logprobs.")
        token_rewards = (
            list(t.token_rewards) if t.token_rewards is not None else [0.0] * len(mask)
        )
        suppress_reward = config.mask_no_eos_with_zero and t.truncated
        if suppress_reward:
            token_rewards = [0.0] * len(mask)
        for i, active in enumerate(mask):
            if active and config.kl_ctl:
                log_ratio = t.logprobs[i] - t.reference_logprobs[i]
                kl = (
                    log_ratio
                    if config.kl_estimator == "k1"
                    else 0.5 * log_ratio**2
                    if config.kl_estimator == "k2"
                    else math.expm1(min(700.0, -log_ratio)) + log_ratio
                )
                token_rewards[i] -= config.kl_ctl * max(-10.0, min(10.0, kl))
        active_indices = [i for i, active in enumerate(mask) if active]
        if active_indices and not suppress_reward:
            token_rewards[active_indices[-1]] += score[0]
        adv, ret = compute_gae(
            token_rewards,
            t.values if config.use_critic else [0.0] * (len(mask) + 1),
            mask,
            discount=config.discount,
            gae_lambda=config.gae_lambda,
            truncated=t.truncated,
        )
        advantages.append(adv)
        returns.append(ret)
    if not any(any(mask) for mask in masks):
        raise ValueError(
            "All trajectories were masked; no policy update can be performed."
        )
    advantages = normalize(advantages, ids, config.adv_norm, masks)
    return PreparedBatch(
        list(trajectories),
        advantages,
        returns,
        masks,
        [list(t.logprobs) for t in trajectories],
    )
