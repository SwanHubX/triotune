"""Reward shaping and advantage calculation for PPO-family algorithms."""

import inspect
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ...api.cli_args import NormConfig, PPOConfig
from .gae import compute_gae
from .trajectory import PreparedBatch, Trajectory


class RewardShaper(Protocol):
    """Shape outcome scalars and completion-aligned process rewards separately.

    Outcome shaping precedes candidate-level reward normalization. Process
    shaping precedes GAE and leaves direct advantages outside the return target.
    An explicit ``PPOAlgorithmPlan.process_shaper`` overrides ``shape_process``.
    """

    def shape(self, trajectory: Trajectory, reward: float) -> float: ...

    def shape_process(
        self, trajectory: Trajectory, rewards: Sequence[float]
    ) -> Sequence[float]: ...


class RewardNormalizer(Protocol):
    """Normalize candidate-level rewards with optional population masks.

    ``masks`` describes entries in ``rows``, not completion-token loss masks.
    Implementations may additionally declare ``config`` to receive NormConfig;
    a protocol-only implementation never receives that object as its mask.
    """

    def normalize(
        self,
        rows: Sequence[Sequence[float]],
        group_ids: Sequence[str],
        masks: Sequence[Sequence[bool]] | None = None,
    ) -> list[list[float]]: ...


class AdvantageEstimator(Protocol):
    """Estimate token-aligned advantages and critic returns."""

    def estimate(
        self, rewards, values, mask, **kwargs
    ) -> tuple[list[float], list[float]]: ...


class ReturnEstimator(Protocol):
    """Estimate critic targets independently from policy advantages."""

    def estimate(self, rewards, values, mask, **kwargs) -> list[float]: ...


class LossWeightPolicy(Protocol):
    """Assign a physical training-row weight before loss reduction."""

    def weight(self, trajectory: Trajectory, mask: Sequence[bool]) -> float: ...


class Objective(Protocol):
    """Build a policy objective callback for one prepared batch."""

    def make_loss(self, batch, config, diagnostics=None): ...


class _DefaultRewardNormalizer:
    def normalize(self, rows, group_ids, masks=None, config=None):
        return normalize(rows, group_ids, config, masks)


class _DefaultAdvantageEstimator:
    def estimate(self, rewards, values, mask, **kwargs):
        return compute_gae(rewards, values, mask, **kwargs)


class _DefaultReturnEstimator:
    def estimate(self, rewards, values, mask, **kwargs):
        return compute_gae(rewards, values, mask, **kwargs)[1]


def _call_shaper(method, trajectory, values):
    """Select the legacy one-argument or public two-argument call before execution.

    Only signature binding is allowed to fail over. A TypeError raised inside
    the plugin must propagate unchanged; repeating a stateful plugin could both
    hide its real error and submit the same shaping side effect twice.
    """
    try:
        inspect.signature(method).bind(trajectory, values)
    except TypeError:
        return method(values)
    return method(trajectory, values)


@dataclass(slots=True)
class PPOAlgorithmPlan:
    """Public dependency-injection plan for PPO reward and objective stages."""

    reward_shaper: Any | None = None
    process_shaper: Any | None = None
    normalizer: Any | None = None
    advantage_estimator: Any | None = None
    return_estimator: Any | None = None
    loss_weight_policy: Any | None = None
    objective: Any | None = None

    def shape_reward(self, trajectory: Trajectory, reward: float) -> float:
        shaper = self.reward_shaper
        if shaper is None:
            return reward
        method = getattr(shaper, "shape", shaper)
        return float(_call_shaper(method, trajectory, reward))

    def shape_process(self, trajectory: Trajectory, rewards):
        shaper = self.process_shaper
        if shaper is None:
            # A combined RewardShaper owns both stages unless a standalone
            # process shaper was supplied. An outcome-only callable must not
            # accidentally receive process vectors through its scalar method.
            method = getattr(self.reward_shaper, "shape_process", None)
            if method is None:
                return rewards
        else:
            # Prefer the dedicated protocol method while retaining historical
            # process callables and objects exposing only shape(). Do not chain
            # this hook with the combined object: each signal is shaped once.
            method = getattr(shaper, "shape_process", getattr(shaper, "shape", shaper))
        values = _call_shaper(method, trajectory, rewards)
        values = list(values)
        if len(values) != len(rewards) or any(
            not math.isfinite(float(v)) for v in values
        ):
            raise ValueError("Process shaper must return finite aligned rewards.")
        return [float(value) for value in values]

    def normalize_rewards(self, rows, group_ids, config, masks=None):
        normalizer = self.normalizer
        if normalizer is None:
            return normalize(rows, group_ids, config, masks)
        method = getattr(normalizer, "normalize", normalizer)
        # Parameter names distinguish the public third argument (masks) from
        # legacy hooks whose third argument is config. Binding first also keeps
        # positional-only, keyword-only, partial, and two-argument hooks usable
        # without executing a plugin repeatedly to guess its signature.
        signature = inspect.signature(method)
        bound = signature.bind_partial(rows, group_ids)
        bound.apply_defaults()
        for name, value in (("config", config), ("masks", masks)):
            parameter = signature.parameters.get(name)
            if parameter is not None and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                bound.arguments[name] = value
        # **kwargs-only adapters receive the public mask argument. Config is
        # supplied only when explicitly declared, so it cannot be mistaken for
        # a population mask by a protocol adapter or a forwarding wrapper.
        if "masks" not in signature.parameters:
            for parameter in signature.parameters.values():
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    bound.arguments[parameter.name]["masks"] = masks
        return method(*bound.args, **bound.kwargs)

    def row_weight(self, trajectory: Trajectory, mask: Sequence[bool]) -> float:
        policy = self.loss_weight_policy
        if policy is None:
            return 1.0
        method = getattr(policy, "weight", policy)
        value = float(method(trajectory, mask))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "LossWeightPolicy must return a finite nonnegative weight."
            )
        return value


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
    trajectories: Sequence[Trajectory],
    config: PPOConfig,
    plan: PPOAlgorithmPlan | None = None,
) -> PreparedBatch:
    """Build frozen rewards, returns, and normalized advantages for one update.

    ``plan`` is optional so existing PPO callers retain the historical
    numerical path while custom algorithms can replace each stage independently.
    """

    if not trajectories:
        raise ValueError("An update requires at least one trajectory.")
    # Branch groups compare sibling counterfactuals from one fork. Ordinary
    # trajectories continue to use their prompt group, preserving the legacy
    # normalization path exactly.
    group_ids = [trajectory.comparison_group_id for trajectory in trajectories]
    rewards, reward_groups, reward_indices = [], [], {}
    row_indices = []
    masks = []
    action_counts = defaultdict(int)
    for index, trajectory in enumerate(trajectories):
        key = (
            ("sequence", *trajectory.logical_sequence_id)
            if trajectory.rollout_id
            else ("row", index)
        )
        action_counts[key] += sum(trajectory.loss_mask)
    references = {}
    for index, trajectory in enumerate(trajectories):
        key = (
            ("sequence", *trajectory.logical_sequence_id)
            if trajectory.rollout_id
            else ("row", index)
        )
        reward = (
            plan.shape_reward(trajectory, trajectory.reward)
            if plan
            else trajectory.reward
        )
        reference = (trajectory.comparison_group_id, reward)
        if key in references and references[key] != reference:
            raise ValueError(
                "Fragments of one rollout must share group and outcome reward."
            )
        references[key] = reference
        if config.overlong_tokens:
            excess = max(
                0,
                action_counts[key] - (config.max_tokens - config.overlong_tokens),
            )
            reward -= (
                min(1.0, excess / config.overlong_tokens)
                * config.overlong_penalty_factor
            )
        reward = (reward + config.reward_bias) * config.reward_scaling
        # A logical candidate contributes one normalization observation even
        # when it exports several rows. Leave-one-out baselines therefore
        # exclude the candidate, not an arbitrary physical fragment of it.
        if key not in reward_indices:
            reward_indices[key] = len(rewards)
            rewards.append([max(-config.reward_clip, min(config.reward_clip, reward))])
            reward_groups.append(trajectory.comparison_group_id)
        row_indices.append(reward_indices[key])
        masks.append(list(trajectory.loss_mask))

    normalized = (
        plan.normalize_rewards(rewards, reward_groups, config.reward_norm)
        if plan
        else normalize(rewards, reward_groups, config.reward_norm)
    )
    outcome_scores = [normalized[index] for index in row_indices]
    advantages, returns = [], []
    for trajectory, score, mask in zip(
        trajectories, outcome_scores, masks, strict=True
    ):
        if config.use_critic and trajectory.values is None:
            raise ValueError(
                "Critic-based PPO requires T+1 value predictions per trajectory."
            )
        if (
            config.kl_ctl or config.reference_kl_coef
        ) and trajectory.reference_logprobs is None:
            raise ValueError(
                "Reference KL requires fixed-reference completion logprobs."
            )
        # Keep task/process rewards separate from policy-reference KL.  Token
        # GAE consumes both signals at active tokens. Task rewards enter the turn
        # recurrence, while KL remains a token-local actor penalty and never
        # enters the critic's turn-level return target.
        task_rewards = (
            list(trajectory.token_rewards)
            if trajectory.token_rewards is not None
            else [0.0] * len(mask)
        )
        if plan:
            task_rewards = plan.shape_process(trajectory, task_rewards)
        suppress_reward = config.mask_no_eos_with_zero and trajectory.truncated
        if suppress_reward:
            task_rewards = [0.0] * len(mask)
        kl_rewards = [0.0] * len(mask)
        for index, active in enumerate(mask):
            if active and config.kl_ctl:
                log_ratio = (
                    trajectory.logprobs[index] - trajectory.reference_logprobs[index]
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
        estimator = plan.advantage_estimator if plan else None
        if estimator is None:
            adv, ret = compute_gae(
                gae_rewards,
                trajectory.values if config.use_critic else [0.0] * (len(mask) + 1),
                mask,
                discount=config.discount,
                gae_lambda=config.gae_lambda,
                truncated=trajectory.truncated,
                timestep_unit=config.gae_timestep_unit,
                turn_ids=trajectory.turn_ids,
            )
        else:
            method = getattr(estimator, "estimate", estimator)
            adv, ret = method(
                gae_rewards,
                trajectory.values if config.use_critic else [0.0] * (len(mask) + 1),
                mask,
                discount=config.discount,
                gae_lambda=config.gae_lambda,
                truncated=trajectory.truncated,
                timestep_unit=config.gae_timestep_unit,
                turn_ids=trajectory.turn_ids,
            )
        if plan and plan.return_estimator is not None:
            method = getattr(plan.return_estimator, "estimate", plan.return_estimator)
            ret = method(
                gae_rewards,
                trajectory.values if config.use_critic else [0.0] * (len(mask) + 1),
                mask,
                discount=config.discount,
                gae_lambda=config.gae_lambda,
                truncated=trajectory.truncated,
                timestep_unit=config.gae_timestep_unit,
                turn_ids=trajectory.turn_ids,
            )
        if trajectory.token_advantages is not None:
            if trajectory.token_advantage_mode == "replace":
                adv = list(trajectory.token_advantages)
            else:
                adv = [
                    value + direct
                    for value, direct in zip(
                        adv, trajectory.token_advantages, strict=True
                    )
                ]
        if config.gae_timestep_unit == "turn":
            adv = [value + kl for value, kl in zip(adv, kl_rewards, strict=True)]
        advantages.append(adv)
        returns.append(ret)
    if not any(any(mask) for mask in masks):
        raise ValueError(
            "All trajectories were masked; no policy update can be performed."
        )
    advantages = normalize(advantages, group_ids, config.adv_norm, masks)
    weights = [
        plan.row_weight(trajectory, mask) if plan else 1.0
        for trajectory, mask in zip(trajectories, masks, strict=True)
    ]
    return PreparedBatch(
        list(trajectories),
        advantages,
        returns,
        masks,
        [list(trajectory.logprobs) for trajectory in trajectories],
        loss_weights=weights,
    )
