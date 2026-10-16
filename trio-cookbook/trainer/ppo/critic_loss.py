"""Train an unbounded scalar value through two existing vocabulary logits.

V(s) = scale * (log p(positive|s) - log p(negative|s)) + bias.
The log-softmax normalizer cancels. Both logprobs remain differentiable through
PyTRIO's custom CE gradient bridge, without a new model head or token sampling.
"""

import math

import numpy as np

from ...utils.datums import materialize_datum


def critic_datums(trajectories, token_ids):
    """Two teacher-forced passes per trajectory, preserving actual contexts.

    Input is the full prompt+completion; targets at positions P-1..P+T-1
    are a fixed marker. These positions read V(s_0)..V(s_T). In particular
    the first value cannot see the first action, and the last is bootstrap.
    """
    rows = []
    for t in trajectories:
        tokens = t.prompt_tokens + t.completion_tokens
        for marker in token_ids:
            labels = [-100] * (len(t.prompt_tokens) - 1) + [marker] * (
                len(t.completion_tokens) + 1
            )
            rows.append(
                materialize_datum(
                    tokens, {"target_tokens": np.asarray(labels, dtype=np.int32)}
                )
            )
    return rows


def values_from_logprobs(logprobs, trajectories, config):
    """Works with numpy arrays for scoring and torch tensors for autograd."""
    if len(logprobs) != 2 * len(trajectories):
        raise ValueError(
            "Critic needs positive and negative logprobs for every trajectory."
        )
    values = []
    for i, t in enumerate(trajectories):
        positive, negative = [row.reshape(-1) for row in logprobs[2 * i : 2 * i + 2]]
        expected = len(t.prompt_tokens) + len(t.completion_tokens)
        if len(positive) != expected or len(negative) != expected:
            raise ValueError(
                "Critic logprobs must align with full prompt+completion inputs."
            )
        start = len(t.prompt_tokens) - 1
        values.append(
            config.critic_value_scale * (positive[start:] - negative[start:])
            + config.critic_value_bias
        )
    return values


def value_loss(values, old_values, returns, mask, *, clip: float = 0.2):
    """Clipped value regression on action states, excluding bootstrap."""
    import torch

    if not math.isfinite(clip) or clip < 0:
        raise ValueError("Value clip must be finite and nonnegative.")
    if not (values.shape == old_values.shape == returns.shape == mask.shape):
        raise ValueError("Value predictions, targets and masks must align.")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("Value loss requires active tokens.")
    bounded = old_values.detach() + (values - old_values.detach()).clamp(-clip, clip)
    return (
        0.5
        * torch.maximum(
            (values - returns.detach()).square(), (bounded - returns.detach()).square()
        )[valid].mean()
    )


def make_value_loss_fn(trajectories, returns, masks, config, token_ids):
    """Bind frozen GAE targets and old values for every critic update epoch."""
    if (
        len(trajectories) != len(returns)
        or len(returns) != len(masks)
        or not trajectories
    ):
        raise ValueError("Critic trajectories, returns and masks must align.")
    old, targets, valid = [], [], []
    for t, ret, mask in zip(trajectories, returns, masks, strict=True):
        t.__post_init__()
        if (
            t.values is None
            or len(ret) != len(t.completion_tokens)
            or len(mask) != len(ret)
        ):
            raise ValueError("Critic needs frozen T+1 values, T returns and T masks.")
        if any(not math.isfinite(x) for x in ret) or any(
            not isinstance(x, bool) for x in mask
        ):
            raise ValueError("Critic returns must be finite and masks boolean.")
        old.extend(t.values[:-1])
        targets.extend(ret)
        valid.extend(mask)
    if not any(valid):
        raise ValueError("Critic update has no active action states.")
    expected_targets = [
        d.loss_fn_inputs["target_tokens"].data
        for d in critic_datums(trajectories, token_ids)
    ]

    def loss(datums, logprobs):
        import torch

        if len(datums) != len(expected_targets) or any(
            list(d.loss_fn_inputs["target_tokens"].data) != labels
            for d, labels in zip(datums, expected_targets, strict=True)
        ):
            raise ValueError("Critic datum ordering or marker targets changed.")
        rows = values_from_logprobs(logprobs, trajectories, config)
        prediction = torch.cat([row[:-1] for row in rows])
        mask = torch.tensor(valid, device=prediction.device, dtype=torch.bool)
        previous, target = prediction.new_tensor(old), prediction.new_tensor(targets)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Critic values are nonfinite.")
        objective = value_loss(
            prediction, previous, target, mask, clip=config.critic_eps_clip
        )
        objective = objective + sum(lp.sum() * 0 for lp in logprobs)
        if not torch.isfinite(objective):
            raise FloatingPointError("Critic loss is nonfinite.")
        with torch.no_grad():
            bounded = previous + (prediction - previous).clamp(
                -config.critic_eps_clip, config.critic_eps_clip
            )
            clip_mask = (bounded - target).square() > (prediction - target).square()
        return objective, {
            "critic/loss": objective.detach().item(),
            "critic/value_mean": prediction.detach()[mask].mean().item(),
            "critic/return_mean": target[mask].mean().item(),
            "critic/clip_fraction": clip_mask[mask].float().mean().item(),
        }

    return loss
