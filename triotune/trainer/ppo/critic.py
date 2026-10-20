"""PPO critic algorithm and its PyTRIO logprob-based value implementation.

The critic reuses two existing vocabulary markers instead of requiring a new
value head.  It builds causal marker targets, reads an unbounded scalar value
from their logprob difference, and sends clipped value-loss gradients through
PyTRIO's custom loss bridge.  The actor and critic remain independent remote
training clients, while both consume the same frozen rollout targets.
"""

import math
import time
from dataclasses import replace
from typing import Protocol

import numpy as np

from ...api.client_api import call_async, optimizer_params
from ...utils.checkpoint import save_checkpoint_async
from ...utils.datums import materialize_datum
from ...utils.token_usage import compute_train_tokens
from .contracts import Trajectory
from .datum import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Marker-based value readout and clipped value objective
# ---------------------------------------------------------------------------

def critic_datums(trajectories, token_ids):
    """Build positive/negative marker targets on the real causal contexts.

    Marker labels start at the final prompt position and continue through the
    completion plus one bootstrap position.  The marker is never inserted into
    the model input, so each predicted value sees only its preceding context.
    """

    rows = []
    for trajectory in trajectories:
        tokens = trajectory.prompt_tokens + trajectory.completion_tokens
        for marker in token_ids:
            labels = [IGNORE_INDEX] * (len(trajectory.prompt_tokens) - 1) + [marker] * (
                len(trajectory.completion_tokens) + 1
            )
            rows.append(
                materialize_datum(
                    tokens, {"target_tokens": np.asarray(labels, dtype=np.int32)}
                )
            )
    return rows


def values_from_logprobs(logprobs, trajectories, config):
    """Convert marker logprob pairs into one value row per trajectory."""

    if len(logprobs) != 2 * len(trajectories):
        raise ValueError(
            "Critic needs positive and negative logprobs for every trajectory."
        )
    values = []
    for index, trajectory in enumerate(trajectories):
        positive, negative = [
            row.reshape(-1) for row in logprobs[2 * index : 2 * index + 2]
        ]
        expected = len(trajectory.prompt_tokens) + len(trajectory.completion_tokens)
        if len(positive) != expected or len(negative) != expected:
            raise ValueError(
                "Critic logprobs must align with full prompt+completion inputs."
            )
        start = len(trajectory.prompt_tokens) - 1
        values.append(
            config.critic_value_scale * (positive[start:] - negative[start:])
            + config.critic_value_bias
        )
    return values


def value_loss(values, old_values, returns, mask, *, clip: float = 0.2):
    """Compute clipped value regression over active action states."""

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
            (values - returns.detach()).square(),
            (bounded - returns.detach()).square(),
        )[valid].mean()
    )


def make_value_loss_fn(trajectories, returns, masks, config, token_ids):
    """Bind frozen returns and old values for every critic update epoch."""

    if (
        len(trajectories) != len(returns)
        or len(returns) != len(masks)
        or not trajectories
    ):
        raise ValueError("Critic trajectories, returns and masks must align.")
    old, targets, valid = [], [], []
    for trajectory, ret, mask in zip(trajectories, returns, masks, strict=True):
        trajectory.__post_init__()
        if (
            trajectory.values is None
            or len(ret) != len(trajectory.completion_tokens)
            or len(mask) != len(ret)
        ):
            raise ValueError("Critic needs frozen T+1 values, T returns and T masks.")
        if any(not math.isfinite(x) for x in ret) or any(
            not isinstance(x, bool) for x in mask
        ):
            raise ValueError("Critic returns must be finite and masks boolean.")
        old.extend(trajectory.values[:-1])
        targets.extend(ret)
        valid.extend(mask)
    if not any(valid):
        raise ValueError("Critic update has no active action states.")
    expected_targets = [
        datum.loss_fn_inputs["target_tokens"].data
        for datum in critic_datums(trajectories, token_ids)
    ]

    def loss(datums, logprobs):
        import torch

        if len(datums) != len(expected_targets) or any(
            list(datum.loss_fn_inputs["target_tokens"].data) != labels
            for datum, labels in zip(datums, expected_targets, strict=True)
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
        # Keep a differentiable dependency for every marker datum, including
        # rows whose action positions are fully masked by the rollout.
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


class Critic(Protocol):
    async def compute_values(self, trajectories: list[Trajectory]) -> list[list[float]]:
        """Return T+1 values per completion, including bootstrap."""
        ...

    async def update(
        self,
        trajectories: list[Trajectory],
        returns: list[list[float]],
        masks: list[list[bool]],
    ) -> dict[str, float]:
        """Fit frozen returns; an injected implementation owns its lifecycle."""
        ...


class PyTRIOCritic:
    """Separate LM/LoRA/Adam run, sharing the actor's base model/tokenizer.

    The two marker scores are computed on actual causal trajectory contexts,
    not sampled numerical answers. Their difference is an unbounded value.
    """

    def __init__(self, client, config, token_ids, *, metrics=None):
        """Bind an independent PyTRIO client and two value marker token IDs."""

        self.client, self.config, self.token_ids = client, config, tuple(token_ids)
        if (
            len(self.token_ids) != 2
            or self.token_ids[0] == self.token_ids[1]
            or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0
                for t in self.token_ids
            )
        ):
            raise ValueError("Critic needs two distinct positive/negative marker IDs.")
        self._optimizer_config = replace(
            config, learning_rate=config.critic_learning_rate or config.learning_rate
        )
        self.metrics = metrics
        self.global_step = 0
        self.optimizer_steps = 0
        self.train_tokens = 0
        self.checkpoint_path = None

    async def compute_values(self, trajectories):
        """Read one causal T+1 value row per trajectory from remote logprobs."""

        if not trajectories:
            raise ValueError("Critic scoring requires trajectories.")
        if self.config.critic_n_minibatches > len(trajectories):
            raise ValueError("critic_n_minibatches must not exceed trajectory count.")
        for t in trajectories:
            t.__post_init__()
        data = critic_datums(trajectories, self.token_ids)
        result = await call_async(
            self.client,
            "forward",
            data,
            loss_fn="cross_entropy",
            auto_shift=False,
        )
        self.train_tokens += compute_train_tokens(data)
        logprobs = [
            np.asarray(output["logprobs"].tolist(), dtype=np.float64)
            for output in result.loss_fn_outputs
        ]
        if any(not np.isfinite(row).all() for row in logprobs):
            raise FloatingPointError("Critic logprobs are nonfinite.")
        return [
            row.tolist()
            for row in values_from_logprobs(logprobs, trajectories, self.config)
        ]

    async def update(self, trajectories, returns, masks):
        """Fit frozen returns with clipped value regression on the remote client."""

        start = time.perf_counter()
        if (
            not trajectories
            or len(returns) != len(trajectories)
            or len(masks) != len(trajectories)
        ):
            raise ValueError("Critic update inputs must align.")
        active = [i for i, mask in enumerate(masks) if any(mask)]
        if self.config.critic_n_minibatches > len(active):
            raise ValueError(
                "critic_n_minibatches must not exceed active trajectories."
            )
        chunks = []
        for indices in np.array_split(active, self.config.critic_n_minibatches):
            ts, ret, mask = [
                [rows[i] for i in indices] for rows in (trajectories, returns, masks)
            ]
            chunks.append(
                (
                    critic_datums(ts, self.token_ids),
                    make_value_loss_fn(ts, ret, mask, self.config, self.token_ids),
                )
            )
        rows = []
        for _ in range(self.config.critic_update_epochs):
            for data, loss_fn in chunks:
                result = await call_async(
                    self.client, "forward_backward_custom", data, loss_fn=loss_fn
                )
                self.train_tokens += compute_train_tokens(data)
                await call_async(
                    self.client, "optim_step", optimizer_params(self._optimizer_config)
                )
                self.optimizer_steps += 1
                rows.append(
                    {
                        key: float(value)
                        for key, value in result.metrics.items()
                        if key.startswith("critic/")
                        and isinstance(value, (int, float))
                        and np.isfinite(value)
                    }
                )
        self.global_step += 1
        metrics = {
            key: float(np.mean([row[key] for row in rows if key in row]))
            for key in {k for row in rows for k in row}
        }
        metrics.update(
            {
                "critic/optimizer_steps": float(self.optimizer_steps),
                "critic/learning_rate": self._optimizer_config.learning_rate,
                "critic/action_tokens": float(sum(sum(mask) for mask in masks)),
                "time/critic_update": time.perf_counter() - start,
            }
        )
        if self.metrics is not None:
            self.metrics.record(metrics, step=self.global_step)
        return metrics

    async def save_checkpoint(self, step):
        """Persist the critic adapter and publish its checkpoint when enabled."""

        self.checkpoint_path = await save_checkpoint_async(
            self.client, "ppo-critic", step
        )
        if self.metrics is not None:
            self.metrics.log_text(
                key="checkpoint/path", data=self.checkpoint_path, step=step
            )
        return self.checkpoint_path

    def finish(self, *, state="success", error=None):
        """Finish the optional critic metrics reporter."""

        if self.metrics is not None:
            self.metrics.finish(state=state, error=error)
