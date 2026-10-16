"""Independent PyTRIO critic using a logprob-based scalar readout."""

import time
from dataclasses import replace
from typing import Protocol

import numpy as np

from ...api.client_api import call_async, optimizer_params
from ...utils.checkpoint import save_checkpoint_async
from .contracts import Trajectory
from .critic_loss import critic_datums, make_value_loss_fn, values_from_logprobs


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
        self.checkpoint_path = None

    async def compute_values(self, trajectories):
        if not trajectories:
            raise ValueError("Critic scoring requires trajectories.")
        if self.config.critic_n_minibatches > len(trajectories):
            raise ValueError("critic_n_minibatches must not exceed trajectory count.")
        for t in trajectories:
            t.__post_init__()
        result = await call_async(
            self.client,
            "forward",
            critic_datums(trajectories, self.token_ids),
            loss_fn="cross_entropy",
            auto_shift=False,
        )
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
        self.checkpoint_path = await save_checkpoint_async(
            self.client, "ppo-critic", step
        )
        if self.metrics is not None:
            self.metrics.log_text(
                key="checkpoint/path", data=self.checkpoint_path, step=step
            )
        return self.checkpoint_path

    def finish(self, *, state="success", error=None):
        if self.metrics is not None:
            self.metrics.finish(state=state, error=error)
