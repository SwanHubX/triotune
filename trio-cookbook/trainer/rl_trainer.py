"""Asynchronous rollout and optimization orchestration for PPO-family trainers.

The algorithm-specific building blocks live under :mod:`trainer.ppo`; this
module owns the lifecycle that connects them to PyTRIO clients.  A trainer can
run actor-only updates, or coordinate an independent PyTRIO critic alongside
the actor update.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytrio as trio

from ..api.cli_args import PPOConfig
from ..api.client_api import call_async
from ..utils.checkpoint import save_checkpoint_async
from ..utils.metrics import MetricsAPI
from .ppo.actor import PPOActor
from .ppo.contracts import PPOState, Trajectory
from .ppo.critic import Critic, PyTRIOCritic
from ..utils.reporting import isolated_metrics as critic_metrics
from .ppo.rollout import (
    filter_informative_groups,
    gather_cancel_on_error,
    generate_rollouts,
    score_reference,
)


class PPOTrainer:
    """Async PyTRIO trainer with config-selected advantages and policy objectives.

    ``train`` consumes raw prompt rows once; ``step`` accepts custom workflows'
    complete trajectories. PPO creates a separate PyTRIO critic by default;
    callers can still inject an alternative Critic. Initialization is lazy.
    ``max_steps``/``save_every`` count rollout updates, not optimizer minibatches.
    """

    # ------------------------------------------------------------------
    # Trainer construction and dependency ownership
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: str,
        args: PPOConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        reward_fn: Callable | None = None,
        *,
        critic: Critic | None = None,
        lora_rank: int = 32,
        training_client=None,
        critic_training_client=None,
        tokenizer=None,
        reference_client=None,
        service_client=None,
    ) -> None:
        """Create a lazy PPO trainer and validate actor/critic ownership.

        Clients are accepted as dependencies for tests and custom workflows;
        missing clients are created during :meth:`initialize`.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        if (
            isinstance(lora_rank, bool)
            or not isinstance(lora_rank, int)
            or lora_rank < 1
        ):
            raise ValueError("lora_rank must be a positive integer.")
        self.model, self.args = model.strip(), args or PPOConfig()
        if (
            critic is not None or critic_training_client is not None
        ) and not self.args.use_critic:
            raise ValueError("A critic requires use_critic=True.")
        if critic is not None and critic_training_client is not None:
            raise ValueError(
                "Supply either critic or critic_training_client, not both."
            )
        if training_client is not None and training_client is critic_training_client:
            raise ValueError("Actor and critic must use independent training clients.")
        self.train_dataset, self.reward_fn, self.critic = (
            train_dataset,
            reward_fn,
            critic,
        )
        self.lora_rank = lora_rank
        self._client, self._tokenizer = training_client, tokenizer
        self._reference, self._service = reference_client, service_client
        self._critic_client = critic_training_client
        self._owns_critic = self.args.use_critic and critic is None
        self._actor = None
        self._sampling = None
        self._next_group = 0
        self._metrics = None
        self.state = PPOState()

    # ------------------------------------------------------------------
    # Client initialization and reference-model setup
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create clients, tokenizers, reporters, and the actor exactly once."""

        if self._actor is not None:
            return

        # Create the actor client and tokenizer lazily so construction remains
        # side-effect free for configuration and dependency-injection tests.
        if self._client is None:
            if self._service is None:
                self._service = trio.ServiceClient()
            self._client = await call_async(
                self._service,
                "create_lora_training_client",
                base_model=self.model,
                rank=self.lora_rank,
                seed=self.args.seed,
            )
        if self._tokenizer is None:
            # The SDK exposes no asynchronous tokenizer getter, so keep this
            # blocking call off the event loop.
            self._tokenizer = await asyncio.to_thread(self._client.get_tokenizer)

        # The built-in critic is a separate LoRA run with its own optimizer and
        # random seed; an injected critic remains fully owned by the caller.
        if self._owns_critic:
            token_ids = self.args.critic_token_ids
            if token_ids is None:
                markers = [
                    self._tokenizer.encode(text, add_special_tokens=False)
                    for text in ("1", "0")
                ]
                if any(len(ids) != 1 for ids in markers) or markers[0] == markers[1]:
                    raise ValueError(
                        "Set critic_token_ids to two distinct existing single-token markers."
                    )
                token_ids = (int(markers[0][0]), int(markers[1][0]))
            if self._critic_client is None:
                if self._service is None:
                    self._service = trio.ServiceClient()
                self._critic_client = await call_async(
                    self._service,
                    "create_lora_training_client",
                    base_model=self.model,
                    rank=self.args.critic_lora_rank or self.lora_rank,
                    seed=None if self.args.seed is None else self.args.seed + 1,
                )
            if self._critic_client is self._client:
                raise ValueError(
                    "Actor and critic must use independent training clients."
                )
            self.critic = PyTRIOCritic(self._critic_client, self.args, token_ids)

        # Keep actor and critic metrics in separate runs when the trainer owns
        # both clients, while preserving one public actor metrics stream.
        name = self.args.run_name or f"{self.model.rsplit('/', 1)[-1]}-ppo-family"
        actor_args = (
            replace(self.args, run_name=f"{name}-actor")
            if self._owns_critic
            else self.args
        )
        self._metrics = MetricsAPI.from_config(
            actor_args,
            default_name=name,
            metadata={
                "model": self.model,
                "lora_rank": self.lora_rank,
                "role": "actor",
            },
        )
        if self._owns_critic:
            self.critic.metrics = await asyncio.to_thread(
                critic_metrics,
                self.args,
                name=f"{name}-critic",
                metadata={
                    "role": "critic",
                    "model": self.model,
                    "lora_rank": self.args.critic_lora_rank or self.lora_rank,
                    "learning_rate": self.args.critic_learning_rate
                    or self.args.learning_rate,
                    "seed": None if self.args.seed is None else self.args.seed + 1,
                    "critic_token_ids": list(self.critic.token_ids),
                    "value_mapping": "scale * (logp_positive - logp_negative) + bias",
                },
            )
        self._actor = PPOActor(self._client, self.args)

    async def _references(self, trajectories):
        """Populate missing reference logprobs when KL control is enabled."""

        if not self.args.kl_ctl or all(
            t.reference_logprobs is not None for t in trajectories
        ):
            return
        if self._reference is None:
            if self._service is None:
                self._service = trio.ServiceClient()
            # A separate base-model client stays fixed and is never refreshed
            # with policy weights. Callers may inject a frozen reference.
            self._reference = await call_async(
                self._service, "create_sampling_client", base_model=self.model
            )
        await score_reference(self._reference, trajectories)

    # ------------------------------------------------------------------
    # Rollout-batch preparation and actor/critic optimization
    # ------------------------------------------------------------------

    async def step(self, trajectories: list[Trajectory]) -> dict[str, float]:
        """Update one frozen rollout batch.

        The actor and critic consume the same prepared targets.  When both are
        present they use independent clients and can therefore update in
        parallel.  A trainer instance must not receive concurrent ``step``
        calls.
        """
        if not trajectories:
            raise ValueError("An update requires nonempty trajectories.")
        await self.initialize()
        start = time.perf_counter()
        await self._references(trajectories)
        if self.critic is not None:
            values = await self.critic.compute_values(trajectories)
            if len(values) != len(trajectories):
                raise ValueError("Critic must return one T+1 value row per trajectory.")
            for t, row in zip(trajectories, values, strict=True):
                t.values = list(row)
        batch = await self._actor.prepare(trajectories)
        prepared_at = time.perf_counter()
        try:
            if self.critic is None:
                metrics = await self._actor.update(batch)
            else:
                # Both optimizers consume frozen targets and use separate
                # clients, so their remote updates can safely run concurrently.
                metrics, critic_stats = await gather_cancel_on_error(
                    [
                        self._actor.update(batch),
                        self.critic.update(
                            batch.trajectories, batch.returns, batch.masks
                        ),
                    ]
                )
                metrics.update(critic_stats)
        finally:
            # Preserve optimizer counters even if a later remote update fails.
            self.state.optimizer_steps = self._actor.optimizer_steps
            self.state.critic_optimizer_steps = getattr(
                self.critic, "optimizer_steps", 0
            )
            self._sampling = None
        self.state.global_step += 1
        rewards = [t.reward for t in trajectories]
        metrics.update(
            {
                "reward/mean": float(np.mean(rewards)),
                "reward/std": float(np.std(rewards)),
                "train/trajectories": float(len(trajectories)),
                "train/action_tokens": float(sum(sum(mask) for mask in batch.masks)),
                "train/prompt_groups": float(len({t.group_id for t in trajectories})),
                "train/optimizer_steps": float(self.state.optimizer_steps),
                "train/learning_rate": self.args.learning_rate,
                "time/prepare": prepared_at - start,
                "time/update": time.perf_counter() - prepared_at,
            }
        )
        self.state.metrics = self._metrics.record(metrics, step=self.state.global_step)
        if self.args.save_every and self.state.global_step % self.args.save_every == 0:
            await self.save_checkpoint()
        return self.state.metrics

    # ------------------------------------------------------------------
    # Checkpointing and run finalization
    # ------------------------------------------------------------------

    async def save_checkpoint(self) -> str:
        """Save owned actor and critic runs and return the actor checkpoint."""
        await self.initialize()

        async def actor_save():
            self.state.checkpoint_path = await save_checkpoint_async(
                self._client, "ppo", self.state.global_step
            )

        async def critic_save():
            self.state.critic_checkpoint_path = await self.critic.save_checkpoint(
                self.state.global_step
            )

        jobs = [actor_save()]
        if self._owns_critic:
            jobs.append(critic_save())
        await gather_cancel_on_error(jobs)
        return self.state.checkpoint_path

    async def close(self, *, state="success", error=None) -> None:
        """Finish owned reporters; injected critics retain lifecycle ownership."""
        try:
            if self._owns_critic and self.critic is not None:
                await asyncio.to_thread(self.critic.finish, state=state, error=error)
        finally:
            if self._metrics is not None:
                # SwanLab restores signal handlers on finish. The actor run was
                # initialized in this process and must finish on the main thread.
                self._metrics.finish(state=state, error=error)

    # ------------------------------------------------------------------
    # Dataset-driven rollout collection and training entry point
    # ------------------------------------------------------------------

    async def _collect(self, rows):
        """Generate and filter rollouts until one complete batch is available."""

        if self._sampling is None:
            self._sampling = await call_async(
                self._client, "save_weights_and_get_sampling_client"
            )
        accepted, empty = [], 0
        group_count = 0
        while group_count < self.args.batch_size:
            # Dynamic sampling may reject entire prompt groups, so keep reading
            # source rows until the target number of informative groups exists.
            source = list(itertools.islice(rows, self.args.batch_size - group_count))
            if not source:
                break
            generated = await generate_rollouts(
                self._sampling,
                self._tokenizer,
                source,
                self.reward_fn,
                self.args,
                first_group=self._next_group,
            )
            self._next_group += len(source)
            if self.args.dynamic_sampling:
                generated = filter_informative_groups(generated)
            accepted.extend(generated)
            group_count = len({t.group_id for t in accepted})
            empty = empty + 1 if not generated else 0
            if empty >= self.args.max_empty_batches:
                raise RuntimeError(
                    "Dynamic sampling found too many consecutive uninformative batches."
                )
        return accepted

    async def train(self) -> PPOState:
        """Collect rollout batches until ``max_steps`` and return final state."""
        if self.train_dataset is None or not callable(self.reward_fn):
            raise ValueError(
                "train() requires train_dataset and reward_fn; use step() for custom trajectories."
            )
        try:
            await self.initialize()
            rows = iter(self.train_dataset)
            while (
                self.args.max_steps is None
                or self.state.global_step < self.args.max_steps
            ):
                start = time.perf_counter()
                trajectories = await self._collect(rows)
                rollout_time = time.perf_counter() - start
                if not trajectories:
                    break
                await self.step(trajectories)
                self.state.metrics.update(
                    self._metrics.publish(
                        {
                            "time/rollout": rollout_time,
                            "time/step": time.perf_counter() - start,
                        },
                        step=self.state.global_step,
                    )
                )
            if not self.state.global_step:
                raise ValueError("No trainable rollout batch was collected.")
            if (
                not self.args.save_every
                or self.state.global_step % self.args.save_every
            ):
                await self.save_checkpoint()
            self._metrics.log_text(
                key="checkpoint/path",
                data=self.state.checkpoint_path,
                step=self.state.global_step,
            )
            await self.close()
        except BaseException as error:
            try:
                await self.close(state="crashed", error=str(error))
            except Exception as cleanup_error:
                error.add_note(f"Reporter cleanup also failed: {cleanup_error}")
            raise
        return self.state


# Keep ``trio_cookbook.trainer.ppo.trainer`` as an identity-preserving alias
# for integrations that patch or pickle the historical module path.
_ppo_module = __import__(f"{__package__}.ppo", fromlist=["*"])
_ppo_module.PPOTrainer = PPOTrainer

sys.modules[f"{__package__}.ppo.trainer"] = sys.modules[__name__]
del _ppo_module
