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
from typing import Any

import numpy as np
import pytrio as trio
from tqdm import tqdm

from ..api.cli_args import PPOConfig
from ..api.client_api import call_async
from ..utils.checkpoint import save_checkpoint_async
from ..utils.dataloader import iter_batches
from ..utils.metrics import MetricsAPI
from ..utils.token_usage import SampleTokens
from .ppo.actor import PPOActor
from .ppo.contracts import PPOState, Trajectory
from .ppo.critic import Critic, PyTRIOCritic
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
    ``eval_dataset`` must be replayable when periodic evaluation is enabled;
    evaluation samples the current actor and scores the same task reward without
    reference/critic scoring or any optimizer update.
    """

    # ------------------------------------------------------------------
    # Trainer construction and dependency ownership
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: str | None = None,
        args: PPOConfig | None = None,
        train_dataset: Iterable[Mapping[str, Any]] | None = None,
        reward_fn: Callable | None = None,
        *,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
        reference_model: str | None = None,
        critic: Critic | None = None,
        lora_rank: int | None = None,
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
        self.args = args or PPOConfig()
        configured_model = self.args.model
        if model is None and configured_model is not None:
            model = configured_model.name
        if reference_model is None and self.args.reference is not None:
            reference_model = self.args.reference.name
        if lora_rank is None:
            lora_rank = configured_model.lora_rank if configured_model is not None else 32
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        if reference_model is not None and (
            not isinstance(reference_model, str) or not reference_model.strip()
        ):
            raise ValueError("reference_model must be a non-empty model ID when provided.")
        if (
            isinstance(lora_rank, bool)
            or not isinstance(lora_rank, int)
            or lora_rank < 1
        ):
            raise ValueError("lora_rank must be a positive integer.")
        self.model, self.reference_model = (
            model.strip(),
            reference_model.strip() if reference_model is not None else None,
        )
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
        self.train_dataset, self.eval_dataset, self.reward_fn, self.critic = (
            train_dataset,
            eval_dataset,
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
        self._sample_input_tokens = 0
        self._sample_output_tokens = 0
        self._eval_sample_input_tokens = 0
        self._eval_sample_output_tokens = 0
        self._last_eval_step: int | None = None
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

        # Actor and critic remain independent training clients, but one PPO
        # experiment owns one metric run with component-specific namespaces.
        name = self.args.run_name or f"{self.model.rsplit('/', 1)[-1]}-ppo-family"
        self._metrics = MetricsAPI.from_config(
            self.args,
            default_name=name,
            metadata={
                "model": self.model,
                "lora_rank": self.lora_rank,
                "components": ["actor", "critic"] if self._owns_critic else ["actor"],
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
                self._service,
                "create_sampling_client",
                base_model=self.reference_model or self.model,
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
        initial_train_tokens = self._actor.train_tokens + int(
            getattr(self.critic, "train_tokens", 0)
        )
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
        total_train_tokens = self._actor.train_tokens + int(
            getattr(self.critic, "train_tokens", 0)
        )
        train_tokens = total_train_tokens - initial_train_tokens
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
                "usage/train_tokens": float(train_tokens),
                "usage/train_tokens_total": float(total_train_tokens),
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
        """Save owned actor and critic states and return the actor checkpoint."""
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
        self._metrics.log_text(
            key="checkpoint/actor_path",
            data=self.state.checkpoint_path,
            step=self.state.global_step,
        )
        if self.state.critic_checkpoint_path is not None:
            self._metrics.log_text(
                key="checkpoint/critic_path",
                data=self.state.critic_checkpoint_path,
                step=self.state.global_step,
            )
        return self.state.checkpoint_path

    async def close(self, *, state="success", error=None) -> None:
        """Finish the experiment reporter; injected critics retain ownership."""

        if self._metrics is not None:
            # SwanLab restores signal handlers on finish and therefore remains
            # on the main thread even though actor/critic updates are concurrent.
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
                sample_usage_callback=self._record_sample_usage,
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

    def _record_sample_usage(self, usage: SampleTokens) -> None:
        """Accumulate authoritative PyTRIO sampling usage across refill calls."""

        self._sample_input_tokens += usage.input_tokens
        self._sample_output_tokens += usage.output_tokens

    # ------------------------------------------------------------------
    # Reward evaluation shared by every PPO-family preset
    # ------------------------------------------------------------------

    def _evaluation_batches(
        self, dataset: Iterable[Mapping[str, Any]]
    ) -> Iterable[list[Mapping[str, Any]]]:
        """Yield validation prompts under the shared evaluation limits."""

        batch_size = self.args.eval_batch_size or self.args.batch_size
        yield from iter_batches(
            dataset,
            batch_size,
            max_batches=self.args.eval_max_batches,
        )

    def _publish_evaluation(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Store validation metrics without advancing any training counter."""

        values = {str(name): float(value) for name, value in metrics.items()}
        self._last_eval_step = self.state.global_step
        self.state.metrics.update(values)
        self._metrics.publish(values, step=self.state.global_step)
        return values

    async def evaluate(
        self,
        eval_dataset: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Sample the current actor and aggregate task reward without updates.

        The method is algorithm-agnostic: PPO, GRPO, CISPO and every other
        preset share the same prompt rendering, rollout and reward contract.
        Evaluation deliberately skips reference scoring, critic values,
        advantage construction, backward and optimizer calls.
        """

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("eval_dataset is required for evaluate().")
        if not callable(self.reward_fn):
            raise ValueError("evaluate() requires reward_fn.")
        await self.initialize()
        sampler = await call_async(
            self._client, "save_weights_and_get_sampling_client"
        )
        started = time.perf_counter()
        trajectories: list[Trajectory] = []
        prompt_count = 0
        batch_count = 0
        sample_input_tokens = 0
        sample_output_tokens = 0

        def record_usage(usage: SampleTokens) -> None:
            nonlocal sample_input_tokens, sample_output_tokens
            sample_input_tokens += usage.input_tokens
            sample_output_tokens += usage.output_tokens

        for rows in self._evaluation_batches(dataset):
            batch = await generate_rollouts(
                sampler,
                self._tokenizer,
                rows,
                self.reward_fn,
                self.args,
                first_group=prompt_count,
                sample_usage_callback=record_usage,
            )
            trajectories.extend(batch)
            prompt_count += len(rows)
            batch_count += 1
        if not trajectories:
            raise ValueError("eval_dataset must contain at least one prompt.")

        rewards = np.asarray([item.reward for item in trajectories], dtype=np.float64)
        lengths = np.asarray(
            [len(item.completion_tokens) for item in trajectories],
            dtype=np.float64,
        )
        self._eval_sample_input_tokens += sample_input_tokens
        self._eval_sample_output_tokens += sample_output_tokens
        metrics = {
            "eval/reward_mean": float(np.mean(rewards)),
            "eval/reward_std": float(np.std(rewards)),
            "eval/prompts": float(prompt_count),
            "eval/trajectories": float(len(trajectories)),
            "eval/batches": float(batch_count),
            "eval/completion_tokens": float(np.sum(lengths)),
            "eval/completion_length_mean": float(np.mean(lengths)),
            "eval/completion_length_min": float(np.min(lengths)),
            "eval/completion_length_max": float(np.max(lengths)),
            "eval/truncated_fraction": float(
                np.mean([item.truncated for item in trajectories])
            ),
            "usage/eval_sample_input_tokens": float(sample_input_tokens),
            "usage/eval_sample_output_tokens": float(sample_output_tokens),
            "usage/eval_sample_input_tokens_total": float(
                self._eval_sample_input_tokens
            ),
            "usage/eval_sample_output_tokens_total": float(
                self._eval_sample_output_tokens
            ),
            "time/eval": time.perf_counter() - started,
        }
        return self._publish_evaluation(metrics)

    async def train(self) -> PPOState:
        """Collect rollout batches until ``max_steps`` and return final state."""
        if self.train_dataset is None or not callable(self.reward_fn):
            raise ValueError(
                "train() requires train_dataset and reward_fn; use step() for custom trajectories."
            )
        started = time.perf_counter()
        initial_step = self.state.global_step
        progress = tqdm(
            total=self.args.max_steps,
            initial=initial_step,
            desc="PPO train",
            unit="step",
        )
        try:
            await self.initialize()
            if (
                self.eval_dataset is not None
                and self.args.eval_before_train
                and self._last_eval_step is None
            ):
                await self.evaluate()
            rows = iter(self.train_dataset)
            while (
                self.args.max_steps is None
                or self.state.global_step < self.args.max_steps
            ):
                start = time.perf_counter()
                initial_sample_input = self._sample_input_tokens
                initial_sample_output = self._sample_output_tokens
                trajectories = await self._collect(rows)
                rollout_time = time.perf_counter() - start
                if not trajectories:
                    break
                await self.step(trajectories)
                sample_metrics = {
                    "usage/sample_input_tokens": float(
                        self._sample_input_tokens - initial_sample_input
                    ),
                    "usage/sample_output_tokens": float(
                        self._sample_output_tokens - initial_sample_output
                    ),
                    "usage/sample_input_tokens_total": float(
                        self._sample_input_tokens
                    ),
                    "usage/sample_output_tokens_total": float(
                        self._sample_output_tokens
                    ),
                }
                self.state.metrics.update(
                    self._metrics.publish(
                        sample_metrics,
                        step=self.state.global_step,
                    )
                )
                progress.set_postfix(
                    {
                        label: f"{self.state.metrics[name]:.4f}"
                        for label, name in (
                            ("loss", "train/loss"),
                            ("reward", "reward/mean"),
                            ("critic_loss", "critic/loss"),
                            ("grad_norm", "train/grad_norm"),
                            ("tokens", "usage/train_tokens_total"),
                        )
                        if name in self.state.metrics
                    },
                    refresh=False,
                )
                progress.update(1)
                self.state.metrics.update(
                    self._metrics.publish(
                        {
                            "time/rollout": rollout_time,
                            "time/step": time.perf_counter() - start,
                        },
                        step=self.state.global_step,
                    )
                )
                if (
                    self.eval_dataset is not None
                    and self.args.eval_steps > 0
                    and self.state.global_step % self.args.eval_steps == 0
                    and self._last_eval_step != self.state.global_step
                ):
                    await self.evaluate()
            if not self.state.global_step:
                raise ValueError("No trainable rollout batch was collected.")
            if (
                not self.args.save_every
                or self.state.global_step % self.args.save_every
            ):
                await self.save_checkpoint()
            if (
                self.eval_dataset is not None
                and self.args.eval_after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self.evaluate()
            elapsed = time.perf_counter() - started
            summary = {
                "time/train_runtime": elapsed,
                "time/train_steps_per_second": (
                    self.state.global_step - initial_step
                )
                / elapsed,
                "usage/train_tokens_total": self.state.metrics.get(
                    "usage/train_tokens_total", 0.0
                ),
                "usage/sample_input_tokens_total": float(self._sample_input_tokens),
                "usage/sample_output_tokens_total": float(self._sample_output_tokens),
            }
            self.state.metrics.update(summary)
            self._metrics.publish(
                summary,
                step=self.state.global_step,
            )
            await self.close()
        except BaseException as error:
            try:
                await self.close(state="crashed", error=str(error))
            except Exception as cleanup_error:
                error.add_note(f"Reporter cleanup also failed: {cleanup_error}")
            raise
        finally:
            progress.close()
        return self.state


# Keep ``triotune.trainer.ppo.trainer`` as an identity-preserving alias
# for integrations that patch or pickle the historical module path.
_ppo_module = __import__(f"{__package__}.ppo", fromlist=["*"])
_ppo_module.PPOTrainer = PPOTrainer

sys.modules[f"{__package__}.ppo.trainer"] = sys.modules[__name__]
del _ppo_module
