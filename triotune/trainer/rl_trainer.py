"""Asynchronous rollout and optimization orchestration for PPO-family trainers.

The algorithm-specific building blocks live under :mod:`trainer.ppo`; this
module owns the lifecycle that connects them to PyTRIO clients.  A trainer can
run actor-only updates, or coordinate an independent PyTRIO critic alongside
the actor update.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pytrio as trio
from tqdm import tqdm

from ..api.cli_args import PPOConfig
from ..api.client_api import call_async
from ..utils.checkpoint import TrainingCheckpoint, save_checkpoint_async, training_limit
from ..utils.dataloader import iter_batches, num_batches
from ..utils.metrics import MetricsAPI
from ..utils.token_usage import SampleTokens
from .ppo.actor import PPOActor
from .ppo.advantages import PPOAlgorithmPlan
from .ppo.critic import Critic, PyTRIOCritic
from .ppo.rollout import (
    filter_informative_groups,
    gather_cancel_on_error,
    generate_rollouts,
    score_reference,
)
from .ppo.trajectory import PPOState, Trajectory, validate_policy_versions

if TYPE_CHECKING:
    from ..agentrl.runtime.sampling import SamplingSnapshot


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
        algorithm_plan: PPOAlgorithmPlan | None = None,
        plan: PPOAlgorithmPlan | None = None,
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
            lora_rank = (
                configured_model.lora_rank if configured_model is not None else 32
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a PyTRIO base-model ID.")
        if reference_model is not None and (
            not isinstance(reference_model, str) or not reference_model.strip()
        ):
            raise ValueError(
                "reference_model must be a non-empty model ID when provided."
            )
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
        if algorithm_plan is not None and plan is not None:
            raise ValueError("Supply either algorithm_plan or plan, not both.")
        configured_plan = getattr(self.args, "algorithm_plan", None)
        if (
            sum(value is not None for value in (algorithm_plan, plan, configured_plan))
            > 1
        ):
            raise ValueError("Supply only one PPO algorithm plan.")
        self.algorithm_plan = algorithm_plan or plan or configured_plan
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
        self._rollout_metrics: dict[str, float] = {}
        self.state = PPOState()
        self._checkpoint = TrainingCheckpoint(
            self.args, "ppo", train_dataset, eval_dataset,
            identity={"model": self.model, "lora_rank": lora_rank, "reference": self.reference_model},
        )
        if self._checkpoint.enabled and any(value is not None for value in (
            critic, self.algorithm_plan, training_client, critic_training_client,
            reference_client, tokenizer,
        )):
            raise ValueError("Complete PPO checkpoints require owned PyTRIO clients and built-in algorithm plans.")

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
            self._client = await self._checkpoint.create_client_async(
                self._service,
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
                self._critic_client = await self._checkpoint.create_client_async(
                    self._service, component="critic",
                    base_model=self.model,
                    rank=self.args.critic_lora_rank or self.lora_rank,
                    seed=None if self.args.seed is None else self.args.seed + 1,
                )
            if self._critic_client is self._client:
                raise ValueError(
                    "Actor and critic must use independent training clients."
                )
            self.critic = PyTRIOCritic(self._critic_client, self.args, token_ids, total_steps=self._checkpoint.optimizer_horizons["critic"])

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
            logging_step=self.args.logging_step,
        )
        self._actor = PPOActor(self._client, self.args, self.algorithm_plan, total_steps=self._checkpoint.optimizer_horizons["actor"])
        self._checkpoint.restore(self)

    async def _references(self, trajectories):
        """Populate fixed-reference scores for reward or actor-loss KL."""

        if not (self.args.kl_ctl or self.args.reference_kl_coef) or all(
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

    async def create_sampling_snapshot(self, *, stop=None) -> SamplingSnapshot:
        """Export current actor weights for an entire custom agent batch.

        The returned sampling client is a fixed exported policy. Collection may
        continue from that client while the training client updates newer actor
        weights, but one logical rollout must never switch snapshots mid-run.
        Consumers must enforce policy-version staleness when the completed
        rollout later enters ``step``.
        """
        from ..agentrl.runtime.sampling import PyTRIOAgentSampler, SamplingSnapshot

        await self.initialize()
        self._sampling = await call_async(
            self._client, "save_weights_and_get_sampling_client"
        )
        return SamplingSnapshot(
            f"step-{self.state.global_step}-{uuid4().hex}",
            PyTRIOAgentSampler(self._sampling, self._tokenizer, stop=stop),
            self._tokenizer,
            policy_version=self.state.global_step,
        )

    async def publish_metrics(self, metrics: Mapping[str, float]) -> dict[str, float]:
        """Publish custom workflow metrics through the trainer-owned reporter."""
        await self.initialize()
        payload = self._metrics.publish(metrics, step=self.state.global_step)
        self.state.metrics.update(payload)
        return payload

    # ------------------------------------------------------------------
    # Rollout-batch preparation and actor/critic optimization
    # ------------------------------------------------------------------

    async def step(
        self, trajectories: list[Trajectory], *, max_staleness: int = 0
    ) -> dict[str, float]:
        """Update one frozen rollout batch.

        The actor and critic consume the same prepared targets.  When both are
        present they use independent clients and can therefore update in
        parallel.  A trainer instance must not receive concurrent ``step``
        calls.
        """
        self._checkpoint.require_training_loop()
        if not trajectories:
            raise ValueError("An update requires nonempty trajectories.")
        # Validate provenance before reference scoring or critic work. The
        # actor repeats this check as a defense for callers using it directly.
        validate_policy_versions(
            trajectories,
            current_version=self.state.global_step,
            max_staleness=max_staleness,
        )
        policy_ages = [
            self.state.global_step - trajectory.policy_version
            for trajectory in trajectories
            if type(trajectory.policy_version) is int
        ]
        await self.initialize()
        initial_actor_tokens = self._actor.train_tokens
        initial_critic_tokens = int(getattr(self.critic, "train_tokens", 0))
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
        total_actor_tokens = self._actor.train_tokens
        total_critic_tokens = int(getattr(self.critic, "train_tokens", 0))
        actor_tokens = total_actor_tokens - initial_actor_tokens
        critic_tokens = total_critic_tokens - initial_critic_tokens
        # Outcome statistics describe logical candidates; token usage and
        # physical trajectory counts continue to describe actual remote work.
        logical = {
            t.logical_sequence_id if t.rollout_id else ("row", i): t
            for i, t in enumerate(trajectories)
        }
        rewards = [t.reward for t in logical.values()]
        active_logprobs = [
            logprob
            for trajectory in batch.trajectories
            for logprob, active in zip(
                trajectory.logprobs, trajectory.loss_mask, strict=True
            )
            if active
        ]
        active_advantages = [
            value
            for row, mask in zip(batch.advantages, batch.masks, strict=True)
            for value, active in zip(row, mask, strict=True)
            if active
        ]
        active_returns = [
            value
            for row, mask in zip(batch.returns, batch.masks, strict=True)
            for value, active in zip(row, mask, strict=True)
            if active
        ]
        lengths = [len(t.completion_tokens) for t in trajectories]
        reward_components = {
            name: [
                trajectory.reward_components[name]
                for trajectory in logical.values()
                if name in trajectory.reward_components
            ]
            for name in {
                name
                for trajectory in logical.values()
                for name in trajectory.reward_components
            }
        }
        reference_differences = [
            behavior - reference
            for trajectory in batch.trajectories
            if trajectory.reference_logprobs is not None
            for behavior, reference, active in zip(
                trajectory.logprobs,
                trajectory.reference_logprobs,
                trajectory.loss_mask,
                strict=True,
            )
            if active
        ]
        overlong_penalties = []
        if self.args.overlong_tokens:
            logical_lengths = {}
            for index, trajectory in enumerate(trajectories):
                key = (
                    trajectory.logical_sequence_id
                    if trajectory.rollout_id
                    else ("row", index)
                )
                logical_lengths[key] = logical_lengths.get(key, 0) + sum(
                    trajectory.loss_mask
                )
            for length in logical_lengths.values():
                excess = max(
                    0,
                    length - (self.args.max_tokens - self.args.overlong_tokens),
                )
                overlong_penalties.append(
                    min(1.0, excess / self.args.overlong_tokens)
                    * self.args.overlong_penalty_factor
                )
        metrics.update(
            {
                "reward/mean": float(np.mean(rewards)),
                "reward/std": float(np.std(rewards)),
                "reward/min": float(np.min(rewards)),
                "reward/max": float(np.max(rewards)),
                "policy/entropy_sampled": float(-np.mean(active_logprobs)),
                "advantage/mean": float(np.mean(active_advantages)),
                "advantage/std": float(np.std(active_advantages)),
                "return/mean": float(np.mean(active_returns)),
                "return/std": float(np.std(active_returns)),
                "rollout/completion_length_mean": float(np.mean(lengths)),
                "rollout/completion_length_min": float(np.min(lengths)),
                "rollout/completion_length_max": float(np.max(lengths)),
                "rollout/truncated_fraction": float(
                    np.mean([t.truncated for t in trajectories])
                ),
                "train/trajectories": float(len(trajectories)),
                "train/logical_rollouts": float(
                    len(
                        {t.rollout_id or ("row", i) for i, t in enumerate(trajectories)}
                    )
                ),
                "train/logical_sequences": float(len(logical)),
                "train/action_tokens": float(
                    sum(sum(t.effective_action_weights) for t in batch.trajectories)
                ),
                "train/prompt_groups": float(len({t.group_id for t in trajectories})),
                "train/reward_groups": float(
                    len({t.comparison_group_id for t in trajectories})
                ),
                "train/policy_staleness_max": float(max(policy_ages, default=0)),
                "train/policy_staleness_mean": float(
                    np.mean(policy_ages) if policy_ages else 0.0
                ),
                "train/optimizer_steps": float(self.state.optimizer_steps),
                "train/learning_rate": metrics.pop("optim/lr", self.args.learning_rate),
                "usage/actor_train_tokens": float(actor_tokens),
                "usage/actor_train_tokens_total": float(total_actor_tokens),
                "usage/critic_train_tokens": float(critic_tokens),
                "usage/critic_train_tokens_total": float(total_critic_tokens),
                "usage/train_tokens": float(actor_tokens + critic_tokens),
                "usage/train_tokens_total": float(
                    total_actor_tokens + total_critic_tokens
                ),
                "time/prepare": prepared_at - start,
                "time/update": time.perf_counter() - prepared_at,
            }
        )
        if reference_differences:
            sampled_kl = float(np.mean(reference_differences))
            estimated_kl = [
                difference
                if self.args.kl_estimator == "k1"
                else 0.5 * difference**2
                if self.args.kl_estimator == "k2"
                else math.expm1(min(700.0, -difference)) + difference
                for difference in reference_differences
            ]
            metrics["policy/reference_kl_sampled"] = sampled_kl
            if self.args.kl_ctl:
                metrics["reward/kl_penalty_mean"] = float(
                    np.mean(
                        [
                            -self.args.kl_ctl * max(-10.0, min(10.0, value))
                            for value in estimated_kl
                        ]
                    )
                )
        if overlong_penalties:
            metrics.update(
                {
                    "reward/overlong_penalty_mean": float(np.mean(overlong_penalties)),
                    "rollout/overlong_fraction": float(
                        np.mean([value > 0 for value in overlong_penalties])
                    ),
                }
            )
        if set(rewards) <= {0.0, 1.0}:
            metrics["reward/success_rate"] = float(np.mean(rewards))
        for name, values in reward_components.items():
            metrics[f"reward/{name}/mean"] = float(np.mean(values))
        if "success" in reward_components:
            metrics["reward/success_rate"] = float(
                np.mean(reward_components["success"])
            )
        metrics.update(self._rollout_metrics)
        self._rollout_metrics = {}
        action_weight = metrics["train/action_tokens"]
        loss_weight = (
            action_weight
            if self.args.loss_aggregation == "token_mean"
            else metrics["train/trajectories"]
            if self.args.loss_aggregation in {"seq_mean", "constant"}
            else metrics["train/prompt_groups"]
        )
        weights = {
            name: action_weight
            for name in metrics
            if name.startswith(("policy/", "advantage/", "return/"))
            or (
                name.startswith("critic/")
                and name
                not in {
                    "critic/grad_norm",
                    "critic/learning_rate",
                    "critic/optimizer_steps",
                }
            )
        }
        weights["train/loss"] = loss_weight
        weights.update(
            {
                name: metrics["train/trajectories"]
                for name in metrics
                if name.startswith(("reward/", "rollout/completion_length"))
                or name == "rollout/truncated_fraction"
            }
        )
        self.state.metrics = self._metrics.record(
            metrics,
            step=self.state.global_step,
            weights=weights,
        )
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
        self._checkpoint.remote_saved(self)
        return self.state.checkpoint_path

    async def close(self, *, state="success", error=None, flush=True) -> None:
        """Finish the experiment reporter; injected critics retain ownership."""

        if self._metrics is not None:
            # SwanLab restores signal handlers on finish and therefore remains
            # on the main thread even though actor/critic updates are concurrent.
            self._metrics.finish(state=state, error=error, flush=flush)

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
        sampled_groups = rejected_groups = refill_count = 0
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
            # The sampler is an export of this exact completed policy version.
            # Recreated clients after resume receive a new physical run ID but
            # retain the same logical version and the same per-group seed range.
            for trajectory in generated:
                trajectory.policy_version = self.state.global_step
                trajectory.snapshot_id = getattr(
                    self._sampling, "sampling_run_cuid", f"step-{self.state.global_step}"
                )
            self._next_group += len(source)
            sampled_groups += len(source)
            if self.args.dynamic_sampling:
                filtered = filter_informative_groups(generated)
                rejected_groups += len(source) - len(
                    {trajectory.group_id for trajectory in filtered}
                )
                generated = filtered
                if (
                    len({trajectory.group_id for trajectory in accepted + generated})
                    < self.args.batch_size
                ):
                    refill_count += 1
            accepted.extend(generated)
            group_count = len({t.group_id for t in accepted})
            empty = empty + 1 if not generated else 0
            if empty >= self.args.max_empty_batches:
                raise RuntimeError(
                    "Dynamic sampling found too many consecutive uninformative batches."
                )
        if self.args.dynamic_sampling:
            self._rollout_metrics = {
                "rollout/degenerate_group_fraction": (
                    rejected_groups / sampled_groups if sampled_groups else 0.0
                ),
                "rollout/rejected_groups": float(rejected_groups),
                "rollout/refill_count": float(refill_count),
            }
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
        sampler = await call_async(self._client, "save_weights_and_get_sampling_client")
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

        batch_size = self.args.eval_batch_size or self.args.batch_size
        for rows in tqdm(
            self._evaluation_batches(dataset),
            total=num_batches(
                dataset,
                batch_size,
                max_batches=self.args.eval_max_batches,
            ),
            desc="PPO eval",
            unit="batch",
        ):
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
            "eval/reward_min": float(np.min(rewards)),
            "eval/reward_max": float(np.max(rewards)),
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
        components = {
            name: [
                trajectory.reward_components[name]
                for trajectory in trajectories
                if name in trajectory.reward_components
            ]
            for name in {
                name
                for trajectory in trajectories
                for name in trajectory.reward_components
            }
        }
        if set(rewards.tolist()) <= {0.0, 1.0}:
            metrics["eval/success_rate"] = float(np.mean(rewards))
        for name, values in components.items():
            metrics[f"eval/reward_{name}_mean"] = float(np.mean(values))
        if "success" in components:
            metrics["eval/success_rate"] = float(np.mean(components["success"]))
        return self._publish_evaluation(metrics)

    async def train(self, *, stop_after_step: int | None = None) -> PPOState:
        """Collect rollout batches until ``max_steps`` and return final state."""
        if self.train_dataset is None or not callable(self.reward_fn):
            raise ValueError(
                "train() requires train_dataset and reward_fn; use step() for custom trajectories."
            )
        self._checkpoint.bind_loop()
        started = time.perf_counter()
        progress = None
        try:
            await self.initialize()
            limit, paused = training_limit(self.args, self.state.global_step, stop_after_step)
            self._checkpoint.begin()
            initial_step = self.state.global_step
            progress = tqdm(
                total=self.args.max_steps,
                initial=initial_step,
                desc="PPO train",
                unit="step",
            )
            if (
                self._checkpoint.evaluation_due(self, self._last_eval_step)
            ):
                await self.evaluate()
            rows = self._checkpoint.rows(self)
            while (
                limit is None
                or self.state.global_step < limit
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
                    "usage/sample_input_tokens_total": float(self._sample_input_tokens),
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
                    self._checkpoint.publish(self)
            if not self.state.global_step:
                raise ValueError("No trainable rollout batch was collected.")
            if (
                not self.args.save_every
                or self.state.global_step % self.args.save_every
            ):
                await self.save_checkpoint()
            if (
                self.eval_dataset is not None
                and not paused
                and self.args.eval_after_train
                and self._last_eval_step != self.state.global_step
            ):
                await self.evaluate()
            elapsed = time.perf_counter() - started
            summary = {
                "time/train_runtime": elapsed,
                "time/train_steps_per_second": (self.state.global_step - initial_step)
                / elapsed,
                "usage/train_tokens_total": self.state.metrics.get(
                    "usage/train_tokens_total", 0.0
                ),
                "usage/actor_train_tokens_total": self.state.metrics.get(
                    "usage/actor_train_tokens_total", 0.0
                ),
                "usage/critic_train_tokens_total": self.state.metrics.get(
                    "usage/critic_train_tokens_total", 0.0
                ),
                "usage/sample_input_tokens_total": float(self._sample_input_tokens),
                "usage/sample_output_tokens_total": float(self._sample_output_tokens),
            }
            self.state.metrics.update(summary)
            self._metrics.publish(
                summary,
                step=self.state.global_step,
            )
            self._checkpoint.publish(self)
            await self.close(flush=not paused)
        except BaseException as error:
            try:
                await self.close(state="crashed", error=str(error))
            except Exception as cleanup_error:
                error.add_note(f"Reporter cleanup also failed: {cleanup_error}")
            raise
        finally:
            self._checkpoint.end()
            if progress is not None:
                progress.close()
        return self.state


# Keep ``triotune.trainer.ppo.trainer`` as an identity-preserving alias
# for integrations that patch or pickle the historical module path.
_ppo_module = __import__(f"{__package__}.ppo", fromlist=["*"])
_ppo_module.PPOTrainer = PPOTrainer

sys.modules[f"{__package__}.ppo.trainer"] = sys.modules[__name__]
del _ppo_module
