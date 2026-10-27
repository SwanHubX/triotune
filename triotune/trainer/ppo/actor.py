"""PPO actor optimization and PyTRIO updates.

Advantage preparation lives in :mod:`trainer.ppo.advantages`; this module
keeps prompt-group minibatching and remote actor updates.
"""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ...api.cli_args import PPOConfig
from ...api.client_api import call_async, optimizer_params
from ...utils.token_usage import compute_train_tokens
from .advantages import PPOAlgorithmPlan, prepare_advantages
from .datum import batch_datums, build_datum, completion_logprobs
from .loss import make_loss_fn
from .trajectory import PreparedBatch, Trajectory, validate_policy_versions


def _native_policy_metrics(result, batch: PreparedBatch, config: PPOConfig):
    """Recover policy-drift diagnostics from a built-in PPO response."""

    outputs = getattr(result, "loss_fn_outputs", None)
    if not isinstance(outputs, list) or len(outputs) != len(batch.trajectories):
        return {}
    deltas = []
    for output, trajectory, proximal, mask in zip(
        outputs,
        batch.trajectories,
        batch.proximal_logprobs,
        batch.masks,
        strict=True,
    ):
        if not isinstance(output, Mapping):
            return {}
        values = getattr(output.get("logprobs"), "data", None)
        if not isinstance(values, list):
            return {}
        start = len(trajectory.prompt_tokens) - 1
        current = values[start:]
        if len(current) != len(mask):
            return {}
        deltas.extend(
            float(now) - float(old)
            for now, old, active in zip(current, proximal, mask, strict=True)
            if active
        )
    if not deltas:
        return {}
    delta = np.asarray(deltas, dtype=np.float64)
    ratio = np.exp(np.clip(delta, -700, 700))
    low = 1 - config.eps_clip
    high = 1 + (
        config.eps_clip if config.eps_clip_higher is None else config.eps_clip_higher
    )
    return {
        "policy/ratio": float(np.mean(ratio)),
        "policy/ratio_std": float(np.std(ratio)),
        "policy/ratio_p95": float(np.quantile(ratio, 0.95)),
        "policy/ratio_max": float(np.max(ratio)),
        "policy/update_approx_kl": float(np.mean(ratio - 1 - delta)),
        "policy/clip_fraction": float(np.mean((ratio < low) | (ratio > high))),
        "policy/clip_fraction_low": float(np.mean(ratio < low)),
        "policy/clip_fraction_high": float(np.mean(ratio > high)),
        "_delta": delta.tolist(),
        "_ratio": ratio.tolist(),
        "_clipped": ((ratio < low) | (ratio > high)).astype(float).tolist(),
        "_rejected": np.zeros_like(ratio).tolist(),
    }


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
            groups.setdefault(t.comparison_group_id, []).append(i)
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
        if batch.loss_weights:
            chunk_batch.loss_weights = [batch.loss_weights[i] for i in indices]
        chunk_batch.prompt_mean_scale = count * len(chunk) / len(groups)
        yield chunk_batch


class PPOActor:
    """Prepare frozen policy targets and execute actor updates through PyTRIO."""

    def __init__(self, client, config: PPOConfig, plan: PPOAlgorithmPlan | None = None):
        """Bind one remote training client and its immutable PPO configuration."""

        self.client, self.config = client, config
        self.plan = plan or getattr(config, "algorithm_plan", None)
        self.optimizer_steps = 0
        self.train_tokens = 0
        self.datum_provenance: tuple[dict, ...] = ()
        self.datum_prefix_groups: tuple[dict, ...] = ()

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
            # group_id is the physical tokenized-prompt identity. Reward groups
            # may intentionally span compacted causal segments with different
            # prompts, but those rows remain separate PyTRIO datums.
            previous = prompts.setdefault(t.group_id, t.prompt_tokens)
            if previous != t.prompt_tokens:
                raise ValueError(
                    "One PPO group must identify one identical tokenized prompt."
                )
        validate_policy_versions(trajectories)
        proximal = None
        if config.recompute_logprob or config.use_decoupled_loss:
            datums = [build_datum(t) for t in trajectories]
            result = await call_async(
                self.client,
                "forward",
                datums,
                loss_fn="cross_entropy",
                auto_shift=False,
            )
            self.train_tokens += compute_train_tokens(datums)
            proximal = completion_logprobs(result, trajectories)
        if proximal is not None and not config.use_decoupled_loss:
            # Match AReaL: ordinary recomputation replaces old logprobs for
            # both KL rewards and the frozen PPO ratio denominator.
            trajectories = [
                replace(t, logprobs=lp)
                for t, lp in zip(trajectories, proximal, strict=True)
            ]
        batch = prepare_advantages(trajectories, config, self.plan)
        if proximal is not None:
            batch.proximal_logprobs = proximal
        return batch

    async def update(self, batch: PreparedBatch) -> dict[str, float]:
        """Run configured minibatches and epochs using PyTRIO loss callbacks."""

        config = self.config
        chunks = list(minibatches(batch, config.ppo_n_minibatches))
        metrics = []
        diagnostics = []
        native_backend = (
            config.loss_backend == "native"
            and not (
                self.plan
                and (
                    self.plan.loss_weight_policy is not None
                    or self.plan.objective is not None
                )
            )
            and all(
                all(
                    weight == 1.0
                    for weight, active in zip(
                        trajectory.effective_action_weights,
                        mask,
                        strict=True,
                    )
                    if active
                )
                for trajectory, mask in zip(
                    batch.trajectories, batch.masks, strict=True
                )
            )
        )
        for _ in range(config.update_epochs):
            for chunk in chunks:
                data = batch_datums(chunk, native=native_backend)
                self.datum_provenance = data.provenance
                self.datum_prefix_groups = data.prefix_groups
                if native_backend:
                    result = await call_async(
                        self.client,
                        "forward_backward",
                        data,
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
                    objective = self.plan.objective if self.plan else None
                    factory = (
                        getattr(objective, "make_loss", objective)
                        if objective is not None
                        else make_loss_fn
                    )
                    if not callable(factory):
                        factory = make_loss_fn
                    result = await call_async(
                        self.client,
                        "forward_backward_custom",
                        data,
                        loss_fn=factory(chunk, config, diagnostics),
                    )
                # PyTRIO custom loss submits a forward, then a forward_backward.
                passes = 1 if native_backend else 2
                self.train_tokens += passes * compute_train_tokens(data)
                # The custom SDK path needs its current forward result to build
                # backward weights. Do not mutate policy weights between them.
                optimizer = await call_async(
                    self.client, "optim_step", optimizer_params(config)
                )
                self.optimizer_steps += 1
                row = dict(result.metrics)
                if native_backend:
                    native = _native_policy_metrics(result, chunk, config)
                    diagnostics.append(
                        {
                            "delta": native.pop("_delta", []),
                            "ratio": native.pop("_ratio", []),
                            "clipped": native.pop("_clipped", []),
                            "rejected": native.pop("_rejected", []),
                        }
                    )
                    row.update(native)
                for name in ("grad_norm", "gradient_norm", "optim/grad_norm"):
                    value = getattr(optimizer, "metrics", {}).get(name)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        row["optim/grad_norm"] = float(value)
                        break
                if "train/loss" in row:
                    # The SDK may also return the CE surrogate's raw `loss`.
                    # MetricsAPI maps that name to train/loss, so exclude it
                    # rather than overwrite the actual nonlinear objective.
                    row.pop("loss", None)
                    row.pop("scalar/loss", None)
                    row.pop("trainer/loss", None)
                row.setdefault(
                    "train/action_tokens",
                    float(sum(sum(mask) for mask in chunk.masks)),
                )
                row.setdefault(
                    "train/max_shared_prefix_tokens",
                    float(
                        max(
                            (group["prefix_length"] for group in data.prefix_groups),
                            default=0,
                        )
                    ),
                )
                row["_loss_weight"] = float(
                    (
                        sum(
                            weight * sum(trajectory.effective_action_weights)
                            for trajectory, weight, mask in zip(
                                chunk.trajectories,
                                chunk.loss_weights or [1.0] * len(chunk.masks),
                                chunk.masks,
                                strict=True,
                            )
                        )
                        if config.loss_aggregation == "token_mean"
                        else sum(
                            weight
                            for weight, mask in zip(
                                chunk.loss_weights or [1.0] * len(chunk.masks),
                                chunk.masks,
                                strict=True,
                            )
                            if any(mask)
                        )
                    )
                    if self.plan and self.plan.loss_weight_policy is not None
                    else sum(
                        sum(trajectory.effective_action_weights)
                        for trajectory in chunk.trajectories
                    )
                    if config.loss_aggregation == "token_mean"
                    else len(
                        {
                            t.logical_sequence_id if t.rollout_id else ("row", i)
                            for i, (t, mask) in enumerate(
                                zip(chunk.trajectories, chunk.masks, strict=True)
                            )
                            if any(mask)
                        }
                    )
                    if config.loss_aggregation in {"seq_mean", "constant"}
                    else 1.0
                )
                metrics.append(row)
        keys = {
            key
            for row in metrics
            for key, value in row.items()
            if not key.startswith("_")
            and isinstance(value, (int, float))
            and np.isfinite(value)
        }
        aggregated = {}
        for key in keys:
            rows = [row for row in metrics if key in row]
            if key == "train/loss":
                weights = np.asarray(
                    [row["_loss_weight"] for row in rows], dtype=np.float64
                )
                aggregated[key] = float(
                    np.average([row[key] for row in rows], weights=weights)
                )
            elif key.startswith("policy/"):
                weights = np.asarray(
                    [row["train/action_tokens"] for row in rows], dtype=np.float64
                )
                aggregated[key] = float(
                    np.average([row[key] for row in rows], weights=weights)
                )
            else:
                aggregated[key] = float(np.mean([row[key] for row in rows]))
        ratios = np.asarray(
            [value for item in diagnostics for value in item["ratio"]],
            dtype=np.float64,
        )
        deltas = np.asarray(
            [value for item in diagnostics for value in item["delta"]],
            dtype=np.float64,
        )
        if ratios.size:
            clipped = [value for item in diagnostics for value in item["clipped"]]
            rejected = [value for item in diagnostics for value in item["rejected"]]
            low = 1 - config.eps_clip
            high = 1 + (
                config.eps_clip
                if config.eps_clip_higher is None
                else config.eps_clip_higher
            )
            aggregated.update(
                {
                    "policy/ratio": float(np.mean(ratios)),
                    "policy/ratio_std": float(np.std(ratios)),
                    "policy/ratio_p95": float(np.quantile(ratios, 0.95)),
                    "policy/ratio_max": float(np.max(ratios)),
                    "policy/update_approx_kl": float(np.mean(ratios - 1 - deltas)),
                    "policy/clip_fraction": float(np.mean(clipped)),
                    "policy/clip_fraction_low": float(np.mean(ratios < low)),
                    "policy/clip_fraction_high": float(np.mean(ratios > high)),
                    "policy/rejected_fraction": float(np.mean(rejected)),
                }
            )
        return aggregated
