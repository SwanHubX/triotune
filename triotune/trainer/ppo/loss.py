"""PPO-family policy objectives through PyTRIO's logprob gradient bridge.

The SDK forwards remotely, differentiates this callback locally, then sends
``weights=-d(loss)/d(logprob)`` to remote weighted cross-entropy backward.
No model parameters or full vocabulary logits are downloaded.
"""

from collections import defaultdict

from ...api.cli_args import PPOConfig
from .datum import IGNORE_INDEX
from .trajectory import PreparedBatch


def make_loss_fn(
    batch: PreparedBatch,
    config: PPOConfig,
    diagnostics: list[dict[str, list[float]]] | None = None,
):
    """Create a PyTRIO custom-loss callback for one frozen actor batch.

    The callback receives remote current-policy logprobs and returns a local
    scalar objective plus metrics. PyTRIO converts its logprob gradients into
    the remote weighted cross-entropy update; model parameters never leave the
    training service.
    """

    def loss(datums, logprobs):
        """Evaluate PPO, GSPO, SAPO, or CISPO on SDK-produced logprobs."""

        import torch

        if len(datums) != len(batch.trajectories) or len(logprobs) != len(datums):
            raise ValueError("Custom loss needs one logprob tensor per trajectory.")
        sequence_targets = {}
        if config.importance_sampling_level == "sequence":
            pooled_delta, pooled_advantage = defaultdict(list), defaultdict(list)
            for i, (t, adv, prox, mask, full_lp) in enumerate(
                zip(
                    batch.trajectories,
                    batch.advantages,
                    batch.proximal_logprobs,
                    batch.masks,
                    logprobs,
                    strict=True,
                )
            ):
                lp = full_lp.reshape(-1)
                if lp.numel() != len(t.prompt_tokens) + len(t.completion_tokens) - 1:
                    raise ValueError("Custom loss logprobs are not causally aligned.")
                lp = lp[len(t.prompt_tokens) - 1 :]
                valid = torch.tensor(mask, dtype=torch.bool, device=lp.device)
                previous = lp.new_tensor(prox)[valid]
                keep = torch.ones_like(previous, dtype=torch.bool)
                correction = (previous - lp.new_tensor(t.logprobs)[valid]).exp()
                if config.rejection_lower is not None:
                    keep &= correction >= config.rejection_lower
                if config.rejection_upper is not None:
                    keep &= correction <= config.rejection_upper
                key = t.rollout_id or ("row", i)
                pooled_delta[key].append((lp[valid] - previous)[keep])
                pooled_advantage[key].append(lp.new_tensor(adv)[valid][keep])
            for key, rows in pooled_delta.items():
                deltas_for_rollout = torch.cat(rows)
                advantages_for_rollout = torch.cat(pooled_advantage[key])
                denom = max(1, deltas_for_rollout.numel())
                sequence_targets[key] = (
                    (deltas_for_rollout.sum() / denom).exp(),
                    advantages_for_rollout.sum() / denom,
                )
        sums, counts, deltas, ratios, clipped, rejected = [], [], [], [], [], []
        row_weights = batch.loss_weights or [1.0] * len(batch.trajectories)
        if len(row_weights) != len(batch.trajectories):
            raise ValueError("loss_weights must align with trajectories.")
        # Evaluate each trajectory independently so variable-length completion
        # masks remain aligned before the configured reduction is applied.
        for row_index, (t, adv, prox, mask, datum, full_lp) in enumerate(
            zip(
                batch.trajectories,
                batch.advantages,
                batch.proximal_logprobs,
                batch.masks,
                datums,
                logprobs,
                strict=True,
            )
        ):
            lp = full_lp.reshape(-1)
            if lp.numel() != len(t.prompt_tokens) + len(t.completion_tokens) - 1:
                raise ValueError("Custom loss logprobs are not causally aligned.")
            expected = [IGNORE_INDEX] * (len(t.prompt_tokens) - 1) + [
                token if active else IGNORE_INDEX
                for token, active in zip(t.completion_tokens, mask, strict=True)
            ]
            if list(datum.loss_fn_inputs["target_tokens"].data) != expected:
                raise ValueError(
                    "Custom loss datum targets differ from its bound trajectory."
                )
            lp = lp[len(t.prompt_tokens) - 1 :]
            valid = torch.tensor(mask, dtype=torch.bool, device=lp.device)
            n = int(valid.sum())
            counts.append(n)
            # Keep zero dependencies for fully masked rows: the SDK requires a
            # gradient tensor for every datum, including deliberately ignored rows.
            if n == 0:
                sums.append(lp.sum() * 0)
                continue
            current, previous = lp[valid], lp.new_tensor(prox)[valid]
            advantage = lp.new_tensor(adv)[valid]
            keep = torch.ones_like(current, dtype=torch.bool)
            correction = torch.ones_like(current)
            if config.rejection_lower is not None or config.rejection_upper is not None:
                correction = (previous - lp.new_tensor(t.logprobs)[valid]).exp()
                if config.rejection_lower is not None:
                    keep &= correction >= config.rejection_lower
                if config.rejection_upper is not None:
                    keep &= correction <= config.rejection_upper
            delta = current - previous
            if config.importance_sampling_level == "sequence":
                # A sequence ratio covers the logical candidate, including
                # every fragment in this minibatch. Group-preserving batching
                # keeps its pieces together, so fragment boundaries cannot
                # change the geometric ratio or mean candidate advantage.
                seq_ratio, seq_advantage = sequence_targets[
                    t.rollout_id or ("row", row_index)
                ]
                ratio, advantage = (
                    seq_ratio.expand_as(delta),
                    seq_advantage.expand_as(delta),
                )
            else:
                ratio = delta.exp()
            if not torch.isfinite(ratio).all() or not torch.isfinite(correction).all():
                raise FloatingPointError(
                    "Importance ratios are nonfinite; reject or recollect stale trajectories."
                )
            # Ordinary PPO clips the surrogate; CISPO detaches the clipped
            # ratio, while SAPO replaces the hard clip with a smooth gate.
            bounded = ratio.clamp(
                1 - config.eps_clip,
                1
                + (
                    config.eps_clip
                    if config.eps_clip_higher is None
                    else config.eps_clip_higher
                ),
            )
            if config.loss_type == "sapo":
                tau = torch.where(
                    advantage > 0, config.sapo_tau_pos, config.sapo_tau_neg
                )
                terms = -4 / tau * torch.sigmoid(tau * (ratio - 1)) * advantage
                clip = torch.zeros_like(keep)
            elif config.loss_type == "cispo":
                terms = -bounded.detach() * advantage * current
                clip = ratio != bounded
            else:
                unclipped, clipped_terms = -advantage * ratio, -advantage * bounded
                terms = torch.maximum(unclipped, clipped_terms)
                clip = unclipped < clipped_terms
                if config.c_clip is not None:
                    terms = torch.minimum(terms, config.c_clip * advantage.abs())
            # Filter the numerator only; all reductions retain original counts.
            terms = (terms * correction).masked_fill(~keep, 0)
            sums.append(row_weights[row_index] * terms.sum() + lp.sum() * 0)
            deltas.append(delta.detach())
            ratios.append(ratio.detach())
            clipped.append(clip.detach())
            rejected.append((~keep).detach())
        if not sum(counts):
            raise ValueError("Custom loss has no active action tokens.")
        active = [
            (s, n, t.group_id, t.rollout_id or ("row", i), row_weights[i])
            for i, (s, n, t) in enumerate(
                zip(sums, counts, batch.trajectories, strict=True)
            )
            if n
        ]
        # Sequence reductions treat all fragments of one rollout as one
        # candidate. Splitting a long candidate into more rows cannot increase
        # its sequence weight. Token reductions retain their natural action
        # token weighting; masked shared prefixes contribute to neither count.
        rollout_sums, rollout_counts = {}, defaultdict(float)
        for s, n, _, rollout, weight in active:
            rollout_sums[rollout] = rollout_sums.get(rollout, 0) + s
            rollout_counts[rollout] += n * weight
        zero = sum(sums) * 0
        if config.loss_aggregation == "token_mean":
            objective = sum(sums) / max(
                sum(n * w for n, w in zip(counts, row_weights)), 1e-12
            )
        elif config.loss_aggregation == "seq_mean":
            objective = (
                sum(
                    s / max(rollout_counts[key], 1e-12)
                    for key, s in rollout_sums.items()
                )
                / len(rollout_sums)
                + zero
            )
        elif config.loss_aggregation == "prompt_mean":
            group_tokens = defaultdict(int)
            for _, n, group, _, weight in active:
                group_tokens[group] += n * weight
            objective = (
                sum(
                    s / max(group_tokens[group], 1e-12)
                    for s, _, group, _, _ in active
                )
                / len(group_tokens)
                * batch.prompt_mean_scale
                + zero
            )
        else:
            objective = sum(sums) / (
                len(rollout_sums) * config.loss_aggregation_divisor
            )
        if not torch.isfinite(objective):
            raise FloatingPointError(
                "Policy loss is nonfinite; check logprob ratios and rewards."
            )
        all_deltas = torch.cat(deltas)
        all_ratios = torch.cat(ratios)
        low = 1 - config.eps_clip
        high = 1 + (
            config.eps_clip
            if config.eps_clip_higher is None
            else config.eps_clip_higher
        )
        metrics = {
            "train/loss": objective.detach().item(),
            "policy/ratio": all_ratios.mean().item(),
            "policy/ratio_std": all_ratios.std(unbiased=False).item(),
            "policy/ratio_p95": torch.quantile(all_ratios.float(), 0.95).item(),
            "policy/ratio_max": all_ratios.max().item(),
            "policy/update_approx_kl": (all_ratios - 1 - all_deltas).mean().item(),
            "policy/clip_fraction": torch.cat(clipped).float().mean().item(),
            "policy/clip_fraction_low": (all_ratios < low).float().mean().item(),
            "policy/clip_fraction_high": (all_ratios > high).float().mean().item(),
            "policy/rejected_fraction": torch.cat(rejected).float().mean().item(),
            "train/action_tokens": float(sum(counts)),
        }
        if diagnostics is not None:
            diagnostics.append(
                {
                    "delta": all_deltas.detach().cpu().tolist(),
                    "ratio": all_ratios.detach().cpu().tolist(),
                    "clipped": torch.cat(clipped).float().detach().cpu().tolist(),
                    "rejected": torch.cat(rejected).float().detach().cpu().tolist(),
                }
            )
        return objective, metrics

    return loss
