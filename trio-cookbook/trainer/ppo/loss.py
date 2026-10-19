"""AReaL policy objectives expressed through PyTRIO's logprob gradient bridge.

The SDK forwards remotely, differentiates this callback locally, then sends
``weights=-d(loss)/d(logprob)`` to remote weighted cross-entropy backward.
No model parameters or full vocabulary logits are downloaded.
"""

from collections import defaultdict

from ...api.cli_args import PPOConfig
from .contracts import PreparedBatch
from .datum import IGNORE_INDEX


def make_loss_fn(batch: PreparedBatch, config: PPOConfig):
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
        sums, counts, ratios, clipped, rejected = [], [], [], [], []
        # Evaluate each trajectory independently so variable-length completion
        # masks remain aligned before the configured reduction is applied.
        for t, adv, prox, mask, datum, full_lp in zip(
            batch.trajectories,
            batch.advantages,
            batch.proximal_logprobs,
            batch.masks,
            datums,
            logprobs,
            strict=True,
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
                # AReaL averages both log ratios and advantages over retained
                # actions before broadcasting the sequence objective to tokens.
                denom = keep.sum().clamp_min(1)
                ratio = (
                    (delta.masked_fill(~keep, 0).sum() / denom).exp().expand_as(delta)
                )
                advantage = (advantage.masked_fill(~keep, 0).sum() / denom).expand_as(
                    delta
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
            sums.append(terms.sum() + lp.sum() * 0)
            ratios.append(ratio.detach())
            clipped.append(clip.detach())
            rejected.append((~keep).detach())
        if not sum(counts):
            raise ValueError("Custom loss has no active action tokens.")
        active = [
            (s, n, t.group_id)
            for s, n, t in zip(sums, counts, batch.trajectories, strict=True)
            if n
        ]
        zero = sum(sums) * 0
        if config.loss_aggregation == "token_mean":
            objective = sum(sums) / sum(counts)
        elif config.loss_aggregation == "seq_mean":
            objective = sum(s / n for s, n, _ in active) / len(active) + zero
        elif config.loss_aggregation == "prompt_mean":
            group_tokens = defaultdict(int)
            for _, n, group in active:
                group_tokens[group] += n
            objective = (
                sum(s / group_tokens[group] for s, _, group in active)
                / len(group_tokens)
                * batch.prompt_mean_scale
                + zero
            )
        else:
            objective = sum(sums) / (len(active) * config.loss_aggregation_divisor)
        if not torch.isfinite(objective):
            raise FloatingPointError(
                "Policy loss is nonfinite; check logprob ratios and rewards."
            )
        metrics = {
            "train/loss": objective.detach().item(),
            "policy/ratio": torch.cat(ratios).mean().item(),
            "policy/clip_fraction": torch.cat(clipped).float().mean().item(),
            "policy/rejected_fraction": torch.cat(rejected).float().mean().item(),
            "train/action_tokens": float(sum(counts)),
        }
        return objective, metrics

    return loss
