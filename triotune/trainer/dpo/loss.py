"""Local DPO objective and gradient callback for PyTRIO."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...api.client_api import call, call_async


def _pair_rewards(
    policy_chosen: Any,
    policy_rejected: Any,
    reference_chosen: Any,
    reference_rejected: Any,
    beta: float,
) -> tuple[Any, Any, Any]:
    """Return chosen/rejected implicit rewards and their pairwise margin."""

    chosen = beta * (policy_chosen - reference_chosen)
    rejected = beta * (policy_rejected - reference_rejected)
    return chosen, rejected, chosen - rejected


def dpo_evaluation_metrics(
    policy_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    beta: float,
) -> dict[str, float]:
    """Evaluate the same pairwise DPO objective without tensor gradients."""

    policy = tuple(float(value) for value in policy_scores)
    reference = tuple(float(value) for value in reference_scores)
    if not policy or len(policy) != len(reference) or len(policy) % 2:
        raise ValueError("DPO evaluation scores must contain aligned pairs.")
    if any(not math.isfinite(value) for value in policy + reference):
        raise ValueError("DPO evaluation scores must be finite.")
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive.")

    losses = []
    margins = []
    chosen_rewards = []
    rejected_rewards = []
    for index in range(0, len(policy), 2):
        chosen, rejected, margin = _pair_rewards(
            policy[index],
            policy[index + 1],
            reference[index],
            reference[index + 1],
            float(beta),
        )
        losses.append(max(-margin, 0.0) + math.log1p(math.exp(-abs(margin))))
        margins.append(margin)
        chosen_rewards.append(chosen)
        rejected_rewards.append(rejected)
    count = len(losses)
    return {
        "eval/loss": sum(losses) / count,
        "eval/reward_accuracy": sum(margin > 0 for margin in margins) / count,
        "eval/reward_margin": sum(margins) / count,
        "eval/reward_chosen": sum(chosen_rewards) / count,
        "eval/reward_rejected": sum(rejected_rewards) / count,
        "eval/pairs": float(count),
    }


def make_loss_fn(reference: Sequence[float], beta: float):
    """Return PyTRIO's local custom-loss callback for ordered chosen/rejected pairs."""

    def loss(datums: list[Any], logprobs: list[Any]) -> tuple[Any, dict[str, float]]:
        import torch
        import torch.nn.functional as F

        if len(logprobs) != len(datums) or len(logprobs) != len(reference) or len(logprobs) % 2:
            raise ValueError("DPO requires aligned chosen/rejected datums and reference scores.")
        scores = []
        for datum, token_logprobs in zip(datums, logprobs, strict=True):
            labels = datum.loss_fn_inputs["target_tokens"].data
            if len(labels) != token_logprobs.numel():
                raise ValueError("Policy logprobs and target tokens are not aligned.")
            mask = torch.tensor([token != -100 for token in labels], device=token_logprobs.device)
            scores.append(token_logprobs.reshape(-1)[mask].sum())
        policy_chosen = torch.stack(scores[::2])
        policy_rejected = torch.stack(scores[1::2])
        reference_chosen = policy_chosen.new_tensor(reference[::2])
        reference_rejected = policy_chosen.new_tensor(reference[1::2])
        chosen_rewards, rejected_rewards, margins = _pair_rewards(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta,
        )
        objective = F.softplus(-margins).mean()
        metrics = {
            "train/loss": objective.item(),
            "reward/chosen": chosen_rewards.detach().mean().item(),
            "reward/rejected": rejected_rewards.detach().mean().item(),
            "reward/margin": margins.detach().mean().item(),
            "reward/accuracy": (margins.detach() > 0).float().mean().item(),
            "train/logp_chosen": policy_chosen.detach().mean().item(),
            "train/logp_rejected": policy_rejected.detach().mean().item(),
        }
        return objective, metrics

    return loss


@dataclass(frozen=True, slots=True)
class DPOObjective:
    """Adapt one batch-specific pairwise DPO callback to the Objective protocol."""

    reference_scores: Sequence[float]
    beta: float

    def __post_init__(self) -> None:
        scores = tuple(float(value) for value in self.reference_scores)
        if not scores or len(scores) % 2:
            raise ValueError("DPO reference scores must contain chosen/rejected pairs.")
        if any(not math.isfinite(value) for value in scores):
            raise ValueError("DPO reference scores must be finite.")
        if not math.isfinite(float(self.beta)) or self.beta <= 0:
            raise ValueError("DPO beta must be finite and positive.")
        object.__setattr__(self, "reference_scores", scores)
        object.__setattr__(self, "beta", float(self.beta))

    def forward_backward(self, client: Any, datums: Sequence[Any]) -> Any:
        """Run and resolve the synchronous custom-loss request."""

        return call(
            client,
            "forward_backward_custom",
            list(datums),
            make_loss_fn(self.reference_scores, self.beta),
        )

    async def forward_backward_async(
        self, client: Any, datums: Sequence[Any]
    ) -> Any:
        """Run and resolve the asynchronous custom-loss request."""

        return await call_async(
            client,
            "forward_backward_custom",
            list(datums),
            make_loss_fn(self.reference_scores, self.beta),
        )


__all__ = ["DPOObjective", "dpo_evaluation_metrics", "make_loss_fn"]
