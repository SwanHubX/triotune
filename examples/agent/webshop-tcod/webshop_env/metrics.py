"""WebShop-owned rollout metrics published through Triotune's generic API."""

from __future__ import annotations

from collections.abc import Iterable

from triotune.agentrl import AgentTrainingLoop


def purchase_metrics(episodes: Iterable, *, prefix: str = "agent") -> dict[str, float]:
    """Aggregate purchase coverage and precision with explicit denominators.

    All attempted episodes remain in coverage/rate denominators. System
    failures have no trusted score and therefore contribute zero purchases and
    zero successes, matching AgentRL's generic success-rate convention.
    """

    episodes = list(episodes)
    valid = [episode for episode in episodes if episode.trainable]
    purchased = [
        float(episode.score.components["purchased"])
        for episode in valid
        if "purchased" in episode.score.components
    ]
    if not purchased:
        return {}
    count = len(episodes)
    purchase_count = sum(purchased)
    successful = sum(float(episode.score.success) for episode in valid)
    return {
        f"{prefix}/purchase_count": purchase_count,
        f"{prefix}/successful_purchase_count": successful,
        f"{prefix}/purchase_rate": purchase_count / max(1, count),
        f"{prefix}/purchase_success_rate": successful
        / max(1.0, purchase_count),
        f"{prefix}/incorrect_purchase_rate": (purchase_count - successful)
        / max(1, count),
    }


class WebShopMetricsAgentTrainingLoop(AgentTrainingLoop):
    """Publish domain metrics after each update and complete evaluation."""

    async def _update(self, batch, snapshot, progress, *, max_staleness=0):
        wrapped = _PurchaseMetricsBatch(batch)
        await super()._update(
            wrapped,
            snapshot,
            progress,
            max_staleness=max_staleness,
        )
        # Keep the public evidence object rather than the temporary metrics
        # adapter retained by AgentTrainingLoop._update().
        self.last_train_batch = batch

    async def evaluate(self, dataset=None):
        metrics = await super().evaluate(dataset)
        purchase = purchase_metrics(
            self.last_eval_batch.episodes,
            prefix="eval/agent",
        )
        if purchase:
            metrics.update(await self.trainer.publish_metrics(purchase))
        return metrics


class _PurchaseMetricsBatch:
    """Add WebShop fields to one batch without changing AgentRolloutBatch."""

    def __init__(self, batch):
        self._batch = batch

    def __getattr__(self, name):
        return getattr(self._batch, name)

    def metrics(self, *, prefix="agent"):
        metrics = self._batch.metrics(prefix=prefix)
        metrics.update(purchase_metrics(self._batch.episodes, prefix=prefix))
        return metrics


__all__ = ["WebShopMetricsAgentTrainingLoop", "purchase_metrics"]
