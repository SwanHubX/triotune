"""Persistent Playground exports with a cadence separate from Train saves."""

from ..api.client_api import call, submit_async, wait_async
from .checkpoint import checkpoint_path


class SamplerCheckpoint:
    """Export actor/student weights, never optimizer or critic state.

    Manual exports obey enabled and skip an already exported step. Automatic
    callers use due() for intervals and successful completion. State/reporting
    changes happen only after a valid SDK response has been received.
    """

    def __init__(self, config, algorithm):
        self.config = config
        self.algorithm = algorithm
        self.saved_step = None
        self.saved_ttl_seconds = None

    def restore(self, metadata):
        """Retain export cadence across processes without re-exporting weights."""
        self.saved_step = metadata["step"]
        self.saved_ttl_seconds = metadata["ttl_seconds"]

    def metadata(self, path):
        # Vendored manifests contain a URI without its original export step.
        # Preserve it without guessing which weights it represents.
        return dict(uri=path, step=self.saved_step, ttl_seconds=self.saved_ttl_seconds)

    def due(self, step, *, final=False):
        return (
            self.config.enabled
            and step > 0
            and step != self.saved_step
            and (
                final
                or bool(self.config.save_every and step % self.config.save_every == 0)
            )
        )

    def _kwargs(self, step):
        return {
            "name": f"{self.algorithm}-sampler-step-{step:08d}",
            "ttl_seconds": self.config.ttl_seconds,
        }

    def _publish(self, saved, state, reporter, step):
        path = checkpoint_path(saved, method="save_weights_for_sampler")
        state.sampler_checkpoint_path = path
        self.saved_step = step
        self.saved_ttl_seconds = self.config.ttl_seconds
        reporter.log_text(key="checkpoint/sampler_path", data=path, step=step)
        return path

    def save(self, client, state, reporter):
        step = state.global_step
        if not self.due(step, final=True):
            return state.sampler_checkpoint_path
        saved = call(client, "save_weights_for_sampler", **self._kwargs(step))
        return self._publish(saved, state, reporter, step)

    async def submit_async(self, client, step):
        """Submit immediately after the corresponding optimizer (SFT pipeline)."""
        return await submit_async(
            client, "save_weights_for_sampler", **self._kwargs(step)
        )

    async def collect_async(self, future, state, reporter, step):
        saved = await wait_async(future)
        return self._publish(saved, state, reporter, step)

    async def save_async(self, client, state, reporter):
        step = state.global_step
        if not self.due(step, final=True):
            return state.sampler_checkpoint_path
        future = await self.submit_async(client, step)
        return await self.collect_async(future, state, reporter, step)
