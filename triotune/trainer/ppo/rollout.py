"""PyTRIO sampling helpers for PPO-family rollouts and fixed references.

This module is intentionally limited to online sampling, reward adaptation,
group filtering, and reference scoring.  Workflows that already produce
``Trajectory`` objects can bypass it and call ``PPOTrainer.step`` directly.
Training and inference remain in one PyTRIO client lifecycle for now.
"""

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Mapping
from numbers import Real

import pytrio as trio

from ...api.client_api import call_async
from ...dataset import prepare_prompt
from ...utils.token_usage import (
    SampleUsageCallback,
    compute_sample_tokens,
)
from .trajectory import Trajectory


def _reward_result(value):
    """Normalize a scalar or ``{"reward": ..., component: ...}`` result."""

    if not isinstance(value, Mapping):
        return float(value), {}
    if "reward" not in value:
        raise ValueError("A structured reward result must contain 'reward'.")
    reward = value["reward"]
    components = {name: item for name, item in value.items() if name != "reward"}
    if (
        not isinstance(reward, Real)
        or isinstance(reward, bool)
        or any(
            not isinstance(name, str)
            or not isinstance(item, Real)
            for name, item in components.items()
        )
    ):
        raise ValueError("Structured reward values must be numeric scalars.")
    return float(reward), {str(name): float(item) for name, item in components.items()}


async def gather_cancel_on_error(coroutines):
    """Await concurrent PyTRIO requests and cancel siblings after any failure."""
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def generate_rollouts(
    client,
    tokenizer,
    rows,
    reward_fn,
    config,
    *,
    first_group: int,
    sample_usage_callback: SampleUsageCallback | None = None,
):
    """Sample prompt groups and convert responses into validated trajectories.

    ``reward_fn`` receives ``(row=..., completion=..., completion_ids=...)`` and
    may be synchronous, asynchronous, or return an awaitable.  The sampler is
    still the installed PyTRIO client; reward execution is only moved off the
    event loop when the callable itself is synchronous.

    Sync rewards run in a worker; async rewards are awaited. Each request gets
    a distinct reproducible seed and prompt group ID, including repeated rows.
    """

    async def generate(row, index):
        prompt = prepare_prompt(
            row,
            tokenizer,
            prompt_key=config.prompt_key,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        result = await call_async(
            client,
            "sample",
            prompt=trio.ModelInput.from_ints(prompt.tokens),
            num_samples=config.num_generations,
            sampling_params=trio.SamplingParams(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=None if config.seed is None else config.seed + index,
            ),
        )
        if sample_usage_callback is not None:
            sample_usage_callback(
                compute_sample_tokens(result, prompt_tokens=len(prompt.tokens))
            )
        if len(result.sequences) != config.num_generations:
            raise ValueError("Sampler returned an unexpected number of responses.")
        trajectories = []
        for sequence in result.sequences:
            text = sequence.text
            if text is None:
                text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
            kwargs = dict(
                row=row, completion=text, completion_ids=list(sequence.tokens)
            )
            if inspect.iscoroutinefunction(reward_fn):
                reward = await reward_fn(**kwargs)
            else:
                reward = await asyncio.to_thread(reward_fn, **kwargs)
                if inspect.isawaitable(reward):
                    reward = await reward
            reward, reward_components = _reward_result(reward)
            reason = sequence.stop_reason
            truncated = reason in ("length", "max_tokens", "max_length")
            if reason not in (
                "stop",
                "eos",
                "stop_token",
                "length",
                "max_tokens",
                "max_length",
            ):
                truncated = len(sequence.tokens) >= config.max_tokens
            trajectories.append(
                Trajectory(
                    prompt_tokens=list(prompt.tokens),
                    completion_tokens=list(sequence.tokens),
                    logprobs=list(sequence.logprobs),
                    reward=reward,
                    group_id=f"prompt-{index}",
                    truncated=truncated,
                    reward_components=reward_components,
                    # PyTRIO's sampler returns one assistant completion per
                    # request, so every generated token belongs to turn zero.
                    # Custom multi-turn rollouts can provide richer IDs when
                    # calling PPOTrainer.step() with Trajectory objects.
                    turn_ids=[0] * len(sequence.tokens),
                )
            )
        return trajectories

    groups = await gather_cancel_on_error(
        generate(row, first_group + i) for i, row in enumerate(rows)
    )
    return [t for group in groups for t in group]


def filter_informative_groups(trajectories: list[Trajectory]) -> list[Trajectory]:
    """DAPO: refill the prompt batch after removing equal raw-reward groups."""
    groups = defaultdict(list)
    for t in trajectories:
        groups[t.group_id].append(t)
    return [
        t
        for group in groups.values()
        if len({t.reward for t in group}) > 1
        for t in group
    ]


async def score_reference(client, trajectories: list[Trajectory]) -> None:
    """Score missing completion logprobs with a frozen PyTRIO reference client."""

    async def score(t):
        full = await call_async(
            client,
            "compute_logprobs",
            prompt=trio.ModelInput.from_ints(t.prompt_tokens + t.completion_tokens),
        )
        if len(full) != len(t.prompt_tokens) + len(t.completion_tokens):
            raise ValueError(
                "Reference logprobs must align with the full sequence (first entry None)."
            )
        t.reference_logprobs = list(full[len(t.prompt_tokens) :])
        t.__post_init__()

    await gather_cancel_on_error(
        score(t) for t in trajectories if t.reference_logprobs is None
    )
