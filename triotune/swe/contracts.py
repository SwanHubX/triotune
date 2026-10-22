"""Exact per-call samples and conservative conversion to the PPO contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..trainer.ppo.contracts import Trajectory


@dataclass(frozen=True, slots=True)
class SampledTurn:
    """A response with its actual prompt and behavior distribution scores.

    raw_text is decoded from completion_tokens. No text re-tokenization is
    used to recover training tokens. policy_version identifies one immutable
    sampling snapshot; callers refresh it only between complete episodes.
    """

    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    raw_text: str
    policy_version: str
    truncated: bool = False
    request_retries: int = 0
    sampling_seconds: float = 0.0

    def __post_init__(self):
        if type(self.request_retries) is not int or self.request_retries < 0:
            raise ValueError("request_retries must be a nonnegative integer.")
        if not math.isfinite(self.sampling_seconds) or self.sampling_seconds < 0:
            raise ValueError("sampling_seconds must be finite and nonnegative.")
        for tokens in (self.prompt_tokens, self.completion_tokens):
            if not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError(
                    "Sampled turns require nonempty nonnegative token IDs."
                )
        if len(self.logprobs) != len(self.completion_tokens) or any(
            not isinstance(p, (int, float))
            or isinstance(p, bool)
            or not math.isfinite(p)
            or p > 0
            for p in self.logprobs
        ):
            raise ValueError(
                "Behavior logprobs must be finite, <= 0 and token aligned."
            )
        if (
            not isinstance(self.raw_text, str)
            or not isinstance(self.policy_version, str)
            or not self.policy_version.strip()
            or type(self.truncated) is not bool
        ):
            raise ValueError("Samples require text and a nonempty policy_version.")


def trajectory_from_turns(
    turns: list[SampledTurn],
    *,
    reward: float,
    group_id: str,
) -> Trajectory:
    """Concatenate only samples whose *actual token prompts* extend history.

    Tool observations and template delimiters inserted between calls are
    masked. Their placeholder logprobs are never used as action probabilities.
    Templates that drop earlier reasoning or otherwise rewrite history cannot
    form this single-sequence contract and fail explicitly. Per-call samples
    remain available for a future separately specified training strategy.
    """
    if not turns:
        raise ValueError("An episode needs at least one sampled turn.")
    if len({t.policy_version for t in turns}) != 1:
        raise ValueError("An episode must use a single policy version.")
    prompt = list(turns[0].prompt_tokens)
    full = list(prompt)
    logprobs, masks, turn_ids = [], [], []
    for index, turn in enumerate(turns):
        current = list(turn.prompt_tokens)
        if current[: len(full)] != full:
            raise ValueError(
                "The chat template rewrote an earlier sampled token prefix; "
                "cannot export this episode as one PPO Trajectory. "
                "Keep the per-call samples rather than re-tokenizing history."
            )
        observation = current[len(full) :]
        full.extend(observation)
        logprobs.extend([0.0] * len(observation))
        masks.extend([False] * len(observation))
        turn_ids.extend([-1] * len(observation))
        full.extend(turn.completion_tokens)
        logprobs.extend(turn.logprobs)
        masks.extend([True] * len(turn.completion_tokens))
        turn_ids.extend([index] * len(turn.completion_tokens))
    return Trajectory(
        prompt_tokens=prompt,
        completion_tokens=full[len(prompt) :],
        logprobs=logprobs,
        loss_mask=masks,
        turn_ids=turn_ids,
        reward=reward,
        group_id=group_id,
        truncated=turns[-1].truncated,
    )
