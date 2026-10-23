"""Convert recorded episodes into the existing PPO trajectory contract.

An exported trajectory starts with the first request as its prompt, followed
by all generated actions and intervening context. Model output positions keep
their original behavior probabilities; inserted observations and template
markers remain visible as context but are excluded from policy optimization.
"""

from __future__ import annotations

from ..trainer.ppo.contracts import Trajectory
from .contracts import AgentContractError, EpisodeResult


class EpisodeTrajectoryBuilder:
    """Preserve action tokens verbatim and mask added observations/template gaps.

    Later prompts must extend the exact previous prompt plus sampled output.
    Re-rendering that changes any previous token is rejected, never repaired by
    re-tokenizing text or attaching behavior scores to different tokens.
    """

    def validate_prefix(self, episode: EpisodeResult, prompt_tokens: list[int]) -> None:
        """Reject a next request that changes any previously sampled token.

        The first request has no prior generation to compare. Later requests
        may append observations and role delimiters, but a shorter or rewritten
        prefix cannot share the old action probabilities safely.
        """
        if episode.turns:
            previous = episode.turns[-1]
            prefix = previous.prompt_tokens + previous.generation.tokens
            if tuple(prompt_tokens[: len(prefix)]) != prefix:
                raise AgentContractError(
                    "Chat template changed the sampled token prefix."
                )

    def build(self, episode: EpisodeResult) -> Trajectory:
        """Export one scored episode after checking continuity and reward shape.

        All completion-aligned arrays include both action and context positions.
        Masked context uses zero score/reward placeholders and turn ID -1;
        generated output uses its original scores and zero-based turn index.
        PPO's Datum builder handles the later causal next-token shift.
        """
        if not episode.trainable:
            raise AgentContractError(
                "Only scored episodes with model actions can be trained."
            )
        initial = list(episode.turns[0].prompt_tokens)
        # Track the complete serialized sequence while extending arrays only
        # for positions after the initial prompt, as required by Trajectory.
        full = initial.copy()
        logprobs, masks, turns, process_rewards = [], [], [], []
        score = episode.score
        if any(index >= len(episode.turns) for index in score.process_rewards):
            raise AgentContractError("Process rewards refer to an unknown turn.")
        for index, turn in enumerate(episode.turns):
            prompt = list(turn.prompt_tokens)
            if prompt[: len(full)] != full:
                raise AgentContractError(
                    "Episode prompts do not extend the original token prefix."
                )
            gap = prompt[len(full) :]
            # The gap contains observations and template material introduced
            # between model calls. These tokens have no sampled policy action.
            full.extend(gap)
            logprobs.extend([0.0] * len(gap))
            masks.extend([False] * len(gap))
            turns.extend([-1] * len(gap))
            process_rewards.extend([0.0] * len(gap))
            generation = turn.generation
            # Missing process scores mean no extra process reward for this
            # turn. A supplied vector must cover its original output, including
            # any generated EOS tokens, without interpolation or re-tokenizing.
            rewards = score.process_rewards.get(index, (0.0,) * len(generation.tokens))
            if len(rewards) != len(generation.tokens):
                raise AgentContractError(
                    "Process rewards must align with original output tokens."
                )
            full.extend(generation.tokens)
            logprobs.extend(generation.logprobs)
            masks.extend([True] * len(generation.tokens))
            turns.extend([index] * len(generation.tokens))
            process_rewards.extend(rewards)
        # Keep the terminal task score separate from dense process rewards;
        # existing PPO advantage preparation combines them at the right places.
        return Trajectory(
            prompt_tokens=initial,
            completion_tokens=full[len(initial) :],
            logprobs=logprobs,
            reward=score.reward,
            group_id=episode.group_id,
            loss_mask=masks,
            turn_ids=turns,
            token_rewards=process_rewards if score.process_rewards else None,
            truncated=episode.status in {"max_turns", "token_limit", "context_limit"},
            reward_components={**score.components, "success": float(score.success)},
        )
