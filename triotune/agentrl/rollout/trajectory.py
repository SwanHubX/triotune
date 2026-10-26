"""Convert recorded episodes into the existing PPO trajectory contract.

An exported trajectory starts with the first request as its prompt, followed
by all generated actions and intervening context. Model output positions keep
their original behavior probabilities; inserted observations and template
markers remain visible as context but are excluded from policy optimization.
"""

from __future__ import annotations

from dataclasses import replace

from ...trainer.ppo.trajectory import Trajectory
from ..types import AgentContractError, EpisodeResult, RolloutExport


def _structured_reward_data(score):
    """Aggregate structured submissions while retaining target positions."""
    grouped = {}
    for assignment in score.assignments:
        key = (
            assignment.mode,
            assignment.target,
            assignment.source,
            assignment.position,
        )
        grouped.setdefault(key, []).append(assignment)
    outcome = 0.0
    process, direct = {}, {}
    for (mode, target, _source, position), values in grouped.items():
        weighted = [float(item.value) * float(item.discount) for item in values]
        aggregation = values[0].aggregation
        value = (
            sum(weighted) / len(weighted)
            if aggregation == "mean"
            else weighted[-1]
            if aggregation == "last"
            else sum(weighted)
        )
        if target == "trajectory":
            if mode == "reward":
                outcome += value
            continue
        target_map = direct if mode == "advantage" else process
        target_map[(target, position)] = target_map.get((target, position), 0.0) + value
    return outcome, process, direct


class EpisodeTrajectoryBuilder:
    """Preserve action tokens verbatim and mask added observations/template gaps.

    Later prompts must extend the exact previous prompt plus sampled output.
    Re-rendering that changes any previous token is rejected, never repaired by
    re-tokenizing text or attaching behavior scores to different tokens.
    """

    def __init__(self, *, mode: str = "concat"):
        """Choose whole-episode or disjoint per-turn training rows.

        Individual rows retain the initial prompt and all preceding tokens as
        masked context. Only the selected turn is active. This keeps group
        prompts identical and avoids training shared history repeatedly.
        Each turn receives the candidate outcome and its own process rewards;
        GAE runs independently within each row. This is an explicit temporal
        objective and need not equal concatenated GAE when discount, lambda,
        dense rewards, or critic bootstrap differ across turn boundaries.
        """
        if mode not in {"concat", "individual"}:
            raise ValueError("Export mode must be concat or individual.")
        self.mode = mode

    def export(self, episode: EpisodeResult) -> RolloutExport:
        """Export a candidate with stable fragment-to-interaction mapping.

        Outcome normalization uses one reference reward per logical rollout,
        regardless of its fragment count. Repeated context carries no active
        action mask; the collector additionally checks action keys for custom
        exporters that provide overlapping paths or branching trajectories.
        """
        whole = self.build(episode)
        if self.mode == "concat":
            rows = [whole]
        else:
            rows = []
            for index in range(len(whole.interaction_ids)):
                end = (
                    max(i for i, turn in enumerate(whole.turn_ids) if turn == index) + 1
                )
                mask = [turn == index for turn in whole.turn_ids[:end]]
                rows.append(
                    replace(
                        whole,
                        completion_tokens=whole.completion_tokens[:end],
                        logprobs=whole.logprobs[:end],
                        loss_mask=mask,
                        turn_ids=[index if active else -1 for active in mask],
                        token_rewards=(
                            [
                                value if active else 0.0
                                for value, active in zip(
                                    whole.token_rewards[:end], mask, strict=True
                                )
                            ]
                            if whole.token_rewards is not None
                            else None
                        ),
                        token_advantages=(
                            [
                                value if active else 0.0
                                for value, active in zip(
                                    whole.token_advantages[:end], mask, strict=True
                                )
                            ]
                            if whole.token_advantages is not None
                            else None
                        ),
                        fragment_id=f"{whole.rollout_id}:turn:{index}",
                        interaction_ids=whole.interaction_ids[: index + 1],
                        parent_interaction_id=(
                            whole.interaction_ids[index - 1] if index else None
                        ),
                        action_keys=whole.action_keys[:end],
                        truncated=whole.truncated
                        if index == len(episode.turns) - 1
                        else False,
                    )
                )
        return RolloutExport(
            trajectories=rows,
            logical_rollout_id=whole.rollout_id,
            fragment_map={row.fragment_id: row.interaction_ids for row in rows},
            group_id=whole.group_id,
            metadata={
                "mode": self.mode,
                "outcome_scope": "rollout",
                "shared_prefix": "context_only",
            },
        )

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
        episode.__post_init__()
        interaction_ids = tuple(
            interaction.interaction_id
            for interaction in episode.interactions
            if interaction.turn is not None
        ) or tuple(f"{episode.episode_id}:turn:{i}" for i in range(len(episode.turns)))
        if len(interaction_ids) != len(episode.turns):
            raise AgentContractError("Interactions must identify every generated turn.")
        initial = list(episode.turns[0].prompt_tokens)
        # Track the complete serialized sequence while extending arrays only
        # for positions after the initial prompt, as required by Trajectory.
        full = initial.copy()
        logprobs, masks, turns, process_rewards, action_keys = [], [], [], [], []
        direct_advantages = []
        score = episode.score
        if any(index >= len(episode.turns) for index in score.process_rewards):
            raise AgentContractError("Process rewards refer to an unknown turn.")
        structured_outcome, structured_process, structured_advantages = (
            _structured_reward_data(score)
        )
        reward = score.reward + structured_outcome
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
            direct_advantages.extend([0.0] * len(gap))
            action_keys.extend([None] * len(gap))
            generation = turn.generation
            # Missing process scores mean no extra process reward for this
            # turn. A supplied vector must cover its original output, including
            # any generated EOS tokens, without interpolation or re-tokenizing.
            rewards = list(
                score.process_rewards.get(index, (0.0,) * len(generation.tokens))
            )
            if len(rewards) != len(generation.tokens):
                raise AgentContractError(
                    "Process rewards must align with original output tokens."
                )
            for token_index in range(len(rewards)):
                if token_index == len(rewards) - 1:
                    rewards[token_index] += sum(
                        structured_process.get((target, index), 0.0)
                        for target in ("turn", "interaction")
                    )
                rewards[token_index] += structured_process.get(
                    ("token", (index, token_index)), 0.0
                )
            direct = [0.0] * len(generation.tokens)
            direct[-1] += sum(
                structured_advantages.get((target, index), 0.0)
                for target in ("turn", "interaction")
            )
            for token_index in range(len(direct)):
                direct[token_index] += structured_advantages.get(
                    ("token", (index, token_index)), 0.0
                )
            full.extend(generation.tokens)
            logprobs.extend(generation.logprobs)
            masks.extend([True] * len(generation.tokens))
            turns.extend([index] * len(generation.tokens))
            process_rewards.extend(rewards)
            direct_advantages.extend(direct)
            action_keys.extend(
                (interaction_ids[index], i) for i in range(len(generation.tokens))
            )
        # Keep the terminal task score separate from dense process rewards;
        # existing PPO advantage preparation combines them at the right places.
        return Trajectory(
            prompt_tokens=initial,
            completion_tokens=full[len(initial) :],
            logprobs=logprobs,
            reward=reward,
            group_id=episode.group_id,
            loss_mask=masks,
            turn_ids=turns,
            token_rewards=(
                process_rewards
                if score.process_rewards or structured_process
                else None
            ),
            token_advantages=direct_advantages if any(direct_advantages) else None,
            truncated=episode.status in {"max_turns", "token_limit", "context_limit"},
            reward_components={**score.components, "success": float(score.success)},
            rollout_id=episode.rollout_id or episode.episode_id,
            episode_id=episode.episode_id,
            sample_index=episode.sample_index,
            snapshot_id=episode.snapshot_id,
            policy_version=episode.policy_version,
            task_id=episode.task_id,
            fragment_id=f"{episode.rollout_id or episode.episode_id}:concat",
            interaction_ids=interaction_ids,
            action_keys=action_keys,
        )
