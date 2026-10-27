"""Convert recorded episodes into the existing PPO trajectory contract.

An exported trajectory starts with the first request as its prompt, followed
by all generated actions and intervening context. Model output positions keep
their original behavior probabilities; inserted observations and template
markers remain visible as context but are excluded from policy optimization.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from ...trainer.ppo.trajectory import Trajectory
from ..training.rewards import _project_score
from ..types import AgentContractError, EpisodeResult, EpisodeScore, RolloutExport


class EpisodeTrajectoryBuilder:
    """Preserve action tokens verbatim and mask added observations/template gaps.

    Later prompts must extend the exact previous prompt plus sampled output.
    Re-rendering that changes any previous token is rejected, never repaired by
    re-tokenizing text or attaching behavior scores to different tokens.
    """

    def __init__(self, *, mode: str = "concat"):
        """Choose whole-episode, per-turn, or interaction-tree training rows.

        Individual rows retain the initial prompt and all preceding tokens as
        masked context. Only the selected turn is active. This keeps group
        prompts identical and avoids training shared history repeatedly.
        Each turn receives the candidate outcome and its own process rewards;
        GAE runs independently within each row. This is an explicit temporal
        objective and need not equal concatenated GAE when discount, lambda,
        dense rewards, or critic bootstrap differ across turn boundaries.

        Branch mode emits one logical training path for every completed
        ``EpisodeBranch``. It reconstructs the shared ancestry from immutable
        interaction parent links, preserves original token evidence, and keeps
        unfinished siblings only in branch metadata.
        """
        if mode not in {"concat", "individual", "branch"}:
            raise ValueError("Export mode must be concat, individual, or branch.")
        self.mode = mode

    def export(self, episode: EpisodeResult) -> RolloutExport:
        """Export a candidate with stable fragment-to-interaction mapping.

        Outcome normalization uses one reference reward per logical rollout,
        regardless of its fragment count. Repeated context carries no active
        action mask; the collector additionally checks action keys for custom
        exporters that provide overlapping paths or branching trajectories.
        """
        if self.mode == "branch":
            return self._export_branches(episode)
        rows = self._path_rows(
            episode,
            fragment_prefix=episode.rollout_id or episode.episode_id,
            individual=self.mode == "individual",
        )
        compacted = len({row.context_segment for row in rows}) > 1
        return RolloutExport(
            trajectories=rows,
            logical_rollout_id=episode.rollout_id or episode.episode_id,
            fragment_map={row.fragment_id: row.interaction_ids for row in rows},
            group_id=episode.group_id,
            metadata={
                "mode": self.mode,
                "outcome_scope": "rollout",
                "shared_prefix": "context_only" if not compacted else "segments",
                "context_segments": 1
                + sum(
                    compaction.turn_index < len(episode.turns)
                    for compaction in episode.compactions
                ),
            },
        )

    @staticmethod
    def _slice_score(episode, indices):
        """Remap turn-scoped P1 signals onto one compacted causal segment."""
        positions = {source: target for target, source in enumerate(indices)}
        process = {
            positions[index]: values
            for index, values in episode.score.process_rewards.items()
            if index in positions
        }
        assignments = []
        for assignment in episode.score.assignments:
            if assignment.target == "trajectory":
                assignments.append(assignment)
                continue
            position = assignment.position
            if assignment.target == "token":
                if not isinstance(position, tuple) or position[0] not in positions:
                    continue
                position = (positions[position[0]], position[1])
            else:
                if type(position) is not int or position not in positions:
                    continue
                position = positions[position]
            assignments.append(replace(assignment, position=position))
        return EpisodeScore(
            reward=episode.score.reward,
            success=episode.score.success,
            components=dict(episode.score.components),
            process_rewards=process,
            assignments=tuple(assignments),
        )

    @staticmethod
    def _slice_episode(episode, start, stop):
        """Detach a segment from external parents while preserving action IDs."""
        generated = [
            interaction
            for interaction in episode.interactions
            if interaction.turn is not None
        ]
        interactions = []
        for offset, interaction in enumerate(generated[start:stop]):
            if offset == 0:
                interaction = replace(
                    interaction,
                    context=replace(interaction.context, parent_interaction_id=None),
                )
            interactions.append(interaction)
        return replace(
            episode,
            turns=list(episode.turns[start:stop]),
            interactions=interactions,
            score=EpisodeTrajectoryBuilder._slice_score(
                episode, tuple(range(start, stop))
            ),
            branches=[],
            compactions=[],
        )

    @staticmethod
    def _prompt_group(group_id, prompt_tokens):
        """Create a stable physical group for one exact compacted prompt."""
        digest = hashlib.sha256(
            ",".join(str(token) for token in prompt_tokens).encode("ascii")
        ).hexdigest()[:20]
        return f"{group_id}:prompt:{digest}"

    @staticmethod
    def _individual_rows(whole, turn_count, fragment_prefix):
        """Split one causal segment into independently active turn rows."""
        rows = []
        for index in range(turn_count):
            end = max(i for i, turn in enumerate(whole.turn_ids) if turn == index) + 1
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
                    fragment_id=f"{fragment_prefix}:turn:{index}",
                    interaction_ids=whole.interaction_ids[: index + 1],
                    parent_interaction_id=(
                        whole.interaction_ids[index - 1]
                        if index
                        else whole.parent_interaction_id
                    ),
                    action_keys=whole.action_keys[:end],
                    truncated=whole.truncated if index == turn_count - 1 else False,
                )
            )
        return rows

    def _path_rows(self, episode, *, fragment_prefix, individual=False):
        """Build exact-token rows separated at every recorded context reset."""
        starts = [0] + [
            compaction.turn_index
            for compaction in episode.compactions
            if compaction.turn_index < len(episode.turns)
        ]
        for index in range(1, len(episode.turns)):
            previous = episode.turns[index - 1]
            prefix = previous.prompt_tokens + previous.generation.tokens
            if tuple(episode.turns[index].prompt_tokens[: len(prefix)]) != prefix:
                starts.append(index)
        starts = sorted(set(starts))
        stops = starts[1:] + [len(episode.turns)]
        generated = [
            interaction
            for interaction in episode.interactions
            if interaction.turn is not None
        ]
        rows = []
        for segment, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            sliced = (
                episode
                if len(starts) == 1
                else self._slice_episode(episode, start, stop)
            )
            whole = self.build(sliced)
            whole.context_segment = segment
            whole.parent_interaction_id = (
                generated[start].parent_interaction_id if start else None
            )
            whole.fragment_id = f"{fragment_prefix}:segment:{segment}"
            if len(starts) > 1:
                whole.reward_group_id = episode.group_id
                whole.group_id = self._prompt_group(
                    episode.group_id, whole.prompt_tokens
                )
            whole.__post_init__()
            rows.extend(
                self._individual_rows(
                    whole,
                    stop - start,
                    f"{fragment_prefix}:segment:{segment}",
                )
                if individual
                else [whole]
            )
        return rows

    @staticmethod
    def _branch_path(episode: EpisodeResult, branch) -> tuple[str, ...]:
        """Recover a root-to-leaf path without copying sampled interactions.

        The branch record owns only its suffix. Walking from its fork parent to
        the root ensures that every shared action keeps the exact interaction ID
        used by sibling paths, which is required for collector-side weighting.
        Episode validation has already checked the suffix links; this method
        additionally guards against malformed cycles supplied after validation.
        """
        interactions = {item.interaction_id: item for item in episode.interactions}
        prefix, seen = [], set(branch.interaction_ids)
        current = branch.parent_interaction_id
        while current is not None:
            if current in seen or current not in interactions:
                raise AgentContractError("Branch ancestry is cyclic or incomplete.")
            seen.add(current)
            prefix.append(current)
            current = interactions[current].parent_interaction_id
        return tuple(reversed(prefix)) + branch.interaction_ids

    @staticmethod
    def _branch_score(episode: EpisodeResult, path: tuple[str, ...], branch):
        """Project global process signals onto one branch-local turn sequence.

        Branch reward is already the complete local outcome, so episode outcome
        assignments are not added again. Trajectory advantages apply to every
        completed path. Turn, interaction, and token signals are retained only
        when their interaction lies on this path, with positions remapped to the
        branch-local order. This preserves P1 reward/advantage separation.
        """
        generated = [
            item.interaction_id
            for item in episode.interactions
            if item.turn is not None
        ]
        global_index = {
            interaction_id: index for index, interaction_id in enumerate(generated)
        }
        local_index = {
            interaction_id: index for index, interaction_id in enumerate(path)
        }
        process = {}
        for interaction_id in path:
            source_index = global_index[interaction_id]
            if source_index in episode.score.process_rewards:
                process[local_index[interaction_id]] = episode.score.process_rewards[
                    source_index
                ]
        assignments = []
        for assignment in episode.score.assignments:
            if assignment.target == "trajectory":
                # Branch outcomes replace episode outcome rewards. A direct
                # trajectory advantage remains a policy-only signal on every
                # completed path; collector weights still own shared actions.
                if assignment.mode == "advantage":
                    assignments.append(assignment)
                continue
            position = assignment.position
            if assignment.target == "token":
                if not isinstance(position, tuple):
                    continue
                source_turn, token_index = position
            else:
                if type(position) is not int:
                    continue
                source_turn, token_index = position, None
            if not 0 <= source_turn < len(generated):
                continue
            interaction_id = generated[source_turn]
            if interaction_id not in local_index:
                continue
            target_turn = local_index[interaction_id]
            assignments.append(
                replace(
                    assignment,
                    position=(target_turn, token_index)
                    if token_index is not None
                    else target_turn,
                )
            )
        components = dict(episode.score.components)
        components["branch_reward"] = float(branch.reward)
        return EpisodeScore(
            reward=float(branch.reward),
            success=episode.score.success,
            components=components,
            process_rewards=process,
            assignments=tuple(assignments),
        )

    def _export_branches(self, episode: EpisodeResult) -> RolloutExport:
        """Export completed leaves while retaining the full declared tree.

        Branch groups are scoped by logical rollout before reaching PPO. A
        caller can therefore reuse a readable local group label such as
        ``tool-choice`` in concurrent episodes without accidentally merging
        unrelated reward baselines. Every row starts at the episode's original
        prompt, so the existing PPO prompt-consistency invariant still applies.
        """
        episode.__post_init__()
        if not episode.trainable or not episode.branches:
            raise AgentContractError(
                "Branch export requires a scored episode with declared branches."
            )
        interaction_by_id = {
            interaction.interaction_id: interaction
            for interaction in episode.interactions
        }
        rows = []
        group_metadata = {}
        rollout_id = episode.rollout_id or episode.episode_id
        for branch in episode.branches:
            scoped_group = f"{rollout_id}:branch-group:{branch.group_id}"
            state = group_metadata.setdefault(
                scoped_group,
                {
                    "branch_ids": [],
                    "completed_branch_ids": [],
                    "parent_interaction_id": branch.parent_interaction_id,
                },
            )
            state["branch_ids"].append(branch.branch_id)
            if not branch.completed:
                continue
            state["completed_branch_ids"].append(branch.branch_id)
            path = self._branch_path(episode, branch)
            interactions = [
                interaction_by_id[interaction_id] for interaction_id in path
            ]
            if any(interaction.turn is None for interaction in interactions):
                raise AgentContractError(
                    "A completed branch can contain only sampled interactions."
                )
            branch_episode = replace(
                episode,
                turns=[interaction.turn for interaction in interactions],
                interactions=interactions,
                score=self._branch_score(episode, path, branch),
                branches=[],
                compactions=[],
            )
            branch_rows = self._path_rows(
                branch_episode,
                fragment_prefix=f"{rollout_id}:branch:{branch.branch_id}",
            )
            for row in branch_rows:
                row.parent_interaction_id = branch.parent_interaction_id
                row.branch_id = branch.branch_id
                row.branch_group_id = scoped_group
                row.branch_completed = True
                row.shared_prefix_weight = float(branch.shared_prefix_weight)
                row.__post_init__()
                rows.append(row)
        if not rows:
            raise AgentContractError("Branch export contains no completed branches.")
        for state in group_metadata.values():
            state["complete"] = len(state["branch_ids"]) == len(
                state["completed_branch_ids"]
            )
        return RolloutExport(
            trajectories=rows,
            logical_rollout_id=rollout_id,
            fragment_map={row.fragment_id: row.interaction_ids for row in rows},
            group_id=episode.group_id,
            metadata={
                "mode": "branch",
                "outcome_scope": "branch",
                "shared_prefix": "original_actions",
                "branch_groups": group_metadata,
            },
            branch_map={branch.branch_id: branch for branch in episode.branches},
        )

    def validate_prefix(self, episode: EpisodeResult, prompt_tokens: list[int]) -> None:
        """Reject a next request that changes any previously sampled token.

        The first request has no prior generation to compare. Later requests
        may append observations and role delimiters, but a shorter or rewritten
        prefix cannot share the old action probabilities safely.
        """
        if episode.turns:
            if episode.compactions and episode.compactions[-1].turn_index == len(
                episode.turns
            ):
                if (
                    tuple(prompt_tokens)
                    != episode.compactions[-1].compacted_prompt_tokens
                ):
                    raise AgentContractError(
                        "Rendered prompt differs from its compaction record."
                    )
                return
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
        # Trace and training consume the same raw projection. Shaping,
        # normalization, GAE, and loss weights remain owned by the PPO layer.
        reward, projected_rewards, projected_advantages = _project_score(
            score, episode.turns
        )
        # Preserve the distinction between an explicitly submitted zero
        # process signal and the absence of process supervision.
        has_process_rewards = bool(score.process_rewards) or any(
            assignment.mode == "reward" and assignment.target != "trajectory"
            for assignment in score.assignments
        )
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
            rewards = projected_rewards[index]
            direct = projected_advantages[index]
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
            token_rewards=(process_rewards if has_process_rewards else None),
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
