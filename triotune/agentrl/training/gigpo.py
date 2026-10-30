"""Group-in-Group advantages for complete AgentRL rollout batches.

GiGPO adds state-local credit assignment to an episode-level relative policy
objective. The implementation remains in AgentRL because it depends on trusted
environment observations and cross-trajectory turn grouping. PPO and PyTRIO
receive only the final token-aligned advantage and do not need an algorithm-
specific loss, datum, or remote backward interface.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

from ...trainer.ppo.trajectory import Trajectory
from ..types import AgentBatchAdvantages, AgentContractError


@dataclass(frozen=True, slots=True)
class GiGPOConfig:
    """Configure episode and state-relative GiGPO advantages.

    Parameters:
        step_advantage_weight: Weight ``omega`` applied to the state-local
            advantage before it is added to the episode advantage.
        discount: Turn-level return-to-go discount ``gamma``. Masked tool and
            template tokens never consume a discount step.
        mode: ``mean_norm`` subtracts the group mean. ``mean_std_norm`` also
            divides by the sample standard deviation plus ``epsilon``.
        enable_similarity: Cluster similar text anchors instead of requiring
            exact structural equality.
        similarity_threshold: Minimum SequenceMatcher ratio for joining a text
            anchor to an existing greedy cluster. Similarity mode requires a
            value strictly between zero and one.
        epsilon: Positive numerical stabilizer for standard-deviation scaling.
        episode_statistic_weighting: ``trajectory`` counts each logical policy
            trajectory once, matching Triotune and the paper. ``step`` repeats
            its episode return once per active decision, matching verl-agent's
            current step-row statistics.
    """

    step_advantage_weight: float = 1.0
    discount: float = 0.95
    mode: Literal["mean_norm", "mean_std_norm"] = "mean_norm"
    enable_similarity: bool = False
    similarity_threshold: float = 0.95
    epsilon: float = 1e-6
    episode_statistic_weighting: Literal["trajectory", "step"] = "trajectory"

    def __post_init__(self) -> None:
        for name in ("step_advantage_weight", "discount", "epsilon"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"gigpo.{name} must be finite.")
        if self.step_advantage_weight < 0:
            raise ValueError("gigpo.step_advantage_weight must be nonnegative.")
        if not 0 <= self.discount <= 1:
            raise ValueError("gigpo.discount must be in [0, 1].")
        if self.epsilon <= 0:
            raise ValueError("gigpo.epsilon must be positive.")
        if self.mode not in {"mean_norm", "mean_std_norm"}:
            raise ValueError("gigpo.mode must be mean_norm or mean_std_norm.")
        if type(self.enable_similarity) is not bool:
            raise ValueError("gigpo.enable_similarity must be boolean.")
        if self.enable_similarity and not 0 < self.similarity_threshold < 1:
            raise ValueError(
                "gigpo.similarity_threshold must be in (0, 1) when enabled."
            )
        if not self.enable_similarity and not 0 <= self.similarity_threshold <= 1:
            raise ValueError("gigpo.similarity_threshold must be in [0, 1].")
        if self.episode_statistic_weighting not in {"trajectory", "step"}:
            raise ValueError(
                "gigpo.episode_statistic_weighting must be trajectory or step."
            )


@dataclass(slots=True)
class _Step:
    """One active environment decision reconstructed from physical rows."""

    interaction_id: str
    anchor: Any
    order: tuple[int, int]
    reward: float
    locations: list[tuple[int, int]]
    discounted_return: float = 0.0


@dataclass(slots=True)
class _Candidate:
    """Join fragments that represent one logical episode or branch."""

    comparison_group_id: str
    outcome_reward: float
    steps: dict[str, _Step] = field(default_factory=dict)


def _hashable_anchor(value: Any) -> Any:
    """Freeze a raw structural observation for exact state matching.

    Falling back to ``repr`` would make grouping depend on memory addresses or
    application formatting. Unsupported live objects therefore fail explicitly.
    Array-like values are accepted through ``tolist`` without adding NumPy as a
    runtime dependency of AgentRL.
    """

    if value is None:
        raise AgentContractError(
            "GiGPO requires an anchor observation for every active interaction."
        )
    if isinstance(value, (str, bytes, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AgentContractError(
                "GiGPO anchor observations cannot contain nonfinite numbers."
            )
        return value
    if isinstance(value, Mapping):
        items = [
            (_hashable_anchor(key), _hashable_anchor(item))
            for key, item in value.items()
        ]
        return tuple(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_anchor(item) for item in value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _hashable_anchor(tolist())
    raise AgentContractError(
        "GiGPO anchors must contain strings, finite scalars, mappings, "
        f"sequences, or array-like values; got {type(value).__name__}."
    )


def _relative_advantage(
    value: float,
    population: Sequence[float],
    config: GiGPOConfig,
    *,
    singleton_zero: bool,
) -> float:
    """Center one value with the normalization used by the reference code."""

    if len(population) == 1:
        # A state singleton has no alternative action and contributes no local
        # learning signal. The episode path follows verl-agent: a one-episode
        # group retains its uncentered return rather than forcing it to zero.
        return 0.0 if singleton_zero else value
    mean = sum(population) / len(population)
    centered = value - mean
    if config.mode == "mean_norm":
        return centered
    variance = sum((item - mean) ** 2 for item in population) / (
        len(population) - 1
    )
    return centered / (math.sqrt(variance) + config.epsilon)


class GiGPOAdvantageProvider:
    """Compute complete ``A^E + omega * A^S`` rows for AgentTrainingLoop.

    The provider is deliberately stateless across updates. Every call derives
    grouping from the supplied fixed-policy batch, so asynchronous collection
    cannot leak anchors or rewards between policy versions.
    """

    def __init__(self, config: GiGPOConfig | Mapping[str, Any] | None = None):
        if config is None:
            config = GiGPOConfig()
        elif isinstance(config, Mapping):
            config = GiGPOConfig(**dict(config))
        elif not isinstance(config, GiGPOConfig):
            raise TypeError("config must be GiGPOConfig, a mapping, or None.")
        self.config = config

    def checkpoint_identity(self) -> dict[str, Any]:
        """Return serializable settings that affect resumed advantage targets."""

        return asdict(self.config)

    def validate_trainer(self, trainer) -> None:
        """Reject PPO settings whose effects a pre-step provider cannot see."""

        args = trainer.args
        if args.use_critic:
            raise ValueError("GiGPO requires critic.enabled=false.")
        if args.reward_norm is not None or args.adv_norm is not None:
            raise ValueError(
                "GiGPO computes both relative normalizations; set reward_norm "
                "and adv_norm to null."
            )
        if args.kl_ctl:
            raise ValueError(
                "GiGPO replacement advantages cannot consume KL-as-reward; set "
                "kl_ctl=0 or add an independent reference KL loss."
            )
        if args.mask_no_eos_with_zero or args.overlong_tokens:
            raise ValueError(
                "GiGPO currently requires mask_no_eos_with_zero=false and "
                "overlong_tokens=0 because those reward transforms run later."
            )
        plan = trainer.algorithm_plan
        if plan is not None and any(
            getattr(plan, name) is not None
            for name in (
                "reward_shaper",
                "process_shaper",
                "normalizer",
                "advantage_estimator",
                "return_estimator",
            )
        ):
            raise ValueError(
                "GiGPO cannot precede custom reward/advantage preparation stages."
            )

    @staticmethod
    def _collect_candidates(
        trajectories: Sequence[Trajectory],
    ) -> dict[tuple[Any, ...], _Candidate]:
        """Reassemble logical candidates without counting fragments twice."""

        candidates: dict[tuple[Any, ...], _Candidate] = {}
        for row_index, trajectory in enumerate(trajectories):
            if trajectory.turn_ids is None:
                raise AgentContractError(
                    "GiGPO requires completion-aligned turn_ids."
                )
            if not trajectory.interaction_ids or not trajectory.anchor_observations:
                raise AgentContractError(
                    "GiGPO requires trusted anchors aligned with interaction_ids."
                )
            key = tuple(trajectory.logical_sequence_id)
            candidate = candidates.setdefault(
                key,
                _Candidate(trajectory.comparison_group_id, float(trajectory.reward)),
            )
            if (
                candidate.comparison_group_id != trajectory.comparison_group_id
                or not math.isclose(
                    candidate.outcome_reward,
                    float(trajectory.reward),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise AgentContractError(
                    "Fragments of one GiGPO candidate must share group and outcome."
                )

            positions: dict[int, list[int]] = defaultdict(list)
            for token_index, (turn_id, active) in enumerate(
                zip(trajectory.turn_ids, trajectory.loss_mask, strict=True)
            ):
                if active:
                    positions[turn_id].append(token_index)
            rewards = trajectory.token_rewards or [0.0] * len(
                trajectory.completion_tokens
            )
            for turn_id, token_indices in positions.items():
                if not 0 <= turn_id < len(trajectory.interaction_ids):
                    raise AgentContractError(
                        "GiGPO turn_ids must index interaction_ids."
                    )
                interaction_id = trajectory.interaction_ids[turn_id]
                if interaction_id in candidate.steps:
                    raise AgentContractError(
                        "A GiGPO interaction may be active in only one physical row."
                    )
                candidate.steps[interaction_id] = _Step(
                    interaction_id=interaction_id,
                    anchor=trajectory.anchor_observations[turn_id],
                    order=(trajectory.context_segment, turn_id),
                    reward=sum(float(rewards[index]) for index in token_indices),
                    locations=[(row_index, index) for index in token_indices],
                )
        if not candidates or any(not candidate.steps for candidate in candidates.values()):
            raise AgentContractError(
                "Every GiGPO candidate must contain an active decision step."
            )
        return candidates

    def _state_groups(
        self, candidates: Mapping[tuple[Any, ...], _Candidate]
    ) -> dict[tuple[tuple[Any, ...], str], tuple[Any, ...]]:
        """Group anchors only within the same task/comparison population."""

        config = self.config
        assigned = {}
        if not config.enable_similarity:
            for candidate_key, candidate in candidates.items():
                for step in candidate.steps.values():
                    assigned[(candidate_key, step.interaction_id)] = (
                        candidate.comparison_group_id,
                        _hashable_anchor(step.anchor),
                    )
            return assigned

        clusters: dict[str, list[str]] = defaultdict(list)
        for candidate_key, candidate in candidates.items():
            for step in sorted(candidate.steps.values(), key=lambda item: item.order):
                if not isinstance(step.anchor, str):
                    raise AgentContractError(
                        "Similarity-based GiGPO supports only text anchors."
                    )
                cluster_index = None
                for index, representative in enumerate(
                    clusters[candidate.comparison_group_id]
                ):
                    if (
                        SequenceMatcher(None, step.anchor, representative).ratio()
                        >= config.similarity_threshold
                    ):
                        cluster_index = index
                        break
                if cluster_index is None:
                    cluster_index = len(clusters[candidate.comparison_group_id])
                    clusters[candidate.comparison_group_id].append(
                        step.anchor
                    )
                assigned[(candidate_key, step.interaction_id)] = (
                    candidate.comparison_group_id,
                    cluster_index,
                )
        return assigned

    def advantages(
        self, trajectories: Sequence[Trajectory]
    ) -> AgentBatchAdvantages:
        """Compute episode and state-relative advantages for one update batch."""

        trajectories = tuple(trajectories)
        candidates = self._collect_candidates(trajectories)
        episode_returns = {}
        for candidate_key, candidate in candidates.items():
            ordered = sorted(candidate.steps.values(), key=lambda item: item.order)
            # The scalar outcome is an environment reward on the final decision
            # and must be assigned once even when export produced several rows.
            ordered[-1].reward += candidate.outcome_reward
            running = 0.0
            for step in reversed(ordered):
                running = step.reward + self.config.discount * running
                step.discounted_return = running
            episode_returns[candidate_key] = sum(step.reward for step in ordered)

        episode_populations: dict[str, list[float]] = defaultdict(list)
        for candidate_key, candidate in candidates.items():
            repeats = (
                len(candidate.steps)
                if self.config.episode_statistic_weighting == "step"
                else 1
            )
            episode_populations[candidate.comparison_group_id].extend(
                [episode_returns[candidate_key]] * repeats
            )

        state_group = self._state_groups(candidates)
        state_populations: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for candidate_key, candidate in candidates.items():
            for step in candidate.steps.values():
                state_populations[
                    state_group[(candidate_key, step.interaction_id)]
                ].append(step.discounted_return)

        rows = [[0.0] * len(trajectory.completion_tokens) for trajectory in trajectories]
        for candidate_key, candidate in candidates.items():
            episode_advantage = _relative_advantage(
                episode_returns[candidate_key],
                episode_populations[candidate.comparison_group_id],
                self.config,
                singleton_zero=False,
            )
            for step in candidate.steps.values():
                group = state_group[(candidate_key, step.interaction_id)]
                state_advantage = _relative_advantage(
                    step.discounted_return,
                    state_populations[group],
                    self.config,
                    singleton_zero=True,
                )
                joint = (
                    episode_advantage
                    + self.config.step_advantage_weight * state_advantage
                )
                for row_index, token_index in step.locations:
                    rows[row_index][token_index] = joint
        return AgentBatchAdvantages(rows, mode="replace")
