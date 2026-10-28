"""Trajectory and prepared-batch records shared by the PPO pipeline.

The module owns the token-aligned trajectory data that flows from rollout to
advantage preparation, actor/critic updates, and PyTRIO datums. Validation
remains here so every producer and consumer observes the same token, reward,
grouping, and provenance invariants.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


def validate_policy_versions(
    trajectories: Sequence[Trajectory],
    *,
    current_version: int | str | None = None,
    max_staleness: int = 0,
) -> None:
    """Reject a batch that combines known and unknown policy provenance.

    Behavior log probabilities are meaningful only relative to the policy that
    sampled them. Agent workflows attach a version to every exported row;
    direct legacy callers may omit provenance, but an update must not silently
    mix those two modes or combine two different versions.
    """
    if type(max_staleness) is not int or max_staleness < 0:
        raise ValueError("max_staleness must be a nonnegative integer.")
    versions = set()
    has_unversioned = False
    for trajectory in trajectories:
        trajectory.__post_init__()
        version = trajectory.policy_version
        if version is None:
            has_unversioned = True
        else:
            versions.add(version)
    if len(versions) > 1 or (versions and has_unversioned):
        raise ValueError(
            "One PPO update must use one policy_version for every trajectory."
        )
    if versions and current_version is not None:
        version = next(iter(versions))
        # Numeric versions support a bounded age. Opaque labels support exact
        # equality only: ordering a checkpoint URI or external label would
        # invent a temporal relationship that its producer never guaranteed.
        valid = (
            0 <= current_version - version <= max_staleness
            if type(version) is int and type(current_version) is int
            else version == current_version
        )
        if not valid:
            raise ValueError(
                "Stale or future policy_version cannot enter this PPO update."
            )


@dataclass(slots=True)
class Trajectory:
    """One unpadded prompt/completion sample and its aligned training fields.

    Auxiliary arrays describe completion tokens; ``values`` additionally holds
    one final bootstrap prediction for critic-based PPO. Optional provenance
    fields are carried alongside the numerical arrays so a custom workflow can
    audit which logical rollout, attempt, and policy produced a training row.
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    reward: float
    group_id: str
    loss_mask: list[bool] | None = None
    truncated: bool = False
    reference_logprobs: list[float] | None = None
    values: list[float] | None = None
    token_rewards: list[float] | None = None
    turn_ids: list[int] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    # Provenance is optional for legacy callers that construct trajectories
    # directly. Agent rollouts populate these fields before PPO sees the data,
    # which lets diagnostics distinguish one logical rollout from its physical
    # training rows without changing the loss input shape.
    rollout_id: str | None = None
    episode_id: str | None = None
    sample_index: int | None = None
    snapshot_id: str | None = None
    policy_version: int | str | None = None
    task_id: str | None = None
    fragment_id: str | None = None
    interaction_ids: tuple[str, ...] = ()
    parent_interaction_id: str | None = None
    # Action keys are completion-aligned (interaction ID, original token
    # offset) pairs. Context has None. They identify shared sampled actions
    # independently of their physical position in a branch or fragment.
    action_keys: list[tuple[str, int] | None] | None = None
    # Optional PRM/direct-advantage signal.  It is deliberately separate from
    # token_rewards so critic returns never learn a value target from a signal
    # that the caller declared to be an advantage.  It is appended after the
    # historical fields to preserve positional construction compatibility.
    token_advantages: list[float] | None = None
    # Branch provenance is optional for ordinary linear rollouts. A branch ID
    # turns multiple fragments of that branch into one logical PPO sequence,
    # while branch_group_id identifies sibling counterfactuals for reward and
    # advantage normalization without changing the original prompt group.
    branch_id: str | None = None
    branch_group_id: str | None = None
    branch_completed: bool | None = None
    shared_prefix_weight: float = 1.0
    # Action weights are completion aligned and multiply the policy objective.
    # They are normally one for active tokens and zero for context. Weighted
    # branch export uses fractional values only for actions shared by sibling
    # paths; branch-local actions remain at unit weight.
    action_weights: list[float] | None = None
    # A compacted rollout can contain physical rows with different tokenized
    # prompts. ``group_id`` remains the strict identical-prompt identity, while
    # reward_group_id keeps those fragments in the candidate comparison group.
    # context_segment is the zero-based causal segment within the logical path.
    reward_group_id: str | None = None
    context_segment: int = 0
    # AgentRL records one trusted pre-action environment observation for each
    # interaction. PPO treats these values as opaque provenance: they are
    # available to an AgentRL batch-level advantage provider but never enter a
    # PyTRIO datum, loss input, or remote request. The empty tuple preserves
    # compatibility for ordinary text rollouts and legacy custom producers.
    anchor_observations: tuple[Any, ...] = ()
    # Direct advantages normally augment reward/GAE-derived values. Batch-level
    # algorithms may instead provide the complete policy advantage while still
    # leaving rewards and critic returns untouched. Appending this switch keeps
    # every established positional constructor field unchanged.
    token_advantage_mode: str = "add"

    @property
    def logical_sequence_id(self) -> tuple[str, str | None] | tuple[str, int]:
        """Return the identity used to join fragments of one training path."""
        return (
            (self.rollout_id, self.branch_id)
            if self.rollout_id is not None
            else ("row", id(self))
        )

    @property
    def comparison_group_id(self) -> str:
        """Return the reward-comparison group for this branch or trajectory."""
        return self.branch_group_id or self.reward_group_id or self.group_id

    @property
    def effective_action_weights(self) -> list[float]:
        """Materialize objective weights without mutating legacy trajectories."""
        if self.action_weights is not None:
            return list(self.action_weights)
        return [1.0 if active else 0.0 for active in self.loss_mask]

    @property
    def provenance(self) -> dict:
        """Return a serializable source map for an ordered remote training row.

        This mapping stays beside SDK datums instead of entering loss inputs,
        where strings and identity fields have no numerical training meaning.
        A caller can persist it with a request ID and recover the origin of
        every row in the corresponding remote response.
        """
        source = {
            name: getattr(self, name)
            for name in (
                "rollout_id",
                "episode_id",
                "group_id",
                "sample_index",
                "snapshot_id",
                "policy_version",
                "task_id",
                "fragment_id",
                "interaction_ids",
                "parent_interaction_id",
                "branch_id",
                "branch_group_id",
                "branch_completed",
                "shared_prefix_weight",
                "reward_group_id",
                "context_segment",
            )
        }
        source["action_keys"] = (
            tuple(self.action_keys) if self.action_keys is not None else None
        )
        source["action_weights"] = (
            tuple(self.action_weights) if self.action_weights is not None else None
        )
        return source

    def __post_init__(self) -> None:
        if not self.prompt_tokens or not self.completion_tokens:
            raise ValueError(
                "A trajectory needs nonempty prompt and completion tokens."
            )
        if any(
            isinstance(t, bool) or not isinstance(t, int) or t < 0
            for t in self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("Token IDs must be nonnegative integers.")
        n = len(self.completion_tokens)
        if self.logprobs is None:
            raise ValueError(
                "Behavior logprobs are required for every completion token."
            )
        if self.loss_mask is None:
            self.loss_mask = [True] * n
        if len(self.loss_mask) != n or not all(
            isinstance(x, bool) for x in self.loss_mask
        ):
            raise ValueError("loss_mask must contain one boolean per completion token.")
        if not any(self.loss_mask):
            raise ValueError(
                "A trajectory must contain at least one active action token."
            )
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError(
                "group_id must be a nonempty string identifying the prompt group."
            )
        for name in (
            "rollout_id",
            "episode_id",
            "snapshot_id",
            "task_id",
            "fragment_id",
            "parent_interaction_id",
            "branch_id",
            "branch_group_id",
            "reward_group_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or None.")
        if (
            self.branch_completed is not None
            and type(self.branch_completed) is not bool
        ):
            raise ValueError("branch_completed must be a boolean or None.")
        if self.branch_id is None and (
            self.branch_group_id is not None or self.branch_completed is not None
        ):
            raise ValueError("Branch metadata requires branch_id.")
        if type(self.context_segment) is not int or self.context_segment < 0:
            raise ValueError("context_segment must be a nonnegative integer.")
        if self.branch_id is not None and self.branch_completed is not True:
            raise ValueError("Only completed branches may enter PPO trajectories.")
        if (
            isinstance(self.shared_prefix_weight, bool)
            or not isinstance(self.shared_prefix_weight, Real)
            or not math.isfinite(float(self.shared_prefix_weight))
            or self.shared_prefix_weight < 0
        ):
            raise ValueError("shared_prefix_weight must be finite and nonnegative.")
        if self.sample_index is not None and (
            type(self.sample_index) is not int or self.sample_index < 0
        ):
            raise ValueError("sample_index must be a nonnegative integer or None.")
        if self.policy_version is not None and (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, (int, str))
            or (isinstance(self.policy_version, str) and not self.policy_version)
        ):
            raise ValueError(
                "policy_version must be a nonempty integer/string or None."
            )
        if len(set(self.interaction_ids)) != len(self.interaction_ids) or any(
            not isinstance(value, str) or not value for value in self.interaction_ids
        ):
            raise ValueError("interaction_ids must contain unique nonempty strings.")
        if self.anchor_observations and len(self.anchor_observations) != len(
            self.interaction_ids
        ):
            raise ValueError(
                "anchor_observations must align with interaction_ids when provided."
            )
        if self.token_advantage_mode not in {"add", "replace"}:
            raise ValueError("token_advantage_mode must be add or replace.")
        if self.action_keys is not None:
            if len(self.action_keys) != n:
                raise ValueError("action_keys must align with completion tokens.")
            for key, active in zip(self.action_keys, self.loss_mask, strict=True):
                if key is None:
                    if active:
                        raise ValueError("Every active token needs an action key.")
                elif (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or key[0] not in self.interaction_ids
                    or type(key[1]) is not int
                    or key[1] < 0
                ):
                    raise ValueError(
                        "Action keys must identify an interaction and token offset."
                    )
        if self.action_weights is not None:
            if len(self.action_weights) != n:
                raise ValueError("action_weights must align with completion tokens.")
            for weight, active in zip(self.action_weights, self.loss_mask, strict=True):
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, Real)
                    or not math.isfinite(float(weight))
                    or weight < 0
                    or (active and weight <= 0)
                    or (not active and weight != 0)
                ):
                    raise ValueError(
                        "Active action weights must be positive and context weights zero."
                    )
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite.")
        if not isinstance(self.reward_components, Mapping):
            raise TypeError("reward_components must be a mapping of finite scalars.")
        components = {}
        for name, value in self.reward_components.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "reward_components must be a mapping of finite scalars."
                )
            components[name.strip()] = float(value)
        self.reward_components = components
        for name in (
            "logprobs",
            "reference_logprobs",
            "values",
            "token_rewards",
            "token_advantages",
        ):
            values = getattr(self, name)
            if values is not None and (
                len(values) != n + int(name == "values")
                or any(not math.isfinite(x) for x in values)
            ):
                raise ValueError(
                    f"{name} must be finite and aligned with completion tokens"
                    + (" plus one bootstrap value." if name == "values" else ".")
                )
        if self.turn_ids is not None:
            if len(self.turn_ids) != n or any(
                isinstance(turn_id, bool) or not isinstance(turn_id, int)
                for turn_id in self.turn_ids
            ):
                raise ValueError(
                    "turn_ids must contain one integer per completion token."
                )
            previous = -1
            for turn_id, active in zip(self.turn_ids, self.loss_mask, strict=True):
                if not active:
                    if turn_id != -1:
                        raise ValueError(
                            "Inactive completion tokens must use turn_id -1."
                        )
                    continue
                if not 0 <= turn_id < n:
                    raise ValueError("Active turn_ids must be in [0, T).")
                if turn_id < previous:
                    raise ValueError(
                        "Active turn_ids must be temporally nondecreasing."
                    )
                previous = turn_id


@dataclass(slots=True)
class PreparedBatch:
    """Frozen actor targets prepared once before minibatch optimization."""

    trajectories: list[Trajectory]
    advantages: list[list[float]]
    returns: list[list[float]]
    masks: list[list[bool]]
    proximal_logprobs: list[list[float]]
    prompt_mean_scale: float = 1.0
    loss_weights: list[float] = field(default_factory=list)


@dataclass(slots=True)
class PPOState:
    """Mutable lifecycle state returned by the root PPO trainer."""

    global_step: int = 0
    optimizer_steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    critic_optimizer_steps: int = 0
    critic_checkpoint_path: str | None = None
    data_cursor: int = 0
    checkpoint_manifest_path: str | None = None
