# Agent execution and training data

For inference with the optional OpenAI SDK and registered local tools, see
[OpenAI SDK tools and sessions](agentrl_openai.md). Its API traces are separate
from the exact-token PPO trajectory contracts described here.

## Candidate identity

`ExecutionContext` contains immutable rollout, episode, group, candidate,
snapshot, policy, task, and interaction identity. A rollout identifies one
candidate; candidates of the same question share a group. The collector creates
the context before calling `EpisodeExecutor.run(..., context=context)`.

Tools and synchronous functions can call `current_execution_context()` without
accepting identity in model-generated arguments. Context is isolated between
asyncio tasks and propagated to `asyncio.to_thread`. Process adapters must
serialize and establish it explicitly. Scorers receive the episode context;
loggers receive the same trusted context while writing a completed episode.

Each model request has an `Interaction`, including failed requests without a
generation. It records the parent ID, half-open input/output token spans,
original turn, tool observations, reward, retries, status, latency in seconds,
and resource owner. A resource owner is an identity, not an instruction to
close a shared client. Parents precede children and interaction IDs are unique.

`GenerationEvent`, `ToolCallEvent`, `ToolResultEvent`, and `RewardEvent` record
semantic evidence. Generic `AgentEvent` records retries and terminal states.
Their timestamps are process-local monotonic seconds. Reward events describe
scorer output; they are not added to the reward a second time. JSONL episode
records include interactions, events, and export metadata.

## Export and group policy

```python
from triotune.agentrl import AgentConfig

config = AgentConfig(
    export_mode="branch",
    min_usable_group_size=2,
    shared_prefix_policy="weighted",
)
```

The default export mode is `concat`: one candidate produces one trajectory.
`individual` produces one row per generated turn. Every row retains the same
initial prompt, while preceding model output and tool observations are masked
context. Only the selected turn receives a policy gradient. This preserves
PPO's identical-prompt group constraint.

Individual rows receive the candidate outcome and their own process rewards.
GAE is evaluated independently per row. Individual and concatenated objectives
can differ when discount, lambda, process rewards, or critic bootstrap cross
turn boundaries. Export mode is an explicit temporal modeling choice.

Custom exporters implement `export(episode) -> RolloutExport`, with trajectories,
logical rollout ID, group ID, metadata, and a fragment-to-interaction mapping.
Each trajectory carries a fragment ID, interaction IDs, and completion-aligned
action keys `(interaction_id, original_output_offset)`. Context positions have
no action key. Export validation checks the original token, behavior logprob,
causal position, and conditioning prefix of every active action.

Shared actions are masked after their first occurrence under `deduplicate`.
`repeat` explicitly counts each occurrence. `weighted` retains one occurrence
per logical branch and normalizes shared-action contributions from the branches'
declared prefix weights. Branch-local actions retain unit weight. The actor and
critic consume the same completion-aligned weights; weighted rows use the
custom PyTRIO loss boundary because the native objective has no token-weight
field. Conflicting token, logprob, causal-position, or prefix evidence is always
an error. Fragments with no active actions are omitted. Differing initial
prompts cannot share a PPO group.

## Branch trees

`EpisodeBranch` declares a rollout-local branch suffix. Its
`parent_interaction_id` is the fork state, and `interaction_ids` is the ordered
parent-linked suffix after that state. The exporter walks parent links to recover
the exact shared prefix rather than reconstructing it from displayed text.

```python
from triotune.agentrl import EpisodeBranch

episode.branches = [
    EpisodeBranch(
        branch_id="tool-a",
        parent_interaction_id=state_interaction_id,
        interaction_ids=(tool_a_interaction_id, final_a_interaction_id),
        group_id="tool-choice",
        reward=1.0,
        completed=True,
        shared_prefix_weight=0.5,
    ),
    EpisodeBranch(
        branch_id="tool-b",
        parent_interaction_id=state_interaction_id,
        interaction_ids=(tool_b_interaction_id, final_b_interaction_id),
        group_id="tool-choice",
        reward=0.0,
        completed=True,
        shared_prefix_weight=0.5,
    ),
]
```

Branch reward is the complete branch-local outcome, not an increment to the
episode outcome. Trajectory-level episode assignments are therefore not copied
onto every branch. Process-reward and direct-advantage assignments are projected
onto the interactions present in each path and retain their P1 reward mode.
Each branch group is scoped by logical rollout before PPO normalization, so
concurrent episodes can reuse readable local labels without merging baselines.

Only complete sibling groups enter training. An unfinished sibling remains in
`RolloutExport.branch_map`, group metadata, and the episode log, but its completed
siblings are not trained as though the missing reward were zero.
`AgentRolloutBatch.branch_groups` exposes branch IDs, completed branch IDs, fork
parent, and completeness independently of candidate-level
`AgentRolloutBatch.groups`. PPO counts multiple fragments of one branch once for
reward normalization and sequence reductions, while separate branches of the
same logical rollout retain their distinct local rewards.

By default all candidates must be usable. An explicit `min_usable_group_size`
allows partial groups; the threshold counts candidates, not fragments. Failed
attempts remain in episode diagnostics and never become zero-reward samples.
`AgentRolloutBatch.groups` records expected/usable/failed candidates and whether
each group is complete and accepted. Exports contain accepted candidates only.
Strict collection raises when a group does not meet its threshold. Evaluation
and pass@k continue to count episodes, including failed attempts.

## PPO provenance and policy age

Outcome normalization counts each linear logical rollout once. For a branch
tree, it counts each branch once and combines all fragments of that branch before
assigning sequence weight. Sequence importance ratios follow the same logical
sequence identity. Token reductions use active action weights. Overlong penalties
use the branch or linear candidate's active action count. Fragments of one branch
share its outcome reward; sibling branches may have different outcomes.

`batch_datums(prepared_batch)` returns an SDK-compatible list with an ordered
`provenance` sidecar. The sidecar identifies each physical row without placing
strings in numerical loss inputs. Critic marker rows each retain their source
trajectory. Actor and critic retain the latest submitted row mapping in
`datum_provenance`; callers can persist it alongside their remote request IDs.

`PPOTrainer.step()` rejects stale or future numeric policy versions relative to
its current update step before reference/critic requests. Standalone
`validate_policy_versions(..., current_version=..., max_staleness=...)` supports
bounded numeric age and exact opaque-label matching. One batch cannot mix
versions or mix known with unknown provenance. Entirely unversioned direct
trajectories remain supported; their age cannot be checked. Numeric versions
are scoped to the trainer that exports the snapshot, not global checkpoint IDs.
