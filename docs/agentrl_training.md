# Agent execution and training data

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
    export_mode="individual",
    min_usable_group_size=2,
    shared_prefix_policy="deduplicate",
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
`repeat` explicitly counts each occurrence. Conflicting evidence is always an
error. Fragments with no active actions are omitted. Differing initial prompts
cannot share a PPO group. Interaction lineage can represent branches, but
custom branch exporters must still satisfy these training constraints.

By default all candidates must be usable. An explicit `min_usable_group_size`
allows partial groups; the threshold counts candidates, not fragments. Failed
attempts remain in episode diagnostics and never become zero-reward samples.
`AgentRolloutBatch.groups` records expected/usable/failed candidates and whether
each group is complete and accepted. Exports contain accepted candidates only.
Strict collection raises when a group does not meet its threshold. Evaluation
and pass@k continue to count episodes, including failed attempts.

## PPO provenance and policy age

Outcome normalization counts each logical rollout once, and sequence reductions
combine all its fragments before assigning candidate weight. Sequence importance
ratios also pool retained action tokens across all fragments of a candidate.
Token reductions continue to weight active action tokens. Overlong penalties
use the candidate's total active action count. Fragments of one candidate share
the same outcome reward.

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
