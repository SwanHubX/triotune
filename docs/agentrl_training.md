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
serialize and establish it explicitly. Episode scorers and completed-episode
loggers use the root context. Interaction scorers use the exact child
`interaction.context`, including its interaction ID and parent ID. That scope
surrounds the entire scoring retry/deadline operation, so PRM calls, nested
logging, worker-thread fallbacks, and retry events retain the same identity.
Both EpisodeRunner and OpenAIWorkflow restore the enclosing context after each
interaction, including on exceptions, timeout, and cancellation. SDK terminal
scoring and acceptance hooks use the session root context before the workflow
restores its caller's scope; shared scorers and clients carry no mutable identity.

```python
from triotune.agentrl import RewardAssignment, current_execution_context


async def score_interaction(row, episode, interaction):
    context = current_execution_context()
    assert context == interaction.context
    return RewardAssignment("interaction", 0.1)
```

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

Interaction and RewardEvent reward fields share the exporter's raw projection:

- `outcome_reward` includes `EpisodeScore.reward` and structured trajectory
  rewards, and is attached only to the last generated interaction.
- `process_rewards` merges legacy process vectors and structured interaction,
  turn, and token rewards after their declared aggregation and discount. Turn
  and interaction scalars are placed on that generation's last output token.
- `token_advantages` records policy-only assignments separately, including
  trajectory advantages broadcast to original model-output tokens.
- `reward` is the raw local total: `outcome_reward + sum(process_rewards)`.
  It already includes both components; do not add the component views to it.

These values precede PPO shaping, candidate normalization, GAE, KL penalties,
and loss weighting. They are not expected to equal normalized advantages or
critic returns. Fragment export retains the same per-action source values while
masking repeated context; branch-local outcomes remain in the branch records
and replace the episode outcome according to the branch export contract.

Reward views are committed only after scoring and token alignment validation
succeed. Exact retries of `EpisodeRunner.submit_reward(episode, score)` reuse
the committed result without appending events, including after collector export;
different content for an already committed episode raises `AgentContractError`.
Rescoring requires a separate record, so an existing trace cannot disagree with
already exported training rows. Optional diagnostic fields were appended to
Interaction and RewardEvent; existing positional Interaction arguments retain
their order. SDK provider traces continue to use their separate session format.

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

## Structured rewards and direct advantages

`RewardLedger.submit()` accepts a whole contribution payload atomically. Values
sharing a rollout, source, mode, target, and position are reduced using their
declared `sum`, `mean`, or `last` aggregation. Each value is multiplied by its
`discount` before reduction; contributions from different sources are added.
One aggregation group must use one aggregation rule. Different rollouts keep
independent aggregation populations, including when a runner reuses a ledger.

Use `submission_id` to distinguish retries from new contributions. Within one
rollout, repeating an ID and the same ordered payload returns no new assignments;
reusing it with different content raises `AgentContractError` without changing
the ledger. Different IDs may add values to the same aggregation group.

```python
from triotune.agentrl import EpisodeScore, RewardAssignment, RewardLedger

ledger = RewardLedger()
contributions = (
    RewardAssignment("trajectory", 2.0, aggregation="mean"),
    RewardAssignment("trajectory", 4.0, aggregation="mean"),
)
options = {"logical_rollout_id": "candidate-0", "submission_id": "judge-0"}
assert ledger.submit(contributions, **options) == contributions
assert ledger.submit(contributions, **options) == ()
assert ledger.aggregate()[0] == 3.0

score = EpisodeScore(
    0.0,
    False,
    assignments=ledger.assignments + (
        RewardAssignment("trajectory", 0.5, mode="advantage", source="prm"),
    ),
)
```

Without an explicit submission ID, legacy rollout/source/mode deduplication
still applies to trajectory contributions. It compares the entire same-source
bundle, preserving all values in its first submission. An exact bundle retry is
ignored; a changed bundle raises instead of silently dropping the new value.
Process contributions without an ID remain additive. Supply explicit IDs for
retry-safe process submissions or multiple updates from the same source.

A trajectory assignment with `mode="advantage"` broadcasts its reduced value
to every active original model-output token, including returned stop tokens.
It combines additively with token/turn/interaction advantages, does not enter
reward normalization or GAE, and does not change critic returns or bootstrap.
The configured final advantage normalization still applies. Tool observations
and repeated context positions receive zero direct advantage. Individual rows
and compacted segments retain the value on their own active actions; existing
shared-action deduplication or weights govern branch prefixes.

`aggregate_assignments()` and `RewardLedger.aggregate()` retain their three-part
outcome/process/direct result. A trajectory advantage has no turn position and
is exposed as `direct[None] = (value,)`. The exporter consumes the same reduction
with exact target positions to construct completion-aligned token advantages.

## PPO reward plugin interfaces

`PPOAlgorithmPlan` runs reward plugins locally before building PyTRIO datums.
The public normalizer signature is `normalize(rows, group_ids, masks=None)`.
These are candidate-level outcome rows, with one normalization observation per
logical rollout even when it exports several fragments. During outcome
preparation `masks` is `None`, meaning all outcome entries are valid; it is not
the completion-token loss mask. A normalizer may explicitly declare `config`
to receive `NormConfig`. Legacy two-argument and named-`config` hooks remain
supported; named `config` and `masks` can be positional-only or keyword-only.
Adapters accepting only `**kwargs` receive `masks`, without an implicit config.

```python
from triotune.trainer.ppo import PPOAlgorithmPlan, PPOConfig, Trajectory, prepare_advantages


class Shaper:
    def shape(self, trajectory, reward):
        return reward

    def shape_process(self, trajectory, rewards):
        return [3.0 * reward for reward in rewards]


class Normalizer:
    def normalize(self, rows, group_ids, masks=None):
        return [[2.0 * reward for reward in row] for row in rows]


plan = PPOAlgorithmPlan(reward_shaper=Shaper(), normalizer=Normalizer())
row = Trajectory([1], [2], [-1.0], 1.0, "group", token_rewards=[1.0])
batch = prepare_advantages(
    [row], PPOConfig(use_critic=False, reward_norm=None, adv_norm=None), plan
)
assert batch.returns == [[5.0]]  # Process reward 3 plus normalized outcome 2.
```

An explicit `process_shaper` overrides `reward_shaper.shape_process`; the two
are not chained. Process shapers may expose `shape_process(trajectory, rewards)`
or the legacy `shape(trajectory, rewards)`/callable forms. Legacy single-value
shaping callables remain supported. Outcome-only shapers are not reused for
process vectors. Process output must remain finite and completion-aligned;
direct advantages remain separate from GAE and critic returns. Signature
selection happens before invoking a hook, so an internal `TypeError` propagates
without repeating the plugin. The same prepared numeric targets feed both
custom and native PyTRIO PPO updates; plugin objects are not sent to the SDK.

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
episode outcome. Trajectory-level episode outcome rewards are therefore not
copied onto every branch. A trajectory-level direct advantage applies to every
completed path without changing its branch-local outcome or critic return.
Process-reward and position-specific direct-advantage assignments are projected
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

## Context compaction and prefix reuse

Context compaction is opt-in because a summary changes the policy's observable
state. It can be enabled through `AgentConfig` or supplied as a custom
`ContextCompactor` to `EpisodeRunner`:

```python
config = AgentConfig(
    context_compaction=True,
    compaction_keep_recent_interactions=2,
    compaction_summary_chars=2048,
    prefix_cache_size=256,
)
```

`ContextCompactor` keeps initial instructions, recent interaction blocks,
interactions selected by a preservation hook, and tool-error interactions. Old
tool exchanges are replaced by deterministic summaries by default. A custom
synchronous or asynchronous summarizer receives omitted interactions and any
previous summary messages. It may return text or chat message mappings.

Previously sampled action IDs are never decoded and encoded into a new training
prefix. A compacted request starts a new causal segment and records its exact
prompt IDs in `ContextCompaction`. Export emits one physical trajectory per
segment, with separate prompt group IDs and the original candidate reward group.
Outcome normalization and sequence loss still count the logical rollout once;
process rewards and direct advantages are projected to their original segment.
This segmentation also applies inside P3 branch paths.

`PrefixCache` is a bounded, thread-safe LRU for exact chat-template
tokenizations. It does not cache generations or log probabilities. PPO datum
batches expose a local `prefix_groups` sidecar describing maximal shared input
prefixes. Current PyTRIO accepts standard datum lists and has no portable tree
attention option, so normal execution safely ignores the sidecar while compatible
adapters can use it for prefix scheduling. P3 action weights continue to prevent
shared branch actions from receiving duplicate gradients independently of model
forward optimization.

## Asynchronous rollout scheduling

```python
config = AgentConfig(
    async_rollouts=True,
    rollout_prefetch_capacity=1,
    max_policy_staleness=1,
)
```

The asynchronous training loop exports a fixed `SamplingSnapshot`, starts the
next collection from that snapshot, and overlaps it with the current PPO update.
A logical rollout never changes snapshot. Completed batches are consumed in
launch order, and `PPOTrainer.step(..., max_staleness=...)` rejects future or
over-age policy versions before reference scoring, critic work, or optimization.
Prefetch capacity cannot exceed the configured numeric staleness allowance.

The collector uses one workflow-wide semaphore, so concurrent collect calls do
not multiply sampling, tool, or scorer capacity. `collect_iter()` yields complete
prompt groups as they finish and bounds scheduled group tasks separately.
Completed groups are safe persistence and recovery boundaries.
`workflow.resume(partial_batch, remaining_rows, original_snapshot)` collects only
previously unstarted rows and merges identities; a different snapshot is
rejected. In-flight tool interactions are cancelled and drained rather than
automatically replayed, because a tool may not be retry-safe.

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
