# Agent execution and training data

[中文](agentrl_training_CN.md)

Use this guide when changing how episodes become PPO/GRPO training rows.
For installation and a complete training recipe, start with [README](README.md).
For OpenAI-style agent execution, see [OpenAI tools and sessions](agentrl_openai.md).

## Collect an episode batch

The data path is:

```text
task rows → fixed PyTRIO snapshot → candidate episodes → scorer
          → trajectory export → group validation → PPOTrainer.step
```

The trainer owns sampling snapshots and optimization. The workflow owns tools,
episode execution, scoring, and export. The following helper performs one update
using an existing trainer and workflow; the caller manages their lifecycle.

```python
async def collect_and_update(trainer, workflow, rows):
    snapshot = await trainer.create_sampling_snapshot()
    batch = await workflow.collect(
        rows,
        snapshot,
        n_samples=trainer.args.num_generations,
    )
    print(batch.metrics())
    return await trainer.step(batch.trajectories)
```

Keep the same snapshot throughout collection. Never reconstruct training tokens
by encoding displayed response text: export must preserve original sampled IDs
and their behavior log probabilities.

## Candidate identity

The collector creates ExecutionContext before calling the episode executor.
Use these fields to connect a task, its candidates, and its model requests:

| Field | Meaning |
| --- | --- |
| task_id | Application task identifier |
| group_id | Candidates for the same prompt, compared for relative rewards |
| rollout_id | One logical candidate, even if it exports several rows |
| episode_id | The candidate's episode execution |
| sample_index / group_size | Candidate position and requested candidate count |
| snapshot_id / policy_version | Fixed sampling policy used by the candidate |
| interaction_id / parent_interaction_id | One model request and its parent |

Read identity from the execution context rather than asking the model to supply
IDs as tool arguments:

```python
from triotune.agentrl import current_execution_context


def log_tool_context():
    context = current_execution_context()
    if context is not None:
        print(context.task_id, context.rollout_id, context.interaction_id)
```

Episode scorers use the root context. Interaction scorers receive the exact
request context, including during scoring retries:

```python
from triotune.agentrl import RewardAssignment, current_execution_context


class InteractionBonus:
    async def score(self, row, episode, interaction):
        assert current_execution_context() == interaction.context
        return RewardAssignment("interaction", 0.1, source="interaction_bonus")
```

Pass InteractionBonus() as the interaction_scorer argument to EpisodeRunner.
The runner fills in the generated-turn position for this assignment.

- Context is isolated between asyncio tasks and copied into asyncio.to_thread.
- The runner restores the caller's context on success, failure, and cancellation.
- Process workers must explicitly transfer and establish their context.
- An Interaction is recorded even when a request fails before generating tokens.
- resource_owner identifies ownership; it does not authorize closing shared clients.

Interaction records include token spans [start, end), tool results, status,
retries, and latency in seconds. Generation, tool, and reward events describe
execution steps; their timestamps are process-local monotonic time.

## Export and group policy

### Choose the training row layout

| export_mode | Exported rows | Use when |
| --- | --- | --- |
| concat | One full-episode row, or one row per compacted segment | Optimizing the complete episode |
| individual | One row per generated turn | Optimizing turns separately |
| branch | Paths declared by EpisodeBranch | Comparing continuations from a shared state |

```python
from triotune.agentrl import AgentConfig

config = AgentConfig(
    export_mode="individual",
    min_usable_group_size=2,
    shared_prefix_policy="deduplicate",
)


def build_workflow(snapshot, tools, scorer):
    return config.build_workflow(
        tokenizer=snapshot.tokenizer,
        tools=tools,
        scorer=scorer,
    )
```

For individual, earlier actions and tool observations stay in the row as masked
context. Each row receives the candidate outcome and its own process rewards;
GAE runs independently per row. Do not assume this objective equals concat when
discount, lambda, process rewards, or critic bootstrap cross turn boundaries.

### Decide whether partial candidate groups may train

With min_usable_group_size=None, every requested candidate must be usable. Setting
it to 2 accepts a group with at least two usable candidates, regardless of how
many fragments they export. Request at least that many candidates when collecting.

```python
async def inspect_groups(workflow, rows, snapshot):
    batch = await workflow.collect(
        rows, snapshot, n_samples=4, strict=False,
    )
    print(batch.groups)
    print("dropped_groups", batch.dropped_groups)
    print("accepted_rows", len(batch.trajectories))
    return batch
```

strict=False returns failed attempts for inspection. Only accepted groups enter
batch.trajectories; failures are never replaced with zero-reward samples.
strict=True raises if the group does not meet its acceptance threshold.
Evaluation and pass@k still include failed attempts.

### Implement a custom exporter

Implement export(episode) returning RolloutExport. Preserve these fields:

| Output | Required meaning |
| --- | --- |
| logical_rollout_id / group_id | Original candidate and group identity |
| fragment_map | Each fragment ID mapped to its interaction IDs |
| action_keys | For each active token: (interaction_id, original_output_offset) |
| loss_mask | Active model actions versus inserted or repeated context |
| tokens / logprobs | Original sampled token and its behavior probability |

The collector checks token identity, causal position, and conditioning prefix.
Context positions have no action key. Rows without active actions are omitted;
different initial prompts cannot share a PPO group. See the
[exporter implementation](../../triotune/agentrl/rollout/trajectory.py) for a
complete implementation.

## Structured rewards and direct advantages

### Choose a reward target

| target | position | Effect |
| --- | --- | --- |
| trajectory | None | Episode outcome, or an advantage broadcast to all active actions |
| interaction | Generated-turn index | Signal for that interaction's generation |
| turn | Generated-turn index | Signal for that turn |
| token | (turn index, output-token offset) | Signal for one original output token |

Indices start at zero and refer to original generated output, including returned
stop tokens. Turn and interaction reward scalars land on the last output token.
Interaction positions count generated turns, not failed requests without output.

This scorer combines answer correctness with a small tool-use cost per turn:

```python
from triotune.agentrl import EpisodeScore, RewardAssignment


class OutcomeWithToolCost:
    async def score(self, row, episode):
        success = (
            episode.status == "completed"
            and episode.final_text == row["answer"]
        )
        costs = tuple(
            RewardAssignment(
                "turn", -0.01 * len(turn.tool_results),
                position=index, source="tool_cost",
            )
            for index, turn in enumerate(episode.turns)
            if turn.tool_results and turn.generation.tokens
        )
        return EpisodeScore(float(success), success, assignments=costs)
```

Use legacy process_rewards vectors only when each vector has exactly one value
per output token in its turn. The exporter rejects unknown positions and
misaligned lengths.

### Submit rewards safely when a judge retries

RewardLedger groups values by rollout, source, mode, target, and position.
discount is applied before sum, mean, or last aggregation. Different sources
are added; each aggregation group must use one aggregation rule.

```python
from triotune.agentrl import RewardAssignment, RewardLedger

ledger = RewardLedger()
contributions = (
    RewardAssignment("trajectory", 2.0, aggregation="mean", source="judge"),
    RewardAssignment("trajectory", 4.0, aggregation="mean", source="judge"),
)
identity = {"logical_rollout_id": "candidate-0", "submission_id": "judge-0"}

assert ledger.submit(contributions, **identity) == contributions
assert ledger.submit(contributions, **identity) == ()  # Exact retry.
outcome, process, direct = ledger.aggregate()
assert outcome == 3.0
```

- Reusing a submission ID with changed content raises AgentContractError without
  changing the ledger. A new contribution needs a new ID.
- Without explicit IDs, same-source trajectory bundles are deduplicated within
  a rollout; process contributions remain additive. Use IDs for retryable work.
- Pass a candidate's accepted assignments to EpisodeScore.assignments. Do not
  attach another candidate's ledger contents to the score.

### Add an advantage without changing critic targets

Use mode="advantage" only for signals intended to modify policy advantages
directly. Add the following assignment to the scorer's EpisodeScore.assignments:

```python
from triotune.agentrl import EpisodeScore, RewardAssignment

score = EpisodeScore(
    reward=1.0,
    success=True,
    assignments=(
        RewardAssignment(
            "trajectory", 0.5, mode="advantage", source="prm",
        ),
    ),
)
```

| Signal | Reward normalization / GAE | Critic returns | Final advantage normalization |
| --- | --- | --- | --- |
| Ordinary outcome/process reward | Follows the configured reward pipeline | Contributes | Applies to the resulting advantages |
| Direct advantage | Bypasses reward normalization and GAE | Unchanged | Still applies |

A trajectory advantage reaches every active original model-output token.
Tool observations and repeated masked context get zero direct advantage.
Fragment deduplication or branch weights still control shared actions.
RewardLedger.aggregate returns outcome, process, and direct; a trajectory
advantage appears as direct[None] = (value,).

### Add a batch-level state advantage

Algorithms such as GiGPO need to compare steps across complete trajectories,
which a per-trajectory advantage estimator cannot do. Keep that algorithm in an
AgentRL plugin: capture a trusted environment observation before every action,
then return only the additional token-aligned advantage for the collected batch.

```python
from triotune import PPOConfig
from triotune.agentrl import (
    AgentTrainingLoop,
    GiGPOAdvantageProvider,
    GiGPOConfig,
    current_execution_context,
)


class EnvironmentAnchors:
    async def observe(self, episode, interaction):
        context = current_execution_context()
        return environment.raw_observation(context.rollout_id)


workflow = config.build_workflow(
    tokenizer=tokenizer,
    tools=tools,
    scorer=scorer,
    anchor_observer=EnvironmentAnchors(),
)
loop = AgentTrainingLoop(
    trainer,
    workflow,
    config=config,
    batch_advantage_provider=GiGPOAdvantageProvider(
        GiGPOConfig(
            step_advantage_weight=1.0,
            discount=0.95,
            mode="mean_norm",
            episode_statistic_weighting="trajectory",
        )
    ),
)
```

Build the trainer from the GRPO preset, but disable its separate reward and
final advantage normalization because the GiGPO provider computes the complete
joint advantage:

```python
ppo = PPOConfig.for_algorithm(
    "grpo",
    reward_norm=None,
    adv_norm=None,
    use_critic=False,
    kl_ctl=0,
    reference_kl_coef=0.01,
)
```

The anchor observer runs after the trusted interaction context is installed and
before policy sampling. Returned mutable observations are deep-copied into
optional episode export metadata keyed by interaction ID, then exported as
Trajectory.anchor_observations aligned with interaction_ids. Episodes that
do not install an observer retain the historical Interaction and JSONL schema.
Anchors are provenance only: prompts, PyTRIO datums, and loss inputs never
contain them. When JSONL episode logging is enabled, the provider must return a
JSON-serializable observation (or an application-owned serialized view of a
larger state).

The batch provider runs once after a complete update batch is accepted and
immediately before PPOTrainer.step(). It joins physical fragments by logical
trajectory, aggregates process rewards by active turn, adds the outcome once to
the final turn, calculates discounted return-to-go, and forms
A^E + omega * A^S. Exact matching accepts structural observations; optional
similarity matching supports text anchors. State singleton groups contribute
zero. episode_statistic_weighting="trajectory" counts each logical candidate
once, while "step" reproduces verl-agent's step-row episode statistics.

The provider returns a generic replacement-advantage contract. Existing direct
advantages are still added, configured adv_norm would still run afterward,
and the clipped loss and PyTRIO datum path remain unchanged. The provider
rejects critics, KL-as-reward, late reward transforms, or duplicate
normalization settings that would otherwise be silently bypassed. An
independent reference KL objective may be enabled with reference_kl_coef; it
is evaluated in the custom actor loss and is not part of the frozen GiGPO
advantage provider.

### Read reward fields in episode logs

| Logged field | Meaning before PPO processing |
| --- | --- |
| outcome_reward | EpisodeScore.reward plus structured trajectory rewards; attached to the last generated interaction |
| process_rewards | Aligned legacy and structured process rewards after aggregation and discount |
| token_advantages | Separate policy-only advantage assignments |
| reward | outcome_reward + sum(process_rewards); already includes both components |

RewardEvent records describe these values; do not add them to rewards again.
They precede shaping, normalization, GAE, KL penalties, and loss weighting.
Branch outcomes are recorded on branches and replace the episode outcome for
branch export.

EpisodeRunner.submit_reward commits only after scoring and alignment succeed.
An exact retry reuses the committed result without new events; changed content
raises AgentContractError, including after export. Use a separate record when
rescoring an episode.

## PPO reward plugin interfaces

Use PPOAlgorithmPlan when reward processing must happen in the PPO preparation
stage. Plugins run locally before PyTRIO Datum construction.

| Hook | Purpose |
| --- | --- |
| reward_shaper.shape(trajectory, reward) | Transform a candidate's outcome reward |
| reward_shaper.shape_process(trajectory, rewards) | Transform its aligned process vector |
| process_shaper | Explicit override for process shaping |
| normalizer.normalize(rows, group_ids, masks=None) | Normalize outcome rows across comparison groups |

This small example verifies that outcome and process shaping remain separate:

```python
from triotune.trainer.ppo import (
    PPOAlgorithmPlan, PPOConfig, Trajectory, prepare_advantages,
)


class Shaper:
    def shape(self, trajectory, reward):
        return reward

    def shape_process(self, trajectory, rewards):
        return [3.0 * value for value in rewards]


class Normalizer:
    def normalize(self, rows, group_ids, masks=None):
        return [[2.0 * value for value in row] for row in rows]


plan = PPOAlgorithmPlan(reward_shaper=Shaper(), normalizer=Normalizer())
row = Trajectory([1], [2], [-1.0], 1.0, "group", token_rewards=[1.0])
batch = prepare_advantages(
    [row], PPOConfig(use_critic=False, reward_norm=None, adv_norm=None), plan,
)
assert batch.returns == [[5.0]]  # Process reward 3 + normalized outcome 2.
```

To use the plan in training, pass plan=plan when constructing PPOTrainer.

- Normalizer input counts logical candidates, not exported fragments. During
  outcome preparation masks is None; it is not the token loss mask.
- A normalizer may declare config to receive NormConfig. Legacy two-argument
  and named-config hooks are supported; a **kwargs-only hook gets masks without
  an implicit config argument.
- Explicit process_shaper replaces reward_shaper.shape_process; they are not
  chained. Legacy process shape/callable forms remain supported.
- Process output must be finite and completion-aligned. Hook errors propagate;
  an internal TypeError does not trigger a second plugin invocation.
- Plugin objects stay local; native and custom PyTRIO updates receive the same
  prepared numerical targets.

## Branch trees

Use branches when an executor has actually sampled alternative continuations
from the same state. Selecting export_mode="branch" does not generate them.

```text
shared request
├── tool-a request → final-a request → reward 1.0
└── tool-b request → final-b request → reward 0.0
```

Inside your executor, attach branch records after collecting the interactions.
The IDs passed to this helper must already exist in episode.interactions and
form parent-linked suffixes from fork_id:

```python
from triotune.agentrl import EpisodeBranch


def attach_branches(episode, fork_id, left_ids, right_ids):
    episode.branches = [
        EpisodeBranch(
            branch_id="tool-a",
            parent_interaction_id=fork_id,
            interaction_ids=tuple(left_ids),
            group_id="tool-choice",
            reward=1.0,
            shared_prefix_weight=0.5,
        ),
        EpisodeBranch(
            branch_id="tool-b",
            parent_interaction_id=fork_id,
            interaction_ids=tuple(right_ids),
            group_id="tool-choice",
            reward=0.0,
            shared_prefix_weight=0.5,
        ),
    ]
```

Set export_mode="branch" and choose shared_prefix_policy on AgentConfig:

| Policy | Shared action contribution |
| --- | --- |
| deduplicate | First occurrence is active; later copies are masked |
| repeat | Each path counts its own copy |
| weighted | One occurrence per branch, normalized by declared prefix weights |

Branch-local actions keep unit weight. Actor and critic use the same weights;
weighted rows use the custom PyTRIO loss boundary because the native objective
has no token-weight field. Conflicting tokens, logprobs, or prefixes are always
rejected, including with repeat.

- Branch reward is the full branch outcome, replacing the episode outcome.
- Process rewards apply only to interactions on that path. Trajectory advantages
  apply to all completed paths without changing branch outcomes or critic returns.
- Siblings share a rollout-local group_id; separate rollouts cannot merge their
  branch reward baselines.
- Only complete sibling groups train. Mark unfinished branches completed=False
  with reward=None; never supply zero as a substitute for an unknown reward.
- Inspect batch.branch_groups for branch completeness and batch.groups for
  candidate completeness. RolloutExport.branch_map retains unfinished siblings.
- PPO counts each branch once for outcome normalization and sequence reductions,
  even if context compaction splits it into multiple fragments.

## Context compaction and prefix reuse

Enable compaction before building the workflow when old tool exchanges make
requests too long:

```python
from triotune.agentrl import AgentConfig

config = AgentConfig(
    context_compaction=True,
    compaction_keep_recent_interactions=2,
    compaction_summary_chars=2048,
    prefix_cache_size=256,
)
```

| Setting or behavior | What it does |
| --- | --- |
| compaction_keep_recent_interactions | Keeps recent interaction blocks literal |
| compaction_summary_chars | Bounds the default summary length |
| Initial instructions and tool-error interactions | Always retained |
| Custom preservation hook | Selects other interactions to retain |
| Custom summarizer | Receives omitted interactions and previous summary messages; returns text or messages, synchronously or asynchronously |
| prefix_cache_size | Bounds cached template tokenizations; never caches generations or logprobs |

A custom ContextCompactor can also be supplied to EpisodeRunner.

Compaction changes the context visible to the model, so re-evaluate success
rate after enabling it. Original sampled tokens are not re-encoded: each context
reset starts a new segment with its exact prompt IDs. Segments have separate
prompt groups but retain the original candidate reward group; process signals
stay with their original actions. This also applies within branch paths.

The local prefix_groups sidecar on Datum batches describes shared input
prefixes. Current PyTRIO accepts ordinary Datum lists and exposes no portable
tree-attention option; caching and branch-action weighting do not imply a
shared model forward pass.

## Asynchronous rollout scheduling

Use AgentTrainingLoop for bounded overlap between fixed-snapshot collection and
serialized PPO updates:

```python
from triotune.agentrl import AgentConfig, AgentTrainingLoop


async def train_with_overlap(trainer, tools, scorer):
    config = AgentConfig(
        async_rollouts=True,
        rollout_prefetch_capacity=1,
        max_policy_staleness=1,
        max_concurrency=4,
    )
    snapshot = await trainer.create_sampling_snapshot(stop=config.stop)
    workflow = config.build_workflow(
        tokenizer=snapshot.tokenizer, tools=tools, scorer=scorer,
    )
    return await AgentTrainingLoop(trainer, workflow, config=config).train()
```

| Setting | Meaning |
| --- | --- |
| rollout_prefetch_capacity | Number of batches prefetched ahead of updates |
| max_policy_staleness | Maximum allowed numeric policy age; must cover prefetch capacity |
| max_concurrency | Workflow-wide candidate limit, shared across overlapping collection calls |

Each rollout keeps one snapshot. Batches are consumed in launch order; updates
on one trainer remain serialized. Future or over-age policies are rejected
before reference scoring, critic work, or optimization.

For incremental collection, collect_iter yields completed prompt-group batches.
max_pending_groups separately bounds scheduled group tasks. For recovery, resume
only rows that were never started and reuse the original snapshot:

```python
async def resume_unstarted(workflow, partial_batch, unstarted_rows, snapshot):
    return await workflow.resume(
        partial_batch,
        unstarted_rows,
        snapshot,
        n_samples=4,
    )
```

The caller tracks which rows were started; resume does not find or skip them
automatically. A different snapshot is rejected. In-flight tool work is cancelled
and drained instead of automatically replayed, since side effects may not be safe
to repeat. merge_batches combines completed batches while rejecting duplicate IDs.

## PPO provenance and policy age

Check policy versions before passing a custom batch to the optimizer:

```python
from triotune.trainer.ppo.trajectory import validate_policy_versions


async def update_checked(trainer, batch):
    validate_policy_versions(
        batch.trajectories,
        current_version=trainer.state.global_step,
        max_staleness=1,
    )
    return await trainer.step(batch.trajectories, max_staleness=1)
```

PPOTrainer.step already performs this validation. Use the standalone check when
validating data before handing it to the trainer.

| Version data | Accepted condition |
| --- | --- |
| Integers | 0 ≤ current_version − sampled_version ≤ max_staleness |
| Opaque labels | Exact equality with the current label |
| One batch | One version throughout; no known/unknown version mixture |
| Entirely unversioned direct trajectories | Supported for legacy callers; policy age cannot be checked |

Numeric versions belong to the trainer that exported them; a matching step
number from an unrelated run does not identify the same policy.

For low-level request tracking, batch_datums(prepared_batch) returns an
SDK-compatible Datum list with an ordered provenance sidecar. Actor and critic
retain datum_provenance for their latest submitted rows; persist it alongside
remote request IDs if needed. Provenance strings are not numerical loss inputs.
Critic rows retain their source trajectory, and token reductions use effective
action weights. Overlong penalties count active actions in each logical episode
or branch, rather than counting repeated fragment context.
