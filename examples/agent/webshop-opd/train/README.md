# Multi-turn OPD and TCOD-F2B

The provided scripts use the shared repository-level `triotune/` package.
The TCOD-F2B curriculum lives in the application training entry point and uses
the shared Agent OPD interfaces.

## Data flow

For every optimizer step:

1. Export one immutable student policy snapshot.
2. Run complete WebShop episodes through `AgentRolloutWorkflow`.
3. Convert each generated turn into one `OPDRollout`:
   - the exact model-visible prompt is masked context;
   - the exact sampled response IDs are the only active tokens;
   - original behavior log-probabilities remain token aligned.
4. Score each response under the fixed teacher using the same prompt.
5. Build the existing sampled reverse-KL target and perform one OPD update.
6. Discard the student snapshot before collecting the next batch.

Tool observations and earlier actions condition later decisions but never
receive OPD gradients. WebShop rewards are retained for evaluation metrics and
do not affect the vanilla OPD objective.

## Local verification

The test suite uses deterministic fake PyTRIO clients plus a real multi-turn
AgentRL tool loop. It makes no model, network, Java, or WebShop data requests:

```bash
bash examples/agent/webshop-opd/train/scripts/test-agent-opd.sh
```

It covers exact turn/context alignment, stale-policy rejection, teacher failure
boundaries, optimizer lifecycle, snapshot refresh, WebShop session cleanup,
evaluation isolation, and a two-step train/evaluate/save lifecycle.

Validate the real experiment configuration without opening model services:

```bash
bash examples/agent/webshop-opd/train/scripts/run-vanilla-opd.sh --check-config
```

## Real training

After reviewing the configuration, the configured 200-step run can be started with:

```bash
bash examples/agent/webshop-opd/train/scripts/run-vanilla-opd.sh
```

For a later remote smoke test, override the target without editing YAML:

```bash
bash examples/agent/webshop-opd/train/scripts/run-vanilla-opd.sh --steps 2
```

The default configuration uses Qwen3.5-4B as the student, Qwen3.8-27B as the
fixed teacher, 16 WebShop tasks per update, one training trajectory per task, 15 turns
per trajectory, learning rate 1e-6, and task evaluation every 20 updates.
Training samples at temperature 1; task evaluation samples at temperature 0.4.
Every evaluation task receives one attempt and reports Pass@1 only. No TCOD
trajectory curriculum is applied.

The paper's `batch_size=16` maps directly to 16 task prompts per Triotune
update. Its separate `train_batch_size=64` and `repeat_times=1` describe the
distributed training engine's internal batching/reuse. PyTRIO owns that remote
compute boundary, so this client configuration must not imitate 64 by sampling
four trajectories per task; that would change the data distribution. Likewise,
BF16 precision is selected by the PyTRIO service and is not a local trainer
field. This synchronous baseline always consumes freshly collected rollouts,
so effective staleness is zero; the paper's maximum staleness of two applies
only to its asynchronous scheduler.

The full 200-task validation split is evaluated before training and every 20
updates. Each pass runs 200 complete episodes, producing 2,200 validation
episodes across 11 validation passes in a 200-step run.
For a connectivity
smoke test, use `--steps 2`; periodic step-20 evaluation will not run, while the
configured initial and final evaluations still verify end-to-end task behavior.

Output is written below `output/webshop-opd/train/vanilla-opd/`. Episode JSONL
contains complete token-level evidence for training trajectories. Evaluation
uses `eval-episodes.jsonl`, which stores only task outcomes, parsed actions,
token totals, and errors; prompt IDs, generated token arrays, and interaction
events are omitted. WebShop evaluation stops as soon as a purchase completes,
and one global task queue keeps inference slots occupied across the full split.
The run report records checkpoints, Pass@1, other aggregate metrics, and update
provenance. Complete checkpoint state also retains the next AgentRL seed and
prior update history.

### OPD with the step-400 GRPO teacher

`configs/vanilla-opd-grpo-teacher.yaml` starts a fresh `Qwen/Qwen3.5-4B`
student and uses a fixed `Qwen/Qwen3.5-4B` teacher loaded from
`trio://run_x4uc1nm5nyc8/sampler_weights/ppo-sampler-step-00000400`
(Sampler checkpoint ID: `ckpt_7c56f2uuof26`). The teacher scoring client needs
the sampler URI in `model_path`; the Train checkpoint `ckpt_wth5sxffdght`
is for restoring training state. A bare checkpoint ID is not a valid
`model_path`. It runs 200 OPD updates
with the same batch size, learning rate, rollout limits, and evaluation settings
as the vanilla OPD baseline. Reports, metrics, and checkpoints use a separate
output directory whose name includes `teacher-step400`.

Validate the configuration without opening model services:

```bash
bash examples/agent/webshop-opd/train/scripts/run-vanilla-opd-grpo-teacher.sh --check-config
```

Start the 200-step run:

```bash
bash examples/agent/webshop-opd/train/scripts/run-vanilla-opd-grpo-teacher.sh
```

## TCOD-F2B with the GRPO teacher

[`tcod_f2b.py`](tcod_f2b.py) adds Forward-to-Backward temporal curriculum to the
same OPD loss. Every training task starts from its initial WebShop state. The
student generates at most `k` model turns, and the fixed teacher scores only
those original student responses. Collection stops at `k`; later turns are
never generated. Prefixes that have not purchased anything still train with
dense OPD supervision, even when their task reward is zero. No teacher
demonstrations, environment replay, reward objective, or teacher navigation
are required.

The [default configuration](configs/tcod-f2b-grpo-teacher.yaml) preserves the
GRPO-teacher baseline's fresh Qwen3.5-4B student, step-400 teacher sampler URI,
200 optimizer updates, 16 tasks per update, one trajectory per task, learning
rate 1e-6, and full 200-task Pass@1 evaluation every 20 updates. Its output,
checkpoint, and reporting paths are separate from vanilla OPD.

The alternative
[`tcod-f2b-qwen3.8-27b-teacher.yaml`](configs/tcod-f2b-qwen3.8-27b-teacher.yaml)
uses the base `Qwen/Qwen3.8-27B` as the fixed teacher and a fresh
`Qwen/Qwen3.5-4B` student. It keeps the same 200-update, batch-16, learning-rate,
curriculum, rollout, and evaluation settings, with independent output and
checkpoint paths. Validate or start it with:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-f2b-qwen3.8-27b-teacher.sh --check-config
bash examples/agent/webshop-opd/train/scripts/run-tcod-f2b-qwen3.8-27b-teacher.sh
```

The curriculum follows the zero-based batch-index convention in the
[reference WebShop implementation](https://github.com/kokolerk/TCOD/blob/main/trinity/common/workflows/envs/TCOD/webshop/TCOD_f2b_workflow.py):

```yaml
curriculum:
  k_start: 1
  growth_interval: 4
```

```text
completed_steps = trainer.state.global_step before the next update
k = min(k_start + completed_steps // growth_interval, agent.limits.max_turns)
```

Updates 1–4 use `k=1`, updates 5–8 use `k=2`, updates 53–56 use `k=14`,
and updates 57–200 use `k=15`. `growth_interval` corresponds to the paper's
growth-rate parameter eta and the reference code's `checkpoint_steps`.
`--steps` changes the total update budget without rescaling the curriculum.

The budget counts model generations, including the final JSON reply required
by the current training protocol. Every tool-calling generation contains one
WebShop action. A purchase on the last allowed training turn can therefore
have a positive task score without a protocol-complete success; it still
receives OPD supervision. Evaluation retains the baseline behavior of stopping
immediately at purchase. This counting convention is preserved for comparison
with the existing baseline rather than changing the environment protocol.

Training temporarily applies `k` to both the WebShop wrapper and its actual
runner. Full limits are restored after each update, including on errors or
cancellation. Initial, periodic, and final evaluations always retain the full
15-turn budget, the baseline temperature 0.4, and one attempt per task. Early
short-prefix training success rates are not comparable to full-horizon Pass@1;
use the evaluation metrics for that comparison.

If strict collection rejects a batch solely because student sampling exhausted
its timeout retries, F2B re-exports a fresh student snapshot and retries the
batch up to `agent.retry.max_attempts`. The retry reuses the same seed range and
occurs before teacher scoring or optimization. Tool, scoring, contract, and
model-action failures are not replayed. `curriculum/collection_attempts` records
whether an update needed this recovery path.

The same transaction-level recovery applies when every teacher-scoring failure
in the task group is a PyTRIO Actor HTTP 503. Teacher targets are fully
materialized before optimization, so this service-unavailable response cannot
leave a partial update. Other teacher errors remain fail-fast.

Validate without opening model services or WebShop:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-f2b-grpo-teacher.sh --check-config
```

Start the configured run, or override the update budget for a remote smoke test:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-f2b-grpo-teacher.sh
bash examples/agent/webshop-opd/train/scripts/run-tcod-f2b-grpo-teacher.sh --steps 2
```

SwanLab metrics include `curriculum/horizon`, `curriculum/max_turns`, and
`curriculum/completed_steps`; each update in `report.json` and saved manifests
also records its curriculum horizon. Complete checkpoints retain the inherited
data cursor, seed cursor, optimizer state, and update history. The schedule is
part of the agent checkpoint identity: changing it or the full interaction
budget during resume is rejected. The next `k` is derived from restored
`global_step`, so the curriculum never restarts at one after recovery.

To resume, copy the YAML and set `trainer.checkpoint.resume_from` to the local
complete checkpoint manifest emitted by this F2B run. Keep the curriculum,
model, optimizer, and data settings the same; paths are resolved relative to
the YAML file. A teacher sampler URI or a bare remote checkpoint ID cannot
replace this student training manifest. `--stop-after-step 20` can stop cleanly
at an update boundary while retaining the original total training plan.

The launcher also accepts a shell-relative checkpoint path, so no YAML edit is
required:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-f2b-qwen3.8-27b-teacher.sh \
  --resume-from output/webshop-opd/train/<run-name>/state/opd-step-00000020.json
```

Local verification uses the same offline suite:

```bash
bash examples/agent/webshop-opd/train/scripts/test-agent-opd.sh
```

F2B tests cover real short-prefix collection and token-aligned teacher scoring,
zero-reward prefix updates, independent full-horizon evaluation, failure and
cancellation cleanup, curriculum configuration rejection on resume, and a
pause/resume lifecycle that preserves the next horizon and sampling seeds.
These are deterministic fake-service tests; they do not validate remote model
quality or start a real training job.

## TCOD-B2F with the GRPO teacher

B2F uses the same fixed step-400 GRPO-trained `Qwen/Qwen3.5-4B` teacher
as the F2B baseline. Its sampler URI is
`trio://run_x4uc1nm5nyc8/sampler_weights/ppo-sampler-step-00000400`.
The student starts from the original `Qwen/Qwen3.5-4B` base model, with no
student checkpoint loaded. The teacher is used both for expert generation
and for scoring the student's original action tokens.

[`configs/tcod-b2f-grpo-teacher.yaml`](configs/tcod-b2f-grpo-teacher.yaml)
keeps the F2B model, LoRA rank, optimizer, loss, evaluation, and rollout settings:
200 updates, 16 tasks per update, one student trajectory per task, learning
rate `1e-6`, 15 turns, 512 tokens per response, full 200-task Pass@1 evaluation
before training and every 20 updates, checkpoints every 20 updates, and seed 42.
Training uses temperature 1 and evaluation uses temperature 0.4. The
curriculum is `k=min(1 + completed_steps // 4, 15)`; evaluation always starts
from reset with the full horizon.

B2F additionally needs successful teacher trajectories.
[`collect_tcod_b2f_experts.py`](collect_tcod_b2f_experts.py) samples up to ten
independent attempts per planned training goal, using the fixed teacher sampler
without creating a training client. It retains the shortest strict environment
success with a replayable sequence of successful `webshop` tool calls ending
in exactly one purchase. Partial purchases, model claims, failed tool actions,
and infrastructure failures are not expert successes.

The main launcher collects missing experts and then starts student training:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-b2f-grpo-teacher.sh --check-config
bash examples/agent/webshop-opd/train/scripts/run-tcod-b2f-grpo-teacher.sh
```

For a bounded collection check, process at most 16 previously uncached goals
and exit before student training:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-b2f-grpo-teacher.sh --expert-max-tasks 16
```

Independent collection is also available:

```bash
bash examples/agent/webshop-opd/train/scripts/collect-tcod-b2f-grpo-teacher.sh --max-tasks 16
```

The default cache is
`data/webshop-opd/tcod-b2f/grpo-teacher-step400-experts.jsonl`.
Its `.meta.json` sidecar binds the teacher URI, actual asset hashes, ordered
goals, environment seed, prompt, generation settings, and full agent limits.
Changing these settings requires a new cache path. A cache without matching
metadata is rejected before opening a model client. Concurrent collectors
using the same path are rejected by a file lock.

Each completed successful goal atomically replaces the complete cache, so an
interruption preserves previously acknowledged successes. Re-running collection
skips cached successes; unsuccessful goals can be retried. Per-goal attempt
seeds do not shift when collection is resumed or the pending batch changes.
Keep the cache frozen during student training. Both cache and metadata live
under ignored `data/`; trajectories and model weights are not packaged or
committed.

Training filters the shuffled train split to distinct cached successes and
cycles that set deterministically to fill `max_steps * batch_size`. At least
one full batch of distinct successful goals is required. Repeated epochs of
one goal cannot satisfy the default minimum of 16 goals.

[`tcod_b2f.py`](tcod_b2f.py) replays the first `max(L-k, 0)` actions of a
successful expert path of length `L`, then hands the resulting WebShop state
to the student. The purchase action is never replayed. The student sees the
current observation and the two most recent teacher transitions. Teacher
prefixes and tool observations are context only: only student-generated
responses receive OPD targets and gradients. Replay and student turns share
the original 15-turn budget. As in F2B, training includes the final model reply;
a purchase on the final allowed turn can carry task credit without a
protocol-complete training success. Evaluation stops at purchase immediately.

Resume uses the frozen cache and skips expert collection:

```bash
bash examples/agent/webshop-opd/train/scripts/run-tcod-b2f-grpo-teacher.sh \
  --resume-from output/webshop-opd/train/<run-name>/state/opd-step-00000020.json
```

Complete manifests retain the curriculum, data cursor, seed cursor, optimizer,
and update history. Changing expert actions or the curriculum invalidates a
resume. Unique run directories allow pause and resume within the same second.
Student sampling timeouts and teacher HTTP 503 responses use the same
pre-update, same-seed retry as F2B; other teacher, tool, and replay errors stop
the update. Cancellation closes sessions and preserves completed cache writes.
Metrics include `curriculum/k`, expert length, teacher-prefix mean/min/max,
and collection attempts.

Offline verification covers actual collection through the AgentRL tool loop,
strict success filtering, model identity and cache checks, failure and
cancellation cleanup, atomic cache replacement, duplicate goals, OPD updates,
evaluation, checkpoints, pause and resume:

```bash
bash examples/agent/webshop-opd/train/scripts/test-agent-opd.sh
```

## Train a GRPO teacher

The separate GRPO experiment trains Qwen3.5-4B with eight task groups and eight
trajectories per group (64 trajectories per update). Its training reward is
`(1 - alpha) * task_score + alpha * I(task_score == 1)`, with `alpha: 0.5` in
the default configuration. This retains partial-credit learning while placing
more relative weight on exact purchases. Set `environment.exact_success_alpha`
to `0.0` to reproduce the native WebShop reward. Periodic evaluation is not
shaped: it uses one trajectory per task and preserves the native score and
strict Pass@1 definition.

Validate the configuration:

```bash
bash examples/agent/webshop-opd/train/scripts/run-grpo-teacher.sh --check-config
```

Start the configured run:

```bash
bash examples/agent/webshop-opd/train/scripts/run-grpo-teacher.sh
```

For a later remote smoke test:

```bash
bash examples/agent/webshop-opd/train/scripts/run-grpo-teacher.sh --steps 2
```

The final remote checkpoint URI can be supplied as the fixed teacher
`model_path` in a subsequent OPD/TCOD configuration while retaining
`Qwen/Qwen3.5-4B` as the teacher base-model identity.

## Resume GRPO for teacher precision

`resume_grpo_teacher.py` starts a second phase from the completed dense-reward
GRPO Train checkpoint. The default configuration loads
`trio://run_n9e12csgxlof/weights/ppo-step-00000200`, reconstructs local progress
at step 200, and performs the configured 200 additional optimizer updates to
reach total step 400. PyTRIO restores the model and remote optimizer. The entry
point rebuilds the data cursor and sampling seed offset from the configured
`global_step`.
Remote checkpoint IDs do not contain the former local RNG, reporting windows,
update history, or token accounting; the report records these limitations.
New complete local manifests are created during the precision phase and support
exact recovery of later interruptions.

The precision phase makes a protocol-complete exact purchase the only positive
training outcome. An incorrect purchase receives a negative base reward plus a
penalty proportional to `1 - task_score`; no purchase and invalid actions also
receive configurable negative rewards. Evaluation remains unchanged and uses
the native WebShop score and strict Pass@1.

The defaults in `configs/resume-grpo-teacher.yaml` are:

```yaml
reward:
  exact_purchase_reward: 1.0
  incorrect_purchase_base: -0.3
  mismatch_penalty: 0.4
  no_purchase_reward: -0.2
  invalid_action_reward: -0.4
```

The ordering is validated so every incorrect purchase is worse than no
purchase, while an exact purchase remains the unique positive terminal result.
WebShop currently exposes an aggregate task score rather than separate product,
attribute, and option matches, so mismatch severity uses `1 - task_score`.

Validate the continuation configuration and checkpoint-ID syntax without
remote calls:

```bash
bash examples/agent/webshop-opd/train/scripts/run-resume-grpo-teacher.sh \
  --check-config
```

Start the configured precision phase:

```bash
bash examples/agent/webshop-opd/train/scripts/run-resume-grpo-teacher.sh
```

`--steps` overrides the number of additional updates. For example, `--steps 20`
restores step 200 and runs through total step 220. The entry point
first evaluates the restored step-200 policy using the new purchase metrics.
The default then saves and evaluates every 20 steps through final step 400,
and exports one final Playground Sampler. The external WebShop metrics loop
records purchase count, purchase rate, purchase success rate, incorrect purchase
rate, Pass@1, success rate, and average turns for evaluation and training batches.

### Native-reward continuation control

`resume_grpo_teacher_native.py` provides a control run from the same step-200
Train checkpoint. It preserves the original GRPO terminal reward exactly:

```text
training reward = task_score after purchase, otherwise 0
```

Model, optimizer, rollout, grouping, learning rate, evaluation, and Agent
settings match the original run. Only the training horizon continues. WebShop
purchase metrics are still aggregated outside Triotune by
`webshop_env/metrics.py` and published at every optimizer step and evaluation.

Validate without remote calls:

```bash
bash examples/agent/webshop-opd/train/scripts/run-resume-grpo-teacher-native.sh \
  --check-config
```

Start the default 200-update continuation from step 200 through step 400:

```bash
bash examples/agent/webshop-opd/train/scripts/run-resume-grpo-teacher-native.sh
```

Run a shorter control, for example 20 additional updates through step 220:

```bash
bash examples/agent/webshop-opd/train/scripts/run-resume-grpo-teacher-native.sh \
  --steps 20
```
