# WebShop before-training evaluation

Install this example's environment and prepare its data as described in
[the WebShop OPD setup](../README.md). Configure PyTRIO credentials through
the usual environment settings before starting a real evaluation.

From the repository root:

```bash
bash examples/agent/webshop-opd/eval/scripts/run-student-eval.sh
bash examples/agent/webshop-opd/eval/scripts/run-teacher-eval.sh
```

Add `--check-config` to either command to validate settings without accessing
WebShop or a model service. No optimizer or training client is created.

Both configurations select the same first 200 official validation goals
(goal indexes 500 through 699). `evaluation.pass_k` selects a separate evaluation:

- `pass_k: 1` (default): one attempt per task, 200 episodes, report Pass@1.
- `pass_k: 4`: four independent attempts per task, 800 episodes, report Pass@4.

Override the YAML for one run with `--pass-k 1` or `--pass-k 4`:

```bash
bash examples/agent/webshop-opd/eval/scripts/run-student-eval.sh --pass-k 4
bash examples/agent/webshop-opd/eval/scripts/run-teacher-eval.sh --pass-k 4
```

Each response selects one action. Each episode
permits at most 15 turns, with 512 output tokens per turn, temperature 0.4,
top-p 1, thinking disabled, and at most four concurrent episodes. The complete
conversation is retained up to a 10,240-token context limit; it is not silently
truncated. Attempt and turn seeds are derived from seed 42 and the goal index.

The terminal progress bar counts tasks out of 200 for either mode. A task
advances the bar only after all its requested attempts finish, including failed
attempts. Elapsed time, estimated remaining time, completed rollout count, and
average success over finished attempts are also shown. Progress is written to
stderr; the final JSON report path stays on stdout.

Terminal logging is configured in each policy's YAML:

```yaml
evaluation:
  pass_k: 1
  rollout_log_interval: 20
```

Here an evaluation step means one fully evaluated task; it is separate from
environment interaction turns and optimizer steps. Every 20 completed tasks,
all rollouts for the selected task are printed, including their initial states,
all generated responses, actions, observations,
and errors. Set `rollout_log_interval: 0` to disable full-rollout printing.
These messages use stderr
alongside the progress bar. All trajectories are still saved to `episodes.jsonl`.
With thinking disabled, responses contain the model's actual action output;
the evaluator does not invent an additional reasoning trace.

The evaluator stops as soon as a purchase ends the environment, including on
the last allowed turn. Success requires a terminal environment score of exactly
1. Model claims alone never establish success. Invalid tool actions consume
a turn and receive error feedback; malformed responses end the attempt.

Each run creates a unique subdirectory under its configured output directory:

- `episodes.jsonl`: all attempts, responses, actions, observations, and errors.
- `report.json`: effective settings, pinned asset hashes, selected goals, and metrics.

Rates are in [0, 1]. `score` is the mean environment score multiplied by 100.
Only the selected `pass@k` is reported: the fraction of tasks with at least one
success in exactly k independent attempts. Pass@1 equals the single-attempt
success rate; Pass@4 does not emit a separate Pass@1 metric.
`average_single_attempt_success_rate` reports successes divided by all attempts,
and `first_attempt_success_rate` reports the success rate of attempt zero.
Attempt zero uses the same per-goal seed in both modes. Failed attempts remain in every
denominator; infrastructure failures are reported and produce a nonzero exit.

Purchase quality is counted directly from individual attempts in either mode:
`purchase_count` counts terminal purchases, `successful_purchase_count` counts
purchases marked successful with a score of exactly 1, and `purchase_success_rate`
is their ratio. `purchase_success_fraction` preserves the exact integer fraction
(for example, `53/79`), so the result is not reconstructed from rounded rates.
The decimal ratio is stored without explicit rounding. Both ratio fields are
`null` when no purchases occur, because the conditional rate is undefined.

This is an inference baseline using the existing JSON WebShop action protocol.
It is not a full reproduction of the TCOD paper's prompt or evaluation budget.
