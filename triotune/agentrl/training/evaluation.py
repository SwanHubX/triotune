"""Task-group metrics and paired comparisons with explicit denominators."""

from __future__ import annotations

import math
from collections import defaultdict


def task_statistics(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_id].append(episode)
    return dict(groups)


def evaluation_metrics(episodes, *, pass_ks=(1,), prefix="eval/agent"):
    groups = task_statistics(episodes)
    metrics = {}
    for k in pass_ks:
        eligible = [group for group in groups.values() if len(group) >= k]
        # Unbiased pass@k estimator. System failures count as failed attempts,
        # preventing infrastructure outages from improving reported success.
        estimates = []
        for group in eligible:
            n = len(group)
            correct = sum(bool(e.trainable and e.score.success) for e in group)
            estimates.append(
                1.0
                - (
                    math.comb(n - correct, k) / math.comb(n, k)
                    if n - correct >= k
                    else 0.0
                )
            )
        metrics[f"{prefix}/pass@{k}"] = sum(estimates) / max(1, len(estimates))
        metrics[f"{prefix}/pass@{k}_tasks"] = float(len(eligible))
    first_calls, invalid, valid, tokens, success_turns = [], 0, 0, 0, []
    for episode in episodes:
        results = [r for turn in episode.turns for r in turn.tool_results]
        invalid += sum(r.is_error for r in results)
        valid += sum(not r.is_error for r in results)
        actions = [
            turn for turn in episode.turns if turn.action and turn.action.tool_calls
        ]
        if actions:
            first = actions[0]
            first_calls.append(
                len(first.tool_results) == len(first.action.tool_calls)
                and all(not r.is_error for r in first.tool_results)
            )
        tokens += sum(len(t.generation.tokens) for t in episode.turns)
        if episode.trainable and episode.score.success:
            success_turns.append(len(episode.turns))
    n = max(1, len(episodes))
    metrics.update(
        {
            f"{prefix}/tasks": float(len(groups)),
            f"{prefix}/first_tool_valid_rate": sum(first_calls)
            / max(1, len(first_calls)),
            f"{prefix}/first_tool_attempts": float(len(first_calls)),
            f"{prefix}/valid_tool_calls_mean": valid / n,
            f"{prefix}/invalid_tool_calls_mean": invalid / n,
            f"{prefix}/success_turns_mean": sum(success_turns)
            / max(1, len(success_turns)),
            f"{prefix}/generated_tokens_total": float(tokens),
            f"{prefix}/elapsed_seconds_total": sum(e.elapsed_seconds for e in episodes),
        }
    )
    correctness = [
        value
        for e in episodes
        if e.trainable
        for key, value in e.score.components.items()
        if key.rsplit("/", 1)[-1] == "first_tool_correctness"
    ]
    if correctness:
        metrics[f"{prefix}/first_tool_correctness"] = sum(correctness) / len(
            correctness
        )
        metrics[f"{prefix}/first_tool_scored_episodes"] = float(len(correctness))
    return metrics


def paired_task_summary(episodes):
    """Retain JSON-compatible per-task baselines without serializing episodes."""
    groups = defaultdict(list)
    for episode in episodes:
        if episode.task_id is not None:
            groups[episode.task_id].append(episode)
    return {
        key: {
            "success_rate": sum(float(e.trainable and e.score.success) for e in group) / len(group),
            "generated_tokens": sum(len(t.generation.tokens) for e in group for t in e.turns) / len(group),
            "tool_calls": sum(len(t.action.tool_calls) if t.action else 0 for e in group for t in e.turns) / len(group),
        } for key, group in groups.items()
    }


def paired_task_metrics(baseline, current, *, prefix="eval/agent/paired"):
    """Compare per-task mean success, lengths and costs for matching task IDs.

    IDs must be unique dataset identities; samples of one task share an ID.
    Calls without task IDs cannot form pairs and are counted in neither side.
    """

    before = baseline if isinstance(baseline, dict) else paired_task_summary(baseline)
    after = paired_task_summary(current)
    ids = sorted(before.keys() & after.keys())
    deltas = {"success_rate": [], "generated_tokens": [], "tool_calls": []}
    for key in ids:
        for name, values in deltas.items():
            values.append(after[key][name] - before[key][name])
    result = {f"{prefix}/tasks": float(len(ids))}
    result.update(
        {
            f"{prefix}/{key}_delta": sum(values) / max(1, len(values))
            for key, values in deltas.items()
        }
    )
    result[f"{prefix}/improved_tasks"] = float(
        sum(v > 0 for v in deltas["success_rate"])
    )
    result[f"{prefix}/regressed_tasks"] = float(
        sum(v < 0 for v in deltas["success_rate"])
    )
    return result
