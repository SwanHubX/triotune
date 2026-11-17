"""Collect resumable successful WebShop paths from the configured fixed teacher."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
sys.path.insert(0, str(EXAMPLE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

import pytrio
from tqdm import tqdm
from triotune.agentrl import PyTRIOAgentSampler, SamplingSnapshot
from triotune.agentrl.types import SYSTEM_STATUSES
from triotune.api.client_api import call_async
from triotune.trainer.ppo.rollout import gather_cancel_on_error
from train.tcod_b2f import (
    DEFAULT_CONFIG, load_experiment, load_expert_actions, validate_expert_actions,
)
from webshop_env.data_prepare import asset_manifest, cached_files, goal_fingerprint, task_rows
from webshop_env.environment import INSTRUCTIONS, SessionPool, WebShopBackend, build_workflow


def episode_actions(episode):
    """Extract actual, successful tool executions; omit model-final replies."""
    actions = []
    for turn in episode.turns:
        action = turn.action
        if action is None:
            return None
        if action.final_text is not None:
            continue
        if len(action.tool_calls) != 1 or len(turn.tool_results) != 1:
            return None
        call, result = action.tool_calls[0], turn.tool_results[0]
        if call.name != "webshop" or result.is_error:
            return None
        actions.append(call.arguments.get("action"))
    try:
        validate_expert_actions(actions)
    except ValueError:
        return None
    return actions


def atomic_write(path, text):
    """Replace a complete small cache boundary, including on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cache_identity(experiment, files, goals):
    agent = experiment.agent.to_dict()
    # Collection always uses the complete horizon; training/evaluation cadence
    # and the subsequent curriculum do not change teacher generations.
    for name in ("curriculum", "evaluation", "log_path"):
        agent.pop(name, None)
    return json.loads(json.dumps({
        "format_version": 1,
        "teacher": asdict(experiment.trainer.teacher),
        "assets": asset_manifest(files),
        "goal_sha256": goal_fingerprint(goals),
        "environment_seed": experiment.environment_seed,
        "split_protocol": experiment.data.split_protocol,
        "experts": {key: value for key, value in asdict(experiment.experts).items() if key != "path"},
        "agent": agent,
        "instructions": INSTRUCTIONS,
    }, sort_keys=True, allow_nan=False))


def open_cache(experiment, identity):
    path = experiment.experts.path
    metadata = path.with_suffix(path.suffix + ".meta.json")
    if metadata.exists():
        if json.loads(metadata.read_text()) != identity:
            raise ValueError("Expert cache teacher, assets or collection settings changed; use a new cache path.")
    elif path.exists():
        raise ValueError("Expert cache has no provenance metadata; use a new cache path.")
    else:
        atomic_write(metadata, json.dumps(identity, indent=2, sort_keys=True) + "\n")
    actions = load_expert_actions(path, allow_empty=True) if path.exists() else {}
    for value in actions.values():
        validate_expert_actions(list(value), max_turns=experiment.agent.limits.max_turns)
    return {goal: {"goal_index": goal, "success": True, "actions": list(value)}
            for goal, value in actions.items()}


async def collect(experiment, *, max_tasks=None):
    """Sample only the teacher, persist strict successes, and close every session.

    max_tasks bounds previously uncached goals, not successful goals. Failures
    of the model or environment abort collection after keeping completed cache
    boundaries. Ordinary unsuccessful attempts can be retried on a later run.
    """
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer.")
    path = experiment.experts.path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hold the same lock through reads and replacements. Never unlink a lock
    # file: another process may already be waiting on its inode.
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another collector is using this expert cache.") from None
        return await _collect(experiment, max_tasks=max_tasks)


async def _collect(experiment, *, max_tasks):
    files = cached_files(experiment.data)
    pool = workflow = None
    try:
        backend = await asyncio.to_thread(WebShopBackend, experiment.source_dir,
                                          files, experiment.environment_seed)
        pool = SessionPool(backend)
        identity = cache_identity(experiment, files, backend.goals)
        records = open_cache(experiment, identity)
        rows = task_rows(backend.goals, experiment.data, "train")
        required = experiment.trainer.max_steps
        if required is not None:
            rows = rows[:required * experiment.trainer.batch_size]
        unique = {row["goal_index"]: row for row in rows}
        pending = [row for goal, row in unique.items() if goal not in records]
        if max_tasks is not None:
            pending = pending[:max_tasks]
        report = {"cache": str(experiment.experts.path), "requested_tasks": len(pending),
                  "successful_tasks": 0, "cached_total": len(records)}
        if not pending:
            return report
        service = await asyncio.to_thread(pytrio.ServiceClient)
        teacher = experiment.trainer.teacher
        client = await call_async(service, "create_sampling_client",
                                  base_model=teacher.name, model_path=teacher.model_path)
        tokenizer = await asyncio.to_thread(client.get_tokenizer)
        snapshot = SamplingSnapshot(
            f"b2f-expert-{uuid4().hex}",
            PyTRIOAgentSampler(client, tokenizer, stop=experiment.agent.stop,
                               temperature=experiment.experts.temperature),
            tokenizer, policy_version=0,
        )
        workflow = build_workflow(replace(experiment.agent, log_path=None), tokenizer, pool)
        work = iter(pending)

        async def worker():
            for row in work:
                batch = await workflow.collect(
                    [row], snapshot, n_samples=experiment.experts.pass_k,
                    seed=experiment.experts.seed + row["goal_index"] * experiment.experts.pass_k
                         * experiment.agent.limits.max_turns,
                    is_eval=True, strict=False, log_episodes=False,
                )
                candidates = []
                for episode in batch.episodes:
                    if episode.score is not None and episode.score.success:
                        actions = episode_actions(episode)
                        if actions is not None:
                            candidates.append(actions)
                if candidates:
                    actions = min(candidates, key=lambda value: (len(value), tuple(value)))
                    records[row["goal_index"]] = {
                        "goal_index": row["goal_index"], "success": True, "actions": actions,
                    }
                    atomic_write(experiment.experts.path, "".join(
                        json.dumps(records[goal], ensure_ascii=False) + "\n" for goal in sorted(records)
                    ))
                    report["successful_tasks"] += 1
                    report["cached_total"] = len(records)
                progress.update(1)
                progress.set_postfix(successes=report["successful_tasks"], refresh=False)
                if any(episode.status in SYSTEM_STATUSES
                       or episode.export_metadata.get("system_failure", False)
                       for episode in batch.episodes):
                    raise RuntimeError("Expert collection had infrastructure failures; successful paths were preserved.")

        with tqdm(total=len(pending), desc="B2F teacher experts", unit="goal") as progress:
            await gather_cancel_on_error(
                worker() for _ in range(min(experiment.experts.batch_size, len(pending)))
            )
        return report
    finally:
        try:
            if workflow is not None:
                await workflow.close()
        finally:
            if pool is not None:
                await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    experiment = load_experiment(args.config, steps=args.steps)
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    print(json.dumps(asyncio.run(collect(experiment, max_tasks=args.max_tasks))))


if __name__ == "__main__":
    main()
