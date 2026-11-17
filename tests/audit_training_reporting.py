"""Exercise real SwanLab storage and terminal progress with CPU training.

    PYTHONPATH=.:tests SWANLAB_MODE=offline .venv/bin/python \
        tests/audit_training_reporting.py --output /tmp/triotune-reporting

Model/optimizer transport is in memory; SwanLab is the installed, unmocked SDK.
Each case pauses and resumes, including an unfinished metric logging window.
"""

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import test_sampler_checkpoint as support
from recipes.calculator_agent import calculator_rows, calculator_workflow
from swanlab.proto.swanlab.record.v1.record_pb2 import Record
from swanlab.sdk.internal.core_python.store import DataStoreReader

from triotune.agentrl import (
    AgentConfig,
    AgentOPDTrainingLoop,
    AgentTrainingLoop,
    GiGPOAdvantageProvider,
    JsonlEpisodeLogger,
)
from triotune.utils.checkpoint import load_training_checkpoint


class TurnAnchorObserver:
    def observe(self, episode, interaction):
        return {"turn": len(episode.turns), "task": episode.task_id}


async def audit(root):
    results = {}
    for case in (
        "sft",
        "dpo",
        "opd",
        "ppo",
        "grpo",
        "agent_grpo",
        "gigpo",
        "agent_opd",
    ):
        directory = root / case
        agent = case in {"agent_grpo", "gigpo", "agent_opd"}
        kind = "opd" if case == "agent_opd" else "grpo" if agent else case
        service = support.ExportService(agent=agent)
        data = calculator_rows()[:2] if agent else support.rows(kind, 2)

        def make(resume=None):
            args = support.config(
                kind,
                steps=2,
                every=1,
                train_every=1,
                state_dir=str(directory / "checkpoints"),
                resume_from=resume,
            )
            args.reporting.report_to = "swanlab"
            args.reporting.project = "triotune-integration-audit"
            args.reporting.run_name = case
            args.reporting.log_dir = str(directory / "logs")
            args.reporting.logging_step = 3
            args.evaluation.before_train = True
            args.evaluation.after_train = True
            args.evaluation.steps = 1
            if agent:
                args.max_tokens = 256
            if case == "gigpo":
                args.reward_norm = args.adv_norm = None
            trainer = support.trainer_for(kind, args, data)
            trainer.eval_dataset = data[:1]
            owner = trainer
            if agent:
                workflow = calculator_workflow(
                    logger=JsonlEpisodeLogger(directory / "episodes.jsonl")
                )
                if case == "gigpo":
                    workflow.runner.anchor_observer = TurnAnchorObserver()
                loop = (
                    AgentOPDTrainingLoop if case == "agent_opd" else AgentTrainingLoop
                )
                options = dict(
                    config=AgentConfig(log_path=str(directory / "episodes.jsonl"))
                )
                if case == "gigpo":
                    options["batch_advantage_provider"] = GiGPOAdvantageProvider()
                owner = loop(trainer, workflow, **options)
            return trainer, owner

        with patch("pytrio.ServiceClient", return_value=service):
            first, owner = make()
            method = (
                owner.train_async
                if not agent and kind in {"dpo", "opd"}
                else owner.train
            )
            head = await method(stop_after_step=1)
            resumed, owner = make(head.checkpoint_manifest_path)
            method = (
                owner.train_async
                if not agent and kind in {"dpo", "opd"}
                else owner.train
            )
            state = await method()

        assert state.global_step == 2, case
        assert [export["step"] for export in service.exports] == [1, 2], case
        manifest = load_training_checkpoint(state.checkpoint_manifest_path)
        assert manifest["components"]["actor"]["sampler_checkpoint"]["step"] == 2
        runs = list((directory / "logs").rglob("*.swanlab"))
        assert len(runs) == 2, (case, runs)
        scalars, media, finishes = [], [], 0
        for path in runs:
            reader = DataStoreReader()
            reader.open(path)
            try:
                for raw in reader:
                    record = Record.FromString(raw)
                    if record.HasField("scalar"):
                        scalars.append(
                            (
                                record.scalar.key,
                                record.scalar.step,
                                record.scalar.value.number,
                            )
                        )
                    if record.HasField("media"):
                        media.append(record.media.key)
                    finishes += record.HasField("finish")
            finally:
                reader.close()
        assert finishes == 2, case
        assert any(key == "train/loss" and step == 2 for key, step, _ in scalars), case
        assert any(key.startswith("eval/") and step == 2 for key, step, _ in scalars), (
            case
        )
        assert "checkpoint/sampler_path" in media, case
        history = [
            json.loads(line)
            for path in (directory / "logs").glob("metrics-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert any(item.get("key") == "checkpoint/sampler_path" for item in history), (
            case
        )
        if agent:
            episodes = [
                json.loads(line)
                for line in (directory / "episodes.jsonl").read_text().splitlines()
            ]
            assert {episode["policy_version"] for episode in episodes} >= {0, 1}, case
        results[case] = dict(
            steps=2,
            swanlab_runs=len(runs),
            scalar_records=len(scalars),
            sampler_steps=[1, 2],
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        "REPORTING_AUDIT=" + json.dumps(asyncio.run(audit(args.output)), sort_keys=True)
    )
