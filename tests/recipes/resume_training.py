"""Compare uninterrupted and optimizer-restored runs using P0-01 fixtures.

Real PyTRIO execution is opt-in. Each comparison writes independent full/head/
resumed evidence and manifests, plus fixed-token model and optimizer probes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

if __package__:
    from . import remote_baseline as baseline
else:
    import remote_baseline as baseline

from triotune import PPOTrainer
from triotune.agentrl import AgentConfig, AgentTrainingLoop
from triotune.api.client_api import call_async, optimizer_params
from triotune.dataset import prepare_preference
from triotune.utils.checkpoint import load_training_checkpoint


async def stage(recipe, output, *, resume=None, stop=None):
    """Use the same public trainer entry point for full and resumed execution."""
    output.mkdir(parents=True)
    _, config = baseline.load_recipe(recipe, output)
    config.checkpoint.state_dir = str(output / "states")
    config.checkpoint.resume_from = resume
    # Retain the K-step state from both independent starts. Their difference
    # measures cross-run backend variation before any resume has occurred.
    config.checkpoint.save_every = 1
    train, valid = baseline.rows_for(recipe, "train"), baseline.rows_for(recipe, "eval")
    if recipe in {"native", "sdk", "ppo"}:
        trainer = PPOTrainer(
            args=config,
            train_dataset=train,
            eval_dataset=valid,
            reward_fn=baseline.gsm8k_reward,
        )
        if recipe in {"native", "sdk"}:
            owner = AgentTrainingLoop(
                trainer,
                baseline.make_workflow(recipe, config.model.name, output),
                config=AgentConfig(
                    limits=baseline.LIMITS,
                    max_concurrency=2,
                    evaluation={"num_generations": 1, "pass_ks": [1], "seed": 1729},
                ),
            )
        else:
            owner = trainer
        await trainer.initialize()
    else:
        trainer = await asyncio.to_thread(
            baseline.TRAINERS[recipe],
            args=config,
            train_dataset=train,
            eval_dataset=valid,
        )
        owner = trainer
    audits = baseline.attach_audits(trainer, recipe, output)
    evidence = {"config": asdict(config), "evaluations": []}
    evaluator = (
        "_evaluate_async"
        if recipe == "opd"
        else "evaluate_async"
        if recipe == "dpo"
        else "evaluate"
    )
    baseline.observe_evaluation(owner, evaluator, audits, evidence)
    method = "train_async" if recipe in {"dpo", "opd"} else "train"
    state = await getattr(owner, method)(stop_after_step=stop)
    evidence.update(
        state=asdict(state),
        manifest=load_training_checkpoint(state.checkpoint_manifest_path),
        audit={key: client.summary() for key, client in audits.items()},
    )
    (output / "stage.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    return trainer, audits, evidence


def training_inputs(path):
    """Compare the actual ordered SDK token inputs and supervised labels."""
    records = []
    for line in path.read_text().splitlines():
        call = json.loads(line)
        if call["operation"].startswith("forward_backward"):
            records.append(
                [
                    (row["input_ids"], row["loss_inputs"]["target_tokens"])
                    for row in call["datums"]
                ]
            )
    return records


async def compare_readouts(left, right, probe, *, cross_run_noise=0.0, control=False):
    before, after = (
        await baseline.readout(left, probe),
        await baseline.readout(right, probe),
    )
    delta = max(
        abs(a - b) for a, b in zip(before["logprobs"], after["logprobs"], strict=True)
    )
    tolerance = max(
        1e-4,
        5 * max(cross_run_noise, before["max_repeat_delta"], after["max_repeat_delta"]),
    )
    if not control and delta > tolerance:
        raise RuntimeError(
            f"Restored model differs from continuous training: {delta} > {tolerance}."
        )
    return {
        "continuous": before,
        "resumed": after,
        "max_logprob_delta": delta,
        "tolerance": tolerance,
    }


async def compare(recipe, output, report):
    report["phase"] = "continuous"
    full, full_clients, continuous = await stage(recipe, output / "full")
    report["phase"] = "pause"
    _, _, head = await stage(recipe, output / "head", stop=1)
    report["phase"] = "resume"
    _resumed, _, tail = await stage(
        recipe, output / "resumed", resume=head["state"]["checkpoint_manifest_path"]
    )
    report.update(continuous=continuous, head=head, resumed=tail)
    if recipe == "opd":
        # A frozen teacher can still return different floating-point scores
        # across serving batches. An independent N-step replica measures the
        # resulting optimization variation, beyond the K-step start control.
        report["phase"] = "independent_teacher_control"
        _, _, report["control"] = await stage(recipe, output / "control")
    report["phase"] = "continuation_contracts"
    for key in ("global_step", "optimizer_steps", "data_cursor"):
        if continuous["state"][key] != tail["state"][key]:
            raise RuntimeError(f"Continuation progress differs: {key}.")
    if continuous["manifest"]["runtime"] != tail["manifest"]["runtime"]:
        raise RuntimeError(
            "Continuation counters/seed offsets/evaluation progress differ."
        )
    for component in full_clients:
        name = f"{component}-datums.jsonl"
        if training_inputs(output / "full" / name) != training_inputs(
            output / "head" / name
        ) + training_inputs(output / "resumed" / name):
            raise RuntimeError(
                f"Ordered {component} SDK inputs/labels differ after resume."
            )
    if [e["step"] for e in continuous["evaluations"]] != [
        e["step"] for e in head["evaluations"] + tail["evaluations"]
    ]:
        raise RuntimeError("Resumed evaluation schedule differs.")
    for key, value in continuous["state"]["metrics"].items():
        if key.startswith("usage/") and value != tail["state"]["metrics"].get(key):
            raise RuntimeError(f"Cumulative usage differs: {key}.")
    if recipe in {"native", "sdk"}:
        before = continuous["manifest"]["loop"]["state"]
        after = tail["manifest"]["loop"]["state"]
        if (
            before["seed_offset"] != after["seed_offset"]
            or before["baseline"] != after["baseline"]
        ):
            raise RuntimeError("Agent candidate seed/evaluation baseline differs.")
        if [u["policy_versions"] for u in before["updates"]] != [
            u["policy_versions"] for u in after["updates"]
        ]:
            raise RuntimeError("Agent behavior-policy source versions differ.")
    pair = prepare_preference(
        baseline.rows_for("dpo", "train")[0],
        full._tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )
    probe = {"prompt_ids": pair.prompt, "completion_ids": pair.rejected}
    report["probe"] = probe
    report["phase"] = "model_and_optimizer_probes"
    report["components"] = {}
    service = await asyncio.to_thread(baseline.pytrio.ServiceClient)
    for name in full_clients:
        component = "actor" if name == "policy" else name
        algorithm = continuous["manifest"]["algorithm"]
        control_state = load_training_checkpoint(
            output / "full" / "states" / f"{algorithm}-step-00000001.json"
        )
        control_uris = [
            control_state["components"][component]["uri"],
            head["manifest"]["components"][component]["uri"],
        ]
        controls = [
            await call_async(
                service, "create_training_client_from_state_with_optimizer", path=uri
            )
            for uri in control_uris
        ]
        control_values = await compare_readouts(*controls, probe, control=True)
        cross_run_noise = max(
            control_values["max_logprob_delta"],
            control_values["continuous"]["max_repeat_delta"],
            control_values["resumed"]["max_repeat_delta"],
        )
        report["components"][name] = {"independent_start_control": control_values}
        if "control" in report:
            control_uri = report["control"]["manifest"]["components"][component]["uri"]
            replica_uris = [
                continuous["manifest"]["components"][component]["uri"],
                control_uri,
            ]
            replicas = [
                await call_async(
                    service,
                    "create_training_client_from_state_with_optimizer",
                    path=uri,
                )
                for uri in replica_uris
            ]
            replica_values = await compare_readouts(*replicas, probe, control=True)
            report["components"][name]["independent_full_control"] = replica_values
            cross_run_noise = max(
                cross_run_noise,
                replica_values["max_logprob_delta"],
                replica_values["continuous"]["max_repeat_delta"],
                replica_values["resumed"]["max_repeat_delta"],
            )
        # Fork the final complete states and apply the same diagnostic update.
        # PyTRIO does not expose Adam tensors; this tests their observable next
        # update behavior in addition to exact local Adam-state comparisons.
        left_uri = continuous["manifest"]["components"][component]["uri"]
        right_uri = tail["manifest"]["components"][component]["uri"]
        clones = [
            await call_async(
                service, "create_training_client_from_state_with_optimizer", path=uri
            )
            for uri in (left_uri, right_uri)
        ]
        report["components"][name]["model"] = await compare_readouts(
            *clones, probe, cross_run_noise=cross_run_noise
        )
        datum = baseline.pytrio.Datum(
            model_input=baseline.pytrio.ModelInput.from_ints(
                pair.prompt + pair.rejected[:-1]
            ),
            loss_fn_inputs={
                "target_tokens": [-100] * (len(pair.prompt) - 1) + pair.rejected
            },
        )
        settings = (
            full.args.critic.optimizer if name == "critic" else full.args.optimizer
        )
        for clone in clones:
            await call_async(
                clone,
                "forward_backward",
                [datum],
                loss_fn="cross_entropy",
                auto_shift=False,
            )
            await call_async(
                clone,
                "optim_step",
                optimizer_params(settings, baseline.pytrio.AdamParams),
            )
        report["components"][name]["optimizer_next_update"] = await compare_readouts(
            *clones, probe, cross_run_noise=cross_run_noise
        )
    report["phase"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--recipe", choices=baseline.RECIPES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Run only the continuation in this fresh process",
    )
    args = parser.parse_args(argv)
    if not args.remote:
        parser.error("Real continuation acceptance requires explicit --remote.")
    output = (
        args.output_dir
        or baseline.ROOT
        / "output"
        / "resume-acceptance"
        / f"{args.recipe}-{uuid4().hex[:8]}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "recipe": args.recipe,
        "status": "running",
        "phase": "preflight",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--", "triotune"])
        ).hexdigest(),
    }
    try:
        if args.resume_from is not None:
            _, _, evidence = asyncio.run(
                stage(
                    args.recipe,
                    output / "resumed",
                    resume=str(args.resume_from.resolve()),
                )
            )
            report["resumed_only"] = evidence
            report["phase"] = "complete"
        else:
            asyncio.run(compare(args.recipe, output, report))
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{args.recipe}: {report['status']}; report: {output / 'report.json'}",
            flush=True,
        )
    return report


if __name__ == "__main__":
    main()
