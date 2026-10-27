"""Explicitly enabled, small real-PyTRIO training acceptance recipes.

Run from the repository root with ``--remote --recipe sft`` (or another recipe).
The runner audits the SDK boundary while leaving the trainers' actual sampling,
losses, optimizer ordering, evaluation, and checkpoint lifecycle in control.
Outputs belong under ignored output/; only the compact measured summary should
be copied into the acceptance record after reviewing the real remote results.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

# Direct script execution must resolve the checkout, including the public
# examples namespace, before any installed package with the same name.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytrio

from examples.agent.calculator.baseline import LIMITS, make_workflow
from examples.ppo.gsm8k_ppo import gsm8k_reward
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentTrainingLoop,
    PyTRIOAgentSampler,
    SamplingSnapshot,
)
from triotune.api.client_api import call_async
from triotune.dataset import prepare_preference, prepare_prompt
from triotune.trainer import (
    DPOConfig,
    DPOTrainer,
    OPDConfig,
    OPDTrainer,
    SFTConfig,
    SFTTrainer,
)
from triotune.utils.config import load_yaml_config
from triotune.utils.scoring import score_completion_async

RECIPES = ("native", "sdk", "ppo", "sft", "dpo", "opd")
CONFIG_TYPES = {"sft": SFTConfig, "dpo": DPOConfig, "opd": OPDConfig}
TRAINERS = {"sft": SFTTrainer, "dpo": DPOTrainer, "opd": OPDTrainer}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name):
    """Record optional dependencies without requiring every extra for SFT."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def rows_for(recipe, split):
    """Adapt checked-in, disjoint fixtures without sampling or shuffling rows.

    Every algorithm sees the same questions and answers. SFT trains only the
    assistant answer, DPO compares the explicit correct/incorrect pair, and PPO
    uses the public boxed-answer reward. Agent instructions are supplied by the
    application workflow, not hidden in the data adapter.
    """
    rows = []
    for line in (
        Path(__file__).with_name(f"baseline_{split}.jsonl").read_text().splitlines()
    ):
        raw = json.loads(line)
        question = raw["question"]
        if recipe not in {"native", "sdk"}:
            question += " Reply with only the numerical answer inside \\boxed{}."
        messages = [{"role": "user", "content": question}]
        correct = f"\\boxed{{{raw['answer']}}}"
        row = {"task_id": raw["task_id"], "messages": messages, "answer": raw["answer"]}
        if recipe == "sft":
            row = {
                "prompt": messages,
                "completion": [{"role": "assistant", "content": correct}],
            }
        elif recipe == "dpo":
            row.update(chosen=correct, rejected=f"\\boxed{{{raw['rejected']}}}")
        rows.append(row)
    return rows


def load_recipe(recipe, output):
    """Keep YAML directly compatible with public trainer config constructors."""
    directory = "agent/calculator" if recipe in {"native", "sdk"} else recipe
    path = ROOT / "examples" / directory / "baseline.yaml"
    settings = load_yaml_config(path)
    settings.setdefault("reporting", {})["log_dir"] = str(output / "metrics")
    if recipe in {"native", "sdk", "ppo"}:
        algorithm = settings.pop("algorithm")
        config = PPOConfig.for_algorithm(algorithm, **settings)
    else:
        config = CONFIG_TYPES[recipe](**settings)
    return path, config


# ---------------------------------------------------------------------------
# Passive SDK request auditing
# ---------------------------------------------------------------------------


def datum_record(datum, *, critic=False):
    """Validate the exact payload submitted to PyTRIO, without changing it.

    Targets are already shifted by the trainer. Each active nonfinal target
    must equal the following input token; -100 positions are context only.
    Critic marker targets intentionally predict vocabulary markers instead of
    the next sampled token, so only dimensions and finiteness apply there.
    Auxiliary loss inputs must describe the same positions as target_tokens.
    """
    inputs = list(datum.model_input.to_ints())
    fields = {name: value.tolist() for name, value in datum.loss_fn_inputs.items()}
    targets = fields["target_tokens"]
    if not inputs or len(inputs) != len(targets):
        raise ValueError("Training input and target token lengths do not align.")
    if not any(token != -100 for token in targets):
        raise ValueError("Training datum has no active target tokens.")
    for name, values in fields.items():
        if len(values) != len(inputs) or any(not math.isfinite(v) for v in values):
            raise ValueError(f"Training field {name} does not align with input tokens.")
    if not critic and any(
        target != -100 and target != following
        for target, following in zip(targets[:-1], inputs[1:], strict=True)
    ):
        raise ValueError("Training targets violate causal next-token alignment.")
    return {"input_ids": inputs, "loss_inputs": fields}


class AuditedClient:
    """Delegate SDK calls verbatim, retaining futures and submission ordering.

    This test-only adapter observes the SDK boundary. It deliberately returns
    the original coroutine/future instead of awaiting it: SFT needs submission
    and completion to remain separate. Submission counts alone are never used
    as evidence of completed optimization; successful trainer state and actual
    logprob changes are checked independently.
    """

    def __init__(self, client, path, *, critic=False):
        self.client, self.path, self.critic = client, path, critic
        self.optimizer_submissions = 0
        self.backward_requests = 0
        self.datums = 0
        self.active_targets = 0

    def __getattr__(self, name):
        method = getattr(self.client, name)
        operation = name.removesuffix("_async")
        if operation not in {
            "forward",
            "forward_backward",
            "forward_backward_custom",
            "optim_step",
        }:
            return method

        @wraps(method)
        def observed(*args, **kwargs):
            if operation == "optim_step":
                self.optimizer_submissions += 1
            else:
                data = args[0] if args else kwargs["data"]
                records = [datum_record(d, critic=self.critic) for d in data]
                if operation.startswith("forward_backward"):
                    self.backward_requests += 1
                    self.datums += len(records)
                    self.active_targets += sum(
                        token != -100
                        for record in records
                        for token in record["loss_inputs"]["target_tokens"]
                    )
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"operation": operation, "datums": records}, allow_nan=False
                        )
                        + "\n"
                    )
            return method(*args, **kwargs)

        return observed

    def summary(self):
        return {
            name: getattr(self, name)
            for name in (
                "optimizer_submissions",
                "backward_requests",
                "datums",
                "active_targets",
            )
        }


def attach_audits(trainer, recipe, output):
    """Isolate private test instrumentation from production trainer APIs.

    Public trainers intentionally expose no mutable client accessor. Bind the
    passive proxy once, after construction/initialization and before training;
    keep each trainer and its actor/critic pointing to the same delegated SDK
    object. No alternate optimizer, loss, or sampling implementation is used.
    Layout regressions are covered by local contract tests and live recipes.
    """
    attribute = (
        "_client" if recipe in {"native", "sdk", "ppo", "sft"} else "_training_client"
    )
    policy = AuditedClient(getattr(trainer, attribute), output / "policy-datums.jsonl")
    setattr(trainer, attribute, policy)
    audits = {"policy": policy}
    if isinstance(trainer, PPOTrainer):
        trainer._actor.client = policy
        if trainer.critic is not None:
            critic = AuditedClient(
                trainer.critic.client, output / "critic-datums.jsonl", critic=True
            )
            trainer.critic.client = critic
            audits["critic"] = critic
    return audits


def observe_evaluation(owner, method_name, audits, report):
    """Prove every configured evaluation leaves update counters unchanged."""
    original = getattr(owner, method_name)

    async def evaluated(*args, **kwargs):
        before = [a.optimizer_submissions for a in audits.values()]
        step = (
            owner.trainer.state.global_step
            if isinstance(owner, AgentTrainingLoop)
            else owner.state.global_step
        )
        metrics = await original(*args, **kwargs)
        current_step = (
            owner.trainer.state.global_step
            if isinstance(owner, AgentTrainingLoop)
            else owner.state.global_step
        )
        if (
            before != [a.optimizer_submissions for a in audits.values()]
            or step != current_step
        ):
            raise RuntimeError(
                "Evaluation changed optimizer submissions or global_step."
            )
        report.setdefault("evaluations", []).append({"step": step, "metrics": metrics})
        return metrics

    setattr(owner, method_name, evaluated)


# ---------------------------------------------------------------------------
# Fixed-sequence weight and saved-checkpoint verification
# ---------------------------------------------------------------------------


def policy_change(before, after, *, noise=0.0):
    """Reject missing, nonfinite, misaligned, or unchanged remote weights."""
    if not before or len(before) != len(after):
        raise ValueError("Verification logprobs must be nonempty and aligned.")
    if any(not math.isfinite(v) for v in [*before, *after]):
        raise ValueError("Verification logprobs must be finite.")
    delta = max(abs(a - b) for a, b in zip(before, after, strict=True))
    tolerance = max(1e-5, 5 * noise)
    if delta <= tolerance:
        raise RuntimeError(
            f"Updated policy has no logprob change above measured forward noise: {delta} <= {tolerance}."
        )
    return delta


async def snapshot(client):
    sampler = await call_async(client, "save_weights_and_get_sampling_client")
    return sampler, {
        "sampling_run_cuid": sampler.sampling_run_cuid,
        "model_id": sampler.model_id,
    }


async def witness(sampler, tokenizer, row):
    """Sample once, then retain exact IDs for all later weight comparisons."""
    prompt_row = row if "messages" in row else {"messages": row["prompt"]}
    prompt = prepare_prompt(
        prompt_row, tokenizer, chat_template_kwargs={"enable_thinking": False}
    )
    sampled = await call_async(
        sampler,
        "sample",
        prompt=pytrio.ModelInput.from_ints(prompt.tokens),
        num_samples=1,
        sampling_params=pytrio.SamplingParams(
            max_tokens=32, seed=314159, temperature=1.0, top_p=1.0
        ),
    )
    sequence = sampled.sequences[0]
    tokens, logprobs = list(sequence.tokens), list(sequence.logprobs)
    if (
        not tokens
        or len(tokens) != len(logprobs)
        or any(not math.isfinite(v) for v in logprobs)
    ):
        raise ValueError("Sampled token IDs and behavior logprobs do not align.")
    return {
        "prompt_ids": prompt.tokens,
        "completion_ids": tokens,
        "behavior_logprobs": logprobs,
    }


async def score(sampler, evidence):
    return await score_completion_async(
        sampler, evidence["prompt_ids"], evidence["completion_ids"]
    )


async def forward_scores(client, evidence):
    """Read fixed-target probabilities on the training backend, without updates.

    Remote kernels can vary with execution batches. Compare saved and live
    weights using the same training forward operation, and calibrate repeated
    requests separately. Sampling scores remain in the report as inference
    diagnostics and are not used as exact checkpoint equality evidence.
    """
    prompt, completion = evidence["prompt_ids"], evidence["completion_ids"]
    datum = pytrio.Datum(
        model_input=pytrio.ModelInput.from_ints(prompt + completion[:-1]),
        loss_fn_inputs={"target_tokens": [-100] * (len(prompt) - 1) + completion},
    )
    result = await call_async(
        client, "forward", [datum], loss_fn="cross_entropy", auto_shift=False
    )
    values = result.loss_fn_outputs[0]["logprobs"].tolist()
    if len(values) != len(prompt) + len(completion) - 1:
        raise ValueError("Remote forward logprobs violate causal token alignment.")
    scores = values[len(prompt) - 1 :]
    if any(not math.isfinite(v) or v > 0 for v in scores):
        raise ValueError("Remote forward logprobs must be finite and nonpositive.")
    return scores


async def readout(client, evidence):
    """Measure repeated fixed-weight forwards before judging any weight change.

    A seed fixes sampling choices, not backend floating-point reductions. Three
    identical forward requests estimate an observed numerical noise envelope.
    Keep every vector so the acceptance claim can be independently reviewed.
    """
    repeats = [await forward_scores(client, evidence) for _ in range(3)]
    columns = list(zip(*repeats, strict=True))
    return {
        "logprobs": [sum(values) / len(values) for values in columns],
        "repeat_logprobs": repeats,
        "max_repeat_delta": max(max(values) - min(values) for values in columns),
    }


async def verify_saved(service, path, evidence, expected):
    """Load the saved weights for inference; optimizer recovery is a later gate."""
    if not path:
        raise RuntimeError("Training did not save a checkpoint.")
    # save_state produces a TRAIN checkpoint. PyTRIO 0.2.9 does not accept
    # that URI directly in create_sampling_client; load its weights into a
    # training client and export a SAMPLER snapshot through the supported API.
    # No optimizer state is restored and no update is run on the loaded client.
    loaded = await call_async(service, "create_training_client_from_state", path=path)
    saved, _ = await snapshot(loaded)
    actual = await readout(loaded, evidence)
    delta = max(
        abs(a - b)
        for a, b in zip(actual["logprobs"], expected["logprobs"], strict=True)
    )
    # Compare within the measured repeated-forward envelope. A fixed absolute
    # floor handles serialization rounding when all measured repeats agree.
    tolerance = max(
        1e-5, 5 * max(actual["max_repeat_delta"], expected["max_repeat_delta"])
    )
    if delta > tolerance:
        raise RuntimeError(
            f"Saved checkpoint differs from the final policy snapshot: {delta} > {tolerance}."
        )
    return {
        "path": path,
        "sampling_run_cuid": saved.sampling_run_cuid,
        "max_logprob_delta": delta,
        "tolerance": tolerance,
        "forward": actual,
        "sampling_logprobs": await score(saved, evidence),
    }


async def run_recipe(recipe, output, report):
    """Exercise the public training entry point and collect independent evidence."""
    config_path, config = load_recipe(recipe, output)
    report.update(
        recipe=recipe, config=asdict(config), config_sha256=digest(config_path)
    )
    report["fixtures"] = {
        split: digest(Path(__file__).with_name(f"baseline_{split}.jsonl"))
        for split in ("train", "eval")
    }
    report["phase"] = "client_initialization"
    train, valid = rows_for(recipe, "train"), rows_for(recipe, "eval")
    if recipe in {"native", "sdk", "ppo"}:
        trainer = PPOTrainer(
            args=config, train_dataset=train, eval_dataset=valid, reward_fn=gsm8k_reward
        )
        await trainer.initialize()
    else:
        trainer = await asyncio.to_thread(
            TRAINERS[recipe], args=config, train_dataset=train, eval_dataset=valid
        )
    audits = attach_audits(trainer, recipe, output)
    service = await asyncio.to_thread(pytrio.ServiceClient)
    tokenizer = trainer._tokenizer
    owner = trainer
    if recipe in {"native", "sdk"}:
        workflow = make_workflow(recipe, config.model.name, output)
        agent_config = AgentConfig(
            limits=LIMITS,
            max_concurrency=2,
            evaluation={"num_generations": 1, "pass_ks": [1], "seed": 1729},
        )
        report["agent_config"] = asdict(agent_config)
        owner = AgentTrainingLoop(trainer, workflow, config=agent_config)
    # OPD's public train_async calls its guarded internal evaluator directly;
    # observe that boundary without entering the public re-entrancy guard.
    evaluator = (
        "_evaluate_async"
        if recipe == "opd"
        else "evaluate_async"
        if recipe == "dpo"
        else "evaluate"
    )
    observe_evaluation(owner, evaluator, audits, report)
    try:
        report["phase"] = "initial_sampling"
        initial, initial_id = await snapshot(audits["policy"])
        if isinstance(owner, AgentTrainingLoop):
            # Measure a real tool action under its advertised schemas and
            # original prompt, rather than an unrelated plain-text response.
            # The same fixed action IDs are scored again after optimization.
            batch = await owner.workflow.collect(
                train[:1],
                SamplingSnapshot(
                    "baseline-witness-0",
                    PyTRIOAgentSampler(initial, tokenizer),
                    tokenizer,
                    0,
                ),
                n_samples=1,
                seed=314159,
            )
            turn = batch.episodes[0].turns[0]
            evidence = {
                "prompt_ids": list(turn.prompt_tokens),
                "completion_ids": list(turn.generation.tokens),
                "behavior_logprobs": list(turn.generation.logprobs),
                "task_id": train[0]["task_id"],
            }
        else:
            evidence = await witness(initial, tokenizer, valid[0])
            if recipe == "dpo":
                report["initial_sample"] = evidence
                # DPO directly updates both alternatives. Probe its fixed
                # rejected training response, whose probability is measurable
                # even when sampled correct answers are nearly saturated.
                pair = prepare_preference(
                    train[0],
                    tokenizer,
                    max_length=config.max_length,
                    chat_template_kwargs=config.chat_template_kwargs,
                )
                evidence = {
                    "prompt_ids": pair.prompt,
                    "completion_ids": pair.rejected,
                    "task_id": train[0]["task_id"],
                    "preference": "rejected",
                }
        report["witness"] = evidence
        before = await readout(audits["policy"], evidence)
        report["policy_before"] = {
            **initial_id,
            "version": 0,
            **before,
            "sampling_logprobs": await score(initial, evidence),
        }
        critic_before = None
        if "critic" in audits:
            _, critic_id = await snapshot(audits["critic"])
            critic_before = await readout(audits["critic"], evidence)
            report["critic_before"] = {**critic_id, "version": 0, **critic_before}
        report["phase"] = "training"
        train_method = "train_async" if recipe in {"dpo", "opd"} else "train"
        state = await getattr(owner, train_method)()
        report["state"] = asdict(state)
        report["phase"] = "weight_verification"
        final, final_id = await snapshot(audits["policy"])
        after = await readout(audits["policy"], evidence)
        report["policy_after"] = {
            **final_id,
            "version": state.global_step,
            **after,
            "sampling_logprobs": await score(final, evidence),
        }
        report["policy_noise_tolerance"] = max(
            1e-5, 5 * max(before["max_repeat_delta"], after["max_repeat_delta"])
        )
        report["max_policy_logprob_delta"] = policy_change(
            before["logprobs"],
            after["logprobs"],
            noise=report["policy_noise_tolerance"] / 5,
        )
        if (
            state.global_step != config.max_steps
            or audits["policy"].optimizer_submissions != config.max_steps
        ):
            raise RuntimeError(
                "Training did not complete the configured optimizer updates."
            )
        evaluations = report.get("evaluations", [])
        if len(evaluations) != 2 or [e["step"] for e in evaluations] != [
            0,
            config.max_steps,
        ]:
            raise RuntimeError("Missing initial/final evaluation evidence.")
        if isinstance(owner, AgentTrainingLoop):
            report["updates"] = owner.update_history
            batch = owner.last_train_batch
            # Verify the exported action stream against original generations,
            # excluding inserted tool/context tokens from the policy targets.
            sampled = {
                episode.episode_id: [
                    (token, lp)
                    for turn in episode.turns
                    for token, lp in zip(
                        turn.generation.tokens, turn.generation.logprobs, strict=True
                    )
                ]
                for episode in batch.episodes
            }
            exported = {
                row.episode_id: [
                    (token, lp)
                    for token, lp, active in zip(
                        row.completion_tokens, row.logprobs, row.loss_mask, strict=True
                    )
                    if active
                ]
                for row in batch.trajectories
            }
            if sampled != exported:
                raise RuntimeError(
                    "Exported agent actions differ from original sampled tokens/logprobs."
                )
            report["agent_alignment"] = {
                "sampled_actions": sum(map(len, sampled.values())),
                "exported_actions": sum(map(len, exported.values())),
            }
            from examples.agent.calculator.train import gradient_check

            report["gradient_check"] = gradient_check(batch, config)
            if report["gradient_check"]["action_gradient_l1"] <= 0:
                raise RuntimeError(
                    "Real agent trajectories produced no action gradient."
                )
            if not any(
                e.score and e.score.success and len(e.turns) >= 2
                for e in batch.episodes
            ):
                raise RuntimeError(
                    "No successful real tool episode was observed in the final training batch."
                )
        report["phase"] = "checkpoint_verification"
        report["checkpoint"] = await verify_saved(
            service, state.checkpoint_path, evidence, after
        )
        if critic_before is not None:
            _, critic_id = await snapshot(audits["critic"])
            critic_after = await readout(audits["critic"], evidence)
            report["critic_after"] = {
                **critic_id,
                "version": state.global_step,
                **critic_after,
            }
            report["critic_noise_tolerance"] = max(
                1e-5,
                5
                * max(
                    critic_before["max_repeat_delta"], critic_after["max_repeat_delta"]
                ),
            )
            report["max_critic_logprob_delta"] = policy_change(
                critic_before["logprobs"],
                critic_after["logprobs"],
                noise=report["critic_noise_tolerance"] / 5,
            )
            if state.critic_optimizer_steps != config.max_steps:
                raise RuntimeError(
                    "PPO critic did not complete independent optimizer updates."
                )
            report["critic_checkpoint"] = await verify_saved(
                service, state.critic_checkpoint_path, evidence, critic_after
            )
        report["phase"] = "complete"
    finally:
        report["audit"] = {name: audit.summary() for name, audit in audits.items()}
        # Public train methods own reporter finalization. This also handles a
        # failure in witness sampling before train() acquired that lifecycle.
        if isinstance(trainer, PPOTrainer):
            await trainer.close()
            if isinstance(owner, AgentTrainingLoop):
                await owner.workflow.close()
        else:
            trainer._metrics_api.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Explicitly enable real remote sampling and training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New, empty output directory; never overwrites a run",
    )
    args = parser.parse_args(argv)
    if not args.remote:
        parser.error("Real training requires explicit --remote opt-in.")
    output = (
        args.output_dir
        or ROOT / "output" / "remote-baseline" / f"{args.recipe}-{uuid4().hex[:8]}"
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "phase": "preflight",
        "recipe": args.recipe,
        "started_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: package_version(p) for p in ("pytrio", "triotune", "torch", "openai")
            },
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "examples/agent/calculator/baseline.py",
            )
        },
    }
    started, cpu = time.perf_counter(), time.process_time()
    try:
        asyncio.run(run_recipe(args.recipe, output, report))
        report["status"] = "passed"
    except BaseException as error:
        report.update(
            status="failed",
            error={
                "type": type(error).__name__,
                "code": getattr(error, "code", None),
                "message": str(error),
            },
        )
        raise
    finally:
        report["resources"] = {
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            # The SDK exposes token/timing usage in metric history, but does
            # not expose remote GPU memory, GPU-hours, or billed currency.
            "remote_gpu_memory_bytes": None,
            "remote_gpu_seconds": None,
            "cost_currency": None,
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        print(
            f"{args.recipe}: {report['status']}; report: {output / 'report.json'}",
            flush=True,
        )
    return report


if __name__ == "__main__":
    main()
