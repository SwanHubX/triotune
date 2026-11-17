"""Evaluate an untouched PyTRIO policy on fixed WebShop validation goals.

Each attempt is an independent full shopping episode, with one sampled action
per turn. No training client, teacher scoring, or optimizer is constructed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

# Allow direct execution despite the hyphenated example directory name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from webshop_bootstrap import require_repository_triotune

require_repository_triotune()

import pytrio
from tqdm import tqdm

from webshop_env.data_prepare import (
    DataConfig,
    asset_manifest,
    cached_files,
    goal_fingerprint,
    prepare,
    task_rows,
)
from webshop_env.environment import (
    INSTRUCTIONS,
    SessionPool,
    SingleActionParser,
    WebShopBackend,
)
from triotune.agentrl import (
    ActionParseError,
    ChatTemplateRenderer,
    Generation,
    ToolActionError,
)
from triotune.api.client_api import call_async
from triotune.utils.config import load_yaml_config
from triotune.utils.token_usage import compute_sample_tokens


@dataclass(frozen=True)
class EvaluationSettings:
    pass_k: int = 1
    max_turns: int = 15
    max_tokens_per_turn: int = 512
    max_context_tokens: int = 10240
    temperature: float = 0.4
    top_p: float = 1.0
    enable_thinking: bool = False
    seed: int = 42
    max_concurrency: int = 4
    rollout_log_interval: int = 20
    sampling_timeout: float = 120
    tool_timeout: float = 60
    episode_timeout: float = 1800

    def __post_init__(self):
        if type(self.enable_thinking) is not bool:
            raise ValueError("evaluation.enable_thinking must be boolean.")
        if type(self.rollout_log_interval) is not int or self.rollout_log_interval < 0:
            raise ValueError(
                "evaluation.rollout_log_interval must be a nonnegative integer."
            )
        for name in (
            "pass_k",
            "max_turns",
            "max_tokens_per_turn",
            "max_context_tokens",
            "max_concurrency",
            "seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(
                    f"evaluation.{name} must be a valid positive integer (seed may be zero)."
                )
        if self.pass_k not in {1, 4}:
            raise ValueError("evaluation.pass_k must be 1 or 4.")
        for name in (
            "temperature",
            "top_p",
            "sampling_timeout",
            "tool_timeout",
            "episode_timeout",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"evaluation.{name} must be finite and positive.")
        if self.top_p > 1:
            raise ValueError("evaluation.top_p must be <= 1.")


@dataclass(frozen=True)
class Experiment:
    model: str
    role: str
    data: DataConfig
    source_dir: Path
    environment_seed: int
    evaluation: EvaluationSettings
    output_dir: Path


def load_experiment(path: Path) -> Experiment:
    values = load_yaml_config(path)
    unknown = set(values) - {
        "model",
        "role",
        "data",
        "environment",
        "evaluation",
        "output_dir",
    }
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
    if values.get("role") not in {"student", "teacher"}:
        raise ValueError("role must be student or teacher.")
    model = values.get("model", {})
    if (
        not isinstance(model, dict)
        or set(model) != {"name"}
        or not isinstance(model["name"], str)
        or not model["name"].strip()
    ):
        raise ValueError("model must contain a nonempty base-model name only.")
    data_values = dict(values.get("data", {}))
    if "directory" in data_values:
        directory = data_values["directory"]
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("data.directory must be a nonempty path.")
        data_values["directory"] = (
            path.resolve().parent / Path(directory).expanduser()
        ).resolve()
    data = DataConfig(**data_values)
    if data.split_protocol != "official" or data.eval_limit is None:
        raise ValueError("Use split_protocol=official and an explicit data.eval_limit.")
    environment = dict(values.get("environment", {}))
    source = environment.pop("source_dir", None)
    seed = environment.pop("seed", 42)
    if (
        environment
        or not isinstance(source, str)
        or not source.strip()
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError(
            "environment requires source_dir and a nonnegative integer seed."
        )
    output = values.get("output_dir")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("output_dir must be a nonempty path.")

    def resolve(value):
        target = Path(value).expanduser()
        return (path.resolve().parent / target).resolve()

    return Experiment(
        model["name"].strip(),
        values["role"],
        data,
        resolve(source),
        seed,
        EvaluationSettings(**values.get("evaluation", {})),
        resolve(output),
    )


@dataclass
class Attempt:
    task_id: str
    goal_index: int
    attempt_index: int
    seed: int
    status: str = "max_turns"
    success: bool = False
    purchased: bool = False
    task_score: float = 0.0
    model_turns: int = 0
    environment_steps: int = 0
    invalid_actions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)


async def sample_action(client, tokenizer, prompt, settings, seed):
    """Sample exactly one action at evaluation temperature using original IDs."""
    budget = min(
        settings.max_tokens_per_turn, settings.max_context_tokens - len(prompt)
    )
    response = await call_async(
        client,
        "sample",
        prompt=pytrio.ModelInput.from_ints(prompt),
        num_samples=1,
        sampling_params=pytrio.SamplingParams(
            max_tokens=budget,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=seed,
        ),
    )
    if len(response.sequences) != 1:
        raise ValueError("Expected exactly one sampled action per turn.")
    sequence = response.sequences[0]
    if len(sequence.tokens) > budget:
        raise ValueError("Sampling exceeded the requested token budget.")
    usage = compute_sample_tokens(response, prompt_tokens=len(prompt))
    reason = sequence.stop_reason
    reason = getattr(reason, "value", reason)
    if reason not in {
        "length",
        "max_tokens",
        "max_length",
        "stop",
        "eos",
        "stop_token",
    }:
        reason = "length" if len(sequence.tokens) >= budget else "stop"
    return Generation(
        tuple(sequence.tokens),
        tuple(sequence.logprobs),
        tokenizer.decode(sequence.tokens, skip_special_tokens=True),
        reason,
        usage.input_tokens,
        usage.output_tokens,
    )


async def evaluate_attempt(row, attempt_index, client, tokenizer, pool, settings):
    """Stop at terminal purchase, including a purchase on the final allowed turn.

    WebShop success comes from the environment's terminal reward, not a model's
    final claim. An extra JSON final acknowledgement is unnecessary for eval.
    Invalid tool actions consume one turn and receive visible error feedback.
    """
    seed = settings.seed + (row["goal_index"] * 4 + attempt_index) * settings.max_turns
    result = Attempt(row["task_id"], row["goal_index"], attempt_index, seed)
    identity = uuid4().hex
    renderer = ChatTemplateRenderer(
        tokenizer, advertise_tools=False, enable_thinking=settings.enable_thinking
    )
    parser = SingleActionParser(allow_plain_text=False)
    stage = "environment"
    try:
        async with asyncio.timeout(settings.episode_timeout):
            visible = await asyncio.wait_for(
                pool.open(identity, row["goal_index"]), settings.tool_timeout
            )
            messages = [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
            ]
            result.trace.append({"initial_state": visible})
            for turn in range(settings.max_turns):
                stage = "rendering"
                prompt = await asyncio.to_thread(renderer.render, messages, [])
                if len(prompt) >= settings.max_context_tokens:
                    result.status = "context_limit"
                    break
                stage = "sampling"
                generation = await asyncio.wait_for(
                    sample_action(client, tokenizer, prompt, settings, seed + turn),
                    settings.sampling_timeout,
                )
                result.model_turns += 1
                result.input_tokens += generation.input_tokens
                result.output_tokens += generation.output_tokens
                record = {
                    "turn": turn,
                    "seed": seed + turn,
                    "response": generation.text,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                }
                result.trace.append(record)
                if generation.stop_reason in {"length", "max_tokens", "max_length"}:
                    result.status = "token_limit"
                    break
                try:
                    action = parser.parse(generation)
                    if not action.tool_calls:
                        result.status = "final_without_purchase"
                        break
                    call = action.tool_calls[0]
                    if call.name != "webshop" or set(call.arguments) != {"action"}:
                        raise ActionParseError(
                            "Use one webshop call with an action argument."
                        )
                except ActionParseError as error:
                    result.invalid_actions += 1
                    result.status, result.error = "invalid_action", str(error)
                    break
                stage = "environment"
                result.environment_steps += 1
                record["action"] = call.arguments["action"]
                try:
                    visible = await asyncio.wait_for(
                        pool.step(identity, call.arguments["action"]),
                        settings.tool_timeout,
                    )
                    record["state"] = visible
                except ToolActionError as error:
                    result.invalid_actions += 1
                    record["tool_error"] = str(error)
                    visible = {
                        "error": str(error),
                        "state": await pool.observe(identity),
                    }
                done, score = await pool.outcome(identity)
                result.purchased, result.task_score = done, score
                if done:
                    result.status, result.success = "completed", score == 1.0
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": generation.text},
                        {
                            "role": "user",
                            "content": json.dumps(visible, ensure_ascii=False),
                        },
                    ]
                )
    except TimeoutError as error:
        result.status, result.error = "timeout", f"{stage}: {error}"
    except Exception as error:
        result.status, result.error = (
            f"{stage}_failure",
            f"{type(error).__name__}: {error}",
        )
    finally:
        await asyncio.shield(pool.release(identity))
    return result


def purchase_metrics(attempts: list[Attempt]) -> dict:
    """Count purchased attempts directly, preserving an exact integer ratio."""
    purchased = sum(item.purchased for item in attempts)
    successful = sum(
        item.purchased and item.success and item.task_score == 1.0 for item in attempts
    )
    return {
        "purchase_count": purchased,
        "successful_purchase_count": successful,
        "purchase_success_rate": successful / purchased if purchased else None,
        "purchase_success_fraction": f"{successful}/{purchased}" if purchased else None,
    }


def summarize(attempts: list[Attempt], rows: list[dict], pass_k: int) -> dict:
    """Use all attempts, including failures, with explicit task denominators."""
    if not rows or len({row["goal_index"] for row in rows}) != len(rows):
        raise ValueError("Evaluation requires distinct, nonempty task rows.")
    groups = defaultdict(list)
    for item in attempts:
        if not math.isfinite(item.task_score) or not 0 <= item.task_score <= 1:
            raise ValueError("Task scores must be finite and in [0,1].")
        groups[item.goal_index].append(item)
    if set(groups) != {row["goal_index"] for row in rows} or any(
        len(group) != pass_k
        or {item.attempt_index for item in group} != set(range(pass_k))
        for group in groups.values()
    ):
        raise ValueError("Every task must retain all requested independent attempts.")
    if pass_k not in {1, 4}:
        raise ValueError("pass_k must be 1 or 4.")
    metrics = {
        f"pass@{pass_k}": sum(
            any(item.success for item in group) for group in groups.values()
        )
        / len(groups)
    }
    count = len(attempts)
    statuses = Counter(item.status for item in attempts)
    metrics.update(
        tasks=len(rows),
        attempts=count,
        pass_k=pass_k,
        average_single_attempt_success_rate=sum(item.success for item in attempts)
        / count,
        first_attempt_success_rate=sum(
            item.success for item in attempts if item.attempt_index == 0
        )
        / len(rows),
        task_score_mean=math.fsum(item.task_score for item in attempts) / count,
        score=100 * math.fsum(item.task_score for item in attempts) / count,
        purchase_rate=sum(item.purchased for item in attempts) / count,
        environment_steps_mean=sum(item.environment_steps for item in attempts) / count,
        model_turns_mean=sum(item.model_turns for item in attempts) / count,
        invalid_action_rate=sum(item.invalid_actions for item in attempts)
        / max(1, sum(item.model_turns for item in attempts)),
        input_tokens=sum(item.input_tokens for item in attempts),
        output_tokens=sum(item.output_tokens for item in attempts),
        system_failure_rate=sum(
            number
            for status, number in statuses.items()
            if status.endswith("_failure") or status == "timeout"
        )
        / count,
        status_counts=dict(sorted(statuses.items())),
    )
    metrics.update(purchase_metrics(attempts))
    return metrics


async def collect(
    rows, client, tokenizer, pool, settings, episode_path, *, description="WebShop eval"
):
    """Bound workers and persist each completed attempt before aggregation."""
    pending = iter((row, index) for row in rows for index in range(settings.pass_k))
    results = []
    successes, completed_tasks = 0, 0
    task_attempts = defaultdict(list)
    total_attempts = len(rows) * settings.pass_k
    with (
        episode_path.open("x", encoding="utf-8") as stream,
        tqdm(
            total=len(rows),
            desc=description,
            unit="task",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as progress,
    ):

        async def worker():
            nonlocal successes, completed_tasks
            for row, index in pending:
                item = await evaluate_attempt(
                    row, index, client, tokenizer, pool, settings
                )
                stream.write(
                    json.dumps(asdict(item), ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                results.append(replace(item, trace=[]))
                successes += item.success
                task_attempts[item.goal_index].append(item)
                finished = len(task_attempts[item.goal_index]) == settings.pass_k
                if finished:
                    completed_tasks += 1
                progress.set_postfix(
                    rollouts=f"{len(results)}/{total_attempts}",
                    success=f"{successes / len(results):.1%}",
                    refresh=False,
                )
                if finished:
                    progress.update(1)
                    if (
                        settings.rollout_log_interval
                        and completed_tasks % settings.rollout_log_interval == 0
                    ):
                        progress.write(
                            f"[rollout task={completed_tasks}/{len(rows)}]\n"
                            + json.dumps(
                                [
                                    asdict(attempt)
                                    for attempt in sorted(
                                        task_attempts[item.goal_index],
                                        key=lambda attempt: attempt.attempt_index,
                                    )
                                ],
                                indent=2,
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                            file=sys.stderr,
                        )
                    # Full traces have already been persisted; release them
                    # once this task's optional terminal output is complete.
                    del task_attempts[item.goal_index]

        async with asyncio.TaskGroup() as group:
            for _ in range(min(settings.max_concurrency, len(rows) * settings.pass_k)):
                group.create_task(worker())
    return sorted(results, key=lambda item: (item.goal_index, item.attempt_index))


async def run(experiment):
    started = time.perf_counter()
    # Each run owns a unique directory; reruns cannot mix attempts or overwrite evidence.
    output = (
        experiment.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "config": asdict(experiment),
        "output_dir": str(output),
    }
    pool = None
    try:
        files = cached_files(experiment.data)
        backend = await asyncio.to_thread(
            WebShopBackend, experiment.source_dir, files, experiment.environment_seed
        )
        pool = SessionPool(backend)
        rows = task_rows(backend.goals, experiment.data, "eval")
        if len(rows) != experiment.data.eval_limit:
            raise ValueError(
                "The validation split has fewer tasks than data.eval_limit."
            )
        report.update(
            split="eval",
            assets=asset_manifest(files),
            goal_sha256=goal_fingerprint(backend.goals),
            goal_indices=[row["goal_index"] for row in rows],
            metric_definition=f"Pass@{experiment.evaluation.pass_k} = fraction of tasks with at least one success in {experiment.evaluation.pass_k} independent attempts; success = terminal WebShop score exactly 1.",
        )
        service = await asyncio.to_thread(pytrio.ServiceClient)
        client = await call_async(
            service, "create_sampling_client", base_model=experiment.model
        )
        tokenizer = await asyncio.to_thread(client.get_tokenizer)
        attempts = await collect(
            rows,
            client,
            tokenizer,
            pool,
            experiment.evaluation,
            output / "episodes.jsonl",
            description=f"{experiment.role}: {experiment.model}",
        )
        report["metrics"] = summarize(attempts, rows, experiment.evaluation.pass_k)
        report["status"] = (
            "completed_with_errors"
            if report["metrics"]["system_failure_rate"]
            else "completed"
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if pool is not None:
                await pool.close()
        except BaseException as error:
            report["status"] = "failed"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            (output / "report.json").write_text(
                json.dumps(
                    report, indent=2, ensure_ascii=False, allow_nan=False, default=str
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"report": str(output / "report.json"), "status": report["status"]}
                )
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--pass-k",
        type=int,
        choices=(1, 4),
        help="Override the selected Pass@k evaluation (one or four attempts per task).",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare this example's local assets and search index without model calls.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow data preparation to download assets into data.directory.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate settings without opening WebShop or PyTRIO.",
    )
    args = parser.parse_args()
    if args.download and not args.prepare_data:
        parser.error("--download requires --prepare-data.")
    if args.check_config and args.prepare_data:
        parser.error("Use either --check-config or --prepare-data.")
    experiment = load_experiment(args.config)
    if args.pass_k is not None:
        experiment = replace(
            experiment, evaluation=replace(experiment.evaluation, pass_k=args.pass_k)
        )
    if args.check_config:
        print(json.dumps(asdict(experiment), indent=2, default=str))
        return
    if args.prepare_data:
        print(json.dumps(prepare(experiment, download=args.download), indent=2))
        return
    report = asyncio.run(run(experiment))
    if report["status"] != "completed":
        raise SystemExit(
            "Evaluation includes infrastructure failures; inspect the report before comparing policies."
        )


if __name__ == "__main__":
    main()
