"""Exercise evaluation through PyTRIO-shaped responses and isolated shop sessions."""

import asyncio
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "examples/agent/webshop-opd/eval"
SPEC = importlib.util.spec_from_file_location(
    "webshop_opd_eval", EVAL / "beforetraining_eval.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)
from webshop_env import data_prepare
from webshop_env.data_prepare import cached_files, DataConfig, task_rows
from webshop_env.environment import SessionPool


class AgentTokenizer:
    """Keep inference-only tests independent of optional training dependencies."""

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text]

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(
            ""
            if token == 1 and skip_special_tokens
            else "<eos>"
            if token == 1
            else chr(token - 10)
            for token in tokens
        )

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        tokens = []
        for message in messages:
            tokens.extend(self.encode(f"<{message['role']}>" + message["content"]))
            tokens.append(1)
        if add_generation_prompt:
            tokens.extend(self.encode("<assistant>"))
        return tokens if tokenize else self.decode(tokens)


class FakeShop:
    """Minimal stateful shop with full and partial terminal rewards."""

    def __init__(self):
        self.page, self.product = "initial", None

    def get_available_actions(self):
        return {
            "has_search_bar": self.page == "initial",
            "clickables": {
                "initial": [],
                "results": ["a", "b"],
                "item": ["buy now"],
                "terminal": [],
            }[self.page],
        }

    def step(self, action):
        if action.startswith("search["):
            self.page = "results"
        elif action in {"click[a]", "click[b]"}:
            self.product, self.page = action[6:-1], "item"
        elif action == "click[buy now]":
            self.page = "terminal"
        done = self.page == "terminal"
        reward = (1.0 if self.product == "a" else 0.5) if done else 0.0
        return self.page, reward, done, None


class FakeBackend:
    def __init__(self):
        self.goals = [
            {"instruction_text": f"goal {i}", "query": "shoes"} for i in range(1601)
        ]
        self.created, self.released, self.closed = [], [], False

    def create(self, goal_index, rollout_id):
        shop = FakeShop()
        self.created.append((goal_index, rollout_id, shop))
        return shop, "initial"

    def release(self, env):
        self.released.append(env)

    def close(self):
        self.closed = True


class SamplingClient:
    """Only exposes inference; an accidental training call fails the test."""

    def __init__(self, *, failure=True, responses=None):
        self.tokenizer = AgentTokenizer()
        self.failure, self.responses = failure, responses
        self.calls = []
        self.counts = {}

    def get_tokenizer(self):
        return self.tokenizer

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.calls.append((prompt.to_ints(), num_samples, sampling_params))
        # The integration test uses three-turn budgets and the same seed map.
        attempt = ((sampling_params.seed - 42) // 3) % 4
        if attempt == 3 and self.failure:
            raise RuntimeError("sampling service unavailable")
        if attempt == 2:
            text = '{"final":"I succeeded"}'
        else:
            count = self.counts.get(attempt, 0)
            self.counts[attempt] = count + 1
            product = "a" if attempt == 0 else "b"
            actions = self.responses or [
                "search[shoes]",
                f"click[{product}]",
                "click[buy now]",
            ]
            text = json.dumps(
                {
                    "tool_calls": [
                        {"name": "webshop", "arguments": {"action": actions[count]}}
                    ]
                }
            )
        ids = self.tokenizer.encode(text) + [1]
        await asyncio.sleep(0)
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=ids, logprobs=[-0.5] * len(ids), stop_reason="stop"
                )
            ],
            input_tokens=len(prompt.to_ints()),
            output_tokens=len(ids),
        )


class ConfigurationTests(unittest.TestCase):
    def test_java_setup_selects_valid_jdk_and_rejects_missing_runtime(self):
        helper = EVAL / "scripts/setup-java.sh"
        command = (
            'REPO_ROOT="$1"; source "$2"; printf "%s\\n%s\\n" "$JAVA_HOME" "$JVM_PATH"'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "data/webshop-opd/java"
            other = root / "other-java"
            for jdk in (local, other):
                (jdk / "bin").mkdir(parents=True)
                (jdk / "lib/server").mkdir(parents=True)
                (jdk / "bin/java").write_text("#!/bin/sh\nexit 0\n")
                (jdk / "bin/java").chmod(0o755)
                (jdk / "lib/server/libjvm.so").touch()
            for supplied, selected in (
                (str(other), other),
                ("/missing/java", local),
                ("", local),
            ):
                with self.subTest(supplied=supplied):
                    result = subprocess.run(
                        ["bash", "-euc", command, "--", str(root), str(helper)],
                        env={
                            **os.environ,
                            "JAVA_HOME": supplied,
                            "JVM_PATH": "/missing/libjvm.so",
                        },
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertEqual(
                        result.stdout.splitlines(),
                        [str(selected), str(selected / "lib/server/libjvm.so")],
                    )
            (local / "lib/server/libjvm.so").unlink()
            result = subprocess.run(
                ["bash", "-euc", command, "--", str(root), str(helper)],
                env={**os.environ, "JAVA_HOME": "/missing/java"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Java 11 is required", result.stderr)
            result = subprocess.run(
                [
                    "bash",
                    "-euc",
                    'REPO_ROOT="$1"; source "$2" --check-config',
                    "--",
                    str(root),
                    str(helper),
                ],
                env={**os.environ, "JAVA_HOME": "/missing/java"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)

    def test_policies_share_exactly_200_validation_goals_and_sampling_settings(self):
        student = evaluation.load_experiment(EVAL / "configs/student.yaml")
        teacher = evaluation.load_experiment(EVAL / "configs/teacher.yaml")
        self.assertEqual(student.model, "Qwen/Qwen3.5-4B")
        self.assertEqual(teacher.model, "Qwen/Qwen3.8-27B")
        self.assertEqual(student.evaluation, teacher.evaluation)
        self.assertEqual(student.evaluation.rollout_log_interval, 20)
        self.assertEqual(student.data, teacher.data)
        self.assertEqual(student.source_dir, ROOT / "data/webshop-opd/WebShop")
        self.assertEqual(student.data.directory, ROOT / "data/webshop-opd/assets")
        self.assertEqual(SessionPool.__module__, "webshop_env.environment")
        rows = task_rows([{}] * 2000, student.data, "eval")
        self.assertEqual([row["goal_index"] for row in rows], list(range(500, 700)))
        self.assertEqual(len(rows), 200)
        self.assertIn(student.evaluation.pass_k, (1, 4))
        for name in ("student", "teacher"):
            result = subprocess.run(
                ["bash", str(EVAL / f"scripts/run-{name}-eval.sh"), "--check-config"],
                cwd=tempfile.gettempdir(),
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = json.loads(result.stdout)
            self.assertEqual(parsed["role"], name)
            self.assertEqual(parsed["evaluation"]["pass_k"], student.evaluation.pass_k)
            overridden = subprocess.run(
                [
                    "bash",
                    str(EVAL / f"scripts/run-{name}-eval.sh"),
                    "--pass-k",
                    "4",
                    "--check-config",
                ],
                cwd=tempfile.gettempdir(),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(overridden.stdout)["evaluation"]["pass_k"], 4)

    def test_invalid_sampling_and_insufficient_attempts_fail_before_remote_work(self):
        for settings in (
            {"pass_k": 0},
            {"pass_k": 2},
            {"pass_k": True},
            {"max_turns": 0},
            {"seed": True},
            {"temperature": float("nan")},
            {"top_p": 1.1},
            {"sampling_timeout": -1},
            {"rollout_log_interval": -1},
            {"rollout_log_interval": True},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                evaluation.EvaluationSettings(**settings)

    def test_prepare_cli_uses_local_example_without_opening_model_service(self):
        argv = [
            "beforetraining_eval.py",
            "--config",
            str(EVAL / "configs/student.yaml"),
            "--prepare-data",
            "--download",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(evaluation, "prepare", return_value={}) as prepare,
            patch.object(evaluation.pytrio, "ServiceClient") as service,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            evaluation.main()
            prepare.assert_called_once()
            self.assertTrue(prepare.call_args.kwargs["download"])
            self.assertEqual(
                prepare.call_args.args[0].data.directory,
                ROOT / "data/webshop-opd/assets",
            )
            service.assert_not_called()


class DataTests(unittest.TestCase):
    def test_offline_assets_are_read_only_from_configured_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = DataConfig(directory=Path(directory))
            with patch.object(data_prepare, "hf_hub_download") as download:
                with self.assertRaisesRegex(FileNotFoundError, "prepare-data"):
                    cached_files(config)
                for name in (
                    "items_shuffle_1000.json",
                    "items_ins_v2_1000.json",
                    "items_human_ins.json",
                ):
                    (config.directory / name).write_text("[]")
                files = cached_files(config)
                self.assertEqual(len(files), 3)
                self.assertTrue(
                    all(path.parent == config.directory for path in files.values())
                )
                download.assert_not_called()

    def test_downloads_use_local_data_directory_and_pinned_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            config = DataConfig(directory=Path(directory), local_files_only=False)

            def download(**kwargs):
                self.assertEqual(kwargs["local_dir"], config.directory)
                self.assertEqual(kwargs["revision"], config.revision)
                destination = kwargs["local_dir"] / kwargs["filename"]
                destination.write_text("[]")
                return str(destination)

            with patch.object(
                data_prepare, "hf_hub_download", side_effect=download
            ) as mocked:
                self.assertEqual(len(cached_files(config)), 3)
                self.assertEqual(mocked.call_count, 3)
            with patch.object(
                data_prepare,
                "hf_hub_download",
                return_value="/unexpected/global-cache/file.json",
            ):
                with self.assertRaisesRegex(ValueError, "data.directory"):
                    cached_files(config)

    def test_prepare_builds_local_index_reuses_manifest_and_preserves_conflicting_files(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "WebShop"
            (source / "web_agent_site").mkdir(parents=True)
            (source / "search_engine").mkdir()
            assets = root / "assets"
            assets.mkdir()
            for name in (
                "items_shuffle_1000.json",
                "items_ins_v2_1000.json",
                "items_human_ins.json",
            ):
                (assets / name).write_text("[]")
            experiment = SimpleNamespace(
                source_dir=source,
                data=DataConfig(directory=assets),
                environment_seed=42,
            )
            engine = SimpleNamespace(
                __file__=str(source / "web_agent_site/engine/engine.py"),
                load_products=lambda **kwargs: (
                    [
                        {
                            "asin": "a",
                            "Title": "Shoes",
                            "Description": "Shoes",
                            "BulletPoints": ["Blue"],
                        }
                    ],
                ),
            )

            def build_index(command, **kwargs):
                Path(command[command.index("--index") + 1]).mkdir()

            with (
                patch.object(
                    data_prepare.subprocess,
                    "check_output",
                    return_value=data_prepare.WEBSHOP_REVISION,
                ),
                patch.object(
                    data_prepare.importlib, "import_module", return_value=engine
                ),
                patch.object(
                    data_prepare.subprocess, "run", side_effect=build_index
                ) as indexer,
                patch.object(sys, "path", sys.path.copy()),
            ):
                manifest = data_prepare.prepare(experiment)
                self.assertTrue((source / "search_engine/indexes_1k").is_dir())
                self.assertEqual(data_prepare.prepare(experiment), manifest)
                self.assertEqual(indexer.call_count, 1)
                for path in cached_files(experiment.data).values():
                    self.assertEqual((source / "data" / path.name).resolve(), path)
                (assets / "items_human_ins.json").write_text("[1]")
                with self.assertRaisesRegex(FileExistsError, "unknown/different"):
                    data_prepare.prepare(experiment)
            unrelated = source / "data/unrelated.json"
            unrelated.write_text("keep existing data")
            with self.assertRaises(FileExistsError):
                data_prepare.link_assets(
                    source, {"unrelated.json": assets / "items_human_ins.json"}
                )
            self.assertEqual(unrelated.read_text(), "keep existing data")


class MetricTests(unittest.TestCase):
    def test_purchase_quality_uses_actual_counts_and_preserves_exact_fraction(self):
        attempts = [
            evaluation.Attempt(
                "x",
                500,
                index,
                42,
                purchased=index < 79,
                success=index < 53,
                task_score=1.0 if index < 53 else 0.5 if index < 79 else 0.0,
            )
            for index in range(200)
        ]
        metrics = evaluation.purchase_metrics(attempts)
        self.assertEqual(metrics["purchase_count"], 79)
        self.assertEqual(metrics["successful_purchase_count"], 53)
        self.assertEqual(metrics["purchase_success_rate"], 53 / 79)
        self.assertEqual(metrics["purchase_success_fraction"], "53/79")
        encoded = json.loads(json.dumps(metrics, allow_nan=False))
        self.assertEqual(encoded, metrics)
        zero = evaluation.purchase_metrics([evaluation.Attempt("x", 500, 0, 42)])
        self.assertEqual(zero["purchase_count"], 0)
        self.assertEqual(zero["successful_purchase_count"], 0)
        self.assertIsNone(zero["purchase_success_rate"])
        self.assertIsNone(zero["purchase_success_fraction"])

    def test_pass_metrics_include_partial_rewards_and_failed_attempts(self):
        rows = [{"task_id": str(goal), "goal_index": goal} for goal in (500, 501)]
        attempts = [
            evaluation.Attempt(
                str(goal),
                goal,
                index,
                42,
                success=goal == 500 and index in (0, 2),
                task_score=1.0 if goal == 500 and index in (0, 2) else 0.5,
            )
            for goal in (500, 501)
            for index in range(4)
        ]
        attempts[-1].status = "sampling_failure"
        attempts[-1].task_score = 0
        metrics = evaluation.summarize(attempts, rows, 4)
        self.assertNotIn("pass@1", metrics)
        self.assertEqual(metrics["average_single_attempt_success_rate"], 0.25)
        self.assertEqual(metrics["first_attempt_success_rate"], 0.5)
        self.assertEqual(metrics["pass@4"], 0.5)
        self.assertEqual(metrics["task_score_mean"], 0.5625)
        self.assertEqual(metrics["system_failure_rate"], 1 / 8)
        with self.assertRaisesRegex(ValueError, "all requested"):
            evaluation.summarize(attempts[:-1], rows, 4)
        attempts[-1].attempt_index = 0
        with self.assertRaisesRegex(ValueError, "all requested"):
            evaluation.summarize(attempts, rows, 4)

    def test_selected_pass_k_has_only_its_own_metric(self):
        row = {"task_id": "x", "goal_index": 500}
        for k in (1, 4):
            with self.subTest(k=k):
                attempts = [
                    evaluation.Attempt("x", 500, index, 42) for index in range(k)
                ]
                self.assertEqual(
                    evaluation.summarize(attempts, [row], k)[f"pass@{k}"], 0
                )
                attempts[0].success = True
                metrics = evaluation.summarize(attempts, [row], k)
                self.assertEqual(metrics[f"pass@{k}"], 1)
                self.assertEqual(metrics["average_single_attempt_success_rate"], 1 / k)
                self.assertNotIn(f"pass@{4 if k == 1 else 1}", metrics)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_counts_tasks_after_all_attempts_even_with_failures(self):
        rows = [{"task_id": str(goal), "goal_index": goal} for goal in (500, 501)]
        for k in (1, 4):
            completed = []
            advanced = []

            async def attempt(row, index, *args):
                await asyncio.sleep(0.001 if index == 0 else 0)
                completed.append((row["goal_index"], index))
                return evaluation.Attempt(
                    row["task_id"],
                    row["goal_index"],
                    index,
                    42,
                    status="sampling_failure",
                )

            def update(amount):
                self.assertEqual(amount, 1)
                fully_done = {
                    goal
                    for goal in (500, 501)
                    if sum(g == goal for g, _ in completed) == k
                }
                self.assertEqual(len(fully_done), len(advanced) + 1)
                advanced.append(fully_done.copy())

            with (
                self.subTest(k=k),
                tempfile.TemporaryDirectory() as directory,
                patch.object(evaluation, "evaluate_attempt", side_effect=attempt),
                patch.object(evaluation, "tqdm") as factory,
            ):
                progress = factory.return_value.__enter__.return_value
                progress.update.side_effect = update
                results = await evaluation.collect(
                    rows,
                    None,
                    None,
                    None,
                    evaluation.EvaluationSettings(pass_k=k, rollout_log_interval=0),
                    Path(directory) / "episodes.jsonl",
                )
                self.assertEqual(factory.call_args.kwargs["total"], 2)
                self.assertEqual(factory.call_args.kwargs["unit"], "task")
                self.assertEqual(progress.update.call_count, 2)
                self.assertEqual(len(results), 2 * k)

    async def test_context_limit_malformed_response_and_truncation_never_execute_actions(
        self,
    ):
        backend, client = FakeBackend(), SamplingClient()
        pool = SessionPool(backend)
        row = {"task_id": "x", "goal_index": 500}
        try:
            result = await evaluation.evaluate_attempt(
                row,
                0,
                client,
                client.tokenizer,
                pool,
                evaluation.EvaluationSettings(max_context_tokens=1),
            )
            self.assertEqual(result.status, "context_limit")
            self.assertEqual(client.calls, [])
            for text, reason, expected in (
                ("I bought the right product", "stop", "invalid_action"),
                ('{"final":"done"}', "length", "token_limit"),
            ):

                async def sample(*args, **kwargs):
                    ids = tuple(client.tokenizer.encode(text))
                    return evaluation.Generation(ids, (-1.0,) * len(ids), text, reason)

                with patch.object(evaluation, "sample_action", sample):
                    result = await evaluation.evaluate_attempt(
                        row,
                        0,
                        client,
                        client.tokenizer,
                        pool,
                        evaluation.EvaluationSettings(),
                    )
                self.assertEqual(result.status, expected)
                self.assertEqual(result.environment_steps, 0)
                self.assertFalse(result.success)
                self.assertFalse(pool.sessions)
        finally:
            await pool.close()

    async def test_purchase_on_last_turn_partial_reward_false_claim_and_sampling_error(
        self,
    ):
        backend, client = FakeBackend(), SamplingClient()
        pool = SessionPool(backend)
        settings = evaluation.EvaluationSettings(
            pass_k=4, max_turns=3, rollout_log_interval=1
        )
        rows = [{"task_id": "webshop-eval-500", "goal_index": 500}]
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episodes.jsonl"
                with patch("sys.stderr", new_callable=io.StringIO) as progress:
                    attempts = await evaluation.collect(
                        rows, client, client.tokenizer, pool, settings, path
                    )
                self.assertIn("4/4", progress.getvalue())
                self.assertIn("1/1 [", progress.getvalue())
                self.assertIn("success=25.0%", progress.getvalue())
                self.assertNotIn("[eval step=", progress.getvalue())
                self.assertEqual(
                    re.findall(r"\[rollout task=(\d+)/1\]", progress.getvalue()),
                    ["1"],
                )
                self.assertNotIn("kl=N/A", progress.getvalue())
                self.assertNotIn("loss=N/A", progress.getvalue())
                self.assertIn('"initial_state":', progress.getvalue())
                self.assertIn('"response":', progress.getvalue())
                records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [item.status for item in attempts],
                [
                    "completed",
                    "completed",
                    "final_without_purchase",
                    "sampling_failure",
                ],
            )
            self.assertTrue(attempts[0].success)
            self.assertEqual(attempts[0].environment_steps, 3)
            self.assertFalse(attempts[1].success)
            self.assertEqual(attempts[1].task_score, 0.5)
            self.assertFalse(attempts[2].success)
            metrics = evaluation.summarize(attempts, rows, 4)
            self.assertNotIn("pass@1", metrics)
            self.assertEqual(metrics["pass@4"], 1)
            self.assertEqual(metrics["task_score_mean"], 0.375)
            self.assertEqual(len(records), 4)
            self.assertTrue(all(record["trace"] for record in records))
            self.assertEqual(len(backend.created), 4)
            self.assertEqual(len(backend.released), 4)
            self.assertFalse(pool.sessions)
            for _, samples, params in client.calls:
                self.assertEqual(samples, 1)
                self.assertEqual(params.temperature, 0.4)
                self.assertEqual(params.top_p, 1.0)
                self.assertEqual(params.max_tokens, 512)
        finally:
            await pool.close()

    async def test_invalid_tool_action_consumes_turn_and_timeout_releases_session(self):
        backend, client = (
            FakeBackend(),
            SamplingClient(responses=["click[missing]", "search[shoes]"]),
        )
        pool = SessionPool(backend)
        row = {"task_id": "x", "goal_index": 500}
        try:
            result = await evaluation.evaluate_attempt(
                row,
                0,
                client,
                client.tokenizer,
                pool,
                evaluation.EvaluationSettings(max_turns=2),
            )
            self.assertEqual(result.status, "max_turns")
            self.assertEqual(result.environment_steps, 2)
            self.assertEqual(result.invalid_actions, 1)
            self.assertIn("tool_error", result.trace[1])
            self.assertFalse(result.success)

            async def stalled(*args, **kwargs):
                await asyncio.sleep(10)

            with patch.object(evaluation, "sample_action", stalled):
                result = await evaluation.evaluate_attempt(
                    row,
                    0,
                    client,
                    client.tokenizer,
                    pool,
                    evaluation.EvaluationSettings(sampling_timeout=0.01),
                )
            self.assertEqual(result.status, "timeout")
            self.assertFalse(pool.sessions)
        finally:
            await pool.close()

    async def test_run_uses_only_base_sampling_client_and_writes_auditable_report(self):
        backend, client = FakeBackend(), SamplingClient(failure=False)
        service = SimpleNamespace(create_sampling_client=lambda **kwargs: client)
        config = evaluation.load_experiment(EVAL / "configs/student.yaml")
        config = replace(
            config,
            data=replace(config.data, eval_limit=1),
            evaluation=replace(
                config.evaluation,
                pass_k=1,
                max_turns=3,
                rollout_log_interval=0,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            config = replace(config, output_dir=Path(directory))
            with (
                patch.object(evaluation, "cached_files", return_value={}),
                patch.object(evaluation, "asset_manifest", return_value={}),
                patch.object(evaluation, "WebShopBackend", return_value=backend),
                patch.object(evaluation.pytrio, "ServiceClient", return_value=service),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                report = await evaluation.run(config)
            self.assertIn("student: Qwen/Qwen3.5-4B", stderr.getvalue())
            self.assertNotIn("[eval step=", stderr.getvalue())
            self.assertNotIn("[rollout step=", stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "completed")
            saved = json.loads((Path(report["output_dir"]) / "report.json").read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["goal_indices"], [500])
            self.assertEqual(saved["metrics"]["attempts"], 1)
            self.assertTrue(backend.closed)
            self.assertEqual(saved["config"]["model"], "Qwen/Qwen3.5-4B")


if __name__ == "__main__":
    unittest.main()
