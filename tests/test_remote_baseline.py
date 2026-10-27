"""Acceptance gates and failure paths; real training requires an explicit flag."""

import importlib.util
import json
import math
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytrio

from examples.agent.calculator.baseline import execute_calculator, make_workflow
from triotune.api.client_api import call_async, submit_async, wait_async

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "remote_baseline", ROOT / "tests/recipes/remote_baseline.py"
)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def datum(inputs=(10, 11, 12), targets=(-100, 12, 13), **extra):
    return pytrio.Datum(
        model_input=pytrio.ModelInput.from_ints(list(inputs)),
        loss_fn_inputs={"target_tokens": list(targets), **extra},
    )


class BaselineContractTests(unittest.TestCase):
    def test_fixed_recipes_cover_two_updates_and_disjoint_evaluation(self):
        for recipe in baseline.RECIPES:
            with self.subTest(recipe=recipe):
                _, config = baseline.load_recipe(recipe, Path("/tmp/baseline-unit"))
                self.assertEqual(config.max_steps, 2)
                self.assertEqual(config.seed, 42)
                self.assertEqual(config.batch_size, 2)
                self.assertEqual(config.model.lora_rank, 8)
                self.assertIsNone(config.report_to)
                self.assertTrue(config.eval_before_train)
                self.assertTrue(config.eval_after_train)
                self.assertEqual(
                    len(baseline.rows_for(recipe, "train")),
                    config.max_steps * config.batch_size,
                )
                self.assertEqual(len(baseline.rows_for(recipe, "eval")), 2)
                self.assertTrue(asdict(config))
                if recipe == "ppo":
                    self.assertTrue(config.use_critic)
                elif recipe == "opd":
                    self.assertNotEqual(config.teacher.name, config.model.name)
        paths = [
            ROOT / f"tests/recipes/baseline_{split}.jsonl"
            for split in ("train", "eval")
        ]
        train, valid = [
            [json.loads(line) for line in path.read_text().splitlines()]
            for path in paths
        ]
        self.assertFalse({r["task_id"] for r in train} & {r["task_id"] for r in valid})
        self.assertFalse(
            {r["question"] for r in train} & {r["question"] for r in valid}
        )

    def test_datum_audit_rejects_alignment_and_empty_target_failures(self):
        self.assertEqual(baseline.datum_record(datum())["input_ids"], [10, 11, 12])
        cases = (
            (datum(targets=(-100, 11, 12)), "causal"),
            (datum(targets=(-100, 12)), "lengths"),
            (datum(targets=(-100, -100, -100)), "no active"),
            (datum(advantages=[0.0]), "advantages"),
            (datum(advantages=[0.0, math.nan, 1.0]), "advantages"),
        )
        for bad, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                baseline.datum_record(bad)
        # Critic marker labels have the input length, including bootstrap, but
        # intentionally differ from the next sampled token.
        self.assertTrue(baseline.datum_record(datum(targets=(-100, 1, 1)), critic=True))

    def test_weight_gate_rejects_unchanged_nonfinite_and_missing_evidence(self):
        self.assertAlmostEqual(baseline.policy_change([-1.0], [-0.8]), 0.2)
        for before, after in (
            ([], []),
            ([-1], []),
            ([-1], [math.nan]),
            ([-1], [math.inf]),
            ([-1], [-1]),
        ):
            with (
                self.subTest(before=before, after=after),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                baseline.policy_change(before, after)
        with self.assertRaisesRegex(RuntimeError, "measured forward noise"):
            baseline.policy_change([-1], [-0.99], noise=0.01)

    def test_remote_cli_requires_explicit_opt_in(self):
        with (
            patch.object(baseline, "run_recipe") as remote,
            patch("sys.stderr"),
            self.assertRaises(SystemExit) as error,
        ):
            baseline.main(["--recipe", "sft"])
        self.assertEqual(error.exception.code, 2)
        remote.assert_not_called()

    def test_output_directory_cannot_overwrite_existing_evidence(self):
        with (
            tempfile.TemporaryDirectory() as output,
            patch.object(baseline, "run_recipe") as remote,
            self.assertRaises(FileExistsError),
        ):
            baseline.main(["--remote", "--recipe", "sft", "--output-dir", output])
        remote.assert_not_called()

    def test_failed_remote_phase_still_writes_a_resource_report(self):
        async def fail(recipe, output, report):
            report["phase"] = "client_initialization"
            raise ConnectionError("test remote unavailable")

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "failed"
            with (
                patch.object(baseline, "run_recipe", side_effect=fail),
                patch("builtins.print"),
                self.assertRaises(ConnectionError),
            ):
                baseline.main(
                    ["--remote", "--recipe", "sft", "--output-dir", str(output)]
                )
            report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], "client_initialization")
        self.assertEqual(report["error"]["type"], "ConnectionError")
        self.assertGreaterEqual(report["resources"]["wall_seconds"], 0)
        self.assertIsNone(report["resources"]["remote_gpu_seconds"])


class BaselineAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_readout_records_repeats_and_noise_before_comparing_weights(self):
        with patch.object(
            baseline,
            "forward_scores",
            new=AsyncMock(side_effect=[[-1.0, -2.0], [-0.9, -2.0], [-1.1, -2.0]]),
        ) as forward:
            result = await baseline.readout("client", {})
        self.assertEqual(forward.await_count, 3)
        self.assertEqual(result["logprobs"], [-1.0, -2.0])
        self.assertAlmostEqual(result["max_repeat_delta"], 0.2)
        self.assertEqual(len(result["repeat_logprobs"]), 3)

    async def test_audit_preserves_two_stage_sdk_futures(self):
        completed = []

        class Client:
            async def forward_backward_async(self, data, **kwargs):
                async def result():
                    completed.append("backward")
                    return "backward-result"

                return result()

            async def optim_step_async(self, params):
                async def result():
                    completed.append("optim")
                    return "optim-result"

                return result()

        with tempfile.TemporaryDirectory() as root:
            audited = baseline.AuditedClient(Client(), Path(root) / "datums.jsonl")
            pending = await submit_async(audited, "forward_backward", [datum()])
            self.assertEqual(completed, [])
            self.assertEqual(await wait_async(pending), "backward-result")
            self.assertEqual(
                await call_async(audited, "optim_step", object()), "optim-result"
            )
            self.assertEqual(
                audited.summary(),
                {
                    "optimizer_submissions": 1,
                    "backward_requests": 1,
                    "datums": 1,
                    "active_targets": 2,
                },
            )

    async def test_eval_audit_rejects_optimizer_or_step_mutation(self):
        for mutation in ("none", "optimizer", "step"):
            with self.subTest(mutation=mutation):
                audit = SimpleNamespace(optimizer_submissions=0)
                owner = SimpleNamespace(state=SimpleNamespace(global_step=0))

                async def evaluate(mutation=mutation, audit=audit, owner=owner):
                    if mutation == "optimizer":
                        audit.optimizer_submissions += 1
                    if mutation == "step":
                        owner.state.global_step += 1
                    return {"eval/loss": 0.5}

                owner.evaluate = evaluate
                report = {}
                baseline.observe_evaluation(
                    owner, "evaluate", {"policy": audit}, report
                )
                if mutation == "none":
                    await owner.evaluate()
                    self.assertEqual(report["evaluations"][0]["step"], 0)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Evaluation changed"):
                        await owner.evaluate()

    async def test_checkpoint_gate_uses_train_checkpoint_loader_and_compares_weights(
        self,
    ):
        service = SimpleNamespace(
            create_training_client_from_state_async=AsyncMock(return_value="loaded")
        )
        with patch.object(
            baseline,
            "snapshot",
            new=AsyncMock(
                return_value=(SimpleNamespace(sampling_run_cuid="saved"), {})
            ),
        ) as export:
            for actual, passes in (([-0.9], True), ([-0.8], False)):
                readout = {
                    "logprobs": actual,
                    "max_repeat_delta": 0.0,
                    "repeat_logprobs": [actual] * 3,
                }
                expected = {"logprobs": [-0.9], "max_repeat_delta": 0.0}
                with (
                    self.subTest(passes=passes),
                    patch.object(
                        baseline, "readout", new=AsyncMock(return_value=readout)
                    ),
                    patch.object(baseline, "score", new=AsyncMock(return_value=actual)),
                ):
                    if passes:
                        result = await baseline.verify_saved(
                            service, "trio://train", {}, expected
                        )
                        self.assertEqual(result["max_logprob_delta"], 0)
                        export.assert_awaited_with("loaded")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "differs"):
                            await baseline.verify_saved(
                                service, "trio://train", {}, expected
                            )
            with self.assertRaisesRegex(RuntimeError, "did not save"):
                await baseline.verify_saved(service, None, {}, [])
        service.create_training_client_from_state_async.assert_awaited_with(
            path="trio://train"
        )

    async def test_official_sdk_callback_preserves_tool_ids_and_error_observations(
        self,
    ):
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="calculate", arguments='{"operation":"divide","a":1,"b":0}'
            ),
        )
        first = SimpleNamespace(
            tool_calls=[call],
            model_dump=lambda **kwargs: {
                "role": "assistant",
                "tool_calls": [{"id": "call-1"}],
            },
        )
        final = SimpleNamespace(tool_calls=None, content="undefined")
        create = AsyncMock(
            side_effect=[
                SimpleNamespace(choices=[SimpleNamespace(message=m)])
                for m in (first, final)
            ]
        )
        sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        result = await execute_calculator(sdk, "What is 1 / 0?", model="test-model")
        self.assertEqual(result, "undefined")
        observation = create.call_args.kwargs["messages"][-1]
        self.assertEqual(observation["tool_call_id"], "call-1")
        self.assertIn("Division by zero", observation["content"])
        with self.assertRaisesRegex(ValueError, "native or sdk"):
            make_workflow("invalid", "test-model", Path("unused"))


@unittest.skipUnless(
    os.environ.get("TRIOTUNE_REMOTE_BASELINE") == "1",
    "Set TRIOTUNE_REMOTE_BASELINE=1 to enable real remote training.",
)
class RemoteBaselineTests(unittest.TestCase):
    def test_all_remote_training_recipes(self):
        # Each recipe uses its own ignored output directory. A failure remains
        # a failure (never a skip) once remote acceptance has been enabled.
        for recipe in baseline.RECIPES:
            with self.subTest(recipe=recipe):
                report = baseline.main(["--remote", "--recipe", recipe])
                self.assertEqual(report["status"], "passed")


if __name__ == "__main__":
    unittest.main()
