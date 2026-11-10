"""Strict contracts for the isolated multi-turn AgentRL OPD prototype."""

from __future__ import annotations

import unittest
import inspect
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytrio

from agent_test_support import AgentTokenizer, CalculatorSampler
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_opd_trainer import _Service

from triotune.agentrl import (
    AgentConfig,
    AgentContractError,
    AgentOPDTrainingLoop,
    AgentTurn,
    EpisodeResult,
    EpisodeScore,
    Generation,
    SamplingSnapshot,
    current_execution_context,
    episode_turn_rollouts,
)
from triotune.trainer import OPDConfig, OPDTrainer
from train.vanilla_opd import DEFAULT_CONFIG, load_experiment
from webshop_env.environment import build_workflow


def recorded_episode(*, version=0, status="completed", score=True):
    """Build two turns whose second prompt contains an environment observation."""

    return EpisodeResult(
        "episode",
        "group",
        "snapshot",
        turns=[
            AgentTurn((10, 11), Generation((20, 21), (-2.0, -2.5), "first")),
            AgentTurn(
                # Tokens 90 and 91 represent tool-owned context.  They must be
                # visible to teacher scoring and absent from active OPD labels.
                (10, 11, 20, 21, 90, 91),
                Generation((30, 31), (-3.0, -3.5), "second"),
            ),
        ],
        status=status,
        score=EpisodeScore(1.0, True) if score else None,
        task_id="task",
        sample_index=0,
        rollout_id="rollout",
        policy_version=version,
    )


class TurnConversionTests(unittest.TestCase):
    def test_each_turn_keeps_exact_context_and_only_its_action_tokens(self):
        rollouts = episode_turn_rollouts(
            [recorded_episode()], behavior_policy_version=0
        )

        self.assertEqual(len(rollouts), 2)
        self.assertEqual(rollouts[0].prompt_tokens, [10, 11])
        self.assertEqual(rollouts[0].completion_tokens, [20, 21])
        self.assertEqual(rollouts[0].behavior_logprobs, [-2.0, -2.5])
        self.assertEqual(rollouts[1].prompt_tokens, [10, 11, 20, 21, 90, 91])
        self.assertEqual(rollouts[1].completion_tokens, [30, 31])
        self.assertEqual(rollouts[1].behavior_logprobs, [-3.0, -3.5])
        self.assertTrue(all(item.teacher_group == "default" for item in rollouts))

    def test_conversion_rejects_stale_mixed_unscored_and_invalid_inputs(self):
        current = recorded_episode()
        for episodes, version, error in (
            ([], 0, ValueError),
            ([recorded_episode(version=1)], 0, AgentContractError),
            ([recorded_episode(score=False)], 0, AgentContractError),
            ([current, replace(current, snapshot_id="other")], 0, AgentContractError),
        ):
            with self.subTest(error=error), self.assertRaises(error):
                episode_turn_rollouts(
                    episodes,
                    behavior_policy_version=version,
                )

    def test_only_the_final_action_inherits_episode_budget_truncation(self):
        rollouts = episode_turn_rollouts(
            [recorded_episode(status="max_turns")],
            behavior_policy_version=0,
        )
        self.assertFalse(rollouts[0].truncated)
        self.assertTrue(rollouts[1].truncated)


class ExampleConfigurationTests(unittest.TestCase):
    def test_training_config_selects_isolated_agent_opd_components(self):
        experiment = load_experiment(DEFAULT_CONFIG)

        self.assertEqual(experiment.trainer.model.name, "Qwen/Qwen3.5-4B")
        self.assertEqual(experiment.trainer.teacher.name, "Qwen/Qwen3.8-27B")
        self.assertGreater(experiment.trainer.max_steps, 0)
        self.assertEqual(experiment.trainer.batch_size, 16)
        self.assertEqual(experiment.trainer.num_generations, 1)
        self.assertEqual(experiment.agent.limits.max_turns, 15)
        self.assertEqual(experiment.agent.evaluation.temperature, 0.4)
        self.assertEqual(experiment.agent.evaluation.num_generations, 1)
        self.assertEqual(experiment.agent.evaluation.pass_ks, (1,))
        self.assertEqual(experiment.trainer.eval_steps, 20)
        self.assertEqual(experiment.trainer.report_to, "swanlab")
        self.assertEqual(experiment.trainer.workspace, "pytrio")
        self.assertEqual(experiment.trainer.project, "webshop-agentrl-TCOD")
        self.assertFalse(experiment.agent.async_rollouts)
        self.assertEqual(
            Path(inspect.getfile(OPDTrainer)).resolve().parent,
            Path(__file__).resolve().parents[4] / "triotune/trainer",
        )

    def test_training_step_override_does_not_mutate_yaml_defaults(self):
        short = load_experiment(DEFAULT_CONFIG, steps=2)
        original = load_experiment(DEFAULT_CONFIG)
        self.assertEqual(short.trainer.max_steps, 2)
        self.assertNotEqual(original.trainer.max_steps, 2)


class WebShopProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_webshop_wrapper_retains_terminal_score_and_turn_evidence(self):
        class FakePool:
            def __init__(self):
                self.sessions = {}
                self.released = []

            async def open(self, rollout_id, goal_index):
                self.sessions[rollout_id] = {"done": False, "score": 0.0}
                return {
                    "observation": f"goal-{goal_index}",
                    "available_actions": {
                        "has_search_bar": True,
                        "clickables": [],
                    },
                    "done": False,
                }

            async def step(self, rollout_id, action):
                session = self.sessions[rollout_id]
                if action.startswith("search["):
                    observation = "results"
                    actions = {
                        "has_search_bar": False,
                        "clickables": ["buy now"],
                    }
                elif action.lower() == "click[buy now]":
                    session.update(done=True, score=1.0)
                    observation = "purchased"
                    actions = {}
                else:
                    raise AssertionError(f"Unexpected fake action: {action}")
                return {
                    "observation": observation,
                    "available_actions": actions,
                    "done": session["done"],
                }

            async def outcome(self, rollout_id):
                session = self.sessions[rollout_id]
                return session["done"], session["score"]

            async def release(self, rollout_id):
                self.sessions.pop(rollout_id, None)
                self.released.append(rollout_id)

        class WebShopSampler:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.turns = {}

            async def generate(self, prompt_tokens, *, max_tokens, seed=None):
                rollout_id = current_execution_context().rollout_id
                turn = self.turns.get(rollout_id, 0)
                self.turns[rollout_id] = turn + 1
                response = (
                    '{"tool_calls":[{"name":"webshop","arguments":'
                    '{"action":"search[shoes]"}}]}'
                    if turn == 0
                    else '{"tool_calls":[{"name":"webshop","arguments":'
                    '{"action":"click[buy now]"}}]}'
                    if turn == 1
                    else '{"final":"done"}'
                )
                tokens = tuple(self.tokenizer.encode(response) + [1])
                return Generation(
                    tokens,
                    tuple([-1.0] * len(tokens)),
                    response,
                    input_tokens=len(prompt_tokens),
                    output_tokens=len(tokens),
                )

        tokenizer = AgentTokenizer()
        pool = FakePool()
        workflow = build_workflow(
            AgentConfig(limits={"max_turns": 4}),
            tokenizer,
            pool,
        )
        snapshot = SamplingSnapshot(
            "webshop-policy",
            WebShopSampler(tokenizer),
            tokenizer,
            policy_version=0,
        )
        with patch(
            "asyncio.to_thread", side_effect=FullAgentLoopTests.immediate_to_thread
        ):
            batch = await workflow.collect(
                [{"task_id": "shop-0", "goal_index": 500}],
                snapshot,
            )
        try:
            episode = batch.episodes[0]
            self.assertTrue(episode.score.success)
            self.assertEqual(episode.score.components["task_score"], 1.0)
            self.assertEqual(len(episode.turns), 3)
            self.assertEqual(
                len(
                    episode_turn_rollouts(
                        [episode],
                        behavior_policy_version=0,
                    )
                ),
                3,
            )
            self.assertEqual(len(pool.released), 1)
            self.assertFalse(pool.sessions)

            with patch(
                "asyncio.to_thread",
                side_effect=FullAgentLoopTests.immediate_to_thread,
            ):
                eval_batch = await workflow.collect(
                    [{"task_id": "shop-eval-0", "goal_index": 500}],
                    snapshot,
                    is_eval=True,
                )
            evaluated = eval_batch.episodes[0]
            self.assertTrue(evaluated.score.success)
            self.assertEqual(len(evaluated.turns), 2)
            self.assertIsNone(evaluated.final_text)
            self.assertEqual(len(pool.released), 2)
        finally:
            await workflow.close()

        short_pool = FakePool()
        short_workflow = build_workflow(
            AgentConfig(limits={"max_turns": 2}),
            tokenizer,
            short_pool,
        )
        short_snapshot = SamplingSnapshot(
            "short-webshop-policy",
            WebShopSampler(tokenizer),
            tokenizer,
            policy_version=0,
        )
        with patch(
            "asyncio.to_thread",
            side_effect=FullAgentLoopTests.immediate_to_thread,
        ):
            short_batch = await short_workflow.collect(
                [{"task_id": "shop-boundary", "goal_index": 500}],
                short_snapshot,
            )
        try:
            boundary = short_batch.episodes[0]
            self.assertEqual(boundary.status, "max_turns")
            self.assertFalse(boundary.score.success)
            self.assertEqual(boundary.score.components["task_score"], 1.0)
            self.assertTrue(boundary.trainable)
        finally:
            await short_workflow.close()


class TrainerAgentUpdateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = _Service()
        self.service.teacher_values = {"teacher": -1.0}
        self.patch = patch(
            "triotune.trainer.opd_trainer.trio.ServiceClient",
            return_value=self.service,
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def trainer(self, **kwargs):
        return OPDTrainer(
            "student",
            "teacher",
            args=OPDConfig(batch_size=1, num_generations=1, **kwargs),
        )

    async def test_agent_turns_use_teacher_scoring_and_one_optimizer_update(self):
        trainer = self.trainer()
        rollouts = episode_turn_rollouts(
            [recorded_episode()], behavior_policy_version=0
        )

        result = await trainer.step_rollouts_async(rollouts, prompt_count=1)

        self.assertEqual(result.num_datums, 2)
        self.assertEqual(result.behavior_policy_version, 0)
        self.assertEqual(trainer.state.behavior_policy_version, 1)
        self.assertEqual(trainer.state.optimizer_steps, 1)
        self.assertEqual(self.service.training.events, ["forward", "optim"])
        self.assertEqual(
            self.service.teacher.requests,
            [
                [10, 11, 20, 21],
                [10, 11, 20, 21, 90, 91, 30, 31],
            ],
        )
        first, second = self.service.training.datums
        self.assertEqual(first.model_input.to_ints(), [10, 11, 20])
        self.assertEqual(
            first.loss_fn_inputs["target_tokens"].data,
            [-100, 20, 21],
        )
        self.assertEqual(second.model_input.to_ints(), [10, 11, 20, 21, 90, 91, 30])
        self.assertEqual(
            second.loss_fn_inputs["target_tokens"].data,
            [-100, -100, -100, -100, -100, 30, 31],
        )
        self.assertEqual(result.metrics["data/prompts"], 1.0)
        self.assertEqual(result.metrics["data/completions"], 2.0)
        self.assertEqual(result.metrics["usage/sample_input_tokens"], 8.0)
        self.assertEqual(result.metrics["usage/sample_output_tokens"], 4.0)

    async def test_stale_rollouts_fail_before_teacher_or_optimizer_requests(self):
        trainer = self.trainer()
        rollouts = episode_turn_rollouts(
            [recorded_episode()], behavior_policy_version=0
        )
        trainer.state.behavior_policy_version = 1

        with self.assertRaisesRegex(ValueError, "stale or mixed"):
            await trainer.step_rollouts_async(rollouts, prompt_count=1)

        self.assertEqual(self.service.teacher.requests, [])
        self.assertEqual(self.service.training.events, [])

    async def test_teacher_failure_never_reaches_forward_or_optimizer(self):
        trainer = self.trainer()
        self.service.teacher_error = True
        rollouts = episode_turn_rollouts(
            [recorded_episode()], behavior_policy_version=0
        )

        with self.assertRaises(ExceptionGroup):
            await trainer.step_rollouts_async(rollouts, prompt_count=1)

        self.assertNotIn("forward", self.service.training.events)
        self.assertNotIn("optim", self.service.training.events)
        self.assertFalse(trainer._requires_restart)

    async def test_snapshot_carries_current_policy_and_forces_refresh_after_update(
        self,
    ):
        trainer = self.trainer()
        first = await trainer.create_sampling_snapshot()
        self.assertEqual(first.policy_version, 0)
        self.assertTrue(first.snapshot_id.startswith("opd-step-0-"))
        self.assertEqual(first.sampler.temperature, 1.0)
        await trainer.step_rollouts_async(
            episode_turn_rollouts([recorded_episode()], behavior_policy_version=0),
            prompt_count=1,
        )
        second = await trainer.create_sampling_snapshot(temperature=0.4, top_p=1.0)
        self.assertEqual(second.policy_version, 1)
        self.assertEqual(second.sampler.temperature, 0.4)
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(len(self.service.training.samplers), 2)


class FullAgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = _Service()
        self.service.teacher_values = {"teacher": -0.5}
        self.patch = patch(
            "triotune.trainer.opd_trainer.trio.ServiceClient",
            return_value=self.service,
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    @staticmethod
    def policy(version):
        tokenizer = AgentTokenizer()
        client = CalculatorSampler(tokenizer, [], version)

        class DirectAgentSampler:
            """Avoid a worker thread in this deterministic in-memory test."""

            async def generate(self, prompt_tokens, *, max_tokens, seed=None):
                response = await client.sample_async(
                    prompt=pytrio.ModelInput.from_ints(list(prompt_tokens)),
                    num_samples=1,
                    sampling_params=pytrio.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=1.0,
                        top_p=1.0,
                        seed=seed,
                    ),
                )
                sequence = response.sequences[0]
                return Generation(
                    tuple(sequence.tokens),
                    tuple(sequence.logprobs),
                    tokenizer.decode(sequence.tokens, skip_special_tokens=True),
                    sequence.stop_reason,
                    response.input_tokens,
                    response.output_tokens,
                )

        return SamplingSnapshot(
            f"agent-policy-{version}",
            DirectAgentSampler(),
            tokenizer,
            policy_version=version,
        )

    @staticmethod
    async def immediate_to_thread(function, /, *args, **kwargs):
        """Run tiny deterministic fakes inline and leave no test executor behind."""

        return function(*args, **kwargs)

    async def test_real_workflow_collection_updates_every_turn_once(self):
        trainer = OPDTrainer(
            "student",
            "teacher",
            args=OPDConfig(batch_size=1, num_generations=1),
        )
        workflow = calculator_workflow()
        loop = AgentOPDTrainingLoop(trainer, workflow)

        async def snapshot(*, stop=None, temperature=1.0, top_p=1.0):
            return self.policy(trainer.state.behavior_policy_version)

        trainer.create_sampling_snapshot = snapshot
        with patch("asyncio.to_thread", side_effect=self.immediate_to_thread):
            update = await loop.collect_and_update(calculator_rows()[:1])
        try:
            self.assertEqual(update.result.num_datums, 2)
            self.assertEqual(len(update.batch.episodes), 1)
            self.assertEqual(len(update.rollouts), 2)
            self.assertEqual(update.result.metrics["data/prompts"], 1.0)
            self.assertEqual(update.result.metrics["data/completions"], 2.0)
            self.assertEqual(loop.update_history[0]["turns"], 2)
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(self.service.training.optim_calls, 1)
            self.assertEqual(len(self.service.teacher.requests), 2)
        finally:
            await workflow.close()

    async def test_task_evaluation_never_scores_teacher_or_updates_student(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        trainer = OPDTrainer(
            "student",
            "teacher",
            args=OPDConfig(
                batch_size=1,
                num_generations=1,
                eval_batch_size=1,
            ),
            eval_dataset=calculator_rows()[:3],
        )
        workflow = calculator_workflow()
        loop = AgentOPDTrainingLoop(
            trainer,
            workflow,
            config=AgentConfig(
                log_path=str(Path(directory.name) / "episodes.jsonl"),
                # Seed zero follows the deterministic tool-then-answer path in
                # the calculator sampler, making task success unambiguous.
                evaluation={
                    "num_generations": 1,
                    "pass_ks": [1],
                    "seed": 0,
                    "temperature": 0.4,
                    "top_p": 1.0,
                },
            ),
        )

        sampling_settings = []

        async def snapshot(*, stop=None, temperature=1.0, top_p=1.0):
            sampling_settings.append((temperature, top_p))
            return self.policy(trainer.state.behavior_policy_version)

        trainer.create_sampling_snapshot = snapshot
        original_collect_iter = workflow.collect_iter
        observed = {}

        async def observed_collect_iter(rows, snapshot, **kwargs):
            observed["rows"] = len(rows)
            observed["kwargs"] = dict(kwargs)
            async for batch in original_collect_iter(rows, snapshot, **kwargs):
                yield batch

        workflow.collect_iter = observed_collect_iter
        with patch("asyncio.to_thread", side_effect=self.immediate_to_thread):
            metrics = await loop.evaluate()
        try:
            self.assertEqual(metrics["eval/agent/pass@1"], 1 / 3)
            self.assertEqual(self.service.teacher.requests, [])
            self.assertEqual(self.service.training.events, [])
            self.assertEqual(trainer.state.global_step, 0)
            self.assertEqual(sampling_settings, [(0.4, 1.0)])
            self.assertEqual(observed["rows"], 3)
            self.assertFalse(observed["kwargs"]["log_episodes"])
            self.assertEqual(observed["kwargs"]["max_pending_groups"], 8)
            full_log = Path(directory.name) / "episodes.jsonl"
            compact_log = Path(directory.name) / "eval-episodes.jsonl"
            self.assertFalse(full_log.exists())
            compact_text = compact_log.read_text()
            summaries = [json.loads(line) for line in compact_text.splitlines()]
            self.assertEqual(len(summaries), 3)
            self.assertIn("actions", summaries[0])
            self.assertNotIn("prompt_tokens", compact_text)
            self.assertNotIn("generation", compact_text)
        finally:
            await workflow.close()

    async def test_two_step_training_refreshes_policy_and_saves_final_state(self):
        rows = calculator_rows()
        trainer = OPDTrainer(
            "student",
            "teacher",
            args=OPDConfig(
                batch_size=1,
                num_generations=1,
                max_steps=2,
                eval_steps=1,
                eval_before_train=True,
                eval_after_train=True,
                eval_batch_size=1,
                eval_max_batches=1,
            ),
            train_dataset=rows[:2],
            eval_dataset=rows[2:3],
        )
        workflow = calculator_workflow()
        loop = AgentOPDTrainingLoop(
            trainer,
            workflow,
            config=AgentConfig(
                evaluation={"num_generations": 1, "pass_ks": [1], "seed": 0}
            ),
        )
        snapshot_versions = []

        async def snapshot(*, stop=None, temperature=1.0, top_p=1.0):
            snapshot_versions.append(trainer.state.behavior_policy_version)
            return self.policy(trainer.state.behavior_policy_version)

        trainer.create_sampling_snapshot = snapshot
        with patch("asyncio.to_thread", side_effect=self.immediate_to_thread):
            state = await loop.train()

        self.assertEqual(state.global_step, 2)
        self.assertEqual(state.optimizer_steps, 2)
        self.assertEqual(state.behavior_policy_version, 2)
        self.assertEqual(len(loop.update_history), 2)
        self.assertEqual(
            [record["behavior_policy_version"] for record in loop.update_history],
            [0, 1],
        )
        self.assertEqual(
            [record["turns"] for record in loop.update_history],
            [2, 2],
        )
        self.assertEqual(self.service.training.optim_calls, 2)
        self.assertEqual(self.service.training.events.count("forward"), 2)
        self.assertEqual(self.service.training.events.count("save"), 1)
        self.assertEqual(state.checkpoint_path, "checkpoint://opd-step-00000002")
        # Evaluation runs at step 0, 1, and 2; training snapshots use versions
        # 0 and 1.  No snapshot from one update is reused after optimization.
        self.assertEqual(snapshot_versions, [0, 0, 1, 1, 2])
        self.assertFalse(workflow._resource_started)

    async def test_complete_checkpoint_commits_agent_seed_and_update_history(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = OPDTrainer(
                "student",
                "teacher",
                args=OPDConfig(
                    batch_size=1,
                    num_generations=1,
                    max_steps=1,
                    checkpoint={"save_every": 1, "state_dir": directory},
                ),
                train_dataset=calculator_rows()[:1],
                defer_checkpoint_loop_binding=True,
            )
            workflow = calculator_workflow()
            loop = AgentOPDTrainingLoop(trainer, workflow)

            async def snapshot(*, stop=None, temperature=1.0, top_p=1.0):
                return self.policy(trainer.state.behavior_policy_version)

            trainer.create_sampling_snapshot = snapshot
            with patch("asyncio.to_thread", side_effect=self.immediate_to_thread):
                state = await loop.train()

            manifest = json.loads(Path(state.checkpoint_manifest_path).read_text())
            self.assertEqual(manifest["loop"]["kind"], "agent_opd")
            self.assertEqual(
                manifest["loop"]["state"]["seed_offset"],
                workflow.runner.limits.max_turns,
            )
            self.assertEqual(len(manifest["loop"]["state"]["updates"]), 1)
            self.assertEqual(
                manifest["loop"]["state"]["updates"][0]["turns"],
                2,
            )
            self.assertEqual(manifest["state"]["global_step"], 1)

            self.service.create_training_client_from_state_with_optimizer = (
                lambda **kwargs: self.service.training
            )
            resumed_trainer = OPDTrainer(
                "student",
                "teacher",
                args=OPDConfig(
                    batch_size=1,
                    num_generations=1,
                    max_steps=1,
                    checkpoint={
                        "save_every": 1,
                        "state_dir": directory,
                        "resume_from": state.checkpoint_manifest_path,
                    },
                ),
                train_dataset=calculator_rows()[:1],
                defer_checkpoint_loop_binding=True,
            )
            resumed_loop = AgentOPDTrainingLoop(
                resumed_trainer,
                calculator_workflow(),
            )
            self.assertEqual(resumed_trainer.state.global_step, 1)
            self.assertEqual(
                resumed_loop._seed_offset,
                workflow.runner.limits.max_turns,
            )
            self.assertEqual(len(resumed_loop.update_history), 1)
            await resumed_loop.workflow.close()


if __name__ == "__main__":
    unittest.main()
