"""Validate experiment settings and reproducible grouped/paired evaluation."""

import copy
import json
import unittest
from pathlib import Path

from agent_test_support import AgentTokenizer, AgentTrainingClient
from recipes.calculator_agent import (
    calculator_rows,
    calculator_workflow,
    generated_rows,
)
from test_agent_episode_runner import snapshot

from examples.agent.calculator.train import gradient_check, load_experiment
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    ActionParseError,
    AgentConfig,
    AgentEvaluationConfig,
    AgentTrainingLoop,
    Generation,
    JsonActionParser,
    RetryPolicy,
    evaluation_metrics,
    paired_task_metrics,
)


class AgentConfigTests(unittest.TestCase):
    def test_round_trip_nested_config_and_unknown_options(self):
        config = AgentConfig(
            limits={"max_turns": 4},
            retry={"max_attempts": 2},
            evaluation={"pass_ks": [1, 2], "num_generations": 2},
        )
        self.assertEqual(
            AgentConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config
        )
        self.assertEqual(config.limits.max_turns, 4)
        for kwargs in (
            {"max_concurrency": True},
            {"parser": "native"},
            {"renderer": "typo"},
            {"stop": []},
            {"stop": [1, "stop"]},
            {"template_kwargs": {"tokenize": False}},
            {"on_system_failure": "ignore"},
            {"max_refill_batches": 0},
            {"prompt_key": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentConfig(**kwargs)
        with self.assertRaises(TypeError):
            AgentConfig.from_dict({"unknown": True})
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_strict_json_parser_does_not_reward_plain_text_or_nonfinite_numbers(self):
        parser = JsonActionParser(allow_plain_text=False)
        for text in (
            "5",
            '{"tool_calls":[{"name":"calculate","arguments":{"a":NaN}}]}',
            '{"tool_calls":[{"name":"calculate","arguments":{},"typo":1}]}',
        ):
            with self.subTest(text=text), self.assertRaises(ActionParseError):
                parser.parse(Generation((1,), (-1,), text))

    def test_public_yaml_and_generated_data_have_disjoint_tasks(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "examples/agent/calculator/config.yaml"
        )
        trainer, agent, costs, data, output = load_experiment(path)
        self.assertFalse(trainer.use_critic)
        self.assertEqual(trainer.model.name, "Qwen/Qwen3.5-4B")
        self.assertEqual(trainer.gae_timestep_unit, "turn")
        self.assertEqual(agent.parser, "strict_json")
        self.assertEqual(costs.tool_call, 0.01)
        self.assertEqual(data.eval_tasks, 8)
        self.assertTrue(output.is_absolute())
        train = generated_rows(100, split="train", seed=42)
        evaluate = generated_rows(100, split="eval", seed=42)
        self.assertEqual(train, generated_rows(100, split="train", seed=42))
        questions = lambda rows: {row["messages"][-1]["content"] for row in rows}
        self.assertEqual(len(questions(train)), 100)
        self.assertTrue(questions(train).isdisjoint(questions(evaluate)))
        self.assertTrue(any("(" in row["messages"][-1]["content"] for row in train))
        self.assertTrue(
            all("answer" not in message for row in train for message in row["messages"])
        )
        for invalid in (0, True, 10001):
            with self.assertRaises(ValueError):
                generated_rows(invalid, split="train")


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unbiased_pass_at_k_and_failure_denominators(self):
        policy, _ = snapshot()
        batch = await calculator_workflow().collect(
            calculator_rows()[:1], policy, n_samples=4, seed=0
        )
        # Deterministic test sampler gives 2 successes out of 4 attempts.
        metrics = evaluation_metrics(batch.episodes, pass_ks=(1, 2, 4))
        self.assertAlmostEqual(metrics["eval/agent/pass@1"], 0.5)
        self.assertAlmostEqual(metrics["eval/agent/pass@2"], 5 / 6)
        self.assertEqual(metrics["eval/agent/pass@4"], 1)
        self.assertEqual(metrics["eval/agent/first_tool_valid_rate"], 1)
        self.assertEqual(metrics["eval/agent/success_turns_mean"], 2)
        failed = copy.deepcopy(batch.episodes)
        failed[0].status, failed[0].score = "tool_failure", None
        self.assertEqual(evaluation_metrics(failed)["eval/agent/pass@1"], 0.25)
        paired = paired_task_metrics(batch.episodes, failed)
        self.assertEqual(paired["eval/agent/paired/tasks"], 1)
        self.assertEqual(paired["eval/agent/paired/success_rate_delta"], -0.25)

    async def test_eval_seeds_stable_across_versions_and_batch_boundaries(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm("grpo", batch_size=1, num_generations=2),
            training_client=client,
            eval_dataset=[calculator_rows()[0]] * 3,
        )
        config = AgentConfig(
            evaluation=AgentEvaluationConfig(num_generations=2, pass_ks=(1, 2))
        )
        loop = AgentTrainingLoop(trainer, calculator_workflow(), config=config)
        await loop.evaluate()
        seeds_before = [params.seed for _, params in client.snapshots[-1].requests]
        self.assertEqual(len({e.task_id for e in loop.last_eval_batch.episodes}), 3)
        client.version = 1
        metrics = await loop.evaluate()
        seeds_after = [params.seed for _, params in client.snapshots[-1].requests]
        self.assertEqual(seeds_before, seeds_after)
        self.assertEqual(metrics["eval/agent/paired/tasks"], 3)
        self.assertEqual(metrics["eval/agent/paired/success_rate_delta"], 0)
        self.assertNotIn("submit:optim", client.trace)
        await trainer.close()

    async def test_duplicate_task_ids_and_unsupported_pass_k_fail_explicitly(self):
        client = AgentTrainingClient()
        rows = [{**calculator_rows()[0], "task_id": "duplicate"}] * 2
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm("grpo", batch_size=2),
            training_client=client,
            eval_dataset=rows,
        )
        loop = AgentTrainingLoop(trainer, calculator_workflow())
        with self.assertRaisesRegex(ValueError, "unique"):
            await loop.evaluate()
        loop.config = AgentConfig(
            evaluation=AgentEvaluationConfig(num_generations=1, pass_ks=(2,))
        )
        with self.assertRaisesRegex(ValueError, "pass_k"):
            await loop.evaluate()
        await trainer.close()

    async def test_real_loss_gradient_check_and_configured_adapters(self):
        policy, _ = snapshot()
        batch = await calculator_workflow().collect(
            calculator_rows()[:1], policy, n_samples=2, seed=0
        )
        result = gradient_check(batch, PPOConfig.for_algorithm("grpo"))
        self.assertGreater(result["action_gradient_l1"], 0)
        self.assertEqual(result["context_gradient_l1"], 0)
        # Character tokenizer has its own boundary tokens; the default strict
        # renderer remains available for adapters without a string template.
        config = AgentConfig(renderer="chat_template", template_kwargs={})
        workflow = calculator_workflow(config=config, tokenizer=AgentTokenizer())
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertTrue(batch.episodes[0].score.success)

    async def test_configured_prompt_key_is_used_by_runner(self):
        policy, _ = snapshot()
        config = AgentConfig(
            renderer="chat_template", template_kwargs={}, prompt_key="conversation"
        )
        workflow = calculator_workflow(config=config, tokenizer=AgentTokenizer())
        row = calculator_rows()[0]
        row["conversation"] = row.pop("messages")
        batch = await workflow.collect([row], policy)
        self.assertTrue(batch.episodes[0].score.success)
