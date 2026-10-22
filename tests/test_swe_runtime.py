"""PyTRIO sampling fidelity and the exact PPO trajectory bridge."""

import unittest
from types import SimpleNamespace

from triotune.swe import (
    PyTRIOSWEModel,
    SampledTurn,
    SWESamplingConfig,
    trajectory_from_turns,
)
from triotune.trainer.ppo.datum import build_datum


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [10, 11]}

    def decode(self, tokens, **kwargs):
        return "decoded from actual tokens"


class Sampler:
    async def sample_async(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=[20, 21],
                    logprobs=[-0.5, -0.7],
                    text="do not trust this text",
                    stop_reason="stop",
                )
            ]
        )


class SWEModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampling_uses_public_pytrio_and_preserves_actual_tokens(self):
        tokenizer, client = Tokenizer(), Sampler()
        model = PyTRIOSWEModel(client, tokenizer, policy_version="step-3")
        turn = await model.sample(
            [{"role": "user", "content": "fix"}], [{"tool": "schema"}]
        )
        self.assertEqual(turn.prompt_tokens, (10, 11))
        self.assertEqual(turn.completion_tokens, (20, 21))
        self.assertEqual(turn.logprobs, (-0.5, -0.7))
        self.assertEqual(turn.raw_text, "decoded from actual tokens")
        self.assertEqual(client.kwargs["prompt"].to_ints(), [10, 11])
        self.assertEqual(tokenizer.kwargs["tools"], [{"tool": "schema"}])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
        self.assertTrue(model.supports_ppo)

    async def test_context_budget_fails_before_sampling(self):
        client = Sampler()
        model = PyTRIOSWEModel(
            client,
            Tokenizer(),
            policy_version="base",
            config=SWESamplingConfig(max_tokens=2, max_context_tokens=3),
        )
        with self.assertRaisesRegex(ValueError, "context budget"):
            await model.sample([], [])
        self.assertFalse(hasattr(client, "kwargs"))

    async def test_sample_must_have_aligned_finite_behavior_scores(self):
        for scores in ((float("nan"),), (0.1,), (), (None,)):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                SampledTurn((1,), (2,), scores, "text", "base")

    def test_config_rejects_invalid_limits_and_marks_nonppo_sampling(self):
        for kwargs in (
            {"max_tokens": True},
            {"max_context_tokens": 1},
            {"temperature": float("nan")},
            {"top_p": 0},
            {"chat_template_kwargs": {"tools": []}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SWESamplingConfig(**kwargs)
        model = PyTRIOSWEModel(
            None, None, policy_version="test", config=SWESamplingConfig(temperature=0)
        )
        self.assertFalse(model.supports_ppo)

    async def test_exact_episode_trajectories_update_existing_ppo_trainer(self):
        from test_ppo_family import TrainingClient

        from triotune import PPOConfig, PPOTrainer

        trajectories = [
            trajectory_from_turns(
                [SampledTurn((10, 11), (20, 21), (-1, -1), "action", "v1")],
                reward=reward,
                group_id="independent-attempts",
            )
            for reward in (0, 1)
        ]
        client = TrainingClient()
        trainer = PPOTrainer(
            "fake", args=PPOConfig.for_algorithm("grpo"), training_client=client
        )
        try:
            await trainer.step(trajectories)
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(client.version, 1)
            self.assertTrue(any(call[0] == "custom" for call in client.calls))
        finally:
            await trainer.close()


class SWETrainingBridgeTests(unittest.TestCase):
    def test_observations_masked_and_causal_datum_keeps_exact_tokens(self):
        turns = [
            SampledTurn((10, 11), (20, 21), (-0.5, -0.6), "act1", "v1"),
            SampledTurn((10, 11, 20, 21, 30, 31), (22,), (-0.7,), "act2", "v1"),
        ]
        t = trajectory_from_turns(turns, reward=1, group_id="issue-attempts")
        self.assertEqual(t.completion_tokens, [20, 21, 30, 31, 22])
        self.assertEqual(t.loss_mask, [True, True, False, False, True])
        self.assertEqual(t.turn_ids, [0, 0, -1, -1, 1])
        self.assertEqual(t.logprobs, [-0.5, -0.6, 0, 0, -0.7])
        datum = build_datum(t, advantages=[1, 1, 0, 0, 1], old_logprobs=t.logprobs)
        self.assertEqual(
            datum.loss_fn_inputs["target_tokens"].data, [-100, 20, 21, -100, -100, 22]
        )

    def test_template_rewriting_or_policy_change_cannot_fabricate_a_trajectory(self):
        first = SampledTurn((10,), (20,), (-1,), "a", "v1")
        for second, error in (
            (SampledTurn((10, 30), (21,), (-1,), "b", "v1"), "rewrote"),
            (SampledTurn((10, 20, 30), (21,), (-1,), "b", "v2"), "single policy"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                trajectory_from_turns([first, second], reward=0, group_id="g")


if __name__ == "__main__":
    unittest.main()
