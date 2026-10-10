from __future__ import annotations

import unittest
from unittest.mock import patch

from trio_cookbook.trainer import GRPOConfig, GRPOTrainer
import trio_cookbook.trainer.grpo.trainer as trainer_module
from trio_cookbook.trainer.grpo.rewards import RewardEvaluator
from trio_cookbook.trainer.grpo.types import GRPOBatch, SequenceSample


class RewardEvaluatorTests(unittest.TestCase):
    def test_default_messages_key_is_not_forwarded_to_reward_function(self) -> None:
        observed: dict[str, object] = {}

        def reward(**kwargs: object) -> list[float]:
            observed.update(kwargs)
            return [1.0]

        group = GRPOBatch(
            prompt_id="0",
            prompt="rendered prompt",
            prompt_tokens=[1],
            samples=[SequenceSample(tokens=[2], logprobs=[-0.1], text="answer")],
            row={"messages": [{"role": "user", "content": "question"}], "answer": "42"},
        )

        RewardEvaluator(reward).score([group])

        self.assertNotIn("messages", observed)
        self.assertEqual(observed["answer"], ["42"])

    def test_trainer_propagates_custom_prompt_key_to_reward_evaluator(self) -> None:
        class Tokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return [1]

            def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
                return "decoded"

        class TrainingClient:
            def get_tokenizer(self) -> Tokenizer:
                return Tokenizer()

        class ServiceClient:
            def create_lora_training_client(self, **kwargs: object) -> TrainingClient:
                return TrainingClient()

        with patch.object(trainer_module.trio, "ServiceClient", ServiceClient):
            trainer = GRPOTrainer(
                "model",
                lambda **kwargs: [],
                GRPOConfig(prompt_key="input"),
            )

        self.assertEqual(trainer.reward_evaluator.prompt_key, "input")

    def test_custom_prompt_key_is_not_forwarded_to_reward_function(self) -> None:
        observed: dict[str, object] = {}

        def difficulty_reward(
            prompts: list[str],
            completions: list[str],
            completion_ids: list[list[int]],
            difficulty: list[str],
        ) -> list[float]:
            observed.update(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_ids,
                difficulty=difficulty,
            )
            return [1.0, 0.0]

        group = GRPOBatch(
            prompt_id="0",
            prompt="rendered prompt",
            prompt_tokens=[1],
            samples=[
                SequenceSample(tokens=[2], logprobs=[-0.1], text="first"),
                SequenceSample(tokens=[3], logprobs=[-0.2], text="second"),
            ],
            row={"input": "raw prompt", "difficulty": "easy"},
        )

        evaluator = RewardEvaluator(difficulty_reward, prompt_key="input")
        scores, metrics = evaluator.score([group])

        self.assertEqual(scores, [[1.0, 0.0]])
        self.assertEqual(metrics, {"reward/difficulty_reward/mean": 0.5})
        self.assertEqual(observed["prompts"], ["rendered prompt"] * 2)
        self.assertEqual(observed["difficulty"], ["easy"] * 2)


if __name__ == "__main__":
    unittest.main()
