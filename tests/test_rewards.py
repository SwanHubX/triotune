from __future__ import annotations

import unittest

from trio_cookbook.trainer.rewards import RewardEvaluator
from trio_cookbook.trainer.types import GRPOBatch, SequenceSample


class RewardEvaluatorTests(unittest.TestCase):
    def test_prompt_column_is_not_forwarded_to_reward_function(self) -> None:
        observed: dict[str, object] = {}

        def reward(**kwargs: object) -> list[float]:
            observed.update(kwargs)
            return [1.0]

        group = GRPOBatch(
            prompt_id="0",
            prompt="rendered prompt",
            prompt_tokens=[1],
            samples=[SequenceSample(tokens=[2], logprobs=[-0.1], text="answer")],
            row={"prompt": [{"role": "user", "content": "question"}], "answer": "42"},
        )

        RewardEvaluator(reward).score([group])

        self.assertNotIn("prompt", observed)
        self.assertEqual(observed["answer"], ["42"])

    def test_other_columns_are_forwarded_alongside_prompt(self) -> None:
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
            row={"prompt": "raw prompt", "difficulty": "easy"},
        )

        evaluator = RewardEvaluator(difficulty_reward)
        scores, metrics = evaluator.score([group])

        self.assertEqual(scores, [[1.0, 0.0]])
        self.assertEqual(metrics, {"reward/difficulty_reward/mean": 0.5})
        self.assertEqual(observed["prompts"], ["rendered prompt"] * 2)
        self.assertEqual(observed["difficulty"], ["easy"] * 2)


if __name__ == "__main__":
    unittest.main()
