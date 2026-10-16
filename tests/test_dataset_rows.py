from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from trio_cookbook.trainer import GRPOConfig, GRPOTrainer, TrainerState
from trio_cookbook.utils.metrics import MetricsAPI
from trio_cookbook.trainer.grpo.rewards import RewardEvaluator


class _Tokenizer:
    eos_token = "<|endoftext|>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [len(text)]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        return "decoded tokens"

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str | list[int]:
        return [3, 4] if tokenize else "chat template"


class DatasetRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = GRPOTrainer.__new__(GRPOTrainer)
        self.trainer.config = GRPOConfig()
        self.trainer.state = TrainerState()
        self.trainer._tokenizer = _Tokenizer()
        self.trainer._decode = lambda tokens: "decoded tokens"

    def test_default_messages_accepts_text_tokens_and_chat(self) -> None:
        rows = [
            {"messages": "question"},
            {"messages": [1, 2]},
            {"messages": [{"role": "user", "content": "question"}]},
        ]

        groups = self.trainer._groups(rows)

        self.assertEqual([group.prompt_tokens for group in groups], [[8], [1, 2], [3, 4]])
        self.assertEqual(
            [group.prompt for group in groups],
            ["question", "decoded tokens", "chat template"],
        )

    def test_default_rejects_old_prompt_column(self) -> None:
        with self.assertRaisesRegex(KeyError, "messages"):
            self.trainer._groups([{"prompt": "question"}])

    def test_custom_prompt_key_reads_old_column(self) -> None:
        self.trainer.config = GRPOConfig(prompt_key="prompt")

        groups = self.trainer._groups([{"prompt": "question"}])

        self.assertEqual(groups[0].prompt, "question")

    def test_stop_sequences_follow_config(self) -> None:
        self.assertEqual(self.trainer._stop_sequences(), ["<|endoftext|>", "<|im_end|>"])

        self.trainer.config = GRPOConfig(stop=False)
        self.assertIsNone(self.trainer._stop_sequences())

    def test_validation_samples_and_scores_without_training(self) -> None:
        self.trainer.config = GRPOConfig(num_generations=2)
        self.trainer._last_eval_step = None
        self.trainer._metrics_api = MetricsAPI()
        self.trainer.reward_evaluator = RewardEvaluator(
            lambda completions, **_: [float(index + 1) for index, _ in enumerate(completions)]
        )

        class Sampler:
            def sample(self, **_: object) -> object:
                return SimpleNamespace(
                    sequences=[
                        SimpleNamespace(tokens=[5], logprobs=[-0.5], text="a"),
                        SimpleNamespace(tokens=[6, 7], logprobs=[-0.4, -0.3], text="b"),
                    ]
                )

        self.trainer._new_sampling_client = Sampler

        metrics = self.trainer.evaluate([{"messages": "question"}])

        self.assertEqual(metrics["eval/reward"], 1.5)
        self.assertEqual(metrics["eval/prompts"], 1.0)
        self.assertEqual(metrics["eval/completions"], 2.0)
        self.assertEqual(self.trainer.state.global_step, 0)

    def test_eval_dataset_schedule_in_sync_and_async_training(self) -> None:
        class Sampler:
            def sample(self, **kwargs):
                return SimpleNamespace(sequences=[
                    SimpleNamespace(tokens=[5], logprobs=[-0.5], text="a"),
                    SimpleNamespace(tokens=[6], logprobs=[-0.3], text="b"),
                ])

        for async_mode in (False, True):
            with self.subTest(async_mode=async_mode):
                training = SimpleNamespace(
                    get_tokenizer=_Tokenizer,
                    save_weights_and_get_sampling_client=Sampler,
                    forward_backward=lambda *args, **kwargs: SimpleNamespace(
                        metrics={"loss_mean": 0.25}
                    ),
                    optim_step=lambda *args: SimpleNamespace(metrics={}),
                )
                service = SimpleNamespace(create_lora_training_client=lambda **kwargs: training)
                rows = [{"messages": "question"}]
                with patch(
                    "trio_cookbook.trainer.grpo_trainer.trio.ServiceClient",
                    return_value=service,
                ):
                    trainer = GRPOTrainer(
                        "test-model",
                        reward_funcs=lambda completions, **kwargs: [1.0, 2.0],
                        args=GRPOConfig(
                            num_generations=2,
                            max_steps=4,
                            eval_steps=2,
                            eval_before_train=True,
                            eval_after_train=True,
                        ),
                        train_dataset=rows * 4,
                        eval_dataset=rows,
                    )
                evaluated = []
                publish = trainer._publish_evaluation

                def track_evaluation(metrics):
                    evaluated.append(trainer.state.global_step)
                    return publish(metrics)

                trainer._publish_evaluation = track_evaluation
                if async_mode:
                    asyncio.run(trainer.train_async())
                else:
                    trainer.train()

                self.assertEqual(evaluated, [0, 2, 4])
                self.assertEqual(trainer.state.global_step, 4)


if __name__ == "__main__":
    unittest.main()
