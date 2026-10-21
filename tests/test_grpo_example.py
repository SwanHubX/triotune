"""Exercise the GSM8K GRPO example without remote training services."""

import unittest
from unittest.mock import AsyncMock, patch

from datasets import Dataset

from examples.grpo import gsm8k_grpo as example
from triotune.trainer import PPOConfig
from triotune.utils.config import load_yaml_config


CONFIG_PATH = example.ROOT / "examples/grpo/gsm8k_grpo_args.yaml"


class GRPOExampleTests(unittest.TestCase):
    def test_configuration_uses_a_group_relative_baseline_without_a_critic(self):
        config = PPOConfig(**load_yaml_config(CONFIG_PATH))

        self.assertFalse(config.use_critic)
        self.assertGreater(config.num_generations, 1)
        self.assertEqual(config.reward_norm.mean_level, "group")
        self.assertEqual(config.reward_norm.std_level, "group")
        self.assertEqual(config.reference.name, config.model.name)

    def test_reward_uses_the_last_boxed_answer(self):
        cases = [
            ("42", r"The answer is \boxed{42}.", 1.0),
            ("1200", r"The answer is \boxed{ $1,200. }.", 1.0),
            ("-2.5", r"The answer is \boxed{-2.5}.", 1.0),
            ("42", r"First \boxed{10}, finally \boxed{42}.", 1.0),
            ("42", r"First \boxed{42}, finally \boxed{10}.", 0.0),
            ("42", "The answer is 42.", 0.0),
            ("42", r"\boxed{}", 0.0),
            ("42", "", 0.0),
        ]
        for answer, completion, expected in cases:
            with self.subTest(answer=answer, completion=completion):
                self.assertEqual(
                    example.gsm8k_reward(
                        row={"answer": answer},
                        completion=completion,
                        completion_ids=[],
                    ),
                    expected,
                )

    def test_main_loads_training_and_optional_validation_data(self):
        train = Dataset.from_list([{"messages": [], "answer": "42"}])
        valid = Dataset.from_list([
            {"messages": [], "answer": "42"},
            {"messages": [], "answer": "10"},
        ])
        for enabled, limit in [(False, None), (True, None), (True, 1), (True, 10)]:
            with self.subTest(validation=enabled, max_examples=limit):
                settings = load_yaml_config(CONFIG_PATH)
                if enabled:
                    settings["valid_dataset"]["max_examples"] = limit
                else:
                    settings["valid_dataset"] = None
                with (
                    patch("sys.argv", ["gsm8k_grpo.py", "--config", str(CONFIG_PATH)]),
                    patch.object(example, "load_yaml_config", return_value=settings),
                    patch.object(example, "load_dataset", side_effect=[train, valid]) as load,
                    patch.object(example, "PPOTrainer") as trainer,
                ):
                    trainer.return_value.train = AsyncMock()
                    example.main()

                self.assertEqual(load.call_count, 2 if enabled else 1)
                self.assertEqual(load.call_args_list[0].args, ("json",))
                self.assertEqual(load.call_args_list[0].kwargs, {
                    "data_files": {"train": str(example.ROOT / "data/gsm8k_train.jsonl")},
                    "split": "train",
                })
                inputs = trainer.call_args.kwargs
                self.assertIs(inputs["train_dataset"], train)
                self.assertIs(inputs["reward_fn"], example.gsm8k_reward)
                if enabled:
                    expected_size = min(limit, len(valid)) if limit is not None else len(valid)
                    self.assertEqual(len(inputs["eval_dataset"]), expected_size)
                    self.assertEqual(load.call_args_list[1].kwargs, {
                        "data_files": {"test": str(example.ROOT / "data/gsm8k_test.jsonl")},
                        "split": "test",
                    })
                else:
                    self.assertIsNone(inputs["eval_dataset"])
                trainer.return_value.train.assert_awaited_once_with()

    def test_missing_training_data_fails_before_creating_a_trainer(self):
        with (
            patch("sys.argv", ["gsm8k_grpo.py", "--config", str(CONFIG_PATH)]),
            patch.object(example, "load_dataset", side_effect=FileNotFoundError("missing JSONL")),
            patch.object(example, "PPOTrainer") as trainer,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing JSONL"):
                example.main()
        trainer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
