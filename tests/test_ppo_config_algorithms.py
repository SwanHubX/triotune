"""Configuration files can select every supported PPO-family algorithm."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trio_cookbook.api import NormConfig, PPOConfig
from trio_cookbook.utils.config import load_yaml_config


REPOSITORY_ROOT = Path(__file__).parents[1]


class PPOConfigFileTests(unittest.TestCase):
    """Exercise algorithm selection without creating remote PyTRIO clients."""

    _ALGORITHM_EXPECTATIONS = {
        "ppo": {"use_critic": True, "reward_norm": None},
        "grpo": {"use_critic": False, "reward_norm": NormConfig()},
        "drgrpo": {"use_critic": False, "reward_norm": NormConfig("group", None)},
        "liteppo": {"use_critic": False, "reward_norm": NormConfig("group", "batch")},
        "rloo": {
            "use_critic": False,
            "reward_norm": NormConfig("group", None, mean_leave1out=True),
        },
        "reinforce": {"use_critic": False, "reward_norm": None},
        "reinforce_baseline": {
            "use_critic": False,
            "reward_norm": NormConfig("group", None),
        },
        "gspo": {"use_critic": False, "importance_sampling_level": "sequence"},
        "sapo": {"use_critic": False, "loss_type": "sapo"},
        "cispo": {
            "use_critic": False,
            "loss_type": "cispo",
            "eps_clip": 1.0,
            "eps_clip_higher": 4.0,
        },
        "dapo": {
            "use_critic": False,
            "dynamic_sampling": True,
            "eps_clip": 0.2,
            "eps_clip_higher": 0.28,
            "overlong_tokens": 128,
        },
    }

    @staticmethod
    def _write_config(directory: str, algorithm: str) -> Path:
        path = Path(directory) / f"{algorithm}.yaml"
        path.write_text(
            f"""algorithm: {algorithm}
training:
  batch_size: 2
  max_steps: 7
optimizer:
  lr: 0.00002
rollout:
  group_size: 4
  max_new_tokens: 128
""",
            encoding="utf-8",
        )
        return path

    def test_yaml_algorithm_switch_selects_each_ppo_family_preset(self) -> None:
        with TemporaryDirectory() as directory:
            for algorithm, expected in self._ALGORITHM_EXPECTATIONS.items():
                with self.subTest(algorithm=algorithm):
                    settings = load_yaml_config(self._write_config(directory, algorithm))
                    config = PPOConfig.from_modules(settings)

                    self.assertEqual(config.batch_size, 2)
                    self.assertEqual(config.max_steps, 7)
                    self.assertEqual(config.learning_rate, 2e-5)
                    self.assertEqual(config.num_generations, 4)
                    self.assertEqual(config.max_tokens, 128)
                    for field, value in expected.items():
                        self.assertEqual(getattr(config, field), value, field)

    def test_repository_ppo_yaml_uses_the_same_module_adapter(self) -> None:
        path = REPOSITORY_ROOT / "trio-cookbook" / "configs" / "ppo_gsm8k.yaml"
        settings = load_yaml_config(path)
        config = PPOConfig.from_modules(settings)

        self.assertEqual(settings["algorithm"], "ppo")
        self.assertTrue(config.use_critic)
        self.assertEqual(config.num_generations, 4)
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.critic_update_epochs, 1)
        self.assertEqual(config.critic_n_minibatches, 1)

    def test_unknown_algorithm_is_rejected_from_a_config_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_config(directory, "not_a_ppo_algorithm")
            settings = load_yaml_config(path)

        with self.assertRaisesRegex(ValueError, "Unknown PPO preset"):
            PPOConfig.from_modules(settings)


if __name__ == "__main__":
    unittest.main()
