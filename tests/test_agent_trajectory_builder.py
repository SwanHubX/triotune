"""Exact-token export, process reward alignment, and tool masking contracts."""

import unittest

from triotune.agentrl import (
    AgentContractError,
    AgentTurn,
    EpisodeResult,
    EpisodeScore,
    EpisodeTrajectoryBuilder,
    Generation,
)
from triotune.trainer.ppo import PPOConfig, prepare_advantages
from triotune.trainer.ppo.datum import build_datum


def episode():
    return EpisodeResult(
        "episode",
        "group",
        "snapshot",
        turns=[
            AgentTurn((10, 11), Generation((20, 21), (-0.3, -0.4), "tool")),
            AgentTurn(
                (10, 11, 20, 21, 90, 91), Generation((30, 31), (-0.5, -0.6), "final")
            ),
        ],
        score=EpisodeScore(1.0, True, process_rewards={0: (0.0, 0.25)}),
    )


class AgentTrajectoryTests(unittest.TestCase):
    def test_export_preserves_original_tokens_and_masks_observations(self):
        trajectory = EpisodeTrajectoryBuilder().build(episode())
        self.assertEqual(trajectory.prompt_tokens, [10, 11])
        self.assertEqual(trajectory.completion_tokens, [20, 21, 90, 91, 30, 31])
        self.assertEqual(trajectory.logprobs, [-0.3, -0.4, 0, 0, -0.5, -0.6])
        self.assertEqual(trajectory.loss_mask, [True, True, False, False, True, True])
        self.assertEqual(trajectory.turn_ids, [0, 0, -1, -1, 1, 1])
        self.assertEqual(trajectory.token_rewards, [0, 0.25, 0, 0, 0, 0])
        targets = build_datum(trajectory).loss_fn_inputs["target_tokens"].data
        self.assertEqual(targets, [-100, 20, 21, -100, -100, 30, 31])

    def test_process_reward_reaches_turn_gae_without_rewarding_tool_tokens(self):
        trajectory = EpisodeTrajectoryBuilder().build(episode())
        config = PPOConfig(
            use_critic=False, reward_norm=None, adv_norm=None, gae_timestep_unit="turn"
        )
        batch = prepare_advantages([trajectory], config)
        self.assertEqual(batch.advantages, [[1.25, 1.25, 0, 0, 1, 1]])

    def test_rewritten_or_shorter_prefix_is_rejected(self):
        for prompt in [(10, 12, 20, 21, 90), (10,)]:
            value = episode()
            value.turns[1].prompt_tokens = prompt
            with self.assertRaisesRegex(AgentContractError, "prefix"):
                EpisodeTrajectoryBuilder().build(value)

    def test_reward_alignment_and_rejected_episodes_fail_before_update(self):
        for rewards in ({3: (0.1,)}, {0: (0.1,)}):
            value = episode()
            value.score = EpisodeScore(1, True, process_rewards=rewards)
            with self.assertRaises(AgentContractError):
                EpisodeTrajectoryBuilder().build(value)
        value = episode()
        value.status = "tool_failure"
        with self.assertRaisesRegex(AgentContractError, "scored episodes"):
            EpisodeTrajectoryBuilder().build(value)

    def test_model_limits_mark_export_truncated(self):
        value = episode()
        value.status = "max_turns"
        value.score = EpisodeScore(0, False)
        self.assertTrue(EpisodeTrajectoryBuilder().build(value).truncated)

    def test_nonfinite_scores_or_missing_behavior_logprobs_are_rejected(self):
        for reward in (float("nan"), float("inf"), True):
            with self.assertRaises(AgentContractError):
                EpisodeScore(reward, False)
        with self.assertRaisesRegex(AgentContractError, "logprobs"):
            Generation((1,), (), "text")
