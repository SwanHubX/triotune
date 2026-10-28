"""Numerical and contract tests for the AgentRL GiGPO provider."""

import unittest
from types import SimpleNamespace

from triotune import PPOConfig
from triotune.agentrl import (
    AgentBatchAdvantages,
    AgentContractError,
    GiGPOAdvantageProvider,
    GiGPOConfig,
)
from triotune.trainer.ppo import prepare_advantages
from triotune.trainer.ppo.trajectory import Trajectory


def trajectory(
    rollout_id,
    *,
    outcome,
    anchors,
    step_rewards=None,
    group_id="task",
    context_segment=0,
    interaction_offset=0,
):
    """Build active turns separated by one masked environment token."""

    step_rewards = step_rewards or [0.0] * len(anchors)
    completion_tokens = []
    logprobs = []
    loss_mask = []
    turn_ids = []
    token_rewards = []
    for turn_id, reward in enumerate(step_rewards):
        if turn_id:
            completion_tokens.append(90 + turn_id)
            logprobs.append(0.0)
            loss_mask.append(False)
            turn_ids.append(-1)
            token_rewards.append(0.0)
        completion_tokens.append(10 + turn_id)
        logprobs.append(-1.0)
        loss_mask.append(True)
        turn_ids.append(turn_id)
        token_rewards.append(float(reward))
    interaction_ids = tuple(
        f"{rollout_id}:interaction:{interaction_offset + index}"
        for index in range(len(anchors))
    )
    return Trajectory(
        prompt_tokens=[1],
        completion_tokens=completion_tokens,
        logprobs=logprobs,
        reward=float(outcome),
        group_id=group_id,
        loss_mask=loss_mask,
        token_rewards=token_rewards,
        turn_ids=turn_ids,
        rollout_id=rollout_id,
        interaction_ids=interaction_ids,
        anchor_observations=tuple(anchors),
        context_segment=context_segment,
    )


class GiGPOAdvantageTests(unittest.TestCase):
    def test_joint_advantage_uses_discounted_same_state_returns(self):
        rows = [
            trajectory("a", outcome=1, anchors=["shared", "a-final"]),
            trajectory("b", outcome=0, anchors=["shared", "b-final"]),
        ]
        result = GiGPOAdvantageProvider(
            GiGPOConfig(discount=0.5, step_advantage_weight=2.0)
        ).advantages(rows)

        self.assertIsInstance(result, AgentBatchAdvantages)
        self.assertEqual(result.mode, "replace")
        # Episode returns [1, 0] yield A^E=[0.5, -0.5]. At the shared first
        # state, discounted returns [0.5, 0] yield A^S=[0.25, -0.25]. The
        # distinct final anchors are singleton groups and contribute zero.
        self.assertEqual(result.values[0], [1.0, 0.0, 0.5])
        self.assertEqual(result.values[1], [-1.0, 0.0, -0.5])

    def test_process_rewards_are_aggregated_by_turn_before_discounting(self):
        rows = [
            trajectory(
                "a",
                outcome=0,
                anchors=["state-0", "state-1"],
                step_rewards=[1, 2],
            ),
            trajectory(
                "b",
                outcome=0,
                anchors=["state-0", "state-1"],
                step_rewards=[0, 0],
            ),
        ]
        result = GiGPOAdvantageProvider(
            GiGPOConfig(discount=0.5)
        ).advantages(rows)
        # Episode totals [3, 0] produce +/-1.5. Both state returns are [2, 0]
        # after the turn recurrence, so each state advantage is +/-1.
        self.assertEqual(result.values[0], [2.5, 0.0, 2.5])
        self.assertEqual(result.values[1], [-2.5, 0.0, -2.5])

    def test_similarity_grouping_is_optional_and_text_only(self):
        rows = [
            trajectory("a", outcome=1, anchors=["room alpha", "kitchen"]),
            trajectory("b", outcome=0, anchors=["room alphi", "outside"]),
        ]
        exact = GiGPOAdvantageProvider(
            GiGPOConfig(discount=0.5)
        ).advantages(rows)
        similar = GiGPOAdvantageProvider(
            GiGPOConfig(
                discount=0.5,
                enable_similarity=True,
                similarity_threshold=0.8,
            )
        ).advantages(rows)

        self.assertEqual(exact.values[0], [0.5, 0.0, 0.5])
        self.assertEqual(exact.values[1], [-0.5, 0.0, -0.5])
        self.assertEqual(similar.values[0], [0.75, 0.0, 0.5])
        self.assertEqual(similar.values[1], [-0.75, 0.0, -0.5])
        rows[0].anchor_observations = ({"room": 1}, "kitchen")
        with self.assertRaisesRegex(AgentContractError, "only text"):
            GiGPOAdvantageProvider(
                GiGPOConfig(enable_similarity=True)
            ).advantages(rows)

    def test_episode_weighting_selects_trajectory_or_step_statistics(self):
        rows = [
            trajectory("a", outcome=2, anchors=["a-0", "a-1"]),
            trajectory("b", outcome=0, anchors=["b-0"]),
        ]
        by_trajectory = GiGPOAdvantageProvider(
            GiGPOConfig(episode_statistic_weighting="trajectory")
        ).advantages(rows)
        by_step = GiGPOAdvantageProvider(
            GiGPOConfig(episode_statistic_weighting="step")
        ).advantages(rows)

        self.assertEqual(by_trajectory.values[0], [1.0, 0.0, 1.0])
        self.assertEqual(by_trajectory.values[1], [-1.0])
        self.assertAlmostEqual(by_step.values[0][0], 2 / 3)
        self.assertAlmostEqual(by_step.values[0][2], 2 / 3)
        self.assertAlmostEqual(by_step.values[1][0], -4 / 3)

    def test_mean_std_mode_uses_sample_standard_deviation(self):
        rows = [
            trajectory("a", outcome=1, anchors=["a-state"]),
            trajectory("b", outcome=0, anchors=["b-state"]),
        ]
        result = GiGPOAdvantageProvider(
            GiGPOConfig(
                mode="mean_std_norm",
                epsilon=1e-12,
                step_advantage_weight=0,
            )
        ).advantages(rows)
        self.assertAlmostEqual(result.values[0][0], 1 / (2**0.5), places=9)
        self.assertAlmostEqual(result.values[1][0], -1 / (2**0.5), places=9)

    def test_fragments_join_one_logical_candidate_and_count_outcome_once(self):
        rows = [
            trajectory(
                "a",
                outcome=1,
                anchors=["shared"],
                context_segment=0,
                interaction_offset=0,
            ),
            trajectory(
                "a",
                outcome=1,
                anchors=["a-final"],
                context_segment=1,
                interaction_offset=1,
            ),
            trajectory(
                "b",
                outcome=0,
                anchors=["shared"],
                context_segment=0,
                interaction_offset=0,
            ),
            trajectory(
                "b",
                outcome=0,
                anchors=["b-final"],
                context_segment=1,
                interaction_offset=1,
            ),
        ]
        result = GiGPOAdvantageProvider(
            GiGPOConfig(discount=0.5, step_advantage_weight=2.0)
        ).advantages(rows)
        self.assertEqual([row[0] for row in result.values], [1.0, 0.5, -1.0, -0.5])

    def test_missing_anchors_and_invalid_config_fail_explicitly(self):
        value = trajectory("a", outcome=1, anchors=["state"])
        value.anchor_observations = (None,)
        with self.assertRaisesRegex(AgentContractError, "anchor observation"):
            GiGPOAdvantageProvider().advantages([value])
        for kwargs in (
            {"discount": 1.1},
            {"step_advantage_weight": -1},
            {"epsilon": 0},
            {"mode": "unknown"},
            {"enable_similarity": True, "similarity_threshold": 1.0},
            {"episode_statistic_weighting": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GiGPOConfig(**kwargs)

    def test_trainer_validation_rejects_bypassed_reward_stages(self):
        settings = {
            "use_critic": False,
            "reward_norm": None,
            "adv_norm": None,
            "kl_ctl": 0.0,
            "mask_no_eos_with_zero": False,
            "overlong_tokens": 0,
        }
        provider = GiGPOAdvantageProvider()
        provider.validate_trainer(
            SimpleNamespace(args=SimpleNamespace(**settings), algorithm_plan=None)
        )
        for name, value in (
            ("use_critic", True),
            ("reward_norm", object()),
            ("adv_norm", object()),
            ("kl_ctl", 0.01),
            ("mask_no_eos_with_zero", True),
            ("overlong_tokens", 8),
        ):
            invalid = dict(settings)
            invalid[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                provider.validate_trainer(
                    SimpleNamespace(
                        args=SimpleNamespace(**invalid), algorithm_plan=None
                    )
                )

    def test_replacement_rows_bypass_base_gae_but_reuse_ppo_loss_contract(self):
        rows = [
            trajectory("a", outcome=1, anchors=["shared", "a-final"]),
            trajectory("b", outcome=0, anchors=["shared", "b-final"]),
        ]
        result = GiGPOAdvantageProvider(GiGPOConfig(discount=0.5)).advantages(rows)
        for trajectory_row, values in zip(rows, result.values, strict=True):
            trajectory_row.token_advantages = list(values)
            trajectory_row.token_advantage_mode = result.mode
        config = PPOConfig.for_algorithm(
            "grpo", reward_norm=None, adv_norm=None, use_critic=False
        )
        prepared = prepare_advantages(rows, config)
        self.assertEqual(prepared.advantages, [list(row) for row in result.values])


if __name__ == "__main__":
    unittest.main()
