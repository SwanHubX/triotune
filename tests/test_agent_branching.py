"""Branch-tree export, shared-prefix weighting, and PPO grouping contracts."""

import copy
import unittest
from dataclasses import replace

import torch
from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot
from test_ppo_family import objective, raw_config, trajectory

from triotune import PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentContractError,
    EpisodeBranch,
    EpisodeTrajectoryBuilder,
)
from triotune.trainer.ppo import NormConfig, prepare_advantages, value_loss


def attach_counterfactual_branches(workflow, *, completed=True):
    """Make one deterministic calculator episode expose two sibling leaves.

    The fixture copies the already sampled terminal interaction so both leaves
    retain valid original token/logprob evidence. Only interaction identity and
    branch-local metadata differ; the common tool-producing interaction remains
    the exact same object and therefore has the same stable action keys in both
    exported paths.
    """

    original_run = workflow.runner.run

    async def run(*args, **kwargs):
        episode = await original_run(*args, **kwargs)
        root, left = episode.interactions
        right = copy.deepcopy(left)
        right.context = replace(right.context, interaction_id="right-leaf")
        episode.turns.append(right.turn)
        episode.interactions.append(right)
        episode.branches = [
            EpisodeBranch(
                branch_id="tool-a",
                interaction_ids=(left.interaction_id,),
                parent_interaction_id=root.interaction_id,
                group_id="tool-choice",
                reward=1.0,
                shared_prefix_weight=1.0,
            ),
            EpisodeBranch(
                branch_id="tool-b",
                interaction_ids=(right.interaction_id,),
                parent_interaction_id=root.interaction_id,
                group_id="tool-choice",
                reward=3.0 if completed else None,
                completed=completed,
                shared_prefix_weight=3.0,
            ),
        ]
        episode.__post_init__()
        return episode

    workflow.runner.run = run
    workflow.runner.trajectory_builder = EpisodeTrajectoryBuilder(mode="branch")
    return workflow


class EpisodeBranchContractTests(unittest.TestCase):
    def test_public_config_exposes_branch_export_and_weighted_prefixes(self):
        config = AgentConfig(export_mode="branch", shared_prefix_policy="weighted")
        self.assertEqual(AgentConfig.from_dict(config.to_dict()), config)

    def test_branch_records_reject_invalid_reward_weight_and_identity(self):
        valid = {
            "branch_id": "a",
            "interaction_ids": ("child",),
            "group_id": "fork",
            "reward": 1.0,
        }
        for changed in (
            {"branch_id": ""},
            {"interaction_ids": ()},
            {"completed": True, "reward": None},
            {"reward": float("nan")},
            {"shared_prefix_weight": -1},
        ):
            with self.subTest(changed=changed), self.assertRaises(AgentContractError):
                EpisodeBranch(**{**valid, **changed})

    def test_actor_and_critic_apply_the_same_shared_action_weights(self):
        left = trajectory(
            reward=1,
            rollout_id="tree",
            branch_id="left",
            branch_group_id="fork",
            branch_completed=True,
            action_weights=[0.25, 1.0],
        )
        right = trajectory(
            reward=3,
            rollout_id="tree",
            branch_id="right",
            branch_group_id="fork",
            branch_completed=True,
            action_weights=[0.75, 1.0],
        )
        _, gradients, _ = objective(
            raw_config(reward_norm=NormConfig("group", None)),
            [left, right],
            [[-1.0, -1.0], [-1.0, -1.0]],
        )
        self.assertAlmostEqual(abs(gradients[1][1]) / abs(gradients[0][1]), 3.0)

        values = torch.tensor([1.0, 3.0])
        weighted = value_loss(
            values,
            values,
            torch.zeros(2),
            torch.ones(2, dtype=torch.bool),
            weights=torch.tensor([0.25, 0.75]),
        )
        self.assertAlmostEqual(weighted.item(), 3.5)


class BranchWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_tree_export_preserves_lineage_rewards_groups_and_prefix_weights(
        self,
    ):
        policy, _ = snapshot()
        workflow = attach_counterfactual_branches(calculator_workflow())
        workflow.shared_prefix_policy = "weighted"
        batch = await workflow.collect(calculator_rows()[:1], policy)

        self.assertEqual(len(batch.exports), 1)
        self.assertEqual(len(batch.trajectories), 2)
        exported = batch.exports[0]
        self.assertEqual(set(exported.branch_map), {"tool-a", "tool-b"})
        self.assertEqual({row.reward for row in batch.trajectories}, {1.0, 3.0})
        self.assertEqual(
            {row.branch_id for row in batch.trajectories}, {"tool-a", "tool-b"}
        )
        self.assertEqual(len(batch.branch_groups), 1)
        branch_group = next(iter(batch.branch_groups.values()))
        self.assertTrue(branch_group["complete"])
        self.assertEqual(branch_group["completed_branch_ids"], ["tool-a", "tool-b"])

        root = batch.episodes[0].interactions[0].interaction_id
        shared_weights = {}
        local_weights = {}
        for row in batch.trajectories:
            shared_weights[row.branch_id] = [
                weight
                for key, weight, active in zip(
                    row.action_keys,
                    row.action_weights,
                    row.loss_mask,
                    strict=True,
                )
                if active and key[0] == root
            ]
            local_weights[row.branch_id] = [
                weight
                for key, weight, active in zip(
                    row.action_keys,
                    row.action_weights,
                    row.loss_mask,
                    strict=True,
                )
                if active and key[0] != root
            ]
        self.assertTrue(all(weight == 0.25 for weight in shared_weights["tool-a"]))
        self.assertTrue(all(weight == 0.75 for weight in shared_weights["tool-b"]))
        self.assertTrue(
            all(weight == 1.0 for values in local_weights.values() for weight in values)
        )

        prepared = prepare_advantages(
            batch.trajectories,
            raw_config(reward_norm=NormConfig("group", None)),
        )
        terminal_advantages = {
            row.branch_id: next(
                value
                for value, active in zip(advantages[::-1], mask[::-1], strict=True)
                if active
            )
            for row, advantages, mask in zip(
                prepared.trajectories,
                prepared.advantages,
                prepared.masks,
                strict=True,
            )
        }
        self.assertEqual(terminal_advantages, {"tool-a": -1.0, "tool-b": 1.0})

    async def test_incomplete_sibling_group_is_not_trained_as_zero_reward(self):
        policy, _ = snapshot()
        workflow = attach_counterfactual_branches(
            calculator_workflow(), completed=False
        )
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertFalse(batch.trajectories)
        self.assertFalse(batch.exports)
        self.assertEqual(batch.episodes[0].status, "contract_error")
        self.assertIn("complete branch group", batch.episodes[0].error)
        self.assertEqual(len(batch.episodes[0].branches), 2)

    async def test_weighted_tree_reaches_the_existing_pytrio_update_boundary(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake", args=raw_config(loss_backend="native"), training_client=client
        )
        policy = await trainer.create_sampling_snapshot()
        workflow = attach_counterfactual_branches(calculator_workflow())
        workflow.shared_prefix_policy = "weighted"
        batch = await workflow.collect(calculator_rows()[:1], policy)

        physical_actions = sum(sum(row.loss_mask) for row in batch.trajectories)
        weighted_actions = sum(
            sum(row.effective_action_weights) for row in batch.trajectories
        )
        self.assertLess(weighted_actions, physical_actions)
        metrics = await trainer.step(batch.trajectories)
        self.assertEqual(trainer.state.global_step, 1)
        self.assertEqual(metrics["train/logical_rollouts"], 1)
        self.assertEqual(metrics["train/logical_sequences"], 2)
        self.assertAlmostEqual(metrics["train/action_tokens"], weighted_actions)
        self.assertIn("custom", [name for name, *_ in client.calls])
        self.assertNotIn("native", [name for name, *_ in client.calls])
        self.assertTrue(trainer._actor.datum_provenance)
        await trainer.close()

    async def test_branch_identity_tampering_is_rejected_before_ppo(self):
        class TamperedBranchExporter(EpisodeTrajectoryBuilder):
            def export(self, episode):
                exported = super().export(episode)
                exported.trajectories[0].branch_id = "foreign"
                return exported

        policy, _ = snapshot()
        workflow = attach_counterfactual_branches(calculator_workflow())
        workflow.runner.trajectory_builder = TamperedBranchExporter(mode="branch")
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertFalse(batch.trajectories)
        self.assertEqual(batch.episodes[0].status, "contract_error")
        self.assertIn("undeclared branch identity", batch.episodes[0].error)
