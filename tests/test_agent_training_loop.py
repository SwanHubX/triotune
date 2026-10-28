"""Agent collection -> real local Torch loss bridge -> refresh -> evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentBatchAdvantages,
    AgentContractError,
    AgentRolloutBatch,
    AgentTrainingLoop,
    EpisodeCollectionError,
    GiGPOAdvantageProvider,
    GiGPOConfig,
)
from triotune.trainer.ppo import prepare_advantages
from triotune.trainer.ppo.datum import batch_datums
from triotune.trainer.ppo.loss import make_loss_fn


class AgentTrainingTests(unittest.IsolatedAsyncioTestCase):
    def trainer(self, **kwargs):
        client = AgentTrainingClient()
        config = PPOConfig.for_algorithm(
            "grpo",
            max_steps=2,
            batch_size=1,
            num_generations=2,
            seed=2,
            eval_before_train=True,
            eval_after_train=True,
            **kwargs,
        )
        trainer = PPOTrainer(
            "fake",
            args=config,
            training_client=client,
            train_dataset=[calculator_rows()[0]] * 2,
            eval_dataset=calculator_rows()[:1],
        )
        return trainer, client

    async def test_two_updates_refresh_snapshots_and_share_agent_execution_with_evaluation(
        self,
    ):
        trainer, client = self.trainer()
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await AgentTrainingLoop(trainer, calculator_workflow()).train()
        self.assertEqual(state.global_step, 2)
        self.assertEqual(state.optimizer_steps, 2)
        self.assertEqual([s.version for s in client.snapshots], [0, 0, 1, 2])
        # Evaluation uses snapshots but makes no optimizer call.
        self.assertEqual(client.trace.count("submit:optim"), 2)
        self.assertEqual(state.metrics["eval/agent/episodes"], 2)
        self.assertEqual(state.metrics["agent/system_failure_rate"], 0)
        self.assertTrue(state.checkpoint_path.startswith("trio://"))
        call = next(call for call in client.calls if call[0] == "custom")
        self.assertTrue(
            any(-100 in datum.loss_fn_inputs["target_tokens"].data for datum in call[1])
        )

    async def test_batch_advantage_provider_runs_once_per_complete_update(self):
        trainer, client = self.trainer()

        class StateAdvantageProvider:
            def __init__(self):
                self.batch_sizes = []

            async def advantages(self, trajectories):
                self.batch_sizes.append(len(trajectories))
                return [
                    [0.5 if active else 0.0 for active in trajectory.loss_mask]
                    for trajectory in trajectories
                ]

        provider = StateAdvantageProvider()
        loop = AgentTrainingLoop(
            trainer,
            calculator_workflow(),
            batch_advantage_provider=provider,
        )
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await loop.train()

        self.assertEqual(state.global_step, 2)
        self.assertEqual(provider.batch_sizes, [2, 2])
        self.assertTrue(
            all(
                value == (0.5 if active else 0.0)
                for trajectory in loop.last_train_batch.trajectories
                for value, active in zip(
                    trajectory.token_advantages, trajectory.loss_mask, strict=True
                )
            )
        )
        # The ordinary custom PPO submission succeeds without a new PyTRIO
        # loss type; only its already-supported advantage input has changed.
        self.assertTrue(any(call[0] == "custom" for call in client.calls))

    async def test_gigpo_replacement_advantages_reach_existing_pytrio_loss(self):
        trainer, client = self.trainer(reward_norm=None, adv_norm=None, kl_ctl=0)

        class TurnAnchorObserver:
            def observe(self, episode, interaction):
                return {"turn": len(episode.turns), "task": episode.task_id}

        workflow = calculator_workflow()
        workflow.runner.anchor_observer = TurnAnchorObserver()
        loop = AgentTrainingLoop(
            trainer,
            workflow,
            batch_advantage_provider=GiGPOAdvantageProvider(
                GiGPOConfig(discount=0.95)
            ),
        )
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await loop.train()

        self.assertEqual(state.global_step, 2)
        self.assertTrue(
            all(
                trajectory.token_advantage_mode == "replace"
                for trajectory in loop.last_train_batch.trajectories
            )
        )
        self.assertTrue(any(call[0] == "custom" for call in client.calls))

    async def test_batch_advantages_add_to_existing_direct_signal(self):
        trainer, _ = self.trainer()
        policy = await trainer.create_sampling_snapshot()
        workflow = calculator_workflow()
        collected = await workflow.collect(calculator_rows()[:1], policy)
        original = collected.trajectories[0]
        original.token_advantages = [
            0.25 if active else 0.0 for active in original.loss_mask
        ]

        class SyncProvider:
            def advantages(self, trajectories):
                return AgentBatchAdvantages(
                    [
                        [0.5 if active else 0.0 for active in trajectory.loss_mask]
                        for trajectory in trajectories
                    ],
                    mode="replace",
                )

        loop = AgentTrainingLoop(
            trainer, workflow, batch_advantage_provider=SyncProvider()
        )
        batch = AgentRolloutBatch(collected.episodes, [original])
        await loop._add_batch_advantages(batch)
        self.assertEqual(
            batch.trajectories[0].token_advantages,
            [0.75 if active else 0.0 for active in original.loss_mask],
        )
        self.assertIsNot(batch.trajectories[0], original)
        self.assertEqual(batch.trajectories[0].token_advantage_mode, "replace")
        self.assertEqual(
            batch.trajectories[0].completion_tokens, original.completion_tokens
        )
        await trainer.close()

    async def test_batch_advantage_provider_rejects_misaligned_context_signal(self):
        trainer, _ = self.trainer()
        policy = await trainer.create_sampling_snapshot()
        workflow = calculator_workflow()
        collected = await workflow.collect(calculator_rows()[:1], policy)
        trajectory = collected.trajectories[0]

        class InvalidProvider:
            def __init__(self, values):
                self.values = values

            def advantages(self, trajectories):
                return [self.values]

        cases = (
            [0.0],
            [float("nan")] * len(trajectory.completion_tokens),
            [1.0] * len(trajectory.completion_tokens),
        )
        for values in cases:
            loop = AgentTrainingLoop(
                trainer,
                workflow,
                batch_advantage_provider=InvalidProvider(values),
            )
            with self.assertRaises(AgentContractError):
                await loop._add_batch_advantages(
                    AgentRolloutBatch(collected.episodes, [trajectory])
                )
        await trainer.close()

    async def test_evaluation_only_never_updates_actor_or_critic(self):
        trainer, client = self.trainer()
        metrics = await AgentTrainingLoop(trainer, calculator_workflow()).evaluate()
        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(client.version, 0)
        self.assertEqual(metrics["eval/agent/episodes"], 2)
        self.assertNotIn("submit:optim", client.trace)
        await trainer.close()

    async def test_snapshot_is_fixed_across_turns_and_changes_after_step(self):
        trainer, client = self.trainer()
        first = await trainer.create_sampling_snapshot()
        workflow = calculator_workflow()
        batch = await workflow.collect(
            calculator_rows()[:1], first, n_samples=2, seed=2
        )
        self.assertTrue(all(e.snapshot_id == first.snapshot_id for e in batch.episodes))
        await trainer.step(batch.trajectories)
        second = await trainer.create_sampling_snapshot()
        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(first.policy_version, 0)
        self.assertEqual(second.policy_version, 1)
        self.assertEqual(client.snapshots[0].version, 0)
        self.assertEqual(client.snapshots[1].version, 1)
        await trainer.close()

    async def test_calculator_trajectory_produces_action_gradients_and_zero_tool_gradients(
        self,
    ):
        trainer, _ = self.trainer()
        policy = await trainer.create_sampling_snapshot()
        collected = await calculator_workflow().collect(
            calculator_rows()[:1], policy, n_samples=2, seed=0
        )
        self.assertEqual([t.reward for t in collected.trajectories], [1, 0])
        batch = prepare_advantages(collected.trajectories, trainer.args)
        tensors = [
            torch.full(
                (len(t.prompt_tokens) + len(t.completion_tokens) - 1,),
                -1.0,
                requires_grad=True,
            )
            for t in batch.trajectories
        ]
        loss, _ = make_loss_fn(batch, trainer.args)(batch_datums(batch), tensors)
        loss.backward()
        for trajectory, tensor in zip(batch.trajectories, tensors, strict=True):
            gradients = tensor.grad[len(trajectory.prompt_tokens) - 1 :].tolist()
            self.assertTrue(
                any(
                    abs(g) > 0
                    for g, active in zip(gradients, trajectory.loss_mask, strict=True)
                    if active
                )
            )
            self.assertTrue(
                all(
                    g == 0
                    for g, active in zip(gradients, trajectory.loss_mask, strict=True)
                    if not active
                )
            )
        await trainer.close()

    async def test_system_failure_aborts_before_optimizer_and_dynamic_sampling_is_reused(
        self,
    ):
        trainer, client = self.trainer()
        workflow = calculator_workflow()

        class UnavailableScorer:
            async def score(self, row, episode):
                raise ConnectionError("judge unavailable")

        workflow.runner.scorer = UnavailableScorer()
        with self.assertRaises(EpisodeCollectionError):
            await AgentTrainingLoop(trainer, workflow).train()
        self.assertNotIn("submit:optim", client.trace)
        trainer, client = self.trainer(dynamic_sampling=True)
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await AgentTrainingLoop(trainer, calculator_workflow()).train()
        self.assertEqual(state.global_step, 2)

    async def test_error_reporting_preserves_failure_and_writes_serializable_history(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            trainer, client = self.trainer(log_dir=directory)

            class BrokenScorer:
                async def score(self, row, episode):
                    raise ConnectionError("scoring service unavailable")

            workflow = calculator_workflow()
            workflow.runner.scorer = BrokenScorer()
            with self.assertRaises(EpisodeCollectionError):
                await AgentTrainingLoop(trainer, workflow).train()
            self.assertNotIn("submit:optim", client.trace)
            files = list(Path(directory).rglob("*.jsonl"))
            self.assertTrue(files)
            records = [
                json.loads(line)
                for file in files
                for line in file.read_text().splitlines()
            ]
            self.assertTrue(
                any("scoring service unavailable" in str(record) for record in records)
            )

    async def test_empty_training_or_one_shot_eval_fails_explicitly(self):
        trainer, _ = self.trainer()
        trainer.train_dataset = []
        with (
            patch("triotune.agentrl.training.loop.tqdm"),
            self.assertRaisesRegex(ValueError, "No trainable"),
        ):
            await AgentTrainingLoop(trainer, calculator_workflow()).train()
        trainer, _ = self.trainer()
        trainer.eval_dataset = iter(calculator_rows())
        with self.assertRaisesRegex(ValueError, "replayable"):
            await AgentTrainingLoop(trainer, calculator_workflow()).train()
