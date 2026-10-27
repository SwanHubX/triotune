"""Exercise candidate identity, trace evidence, export boundaries, and PPO math.

All samplers and training clients are local deterministic fixtures. These tests
verify source-to-gradient relationships without requiring a remote service.
"""

import asyncio
import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import AsyncMock

from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot
from test_agent_trajectory_builder import episode as process_episode
from test_ppo_family import objective, raw_config, trajectory

from triotune import PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentContractError,
    EpisodeCollectionError,
    EpisodeScore,
    EpisodeTrajectoryBuilder,
    ExecutionContext,
    FunctionTool,
    FunctionToolExecutor,
    Generation,
    GenerationEvent,
    JsonlEpisodeLogger,
    RetryPolicy,
    RewardEvent,
    RolloutExport,
    ToolCallEvent,
    ToolResultEvent,
    current_execution_context,
)
from triotune.trainer.ppo import (
    NormConfig,
    prepare_advantages,
    validate_policy_versions,
)
from triotune.trainer.ppo.actor import minibatches
from triotune.trainer.ppo.critic import critic_datums
from triotune.trainer.ppo.datum import batch_datums


class ContextTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_is_visible_in_sync_tools_scorer_and_logger_and_isolated(
        self,
    ):
        policy, _ = snapshot()
        tool_contexts, scores, logs = [], [], []

        def calculate(**kwargs):
            context = current_execution_context()
            tool_contexts.append(context)
            with self.assertRaises(FrozenInstanceError):
                context.sample_index = 99
            return 5

        class Scorer:
            async def score(self, row, episode):
                scores.append((current_execution_context(), episode.context))
                await asyncio.sleep(0)
                return EpisodeScore(1, True)

        class Logger:
            async def write(self, episode):
                logs.append((current_execution_context(), episode.context))

        workflow = calculator_workflow(logger=Logger())
        workflow.runner.tools = FunctionToolExecutor(
            [FunctionTool("calculate", "Calculator", {"type": "object"}, calculate)]
        )
        workflow.runner.scorer = Scorer()
        batch = await workflow.collect(calculator_rows()[:1], policy, n_samples=3)
        self.assertEqual(sorted(c.sample_index for c in tool_contexts), [0, 1, 2])
        self.assertEqual(len({c.rollout_id for c in tool_contexts}), 3)
        self.assertTrue(
            all(c.group_size == 3 and c.task_id == "row-0" for c in tool_contexts)
        )
        self.assertTrue(all(c.interaction_id for c in tool_contexts))
        self.assertTrue(all(left == right for left, right in scores + logs))
        self.assertIsNone(current_execution_context())
        self.assertEqual(len(batch.exports), 3)

    async def test_trace_links_tokens_tools_rewards_retries_and_json(self):
        policy, client = snapshot()
        original = client.sample_async
        attempts = 0

        async def transient(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary")
            return await original(**kwargs)

        client.sample_async = transient
        with tempfile.TemporaryDirectory() as directory:
            logger = JsonlEpisodeLogger(Path(directory) / "trace.jsonl")
            workflow = calculator_workflow(logger=logger)
            workflow.runner.retry = RetryPolicy(max_attempts=2, initial_delay=0)
            batch = await workflow.collect(calculator_rows()[:1], policy)
            episode = batch.episodes[0]
            first, last = episode.interactions
            self.assertIsNone(first.parent_interaction_id)
            self.assertEqual(last.parent_interaction_id, first.interaction_id)
            self.assertEqual(first.retries, episode.retries)
            self.assertGreater(first.latency, 0)
            for interaction in episode.interactions:
                turn = interaction.turn
                n = len(turn.prompt_tokens)
                self.assertEqual(interaction.input_token_span, (0, n))
                self.assertEqual(
                    interaction.output_token_span, (n, n + len(turn.generation.tokens))
                )
                self.assertEqual(
                    interaction.resource_owner, episode.context.resource_owner
                )
            self.assertEqual(len(first.tool_calls), 1)
            self.assertEqual(first.tool_results[0].content, "5")
            self.assertEqual(last.reward, episode.score.reward)
            for event_type, count in (
                (GenerationEvent, 2),
                (ToolCallEvent, 1),
                (ToolResultEvent, 1),
                (RewardEvent, 2),
            ):
                self.assertEqual(
                    sum(isinstance(e, event_type) for e in episode.events), count
                )
            saved = json.loads(logger.path.read_text())
            self.assertEqual(
                saved["interactions"][1]["context"]["parent_interaction_id"],
                first.interaction_id,
            )
            self.assertIn("retry", [e["kind"] for e in saved["events"]])
            self.assertEqual(saved["events"][-1]["status"], "completed")
            last.context = replace(last.context, parent_interaction_id="missing")
            with self.assertRaisesRegex(AgentContractError, "lineage"):
                episode.__post_init__()

    async def test_failure_and_cancellation_restore_context_and_preserve_failed_request(
        self,
    ):
        policy, _ = snapshot(failure=ConnectionError("offline"))
        batch = await calculator_workflow().collect(
            calculator_rows()[:1], policy, strict=False
        )
        episode = batch.episodes[0]
        self.assertEqual(episode.interactions[0].status, "sampling_failure")
        self.assertIsNone(episode.interactions[0].output_token_span)
        self.assertFalse(any(isinstance(e, RewardEvent) for e in episode.events))
        self.assertIsNone(current_execution_context())

        policy, _ = snapshot()
        workflow = calculator_workflow()
        entered = asyncio.Event()

        async def wait_forever(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        workflow.runner.tools = FunctionToolExecutor(
            [FunctionTool("calculate", "", {"type": "object"}, wait_forever)]
        )
        task = asyncio.create_task(
            workflow.runner.run(calculator_rows()[0], policy, group_id="g")
        )
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(current_execution_context())

    def test_context_and_lineage_reject_invalid_identity(self):
        base = {
            "rollout_id": "r",
            "episode_id": "e",
            "group_id": "g",
            "sample_index": 0,
            "snapshot_id": "s",
        }
        for values in (
            {"sample_index": 1},
            {"group_size": 0},
            {"policy_version": True},
            {"parent_interaction_id": "i"},
        ):
            with self.subTest(values=values), self.assertRaises(AgentContractError):
                ExecutionContext(**{**base, **values})


class ExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_branch_paths_share_actions_without_rewriting_causal_context(self):
        policy, _ = snapshot()
        workflow = calculator_workflow()
        original = workflow.runner._execute

        async def branch(episode, *args):
            await original(episode, *args)
            sibling = copy.deepcopy(episode.interactions[-1])
            sibling.context = replace(sibling.context, interaction_id="sibling")
            tokens = tuple(policy.tokenizer.encode('{"final":"6"}') + [1])
            sibling.turn.generation = Generation(
                tokens, tuple([-1.0] * len(tokens)), '{"final":"6"}'
            )
            sibling.turn.action = workflow.runner.parser.parse(sibling.turn.generation)
            n = len(sibling.turn.prompt_tokens)
            sibling.output_token_span = (n, n + len(tokens))
            episode.turns.append(sibling.turn)
            episode.interactions.append(sibling)

        class Paths(EpisodeTrajectoryBuilder):
            def export(self, episode):
                rows = []
                for i in (1, 2):
                    path = replace(
                        episode,
                        turns=[episode.turns[0], episode.turns[i]],
                        interactions=[episode.interactions[0], episode.interactions[i]],
                    )
                    row = self.build(path)
                    row.fragment_id += f":path:{i}"
                    rows.append(row)
                return RolloutExport(
                    rows,
                    episode.rollout_id,
                    {row.fragment_id: row.interaction_ids for row in rows},
                    episode.group_id,
                )

        workflow.runner._execute = branch
        workflow.runner.trajectory_builder = Paths()
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(len(batch.trajectories), 2)
        first, second = batch.trajectories
        root = batch.episodes[0].interactions[0].interaction_id
        self.assertTrue(
            any(
                key[0] == root
                for key, active in zip(first.action_keys, first.loss_mask)
                if active
            )
        )
        self.assertFalse(
            any(
                key[0] == root
                for key, active in zip(second.action_keys, second.loss_mask)
                if active
            )
        )
        self.assertEqual(batch.episodes[0].interactions[-1].parent_interaction_id, root)

    async def test_individual_exports_keep_identity_prompt_masks_and_pass_at_k(self):
        policy, _ = snapshot()
        workflow = calculator_workflow()
        workflow.runner.trajectory_builder = EpisodeTrajectoryBuilder(mode="individual")
        batch = await workflow.collect(
            calculator_rows()[:1], policy, n_samples=2, seed=0
        )
        self.assertEqual(len(batch.episodes), 2)
        self.assertEqual(len(batch.trajectories), 3)
        self.assertEqual(len(batch.exports), 2)
        self.assertEqual(batch.metrics()["agent/pass@1"], 0.5)
        self.assertEqual(len({tuple(t.prompt_tokens) for t in batch.trajectories}), 1)
        keys = [
            key
            for t in batch.trajectories
            for key, active in zip(t.action_keys, t.loss_mask)
            if active
        ]
        self.assertEqual(len(keys), len(set(keys)))
        for exported in batch.exports:
            self.assertEqual(
                set(exported.fragment_map),
                {t.fragment_id for t in exported.trajectories},
            )
            self.assertTrue(
                all(
                    t.rollout_id == exported.logical_rollout_id
                    for t in exported.trajectories
                )
            )
        prepared = prepare_advantages(
            batch.trajectories, raw_config(reward_norm=NormConfig("group", None))
        )
        self.assertEqual(
            [
                next(a for a, active in zip(row, mask) if active)
                for row, mask in zip(prepared.advantages, prepared.masks)
            ],
            [0.5, 0.5, -0.5],
        )
        datums = batch_datums(prepared)
        self.assertEqual(
            datums.provenance, tuple(t.provenance for t in prepared.trajectories)
        )
        critic = critic_datums(prepared.trajectories, (1, 2))
        self.assertEqual(len(critic.provenance), 2 * len(prepared.trajectories))
        self.assertEqual(critic.provenance[::2], datums.provenance)
        self.assertEqual(
            batch.trajectories[1].parent_interaction_id,
            batch.episodes[0].interactions[0].interaction_id,
        )
        chunk = next(minibatches(prepared, 1))
        self.assertEqual(batch_datums(chunk).provenance, datums.provenance)

    async def test_partial_group_policy_counts_candidates_not_fragments(self):
        policy, client = snapshot()
        original = client.sample_async

        async def fail_one(**kwargs):
            if kwargs["sampling_params"].seed == 8:
                raise ConnectionError("one candidate failed")
            return await original(**kwargs)

        client.sample_async = fail_one
        workflow = calculator_workflow()
        workflow.runner.trajectory_builder = EpisodeTrajectoryBuilder(mode="individual")
        with self.assertRaises(EpisodeCollectionError):
            await workflow.collect(calculator_rows()[:1], policy, n_samples=2, seed=0)
        workflow.min_usable_group_size = 1
        batch = await workflow.collect(
            calculator_rows()[:1], policy, n_samples=2, seed=0
        )
        self.assertEqual(len(batch.exports), 1)
        self.assertEqual(len(batch.trajectories), 2)
        self.assertEqual(batch.dropped_groups, 0)
        state = next(iter(batch.groups.values()))
        self.assertFalse(state["complete"])
        self.assertTrue(state["accepted"])
        self.assertEqual(state["usable_candidates"], [0])
        workflow.min_usable_group_size = 3
        with self.assertRaises(ValueError):
            await workflow.collect(calculator_rows()[:1], policy, n_samples=2)

    async def test_overlap_policy_and_malformed_exports(self):
        class DuplicateExporter(EpisodeTrajectoryBuilder):
            def export(self, episode):
                result = super().export(episode)
                duplicate = copy.deepcopy(result.trajectories[0])
                duplicate.fragment_id += ":duplicate"
                result.trajectories.append(duplicate)
                result.fragment_map[duplicate.fragment_id] = duplicate.interaction_ids
                return result

        policy, _ = snapshot()
        workflow = calculator_workflow()
        workflow.runner.trajectory_builder = DuplicateExporter()
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(len(batch.trajectories), 1)
        workflow.shared_prefix_policy = "repeat"
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(len(batch.trajectories), 2)
        for field, value in (
            ("prompt_tokens", [999]),
            ("episode_id", "wrong"),
            ("logprobs", None),
        ):

            class InvalidExporter(DuplicateExporter):
                def export(self, episode, field=field, value=value):
                    result = super().export(episode)
                    setattr(result.trajectories[-1], field, value)
                    return result

            workflow.runner.trajectory_builder = InvalidExporter()
            batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
            self.assertEqual(batch.episodes[0].status, "contract_error")
            self.assertFalse(batch.trajectories)

        class RewrittenContext(EpisodeTrajectoryBuilder):
            def export(self, episode):
                result = super().export(episode)
                # The first token in the second row is masked context. A
                # matching active token/logprob must not conceal its rewrite.
                result.trajectories[1].completion_tokens[0] += 1
                return result

        workflow.runner.trajectory_builder = RewrittenContext(mode="individual")
        batch = await workflow.collect(calculator_rows()[:1], policy, strict=False)
        self.assertEqual(batch.episodes[0].status, "contract_error")
        self.assertIn("causal context", batch.episodes[0].error)

    async def test_individual_rows_reach_real_local_ppo_loss_and_refresh(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer("fake", args=raw_config(), training_client=client)
        snapshot = await trainer.create_sampling_snapshot()
        workflow = calculator_workflow()
        workflow.runner.trajectory_builder = EpisodeTrajectoryBuilder(mode="individual")
        batch = await workflow.collect(
            calculator_rows()[:1], snapshot, n_samples=2, seed=0
        )
        metrics = await trainer.step(batch.trajectories)
        self.assertEqual(trainer.state.global_step, 1)
        self.assertEqual(metrics["reward/mean"], 0.5)
        self.assertEqual(metrics["train/logical_rollouts"], 2)
        self.assertTrue(trainer._actor.datum_provenance)
        trainer._references = AsyncMock()
        with self.assertRaisesRegex(ValueError, "Stale"):
            await trainer.step(batch.trajectories)
        trainer._references.assert_not_awaited()
        await trainer.close()


class FragmentMathTests(unittest.TestCase):
    def test_individual_process_rewards_remain_on_their_original_turn(self):
        exported = EpisodeTrajectoryBuilder(mode="individual").export(process_episode())
        first, second = exported.trajectories
        self.assertEqual(sum(first.token_rewards), 0.25)
        self.assertEqual(sum(second.token_rewards), 0)
        batch = prepare_advantages(
            exported.trajectories, raw_config(gae_timestep_unit="turn")
        )
        self.assertEqual(
            [
                value
                for value, active in zip(batch.advantages[0], first.loss_mask)
                if active
            ],
            [1.25, 1.25],
        )
        self.assertEqual(
            [
                value
                for value, active in zip(batch.advantages[1], second.loss_mask)
                if active
            ],
            [1, 1],
        )

    def test_loss_reductions_do_not_weight_a_candidate_by_fragment_count(self):
        whole = [
            trajectory(reward=1, n=4, rollout_id="a"),
            trajectory(reward=3, n=2, rollout_id="b"),
        ]
        fragments = [
            trajectory(reward=1, n=2, rollout_id="a"),
            trajectory(
                reward=1, n=4, rollout_id="a", loss_mask=[False, False, True, True]
            ),
            whole[1],
        ]
        for importance in ("token", "sequence"):
            for reduction in ("token_mean", "seq_mean", "prompt_mean", "constant"):
                config = raw_config(
                    importance_sampling_level=importance,
                    loss_aggregation=reduction,
                    **(
                        {"loss_aggregation_divisor": 5}
                        if reduction == "constant"
                        else {}
                    ),
                )
                base, base_grad, _ = objective(
                    config, whole, [[-0.95, -1.02, -0.9, -1.1], [-1.05] * 2]
                )
                split, split_grad, _ = objective(
                    config,
                    fragments,
                    [[-0.95, -1.02], [-0.95, -1.02, -0.9, -1.1], [-1.05] * 2],
                )
                self.assertAlmostEqual(base, split)
                self.assertEqual(
                    base_grad[0][1:], split_grad[0][1:] + split_grad[1][3:]
                )
                self.assertEqual(base_grad[1], split_grad[2])

    def test_staleness_supports_bounded_numeric_age_and_exact_labels(self):
        validate_policy_versions(
            [trajectory(policy_version=2)], current_version=3, max_staleness=1
        )
        validate_policy_versions(
            [trajectory(policy_version="checkpoint")], current_version="checkpoint"
        )
        for version, current in ((2, 3), (4, 3), ("other", "checkpoint")):
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate_policy_versions(
                    [trajectory(policy_version=version)], current_version=current
                )
        with self.assertRaises(ValueError):
            validate_policy_versions([trajectory(policy_version=1), trajectory()])

    def test_configuration_exposes_export_policy(self):
        config = AgentConfig(export_mode="individual", min_usable_group_size=2)
        self.assertEqual(AgentConfig.from_dict(config.to_dict()), config)
        for values in (
            {"export_mode": "bad"},
            {"min_usable_group_size": 0},
            {"shared_prefix_policy": "bad"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AgentConfig(**values)
