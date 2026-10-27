"""Reward submission identity, aggregation, and policy-only trajectory signals."""

import unittest
from dataclasses import replace

from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_branching import attach_counterfactual_branches
from test_agent_episode_runner import snapshot
from test_agent_trajectory_builder import episode
from test_p4_efficiency import long_snapshot
from test_ppo_family import objective, raw_config

from triotune import PPOTrainer
from triotune.agentrl import (
    AgentContractError,
    ContextCompactor,
    EpisodeLimits,
    EpisodeScore,
    EpisodeTrajectoryBuilder,
    RewardAssignment,
    RewardLedger,
    TokenAppendRenderer,
)
from triotune.agentrl.training.rewards import aggregate_assignments
from triotune.trainer.ppo import prepare_advantages


def advantage_episode(*assignments):
    value = episode()
    value.score = EpisodeScore(0.0, False, assignments=assignments)
    return value


class LedgerSemanticsTests(unittest.TestCase):
    def test_whole_payload_aggregates_before_legacy_rollout_deduplication(self):
        for aggregation, expected in (("sum", 6.0), ("mean", 3.0), ("last", 4.0)):
            for mode in ("reward", "advantage"):
                with self.subTest(aggregation=aggregation, mode=mode):
                    ledger = RewardLedger()
                    values = tuple(
                        RewardAssignment(
                            "trajectory", v, mode=mode, aggregation=aggregation
                        )
                        for v in (2.0, 4.0)
                    )
                    self.assertEqual(
                        ledger.submit(values, logical_rollout_id="r"), values
                    )
                    self.assertEqual(ledger.submit(values, logical_rollout_id="r"), ())
                    outcome, _, direct = ledger.aggregate()
                    self.assertEqual(
                        outcome if mode == "reward" else direct[None][0], expected
                    )
                    self.assertEqual(outcome, 0.0 if mode == "advantage" else expected)

    def test_explicit_ids_distinguish_new_contributions_from_full_payload_retries(self):
        for aggregation, expected in (("sum", 6.0), ("mean", 3.0), ("last", 4.0)):
            with self.subTest(aggregation=aggregation):
                ledger = RewardLedger()
                for number, value in enumerate((2.0, 4.0)):
                    payload = (
                        RewardAssignment("trajectory", value, aggregation=aggregation),
                        RewardAssignment(
                            "token", value, position=(0, 0), aggregation=aggregation
                        ),
                    )
                    options = dict(
                        logical_rollout_id="r", submission_id=f"judge-{number}"
                    )
                    self.assertEqual(ledger.submit(payload, **options), payload)
                    self.assertEqual(ledger.commit(payload, **options), ())
                outcome, process, _ = ledger.aggregate()
                self.assertEqual(outcome, expected)
                self.assertEqual(process, {0: (expected,)})
                self.assertEqual(len(ledger.assignments), 4)

    def test_submission_conflicts_are_atomic_and_compare_all_content(self):
        value = RewardAssignment("trajectory", 2.0)
        for changed in (
            replace(value, value=3.0),
            replace(value, mode="advantage"),
            replace(value, aggregation="mean"),
            replace(value, discount=0.5),
            replace(value, source="other"),
            replace(value, target="turn", position=0),
        ):
            with self.subTest(changed=changed):
                ledger = RewardLedger()
                ledger.submit(value, logical_rollout_id="r", submission_id="score")
                with self.assertRaisesRegex(AgentContractError, "submission_id"):
                    ledger.submit(
                        changed, logical_rollout_id="r", submission_id="score"
                    )
                self.assertEqual(ledger.assignments, (value,))
                self.assertEqual(ledger.aggregate()[0], 2.0)

    def test_implicit_conflict_is_reported_without_partial_append(self):
        ledger = RewardLedger()
        ledger.submit(RewardAssignment("trajectory", 2.0), logical_rollout_id="r")
        with self.assertRaisesRegex(AgentContractError, "conflict"):
            ledger.submit(
                [
                    RewardAssignment("turn", 8.0, position=0),
                    RewardAssignment("trajectory", 4.0),
                ],
                logical_rollout_id="r",
            )
        self.assertEqual(len(ledger.assignments), 1)

    def test_invalid_payload_does_not_claim_submission_identity(self):
        value = RewardAssignment("trajectory", 2.0)
        ledger = RewardLedger()
        for invalid in (
            (value, object()),
            (value, RewardAssignment("token", 1.0)),
            (value, replace(value, aggregation="mean")),
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, AgentContractError)),
            ):
                ledger.submit(invalid, logical_rollout_id="r", submission_id="score")
            self.assertEqual(ledger.assignments, ())
        self.assertEqual(
            ledger.submit(value, logical_rollout_id="r", submission_id="score"),
            (value,),
        )

    def test_inconsistent_aggregation_across_submissions_is_atomic(self):
        ledger = RewardLedger()
        value = RewardAssignment("trajectory", 2.0)
        ledger.submit(value, logical_rollout_id="r", submission_id="a")
        with self.assertRaisesRegex(AgentContractError, "aggregation"):
            ledger.submit(
                replace(value, aggregation="mean"),
                logical_rollout_id="r",
                submission_id="b",
            )
        self.assertEqual(
            ledger.submit(value, logical_rollout_id="r", submission_id="b"), (value,)
        )
        self.assertEqual(ledger.aggregate()[0], 4.0)

    def test_rollouts_keep_independent_mean_populations_and_submission_ids(self):
        ledger = RewardLedger()
        for rollout, values in (("a", (2.0, 4.0)), ("b", (10.0,))):
            ledger.submit(
                [RewardAssignment("trajectory", v, aggregation="mean") for v in values],
                logical_rollout_id=rollout,
                submission_id="same-local-id",
            )
        self.assertEqual(ledger.aggregate()[0], 13.0)

    def test_discount_and_sources_are_applied_before_reduction(self):
        ledger = RewardLedger()
        ledger.submit(
            [
                RewardAssignment("trajectory", 2.0, aggregation="mean", discount=0.5),
                RewardAssignment("trajectory", 4.0, aggregation="mean", discount=0.5),
                RewardAssignment("trajectory", -0.5, source="cost"),
            ],
            logical_rollout_id="r",
        )
        self.assertEqual(ledger.aggregate()[0], 1.0)

    def test_invalid_identity_and_nonfinite_reduction_do_not_mutate(self):
        for key in ("submission_id", "logical_rollout_id"):
            for value in ("", " ", 3, True):
                ledger = RewardLedger()
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises(AgentContractError),
                ):
                    ledger.submit(RewardAssignment("trajectory", 1.0), **{key: value})
                self.assertEqual(ledger.assignments, ())
        ledger = RewardLedger()
        with self.assertRaisesRegex(AgentContractError, "finite"):
            ledger.submit(
                [RewardAssignment("trajectory", 1e308)] * 2, submission_id="score"
            )
        self.assertEqual(ledger.assignments, ())


class TrajectoryAdvantageTests(unittest.TestCase):
    def test_trajectory_advantage_broadcast_combines_with_local_advantages(self):
        value = advantage_episode(
            RewardAssignment(
                "trajectory", 2, mode="advantage", aggregation="mean", discount=0.5
            ),
            RewardAssignment(
                "trajectory", 6, mode="advantage", aggregation="mean", discount=0.5
            ),
            RewardAssignment("token", 3, mode="advantage", position=(0, 0)),
            RewardAssignment("turn", 4, mode="advantage", position=1),
        )
        row = EpisodeTrajectoryBuilder().build(value)
        self.assertEqual(row.token_advantages, [5, 2, 0, 0, 2, 6])
        self.assertEqual(row.reward, 0.0)
        self.assertIsNone(row.token_rewards)
        self.assertEqual(
            aggregate_assignments(value.score.assignments)[2][None], (2.0,)
        )
        batch = prepare_advantages([row], raw_config())
        self.assertEqual(batch.advantages, [[5, 2, 0, 0, 2, 6]])
        self.assertEqual(batch.returns, [[0.0] * 6])

    def test_zero_negative_and_multiple_advantage_sources(self):
        for scalar in (0.0, -2.0, 3.0):
            with self.subTest(scalar=scalar):
                value = advantage_episode(
                    RewardAssignment("trajectory", scalar - 1, mode="advantage"),
                    RewardAssignment("trajectory", 1, mode="advantage", source="other"),
                )
                row = EpisodeTrajectoryBuilder().build(value)
                batch = prepare_advantages([row], raw_config())
                self.assertEqual(
                    batch.advantages,
                    [[scalar if active else 0.0 for active in row.loss_mask]],
                )
                self.assertEqual(batch.returns, [[0.0] * len(row.loss_mask)])

    def test_critic_returns_and_bootstrap_are_unchanged_for_every_gae_unit(self):
        baseline = episode()
        baseline.status = "max_turns"
        baseline.score = EpisodeScore(1.0, False, process_rewards={0: (0.0, 0.25)})
        direct = replace(
            baseline,
            score=replace(
                baseline.score,
                assignments=(RewardAssignment("trajectory", 5, mode="advantage"),),
            ),
        )
        rows = [EpisodeTrajectoryBuilder().build(value) for value in (baseline, direct)]
        for row in rows:
            row.values = [0.2] * (len(row.completion_tokens) + 1)
        for unit in ("token", "turn", "trajectory"):
            with self.subTest(unit=unit):
                args = raw_config(
                    use_critic=True,
                    discount=0.7,
                    gae_lambda=0.4,
                    gae_timestep_unit=unit,
                )
                original, changed = [prepare_advantages([row], args) for row in rows]
                self.assertEqual(original.returns, changed.returns)
                for before, after, active in zip(
                    original.advantages[0], changed.advantages[0], rows[0].loss_mask
                ):
                    self.assertAlmostEqual(after - before, 5.0 if active else 0.0)

    def test_fragment_splitting_preserves_policy_loss_and_each_action_gradient(self):
        value = advantage_episode(RewardAssignment("trajectory", 2, mode="advantage"))
        whole = EpisodeTrajectoryBuilder().export(value).trajectories
        split = EpisodeTrajectoryBuilder(mode="individual").export(value).trajectories
        for importance in ("token", "sequence"):
            for reduction in ("token_mean", "seq_mean", "prompt_mean", "constant"):
                with self.subTest(importance=importance, reduction=reduction):
                    args = raw_config(
                        importance_sampling_level=importance,
                        loss_aggregation=reduction,
                        **(
                            {"loss_aggregation_divisor": 5}
                            if reduction == "constant"
                            else {}
                        ),
                    )
                    losses, gradients = [], []
                    for rows in (whole, split):
                        loss, grads, _ = objective(
                            args, rows, [row.logprobs for row in rows]
                        )
                        losses.append(loss)
                        by_action = {}
                        for row, grad in zip(rows, grads):
                            completion_grad = grad[len(row.prompt_tokens) - 1 :]
                            for key, active, derivative in zip(
                                row.action_keys, row.loss_mask, completion_grad
                            ):
                                if active:
                                    self.assertNotIn(key, by_action)
                                    self.assertNotEqual(derivative, 0.0)
                                    by_action[key] = derivative
                                else:
                                    self.assertEqual(derivative, 0.0)
                        gradients.append(by_action)
                    self.assertAlmostEqual(*losses)
                    self.assertEqual(gradients[0], gradients[1])


class RewardIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_compacted_segments_keep_each_direct_advantage_action_once(self):
        limits = dict(max_turns=4, max_tokens_per_turn=128, max_generated_tokens=512)
        policy, client = long_snapshot(tool_turns=3)
        baseline = calculator_workflow(
            limits=EpisodeLimits(**limits, max_context_tokens=4096),
            renderer=TokenAppendRenderer(policy.tokenizer),
        )
        async with baseline:
            await baseline.collect(calculator_rows()[:1], policy)
        context_limit = len(client.requests[1][0]) + limits["max_tokens_per_turn"]

        policy, _ = long_snapshot(tool_turns=3)
        workflow = calculator_workflow(
            limits=EpisodeLimits(**limits, max_context_tokens=context_limit),
            renderer=TokenAppendRenderer(policy.tokenizer),
        )
        workflow.runner.context_compactor = ContextCompactor(
            keep_recent_interactions=1,
            summarizer=lambda interactions, previous: "Earlier calculator result was recorded.",
        )

        class Scorer:
            async def score(self, row, result):
                return EpisodeScore(
                    0.0,
                    False,
                    assignments=(
                        RewardAssignment("trajectory", 2.0, mode="advantage"),
                    ),
                )

        workflow.runner.scorer = Scorer()
        async with workflow:
            batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertGreater(len(batch.trajectories), 1)
        self.assertTrue(batch.episodes[0].compactions)
        prepared = prepare_advantages(batch.trajectories, raw_config())
        keys = []
        for row, advantages, returns in zip(
            batch.trajectories, prepared.advantages, prepared.returns
        ):
            self.assertEqual(
                advantages, [2.0 if active else 0.0 for active in row.loss_mask]
            )
            self.assertEqual(returns, [0.0] * len(returns))
            keys.extend(
                key for key, active in zip(row.action_keys, row.loss_mask) if active
            )
        expected_count = sum(
            len(turn.generation.tokens) for turn in batch.episodes[0].turns
        )
        self.assertEqual(len(keys), expected_count)
        self.assertEqual(len(keys), len(set(keys)))

    async def test_ledger_export_and_pytrio_ppo_update_share_the_same_signals(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer("fake", args=raw_config(), training_client=client)
        workflow = calculator_workflow()
        ledger = workflow.runner.reward_ledger

        class Scorer:
            async def score(self, row, result):
                return EpisodeScore(
                    0.0,
                    False,
                    assignments=(
                        RewardAssignment("trajectory", 2.0),
                        RewardAssignment("trajectory", 4.0),
                        RewardAssignment("trajectory", 3.0, mode="advantage"),
                    ),
                )

        workflow.runner.scorer = Scorer()
        workflow.runner.trajectory_builder = EpisodeTrajectoryBuilder(mode="individual")
        try:
            policy = await trainer.create_sampling_snapshot()
            async with workflow:
                batch = await workflow.collect(calculator_rows()[:1], policy)
            self.assertEqual(ledger.aggregate(), (6.0, {}, {None: (3.0,)}))
            self.assertTrue(all(row.reward == 6.0 for row in batch.trajectories))
            prepared = prepare_advantages(batch.trajectories, raw_config())
            for values, mask in zip(prepared.advantages, prepared.masks):
                self.assertEqual(values, [9.0 if active else 0.0 for active in mask])
            await trainer.step(batch.trajectories)
            self.assertEqual(trainer.state.global_step, 1)
            refreshed = await trainer.create_sampling_snapshot()
            self.assertEqual(refreshed.policy_version, 1)
            self.assertNotEqual(policy.snapshot_id, refreshed.snapshot_id)
        finally:
            await trainer.close()

    async def test_branch_paths_preserve_trajectory_advantage_and_shared_action_weights(
        self,
    ):
        for policy in ("deduplicate", "weighted"):
            with self.subTest(policy=policy):
                sampling, _ = snapshot()
                workflow = attach_counterfactual_branches(calculator_workflow())
                workflow.shared_prefix_policy = policy
                original = workflow.runner.run

                async def run(*args, **kwargs):
                    result = await original(*args, **kwargs)
                    result.score = replace(
                        result.score,
                        assignments=(
                            RewardAssignment("trajectory", 2.0, mode="advantage"),
                            RewardAssignment(
                                "trajectory", 100.0, source="episode-outcome"
                            ),
                        ),
                    )
                    return result

                workflow.runner.run = run
                async with workflow:
                    batch = await workflow.collect(calculator_rows()[:1], sampling)
                self.assertEqual([row.reward for row in batch.trajectories], [1.0, 3.0])
                totals = {}
                for row in batch.trajectories:
                    self.assertEqual(
                        row.token_advantages,
                        [2.0 if active else 0.0 for active in row.loss_mask],
                    )
                    for key, active, weight in zip(
                        row.action_keys, row.loss_mask, row.effective_action_weights
                    ):
                        if active:
                            totals[key] = totals.get(key, 0.0) + weight
                self.assertTrue(all(weight == 1.0 for weight in totals.values()))


if __name__ == "__main__":
    unittest.main()
