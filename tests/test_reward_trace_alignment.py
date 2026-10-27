"""Raw scorer evidence must match exported signals without becoming rewards twice."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from agent_test_support import AgentTrainingClient
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_episode_runner import snapshot
from test_ppo_family import raw_config

from triotune import PPOTrainer
from triotune.agentrl import (
    AgentContractError,
    EpisodeScore,
    EpisodeTrajectoryBuilder,
    JsonlEpisodeLogger,
    RewardAssignment,
    RewardEvent,
    RewardLedger,
)
from triotune.trainer.ppo import prepare_advantages


class MixedScorer:
    async def score(self, row, episode):
        count = len(episode.turns[0].generation.tokens)
        return EpisodeScore(
            1.0,
            True,
            process_rewards={0: (0.125,) + (0.0,) * (count - 1)},
            assignments=(
                RewardAssignment("trajectory", 2.0, aggregation="mean"),
                RewardAssignment("trajectory", 4.0, aggregation="mean"),
                RewardAssignment(
                    "interaction", 2.0, aggregation="mean", discount=0.5, position=0
                ),
                RewardAssignment(
                    "interaction", 4.0, aggregation="mean", discount=0.5, position=0
                ),
                RewardAssignment("turn", -0.25, source="cost", position=0),
                RewardAssignment("token", 0.5, position=(0, count - 1)),
                RewardAssignment(
                    "interaction", -1.0, aggregation="last", discount=0.5, position=1
                ),
                RewardAssignment(
                    "interaction", -3.0, aggregation="last", discount=0.5, position=1
                ),
                RewardAssignment("token", 0.25, position=(1, 0)),
                RewardAssignment("trajectory", 0.5, mode="advantage"),
                RewardAssignment("turn", 1.0, mode="advantage", position=0),
                RewardAssignment("token", 2.0, mode="advantage", position=(1, 0)),
            ),
        )


class RewardTraceTests(unittest.IsolatedAsyncioTestCase):
    def assert_trace_matches_rows(self, episode, rows):
        interactions = {item.interaction_id: item for item in episode.interactions}
        events = [event for event in episode.events if isinstance(event, RewardEvent)]
        self.assertEqual(len(events), len(interactions))
        for event in events:
            interaction = interactions[event.context.interaction_id]
            self.assertEqual(event.context, interaction.context)
            for name in (
                "reward",
                "outcome_reward",
                "process_rewards",
                "token_advantages",
            ):
                self.assertEqual(getattr(event, name), getattr(interaction, name))
            self.assertEqual(
                event.reward, event.outcome_reward + sum(event.process_rewards)
            )
        active_keys = []
        for row in rows:
            self.assertEqual(
                row.reward, sum(item.outcome_reward for item in interactions.values())
            )
            for offset, (key, active) in enumerate(zip(row.action_keys, row.loss_mask)):
                process = (
                    row.token_rewards[offset] if row.token_rewards is not None else 0.0
                )
                advantage = (
                    row.token_advantages[offset]
                    if row.token_advantages is not None
                    else 0.0
                )
                if active:
                    interaction, token_index = interactions[key[0]], key[1]
                    self.assertEqual(process, interaction.process_rewards[token_index])
                    self.assertEqual(
                        advantage, interaction.token_advantages[token_index]
                    )
                    active_keys.append(key)
                else:
                    self.assertEqual((process, advantage), (0.0, 0.0))
        self.assertEqual(len(active_keys), len(set(active_keys)))

    async def test_structured_and_legacy_signals_match_trace_export_and_json(self):
        for mode in ("concat", "individual"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                logger = JsonlEpisodeLogger(Path(directory) / "trace.jsonl")
                workflow = calculator_workflow(logger=logger)
                workflow.runner.scorer = MixedScorer()
                workflow.runner.trajectory_builder = EpisodeTrajectoryBuilder(mode=mode)
                policy, _ = snapshot()
                async with workflow:
                    batch = await workflow.collect(calculator_rows()[:1], policy)
                episode = batch.episodes[0]
                first, last = episode.interactions
                self.assertEqual(
                    (first.outcome_reward, last.outcome_reward), (0.0, 4.0)
                )
                self.assertEqual(sum(first.process_rewards), 1.875)
                self.assertEqual(sum(last.process_rewards), -1.25)
                self.assertEqual((first.reward, last.reward), (1.875, 2.75))
                self.assertEqual(first.process_rewards[0], 0.125)
                self.assertEqual(first.process_rewards[-1], 1.75)
                self.assertEqual(last.process_rewards[0], 0.25)
                self.assertEqual(last.process_rewards[-1], -1.5)
                self.assertEqual(first.token_advantages[0], 0.5)
                self.assertEqual(first.token_advantages[-1], 1.5)
                self.assertEqual(last.token_advantages[0], 2.5)
                self.assert_trace_matches_rows(episode, batch.trajectories)
                saved = json.loads(logger.path.read_text())
                for original, persisted in zip(
                    episode.interactions, saved["interactions"]
                ):
                    self.assertEqual(persisted["reward"], original.reward)
                    self.assertEqual(
                        persisted["outcome_reward"], original.outcome_reward
                    )
                    self.assertEqual(
                        persisted["process_rewards"], list(original.process_rewards)
                    )
                    self.assertEqual(
                        persisted["token_advantages"], list(original.token_advantages)
                    )
                reward_events = [
                    event for event in saved["events"] if event["kind"] == "reward"
                ]
                self.assertEqual(sum(event["reward"] for event in reward_events), 4.625)
                self.assertEqual(
                    sum(event["outcome_reward"] for event in reward_events), 4.0
                )
                self.assertEqual(episode.score.reward, 1.0)

    async def test_interaction_scorer_reward_is_committed_with_its_process_vector(self):
        class Scorer:
            async def score(self, row, episode, interaction):
                return EpisodeScore(
                    2.0 if interaction.parent_interaction_id is None else 0.0, False
                )

        workflow = calculator_workflow()
        workflow.runner.interaction_scorer = Scorer()
        policy, _ = snapshot()
        async with workflow:
            batch = await workflow.collect(calculator_rows()[:1], policy)
        first, last = batch.episodes[0].interactions
        self.assertEqual((first.reward, last.reward), (2.0, 1.0))
        self.assertEqual(first.process_rewards[-1], 2.0)
        self.assert_trace_matches_rows(batch.episodes[0], batch.trajectories)

    async def test_reward_replays_do_not_duplicate_trace_and_conflicts_are_atomic(self):
        workflow = calculator_workflow()
        workflow.runner.scorer = MixedScorer()
        policy, _ = snapshot()
        async with workflow:
            episode = (await workflow.collect(calculator_rows()[:1], policy)).episodes[
                0
            ]
            saved = copy.deepcopy(asdict(episode))
            assignments = workflow.runner.reward_ledger.assignments
            for score in (None, episode.score, copy.deepcopy(episode.score)):
                self.assertEqual(
                    await workflow.runner.submit_reward(episode, score), episode.score
                )
                self.assertEqual(asdict(episode), saved)
            for score in (
                replace(episode.score, reward=3.0),
                replace(episode.score, assignments=()),
            ):
                with (
                    self.subTest(score=score),
                    self.assertRaisesRegex(AgentContractError, "already submitted"),
                ):
                    await workflow.runner.submit_reward(episode, score)
                self.assertEqual(asdict(episode), saved)
                self.assertEqual(workflow.runner.reward_ledger.assignments, assignments)

    async def test_invalid_targets_fail_before_trace_or_ledger_commit(self):
        workflow = calculator_workflow()
        policy, _ = snapshot()
        async with workflow:
            original = (await workflow.collect(calculator_rows()[:1], policy)).episodes[
                0
            ]
            for assignment in (
                RewardAssignment("interaction", 2.0, position=99),
                RewardAssignment("token", 2.0, position=(0, 9999)),
            ):
                with self.subTest(assignment=assignment):
                    episode = copy.deepcopy(original)
                    episode.export_metadata.pop("reward_submitted")
                    saved = copy.deepcopy(asdict(episode))
                    ledger = RewardLedger()
                    workflow.runner.reward_ledger = ledger
                    score = EpisodeScore(1.0, True, assignments=(assignment,))
                    with self.assertRaisesRegex(AgentContractError, "unknown"):
                        await workflow.runner.submit_reward(episode, score)
                    self.assertEqual(asdict(episode), saved)
                    self.assertEqual(ledger.assignments, ())
                    episode.score = score
                    with self.assertRaisesRegex(AgentContractError, "unknown"):
                        EpisodeTrajectoryBuilder().build(episode)

    async def test_scoring_failure_leaves_no_partial_reward_views_or_events(self):
        class Scorer:
            async def score(self, row, episode, interaction):
                if interaction.parent_interaction_id is not None:
                    raise RuntimeError("second judge failed")
                return EpisodeScore(2.0, False)

        workflow = calculator_workflow()
        workflow.runner.interaction_scorer = Scorer()
        policy, _ = snapshot()
        async with workflow:
            episode = (
                await workflow.collect(calculator_rows()[:1], policy, strict=False)
            ).episodes[0]
        self.assertEqual(episode.status, "scoring_failure")
        self.assertFalse(
            any(isinstance(event, RewardEvent) for event in episode.events)
        )
        self.assertFalse(episode.export_metadata.get("reward_submitted"))
        self.assertEqual(workflow.runner.reward_ledger.assignments, ())
        self.assertTrue(
            all(
                item.reward == 0.0 and not item.process_rewards
                for item in episode.interactions
            )
        )

    async def test_explicit_zero_process_signals_preserve_training_field_presence(self):
        workflow = calculator_workflow()

        class Scorer:
            async def score(self, row, episode):
                return EpisodeScore(
                    0.0, False, assignments=(RewardAssignment("turn", 0.0, position=0),)
                )

        workflow.runner.scorer = Scorer()
        policy, _ = snapshot()
        async with workflow:
            batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertIsNotNone(batch.trajectories[0].token_rewards)
        self.assertEqual(sum(batch.trajectories[0].token_rewards), 0.0)
        self.assert_trace_matches_rows(batch.episodes[0], batch.trajectories)

    async def test_trace_is_raw_and_ppo_consumes_each_signal_once(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer("fake", args=raw_config(), training_client=client)
        workflow = calculator_workflow()
        workflow.runner.scorer = MixedScorer()
        try:
            policy = await trainer.create_sampling_snapshot()
            async with workflow:
                batch = await workflow.collect(calculator_rows()[:1], policy)
            episode, row = batch.episodes[0], batch.trajectories[0]
            original = copy.deepcopy(asdict(episode))
            prepared = prepare_advantages([row], raw_config())
            values = {
                key: value
                for key, value in zip(row.action_keys, prepared.advantages[0])
                if key
            }
            first, last = episode.interactions
            self.assertEqual(values[(first.interaction_id, 0)], 5.125)
            self.assertEqual(
                values[(first.interaction_id, len(first.turn.generation.tokens) - 1)],
                6.0,
            )
            self.assertEqual(values[(last.interaction_id, 0)], 5.25)
            self.assertEqual(
                values[(last.interaction_id, len(last.turn.generation.tokens) - 1)], 3.0
            )
            baseline = replace(
                episode,
                score=replace(
                    episode.score,
                    assignments=tuple(
                        item
                        for item in episode.score.assignments
                        if item.mode == "reward"
                    ),
                ),
            )
            baseline_row = EpisodeTrajectoryBuilder().build(baseline)
            self.assertEqual(
                prepare_advantages([baseline_row], raw_config()).returns,
                prepared.returns,
            )
            await trainer.step(batch.trajectories)
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(asdict(episode), original)
        finally:
            await trainer.close()


if __name__ == "__main__":
    unittest.main()
