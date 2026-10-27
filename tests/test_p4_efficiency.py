"""Long-context segmentation and asynchronous Agent RL scheduling contracts."""

import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from agent_test_support import AgentTokenizer, AgentTrainingClient, CalculatorSampler
from recipes.calculator_agent import calculator_rows, calculator_workflow
from test_agent_branching import attach_counterfactual_branches
from test_agent_episode_runner import snapshot as calculator_snapshot

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentTrainingLoop,
    ContextCompactor,
    EpisodeLimits,
    PrefixCache,
    PyTRIOAgentSampler,
    SamplingSnapshot,
    TokenAppendRenderer,
)
from triotune.trainer.ppo import prepare_advantages
from triotune.trainer.ppo.datum import batch_datums

TOOL_ACTION = json.dumps(
    {
        "tool_calls": [
            {
                "name": "calculate",
                "arguments": {"operation": "add", "a": 2, "b": 3},
            }
        ]
    }
)


def long_snapshot(*, tool_turns=2):
    """Return a fixed sampler that performs several tool turns then answers."""
    tokenizer = AgentTokenizer()
    client = CalculatorSampler(tokenizer, [], 0)
    original = client.sample_async
    calls = 0

    async def sample(**kwargs):
        nonlocal calls
        client.output = TOOL_ACTION if calls < tool_turns else '{"final":"5"}'
        calls += 1
        return await original(**kwargs)

    client.sample_async = sample
    return SamplingSnapshot(
        "long-context",
        PyTRIOAgentSampler(client, tokenizer),
        tokenizer,
        policy_version=0,
    ), client


class ContextEfficiencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefix_cache_is_exact_bounded_and_detaches_results(self):
        tokenizer = AgentTokenizer()
        calls = 0
        original = tokenizer.apply_chat_template

        def render(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        tokenizer.apply_chat_template = render
        cache = PrefixCache(max_entries=1)
        renderer = TokenAppendRenderer(tokenizer, prefix_cache=cache)
        messages = [{"role": "user", "content": "same prompt"}]
        first = renderer.render(messages, [])
        first[0] = 999
        second = renderer.render(messages, [])
        self.assertNotEqual(second[0], 999)
        self.assertEqual(calls, 1)
        self.assertEqual((cache.hits, cache.misses), (1, 1))

        renderer.render([{"role": "user", "content": "new prompt"}], [])
        renderer.render(messages, [])
        self.assertEqual(calls, 3)
        self.assertEqual(cache.misses, 3)

    async def test_tool_summary_creates_exact_segments_without_reencoding_actions(self):
        # Measure the uncompressed third request so the compacting run triggers
        # only after two complete tool interactions.
        baseline_snapshot, baseline_client = long_snapshot(tool_turns=3)
        baseline = calculator_workflow(
            limits=EpisodeLimits(
                max_turns=4,
                max_tokens_per_turn=128,
                max_generated_tokens=512,
                max_context_tokens=4096,
            ),
            renderer=TokenAppendRenderer(baseline_snapshot.tokenizer),
        )
        baseline_batch = await baseline.collect(
            calculator_rows()[:1], baseline_snapshot
        )
        self.assertEqual(baseline_batch.episodes[0].status, "completed")
        prompt_lengths = [len(request[0]) for request in baseline_client.requests]
        context_limit = prompt_lengths[1] + 128
        self.assertGreater(prompt_lengths[2] + 128, context_limit)

        snapshot, _ = long_snapshot(tool_turns=3)

        async def summarize(interactions, previous):
            return "Earlier calculator result was recorded."

        workflow = calculator_workflow(
            limits=EpisodeLimits(
                max_turns=4,
                max_tokens_per_turn=128,
                max_generated_tokens=512,
                max_context_tokens=context_limit,
            ),
            renderer=TokenAppendRenderer(snapshot.tokenizer),
        )
        workflow.runner.context_compactor = ContextCompactor(
            keep_recent_interactions=1,
            summarizer=summarize,
        )
        batch = await workflow.collect(calculator_rows()[:1], snapshot)
        episode = batch.episodes[0]
        self.assertEqual(episode.status, "completed")
        self.assertEqual(len(episode.compactions), 2)
        self.assertEqual(
            [compaction.turn_index for compaction in episode.compactions], [2, 3]
        )
        self.assertEqual(len(batch.trajectories), 3)
        self.assertEqual(len({row.group_id for row in batch.trajectories}), 2)
        self.assertEqual(
            batch.trajectories[1].prompt_tokens,
            batch.trajectories[2].prompt_tokens,
        )
        self.assertTrue(
            all(row.reward_group_id == episode.group_id for row in batch.trajectories)
        )
        self.assertEqual([row.context_segment for row in batch.trajectories], [0, 1, 2])
        self.assertEqual(batch.metrics()["agent/context_compactions"], 2.0)
        self.assertGreater(batch.metrics()["agent/compacted_prompt_tokens_saved"], 0)

        active_keys = [
            key
            for row in batch.trajectories
            for key, active in zip(row.action_keys, row.loss_mask, strict=True)
            if active
        ]
        expected_keys = [
            (interaction.interaction_id, offset)
            for interaction in episode.interactions
            if interaction.turn is not None
            for offset in range(len(interaction.turn.generation.tokens))
        ]
        self.assertCountEqual(active_keys, expected_keys)
        prepared = prepare_advantages(batch.trajectories, PPOConfig(use_critic=False))
        datums = batch_datums(prepared)
        self.assertTrue(datums.prefix_groups)

    async def test_compacted_segments_preserve_branch_groups_and_shared_actions(self):
        policy, _ = calculator_snapshot()
        workflow = attach_counterfactual_branches(calculator_workflow())
        original_run = workflow.runner.run

        async def compact_branches(*args, **kwargs):
            episode = await original_run(*args, **kwargs)
            compacted_prompt = (700, 701, 702)
            for interaction in episode.interactions[1:]:
                interaction.turn.prompt_tokens = compacted_prompt
                interaction.input_token_span = (0, len(compacted_prompt))
                interaction.output_token_span = (
                    len(compacted_prompt),
                    len(compacted_prompt) + len(interaction.turn.generation.tokens),
                )
            episode.__post_init__()
            return episode

        workflow.runner.run = compact_branches
        workflow.shared_prefix_policy = "weighted"
        batch = await workflow.collect(calculator_rows()[:1], policy)
        self.assertEqual(len(batch.trajectories), 4)
        self.assertEqual(
            {row.branch_id for row in batch.trajectories}, {"tool-a", "tool-b"}
        )
        self.assertEqual(
            {
                row.branch_id: {
                    item.context_segment
                    for item in batch.trajectories
                    if item.branch_id == row.branch_id
                }
                for row in batch.trajectories
            },
            {"tool-a": {0, 1}, "tool-b": {0, 1}},
        )
        prepare_advantages(batch.trajectories, PPOConfig(use_critic=False))


class AsyncSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_collection_is_globally_capacity_bounded(self):
        tokenizer = AgentTokenizer()
        client = CalculatorSampler(tokenizer, [], 0)
        snapshot = SamplingSnapshot(
            "stream", PyTRIOAgentSampler(client, tokenizer), tokenizer
        )
        workflow = calculator_workflow()
        workflow.max_concurrency = 2
        original = client.sample_async
        running = peak = 0

        async def delayed(**kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await asyncio.sleep(0.005)
                return await original(**kwargs)
            finally:
                running -= 1

        client.sample_async = delayed
        batches = []
        async for batch in workflow.collect_iter(
            calculator_rows()[:3],
            snapshot,
            max_pending_groups=3,
        ):
            batches.append(batch)
        self.assertEqual(len(batches), 3)
        self.assertEqual(peak, 2)
        self.assertEqual(
            sorted(
                state["source_index"]
                for batch in batches
                for state in batch.groups.values()
            ),
            [0, 1, 2],
        )

    async def test_partial_collection_resumes_only_at_original_snapshot_boundary(self):
        tokenizer = AgentTokenizer()
        first_client = CalculatorSampler(tokenizer, [], 0)
        snapshot = SamplingSnapshot(
            "resume", PyTRIOAgentSampler(first_client, tokenizer), tokenizer, 0
        )
        workflow = calculator_workflow()
        partial = await workflow.collect(calculator_rows()[:1], snapshot)
        recovered = await workflow.resume(partial, calculator_rows()[1:3], snapshot)
        self.assertEqual(len(recovered.episodes), 3)
        self.assertEqual(len(recovered.trajectories), 3)
        self.assertEqual(len(recovered.groups), 3)
        self.assertEqual(
            {episode.snapshot_id for episode in recovered.episodes}, {"resume"}
        )

        foreign = SamplingSnapshot(
            "foreign",
            PyTRIOAgentSampler(CalculatorSampler(tokenizer, [], 1), tokenizer),
            tokenizer,
            1,
        )
        with self.assertRaisesRegex(ValueError, "original snapshot"):
            await workflow.resume(partial, calculator_rows()[1:2], foreign)

    async def test_prefetch_overlaps_update_and_accepts_bounded_staleness(self):
        client = AgentTrainingClient()
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm(
                "grpo",
                max_steps=2,
                batch_size=1,
                num_generations=2,
                seed=0,
                eval_before_train=False,
                eval_after_train=False,
            ),
            training_client=client,
            train_dataset=[copy.deepcopy(calculator_rows()[0]) for _ in range(4)],
        )
        workflow = calculator_workflow()
        update_started = asyncio.Event()
        overlap_seen = asyncio.Event()
        run_count = 0
        original_run = workflow.runner.run

        async def run(*args, **kwargs):
            nonlocal run_count
            run_count += 1
            if run_count > 2:
                await asyncio.wait_for(update_started.wait(), 1)
                overlap_seen.set()
            return await original_run(*args, **kwargs)

        workflow.runner.run = run
        original_backward = client.forward_backward_custom_async

        async def backward(data, loss_fn):
            update_started.set()
            await asyncio.wait_for(overlap_seen.wait(), 1)
            return await original_backward(data, loss_fn)

        client.forward_backward_custom_async = backward
        config = AgentConfig(
            async_rollouts=True,
            rollout_prefetch_capacity=1,
            max_policy_staleness=1,
        )
        loop = AgentTrainingLoop(trainer, workflow, config=config)
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await loop.train()
        self.assertTrue(overlap_seen.is_set())
        self.assertEqual(state.global_step, 2)
        self.assertEqual(
            [entry["policy_versions"] for entry in loop.update_history],
            [["0"], ["0"]],
        )
        self.assertEqual(
            [
                entry["metrics"]["train/policy_staleness_max"]
                for entry in loop.update_history
            ],
            [0.0, 1.0],
        )

        # The second batch was sampled from version zero while step one updated
        # version one. Its age is exactly the configured allowance.
        self.assertEqual(client.trace.count("submit:optim"), 2)

    async def test_async_configuration_rejects_unbounded_policy_age(self):
        valid = AgentConfig(
            context_compaction=True,
            prefix_cache_size=8,
            async_rollouts=True,
            rollout_prefetch_capacity=2,
            max_policy_staleness=2,
        )
        self.assertEqual(AgentConfig.from_dict(valid.to_dict()), valid)
        with self.assertRaisesRegex(ValueError, "staleness"):
            AgentConfig(
                async_rollouts=True,
                rollout_prefetch_capacity=2,
                max_policy_staleness=1,
            )
