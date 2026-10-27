"""SDK requests and original PyTRIO samples reach the existing PPO updater."""

import asyncio
import importlib.util
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from agent_test_support import AgentTrainingClient
from test_openai_native import NativeTokenizer, add, xml_call
from test_ppo_family import objective, raw_config

from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentContractError,
    AgentRolloutWorkflow,
    AgentTrainingLoop,
    EpisodeCollectionError,
    EpisodeLimits,
    EpisodeScore,
    EpisodeTrajectoryBuilder,
    GenerationEvent,
    OpenAIEpisodeExecutor,
    PyTRIOAgentSampler,
    SamplingSnapshot,
    current_execution_context,
)
from triotune.agentrl.integrations.openai import (
    Agent,
    OpenAIClient,
    OpenAIWorkflow,
    PyTRIOBackend,
)
from triotune.trainer.ppo import (
    PPOAlgorithmPlan,
    NormConfig,
    normalize,
    prepare_advantages,
)

MODEL = "Qwen/Qwen3.5-4B"
ROWS = [{"messages": "Compute 2+3", "answer": "5", "task_id": "math"}]


class SamplingClient:
    def __init__(self, tokenizer, version=0):
        self.tokenizer, self.version = tokenizer, version
        self.calls, self.outputs, self.contexts, self.seeds = [], [], [], []
        self.fail_seed = None
        self.logprobs = True

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        inputs = list(prompt.to_ints())
        self.calls.append(inputs)
        self.contexts.append(current_execution_context())
        self.seeds.append(sampling_params.seed)
        if self.fail_seed is not None and sampling_params.seed == self.fail_seed:
            raise ConnectionError("sampler unavailable")
        text = self.tokenizer.decode(inputs, skip_special_tokens=False)
        final = "<tool_response>\n" in text
        output = (
            "</think>\n5<|im_end|>"
            if final
            else "</think>\n" + xml_call() + "<|im_end|>"
        )
        # Deliberately use a noncanonical representation of the same text.
        # A bridge that re-encodes API display text cannot pass these checks.
        ids = [
            token + 1000 if token >= 1000 else token
            for token in self.tokenizer.encode(output)
        ]
        reason = "stop" if len(ids) <= sampling_params.max_tokens else "length"
        ids = ids[: sampling_params.max_tokens]
        self.outputs.append(ids)
        await asyncio.sleep(0)
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=ids,
                    logprobs=[-0.25] * len(ids) if self.logprobs else None,
                    stop_reason=reason,
                )
            ],
            input_tokens=len(inputs),
            output_tokens=len(ids),
        )


class TrainingClient(AgentTrainingClient):
    def get_tokenizer(self):
        return NativeTokenizer()

    async def save_weights_and_get_sampling_client_async(self):
        sampler = SamplingClient(self.get_tokenizer(), self.version)
        self.snapshots.append(sampler)
        return sampler


class Scorer:
    async def score(self, row, episode):
        # Distinct candidate rewards make normalized policy gradients nonzero
        # without introducing nondeterministic model behavior into the fixture.
        return EpisodeScore(
            float(episode.final_text == row["answer"]) + episode.sample_index,
            episode.final_text == row["answer"],
        )


def factory(snapshot, *, api="chat_completions", sdk_scorer=None, callback=False):
    client = OpenAIClient(backend=PyTRIOBackend(snapshot, model=MODEL, max_tokens=1024))
    if callback:
        return OpenAIWorkflow(client=client, scorer=sdk_scorer)
    return OpenAIWorkflow(
        Agent(
            name="calculator",
            instructions="Use add.",
            tools=[add],
            model=MODEL,
            client=client,
            api=api,
        ),
        scorer=sdk_scorer,
    )


class OpenAIPPOBridgeTests(unittest.IsolatedAsyncioTestCase):
    def snapshot(self):
        tokenizer = NativeTokenizer()
        sampling = SamplingClient(tokenizer)
        return (
            SamplingSnapshot(
                "snapshot", PyTRIOAgentSampler(sampling, tokenizer), tokenizer, 0
            ),
            sampling,
        )

    async def test_both_endpoints_export_original_tokens_and_train(self):
        for api in ("chat_completions", "responses"):
            for mode in ("concat", "individual"):
                with self.subTest(api=api, mode=mode):
                    client = TrainingClient()
                    populations = []

                    def normalizer(rows, group_ids, masks=None):
                        populations.append(len(rows))
                        return normalize(
                            rows, group_ids, NormConfig("group", None), masks
                        )

                    trainer = PPOTrainer(
                        "fake",
                        args=raw_config(),
                        training_client=client,
                        plan=PPOAlgorithmPlan(normalizer=normalizer),
                    )
                    runner = OpenAIEpisodeExecutor(
                        lambda snapshot: factory(snapshot, api=api),
                        scorer=Scorer(),
                        trajectory_builder=EpisodeTrajectoryBuilder(mode=mode),
                    )
                    workflow = AgentRolloutWorkflow(runner)
                    try:
                        policy = await trainer.create_sampling_snapshot()
                        async with workflow:
                            batch = await workflow.collect(
                                ROWS, policy, n_samples=2, seed=4
                            )
                        sampling = client.snapshots[0]
                        self.assertEqual(len(batch.episodes), 2)
                        self.assertEqual(
                            len(batch.trajectories), 2 if mode == "concat" else 4
                        )
                        by_id = {
                            context.interaction_id: index
                            for index, context in enumerate(sampling.contexts)
                        }
                        for episode in batch.episodes:
                            self.assertEqual(episode.status, "completed")
                            self.assertEqual(episode.final_text, "5")
                            self.assertNotEqual(
                                episode.episode_id,
                                episode.export_metadata["sdk"]["session_id"],
                            )
                            self.assertEqual(len(episode.turns[0].tool_results), 1)
                            for item in episode.interactions:
                                index = by_id[item.interaction_id]
                                self.assertEqual(
                                    item.turn.prompt_tokens,
                                    tuple(sampling.calls[index]),
                                )
                                self.assertEqual(
                                    item.turn.generation.tokens,
                                    tuple(sampling.outputs[index]),
                                )
                                self.assertEqual(
                                    item.turn.generation.logprobs,
                                    (-0.25,) * len(sampling.outputs[index]),
                                )
                            self.assertEqual(
                                sum(
                                    isinstance(event, GenerationEvent)
                                    for event in episode.events
                                ),
                                2,
                            )
                        keys = [
                            key
                            for row in batch.trajectories
                            for key, active in zip(row.action_keys, row.loss_mask)
                            if active
                        ]
                        self.assertEqual(len(keys), len(set(keys)))
                        await trainer.step(batch.trajectories)
                        self.assertEqual(populations, [2])
                        self.assertEqual(trainer.state.global_step, 1)
                        refreshed = await trainer.create_sampling_snapshot()
                        self.assertEqual(refreshed.policy_version, 1)
                        self.assertNotEqual(refreshed.snapshot_id, policy.snapshot_id)
                        self.assertTrue(trainer._actor.datum_provenance)
                        self.assertIsNone(current_execution_context())
                    finally:
                        await trainer.close()

    async def test_existing_training_loop_refreshes_sdk_sampling_policy(self):
        client = TrainingClient()
        trainer = PPOTrainer(
            "fake",
            args=PPOConfig.for_algorithm(
                "reinforce",
                max_steps=2,
                batch_size=1,
                num_generations=1,
                eval_before_train=False,
                eval_after_train=False,
            ),
            training_client=client,
            train_dataset=ROWS * 2,
        )
        workflow = AgentRolloutWorkflow(OpenAIEpisodeExecutor(factory, scorer=Scorer()))
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await AgentTrainingLoop(trainer, workflow).train()
        self.assertEqual(state.global_step, 2)
        self.assertTrue({0, 1}.issubset({item.version for item in client.snapshots}))

    async def test_system_failure_drops_group_and_keeps_completed_tool_results(self):
        policy, sampling = self.snapshot()
        sampling.fail_seed = 1
        workflow = AgentRolloutWorkflow(OpenAIEpisodeExecutor(factory, scorer=Scorer()))
        async with workflow:
            batch = await workflow.collect(ROWS, policy, strict=False, seed=0)
        episode = batch.episodes[0]
        self.assertEqual(episode.status, "sampling_failure")
        self.assertEqual(batch.trajectories, [])
        self.assertEqual(episode.turns[0].tool_results[0].content, "5")
        self.assertIsNone(episode.score)

    async def test_sdk_fragments_preserve_loss_and_shared_action_gradients(self):
        policy, _ = self.snapshot()
        async with AgentRolloutWorkflow(
            OpenAIEpisodeExecutor(factory, scorer=Scorer())
        ) as workflow:
            batch = await workflow.collect(ROWS, policy)
        episode = batch.episodes[0]
        results = []
        for mode in ("concat", "individual"):
            rows = EpisodeTrajectoryBuilder(mode=mode).export(episode).trajectories
            loss, grads, _ = objective(
                raw_config(), rows, [row.logprobs for row in rows]
            )
            by_action = {}
            for row, grad in zip(rows, grads):
                for key, active, value in zip(
                    row.action_keys, row.loss_mask, grad[len(row.prompt_tokens) - 1 :]
                ):
                    if active:
                        self.assertNotIn(key, by_action)
                        self.assertNotEqual(value, 0.0)
                        by_action[key] = value
                    else:
                        self.assertEqual(value, 0.0)
            results.append((loss, by_action))
        self.assertEqual(results[0], results[1])

    async def test_model_budget_exits_are_partial_training_rows(self):
        for limits, expected in (
            (EpisodeLimits(max_turns=1), "max_turns"),
            (EpisodeLimits(max_tokens_per_turn=2), "token_limit"),
            (EpisodeLimits(max_context_tokens=1), "context_limit"),
        ):
            with self.subTest(expected=expected):
                policy, sampling = self.snapshot()
                async with AgentRolloutWorkflow(
                    OpenAIEpisodeExecutor(factory, scorer=Scorer(), limits=limits)
                ) as workflow:
                    batch = await workflow.collect(ROWS, policy, strict=False)
                episode = batch.episodes[0]
                self.assertEqual(episode.status, expected)
                self.assertFalse(episode.score.success)
                if expected == "context_limit":
                    self.assertEqual(sampling.calls, [])
                    self.assertFalse(batch.trajectories)
                else:
                    self.assertTrue(batch.trajectories[0].truncated)

    async def test_cancellation_preserves_finished_tools_and_restores_context(self):
        policy, sampling = self.snapshot()
        entered, saved = asyncio.Event(), []
        original_sample = sampling.sample_async

        async def sample(**kwargs):
            if sampling.outputs:
                entered.set()
                await asyncio.Event().wait()
            return await original_sample(**kwargs)

        sampling.sample_async = sample
        runner = OpenAIEpisodeExecutor(factory, scorer=Scorer())
        original_record = runner._record

        def record(episode, session, snapshot):
            saved.append(episode)
            return original_record(episode, session, snapshot)

        runner._record = record
        async with runner:
            task = asyncio.create_task(runner.run(ROWS[0], policy, group_id="group"))
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(saved[0].status, "cancelled")
        self.assertIsNone(saved[0].score)
        self.assertEqual(saved[0].turns[0].tool_results[0].content, "5")
        self.assertIsNone(current_execution_context())

    async def test_missing_logprobs_and_snapshot_switch_are_rejected(self):
        for invalid in ("logprobs", "snapshot", "external"):
            with self.subTest(invalid=invalid):
                policy, sampling = self.snapshot()
                if invalid == "logprobs":
                    sampling.logprobs = False

                def build(snapshot):
                    if invalid == "external":
                        return OpenAIWorkflow(
                            Agent(
                                name="fake",
                                instructions="",
                                client=OpenAIClient(SimpleNamespace()),
                            )
                        )
                    return factory(
                        replace(snapshot, snapshot_id="other")
                        if invalid == "snapshot"
                        else snapshot
                    )

                workflow = AgentRolloutWorkflow(
                    OpenAIEpisodeExecutor(build, scorer=Scorer())
                )
                async with workflow:
                    with self.assertRaises(EpisodeCollectionError) as caught:
                        await workflow.collect(ROWS, policy)
                self.assertEqual(
                    caught.exception.batch.episodes[0].status, "contract_error"
                )
                self.assertIsNone(caught.exception.batch.episodes[0].score)
                if invalid != "logprobs":
                    self.assertEqual(sampling.calls, [])

    @unittest.skipUnless(
        importlib.util.find_spec("openai") and importlib.util.find_spec("fastapi"),
        "Install the openai-proxy extra.",
    )
    async def test_third_party_sdk_rewards_flow_through_native_commit_once(self):
        policy, _ = self.snapshot()

        async def sdk_score(input, result):
            from triotune.agentrl import RewardAssignment

            return EpisodeScore(
                0.5,
                False,
                assignments=(
                    RewardAssignment("interaction", 2.0, position=0),
                    RewardAssignment("trajectory", 0.25, mode="advantage"),
                ),
            )

        async def execute(sdk, question):
            tools = [
                {
                    "type": "function",
                    "function": {"name": add.name, "parameters": add.parameters},
                }
            ]
            messages = (
                [{"role": "user", "content": question}]
                if isinstance(question, str)
                else question
            )
            first = await sdk.chat.completions.create(
                model=MODEL, messages=messages, tools=tools
            )
            call = first.choices[0].message.tool_calls[0]
            final = await sdk.chat.completions.create(
                model=MODEL,
                tools=tools,
                messages=messages
                + [first.choices[0].message.model_dump(exclude_none=True)]
                + [{"role": "tool", "tool_call_id": call.id, "content": "5"}],
            )
            return final.choices[0].message.content

        runner = OpenAIEpisodeExecutor(
            lambda snapshot: factory(snapshot, callback=True, sdk_scorer=sdk_score),
            scorer=Scorer(),
            execute=execute,
        )
        async with AgentRolloutWorkflow(runner) as workflow:
            batch = await workflow.collect(ROWS, policy)
        row = batch.trajectories[0]
        self.assertEqual(row.reward, 1.5)
        self.assertEqual(sum(row.token_rewards), 2.0)
        prepared = prepare_advantages([row], raw_config())
        self.assertEqual(
            next(
                value
                for value, active in zip(prepared.advantages[0], row.loss_mask)
                if active
            ),
            3.75,
        )


if __name__ == "__main__":
    unittest.main()
