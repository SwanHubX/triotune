"""Public reward-plugin signatures and their numerical PyTRIO boundary."""

import unittest
from functools import partial
from types import SimpleNamespace

from agent_test_support import AgentTrainingClient
from test_agent_trajectory_builder import episode
from test_ppo_family import raw_config

from triotune import PPOTrainer
from triotune.agentrl import EpisodeScore, EpisodeTrajectoryBuilder
from triotune.trainer.ppo import (
    NormConfig,
    PPOAlgorithmPlan,
    Trajectory,
    prepare_advantages,
)
from triotune.trainer.ppo.advantages import _DefaultRewardNormalizer, normalize


class NormalizerDispatchTests(unittest.TestCase):
    def test_public_masks_are_not_replaced_by_config(self):
        observed = []

        class Normalizer:
            def normalize(self, rows, group_ids, masks=None):
                observed.append(masks)
                return rows

        rows, groups, masks = [[1.0], [2.0]], ["g", "g"], [[True], [False]]
        result = PPOAlgorithmPlan(normalizer=Normalizer()).normalize_rewards(
            rows, groups, NormConfig("group", None), masks
        )
        self.assertIs(result, rows)
        self.assertEqual(observed, [masks])

    def test_declared_config_masks_and_legacy_two_argument_hooks(self):
        rows, groups, masks = [[1.0]], ["g"], [[True]]
        config = NormConfig("group", None)
        observed = []

        def two_args(rows, group_ids):
            observed.append(())
            return rows

        def config_only(rows, group_ids, config):
            observed.append((config,))
            return rows

        def config_first(rows, group_ids, config, masks):
            observed.append((config, masks))
            return rows

        def masks_first(rows, group_ids, masks=None, config=None):
            observed.append((config, masks))
            return rows

        def keyword_only(rows, group_ids, *, masks, config):
            observed.append((config, masks))
            return rows

        def positional_only(rows, group_ids, masks, config, /):
            observed.append((config, masks))
            return rows

        for hook, expected in (
            (two_args, ()),
            (config_only, (config,)),
            (config_first, (config, masks)),
            (masks_first, (config, masks)),
            (keyword_only, (config, masks)),
            (positional_only, (config, masks)),
        ):
            with self.subTest(hook=hook.__name__):
                self.assertIs(
                    PPOAlgorithmPlan(normalizer=hook).normalize_rewards(
                        rows, groups, config, masks
                    ),
                    rows,
                )
                self.assertEqual(observed[-1], expected)

    def test_builtin_and_partial_normalizers_keep_group_mask_semantics(self):
        rows, groups, masks = [[1.0, 100.0], [3.0]], ["g", "g"], [[True, False], [True]]
        config = NormConfig("group", None)
        for hook in (
            _DefaultRewardNormalizer(),
            normalize,
            partial(normalize, masks=masks),
        ):
            with self.subTest(hook=hook):
                result = PPOAlgorithmPlan(normalizer=hook).normalize_rewards(
                    rows, groups, config, masks
                )
                self.assertEqual(result, [[-1.0, 0.0], [1.0]])

    def test_normalizer_body_typeerror_is_not_retried(self):
        calls = []
        failure = TypeError("normalizer implementation failed")

        def hook(rows, group_ids, masks=None):
            calls.append(masks)
            raise failure

        with self.assertRaises(TypeError) as error:
            PPOAlgorithmPlan(normalizer=hook).normalize_rewards(
                [[1.0]], ["g"], NormConfig()
            )
        self.assertIs(error.exception, failure)
        self.assertEqual(calls, [None])

    def test_kwargs_adapter_receives_public_masks_without_invented_config(self):
        observed = []

        def hook(rows, group_ids, **options):
            observed.append(options)
            return rows

        masks = [[True]]
        PPOAlgorithmPlan(normalizer=hook).normalize_rewards(
            [[1.0]], ["g"], NormConfig(), masks
        )
        self.assertEqual(observed, [{"masks": masks}])

    def test_normalizer_still_receives_one_outcome_per_logical_candidate(self):
        value = episode()
        value.score = EpisodeScore(1.0, False)
        rows = EpisodeTrajectoryBuilder(mode="individual").export(value).trajectories
        observed = []

        def hook(rewards, group_ids, masks=None):
            observed.append((rewards, group_ids, masks))
            return [[2.0]]

        batch = prepare_advantages(
            rows, raw_config(), PPOAlgorithmPlan(normalizer=hook)
        )
        self.assertEqual(observed, [([[1.0]], ["group"], None)])
        for advantages, mask in zip(batch.advantages, batch.masks):
            self.assertEqual(advantages, [2.0 if active else 0.0 for active in mask])


class ProcessShaperDispatchTests(unittest.TestCase):
    def test_combined_reward_shaper_changes_process_returns_without_touching_inputs(
        self,
    ):
        class Combined:
            def shape(self, trajectory, reward):
                return 2.0 * reward

            def shape_process(self, trajectory, rewards):
                return [3.0 * reward for reward in rewards]

        row = Trajectory(
            [1],
            [2, 3, 4],
            [-1.0] * 3,
            1.0,
            "g",
            loss_mask=[True, False, True],
            turn_ids=[0, -1, 1],
            token_rewards=[1.0, 0.0, 2.0],
            token_advantages=[0.5, 0.0, 0.5],
        )
        batch = prepare_advantages(
            [row], raw_config(), PPOAlgorithmPlan(reward_shaper=Combined())
        )
        self.assertEqual(batch.returns, [[11.0, 0.0, 8.0]])
        self.assertEqual(batch.advantages, [[11.5, 0.0, 8.5]])
        self.assertEqual(row.token_rewards, [1.0, 0.0, 2.0])

    def test_explicit_process_shaper_overrides_combined_method_exactly_once(self):
        calls = []

        class Combined:
            def shape_process(self, trajectory, rewards):
                raise AssertionError("An explicit process shaper owns this stage")

        class Explicit:
            def shape_process(self, trajectory, rewards):
                calls.append(trajectory)
                return [4.0]

            def shape(self, trajectory, rewards):
                raise AssertionError("The dedicated process method has priority")

        row = Trajectory([1], [2], [-1.0], 0.0, "g", token_rewards=[1.0])
        plan = PPOAlgorithmPlan(reward_shaper=Combined(), process_shaper=Explicit())
        self.assertEqual(plan.shape_process(row, [1.0]), [4.0])
        self.assertEqual(calls, [row])

    def test_existing_callable_and_shape_object_interfaces_remain_supported(self):
        row = Trajectory([1], [2], [-1.0], 0.0, "g")
        for hook in (
            lambda trajectory, rewards: [3.0],
            lambda rewards: [3.0],
            SimpleNamespace(shape=lambda trajectory, rewards: [3.0]),
        ):
            with self.subTest(hook=hook):
                self.assertEqual(
                    PPOAlgorithmPlan(process_shaper=hook).shape_process(row, [1.0]),
                    [3.0],
                )
        self.assertEqual(
            PPOAlgorithmPlan(reward_shaper=lambda reward: 2 * reward).shape_reward(
                row, 1.0
            ),
            2.0,
        )

    def test_outcome_only_hook_does_not_implicitly_shape_process_signals(self):
        rewards = [1.0]
        plan = PPOAlgorithmPlan(reward_shaper=lambda reward: reward * 10)
        self.assertIs(plan.shape_process(None, rewards), rewards)

    def test_process_body_typeerror_is_not_retried(self):
        calls = []
        failure = TypeError("process implementation failed")

        def hook(trajectory, rewards=None):
            calls.append((trajectory, rewards))
            raise failure

        with self.assertRaises(TypeError) as error:
            PPOAlgorithmPlan(process_shaper=hook).shape_process(None, [1.0])
        self.assertIs(error.exception, failure)
        self.assertEqual(calls, [(None, [1.0])])

    def test_combined_process_outputs_keep_finite_and_alignment_validation(self):
        for invalid in ([], [1.0, 2.0], [float("nan")], [float("inf")]):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "finite aligned"),
            ):
                PPOAlgorithmPlan(
                    reward_shaper=SimpleNamespace(
                        shape_process=lambda trajectory, rewards: invalid
                    )
                ).shape_process(None, [1.0])


class PyTRIORewardPluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_plugin_failures_stop_before_pytrio_backward_or_optimizer(self):
        failure = TypeError("plugin implementation failed")
        calls = []

        def broken(*args, **kwargs):
            calls.append((args, kwargs))
            raise failure

        for plan in (
            PPOAlgorithmPlan(normalizer=broken),
            PPOAlgorithmPlan(process_shaper=broken),
        ):
            with self.subTest(plan=plan):
                calls.clear()
                client = AgentTrainingClient()
                trainer = PPOTrainer(
                    "fake", args=raw_config(), training_client=client, plan=plan
                )
                try:
                    with self.assertRaises(TypeError) as error:
                        await trainer.step([Trajectory([1], [2], [-1.0], 1.0, "g")])
                    self.assertIs(error.exception, failure)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(client.calls, [])
                    self.assertEqual(client.version, 0)
                    self.assertEqual(trainer.state.global_step, 0)
                finally:
                    await trainer.close()

    async def test_plugins_reach_custom_and_native_updates_as_numeric_fields(self):
        for backend in ("custom", "native"):
            with self.subTest(backend=backend):
                observed = []

                class Combined:
                    def shape(self, trajectory, reward):
                        return reward

                    def shape_process(self, trajectory, rewards):
                        return [3.0]

                def hook(rows, group_ids, masks=None):
                    observed.append(masks)
                    return [[2.0]]

                client = AgentTrainingClient()
                trainer = PPOTrainer(
                    "fake",
                    args=raw_config(loss_backend=backend),
                    training_client=client,
                    plan=PPOAlgorithmPlan(reward_shaper=Combined(), normalizer=hook),
                )
                try:
                    row = Trajectory([1], [2], [-1.0], 1.0, "g", token_rewards=[1.0])
                    await trainer.initialize()
                    prepared = await trainer._actor.prepare([row])
                    self.assertEqual(prepared.returns, [[5.0]])
                    self.assertEqual(prepared.advantages, [[5.0]])
                    await trainer.step([row])
                    self.assertEqual(trainer.state.global_step, 1)
                    self.assertEqual(observed, [None, None])
                    call = next(call for call in client.calls if call[0] == backend)
                    datum = call[1][0]
                    self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [2])
                    self.assertEqual(
                        set(datum.loss_fn_inputs),
                        (
                            {"target_tokens", "advantages", "logprobs"}
                            if backend == "native"
                            else {"target_tokens"}
                        ),
                    )
                    if backend == "native":
                        self.assertEqual(datum.loss_fn_inputs["advantages"].data, [5.0])
                finally:
                    await trainer.close()


if __name__ == "__main__":
    unittest.main()
