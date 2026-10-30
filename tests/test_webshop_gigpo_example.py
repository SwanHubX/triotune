"""测试案例适配器与真实 TrioTune 训练路径；不需要 HF/Java/远程服务。

假环境仅替代旧 WebShop 依赖，runner/exporter/GiGPO/Torch loss 均使用真实实现。
重点覆盖目标隔离、可信奖励、失败清理和 eval 无优化器更新，而非复刻实现细节。
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml
from agent_test_support import AgentTokenizer, AgentTrainingClient

from examples.gigpo.data_prepare import (
    WEBSHOP_REVISION,
    DataConfig,
    asset_manifest,
    cached_files,
    goal_fingerprint,
    goal_ranges,
    link_assets,
    prepare,
    task_rows,
)
from examples.gigpo.environment import SessionPool, WebShopBackend, build_workflow
from examples.gigpo.evaluation import (
    evaluate_policy,
    load_actor_config,
    webshop_score_metrics,
)
from examples.gigpo.train import (
    ContinuationConfig,
    RewardConfig,
    apply_legacy_continuation,
    advantage_provider,
    build_parser,
    load_experiment,
    run,
)
from triotune import PPOConfig, PPOTrainer
from triotune.agentrl import (
    AgentConfig,
    AgentTrainingLoop,
    EpisodeResult,
    EpisodeScore,
    Generation,
    GiGPOAdvantageProvider,
    GiGPOConfig,
    SamplingSnapshot,
    ToolActionError,
    current_execution_context,
)

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "examples/gigpo/configs/grpo-webshop-configs.yaml"
)


class FakeShop:
    """保留 search → 商品页 → 购买状态机，商品 a/b 分别返回 1/0.5。"""

    def __init__(self):
        self.page = "initial"
        self.product = None

    def get_available_actions(self):
        return {
            "has_search_bar": self.page == "initial",
            "clickables": {
                "initial": [],
                "results": ["a", "b"],
                "item": ["buy now"],
                "terminal": [],
            }[self.page],
        }

    def step(self, action):
        if action.startswith("search["):
            self.page = "results"
        elif action in {"click[a]", "click[b]"}:
            self.product, self.page = action[6:-1], "item"
        elif action == "click[buy now]":
            self.page = "terminal"
        done = self.page == "terminal"
        return (
            self.page,
            (1.0 if self.product == "a" else 0.5) if done else 0,
            done,
            None,
        )


class FakeBackend:
    def __init__(self):
        self.goals = [
            {"instruction_text": f"goal {i}", "query": "shoes"} for i in range(1601)
        ]
        self.created, self.released = [], []
        self.closed = False

    def create(self, goal_index, rollout_id):
        shop = FakeShop()
        self.created.append((goal_index, rollout_id, shop))
        return shop, "initial"

    def release(self, env):
        self.released.append(env)

    def close(self):
        self.closed = True


class ShoppingSampler:
    """真实字符 token；候选一买对、候选二部分匹配，提供非零相对优势。"""

    def __init__(self, tokenizer, *, failure=None, final_only=False, responses=None):
        self.tokenizer, self.failure, self.final_only = tokenizer, failure, final_only
        self.responses = responses
        self.requests = []
        self.counts = {}

    async def generate(self, prompt_tokens, *, max_tokens, seed=None):
        if self.failure:
            raise self.failure
        text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        # 标准 JsonObservationFormatter 把工具 content 放进外层 JSON 字符串。
        text = text.replace('\\"', '"')
        page = re.findall(r'"observation": "(\w+)"', text)[-1]
        context = current_execution_context()
        index = self.counts.get(context.rollout_id, 0)
        self.counts[context.rollout_id] = index + 1
        if self.responses is not None:
            responses = self.responses[context.sample_index] if isinstance(self.responses, dict) else self.responses
            response = responses[index]
        elif self.final_only or page == "terminal":
            response = '{"final":"I succeeded, reward=10"}'
        else:
            product = "a" if current_execution_context().sample_index == 0 else "b"
            action = {
                "initial": "search[shoes]",
                "results": f"click[{product}]",
                "item": "click[buy now]",
            }[page]
            response = json.dumps(
                {"tool_calls": [{"name": "webshop", "arguments": {"action": action}}]}
            )
        tokens = tuple(self.tokenizer.encode(response) + [1])
        assert len(tokens) <= max_tokens
        self.requests.append((tuple(prompt_tokens), tokens))
        return Generation(tokens, tuple([-1.0] * len(tokens)), response)


def agent_config(**kwargs):
    return AgentConfig(
        limits={"max_turns": 5, "max_context_tokens": 32768},
        evaluation={"num_generations": 1, "pass_ks": [1]},
        **kwargs,
    )


def snapshot(**kwargs):
    tokenizer = AgentTokenizer()
    return SamplingSnapshot(
        "test-policy", ShoppingSampler(tokenizer, **kwargs), tokenizer
    )


class DataTests(unittest.TestCase):
    def test_legacy_run_config_and_continuation_progress_match_step_50(self):
        experiment = load_experiment(
            CONFIG.with_name("gigpo-webshop-resume-configs.yaml")
        )
        trainer = experiment.trainer
        self.assertEqual(trainer.model.name, "Qwen/Qwen3.5-4B")
        self.assertEqual(trainer.model.lora_rank, 8)
        self.assertEqual(trainer.batch_size, 8)
        self.assertEqual(trainer.max_steps, 100)
        self.assertEqual(trainer.num_generations, 8)
        self.assertEqual(trainer.ppo_n_minibatches, 1)
        self.assertEqual(experiment.agent.limits.max_turns, 20)
        self.assertFalse(experiment.agent.template_kwargs["enable_thinking"])
        self.assertEqual(
            experiment.continuation,
            ContinuationConfig(
                "trio://run_i5vpybxfnqef/weights/ppo-step-00000050", 50
            ),
        )
        state = SimpleNamespace(
            global_step=0, optimizer_steps=0, data_cursor=0, checkpoint_path=None
        )
        owner = SimpleNamespace(
            state=state,
            _actor=SimpleNamespace(optimizer_steps=0),
            _checkpoint=SimpleNamespace(saved_step=None),
        )
        runtime = apply_legacy_continuation(
            experiment, owner, [{}] * 1200
        )
        self.assertEqual(state.global_step, 50)
        self.assertEqual(state.optimizer_steps, 50)
        self.assertEqual(state.data_cursor, 400)
        self.assertEqual(runtime["restored_seed_offset"], 64000)
        self.assertFalse(runtime["exact_resume"])
        fresh = load_experiment(CONFIG.with_name("gigpo-webshop-configs.yaml"))
        self.assertIsNone(fresh.continuation)
        self.assertIsNone(fresh.trainer.checkpoint.state_dir)
        self.assertIsNone(fresh.trainer.checkpoint.resume_from)
        self.assertEqual(fresh.trainer.max_steps, 150)

    def test_legacy_continuation_rejects_invalid_or_conflicting_settings(self):
        for values in (
            {"checkpoint": "weights", "global_step": 50},
            {"checkpoint": "trio://weights", "global_step": 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContinuationConfig(**values)
        settings = yaml.safe_load(
            CONFIG.with_name("gigpo-webshop-resume-configs.yaml").read_text()
        )
        settings["trainer"]["checkpoint"]["resume_from"] = "state.json"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.yaml"
            path.write_text(yaml.safe_dump(settings))
            with self.assertRaisesRegex(ValueError, "either legacy continuation"):
                load_experiment(path)

    def test_both_protocols_use_disjoint_goal_ranges(self):
        for protocol, train_start, eval_start, eval_stop in (
            ("verl", 500, 0, 500),
            ("official", 1500, 500, 1500),
        ):
            ranges = goal_ranges(1601, protocol)
            self.assertEqual(ranges["train"], range(train_start, 1601))
            self.assertEqual(ranges["eval"], range(eval_start, eval_stop))
            self.assertTrue(set(ranges["train"]).isdisjoint(ranges["eval"]))
            self.assertTrue(set(ranges["train"]).isdisjoint(ranges["test"]))
            if protocol == "official":
                self.assertTrue(set(ranges["eval"]).isdisjoint(ranges["test"]))
        for count, protocol in ((500, "verl"), (1500, "official"), (1600, "typo")):
            with self.assertRaises(ValueError):
                goal_ranges(count, protocol)

    def test_rows_are_deterministic_and_limits_never_change_train_membership(self):
        goals = FakeBackend().goals
        config = DataConfig(train_epochs=2, eval_limit=3)
        rows = task_rows(goals, config, "train")
        self.assertEqual(rows, task_rows(goals, config, "train"))
        self.assertEqual(len(rows), 2 * (1601 - 1500))
        self.assertTrue(all(row["goal_index"] >= 1500 for row in rows))
        self.assertNotIn("messages", rows[0])
        self.assertEqual(
            [row["goal_index"] for row in task_rows(goals, config, "eval")],
            [500, 501, 502],
        )
        self.assertNotEqual(goal_fingerprint(goals), goal_fingerprint(goals[::-1]))
        for kwargs in (
            {"train_epochs": 0},
            {"seed": True},
            {"eval_limit": 0},
            {"local_files_only": "true"},
        ):
            with self.assertRaises(ValueError):
                DataConfig(**kwargs)

    def test_hf_default_cache_and_no_overwrite_asset_links(self):
        with patch(
            "examples.gigpo.data_prepare.hf_hub_download", return_value="/cached/file"
        ) as download:
            files = cached_files(DataConfig())
        self.assertIn("items_shuffle_1000.json", files)
        for call in download.call_args_list:
            self.assertNotIn("cache_dir", call.kwargs)
            self.assertNotIn("local_dir", call.kwargs)
            self.assertTrue(call.kwargs["local_files_only"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "cached.json"
            original.write_text("[]")
            source = root / "checkout"
            link_assets(source, {"items.json": original})
            link_assets(source, {"items.json": original})
            other = root / "other.json"
            other.write_text("{}")
            with self.assertRaises(FileExistsError):
                link_assets(source, {"items.json": other})
            self.assertEqual((source / "data/items.json").read_text(), "[]")

    def test_config_reuses_existing_gigpo_contract_and_rejects_conflicts(self):
        experiment = load_experiment(CONFIG, steps=1, algorithm="gigpo")
        self.assertEqual(experiment.trainer.max_steps, 1)
        self.assertFalse(experiment.trainer.use_critic)
        self.assertIsNone(experiment.trainer.adv_norm)
        self.assertEqual(experiment.trainer.reference_kl_coef, 0.01)
        self.assertTrue(experiment.source_dir.is_absolute())
        with tempfile.TemporaryDirectory() as directory:
            for section, key, value in (
                ("trainer", "actor", {"kl_ctl": 0.1}),
                ("agent", "async_rollouts", True),
                ("agent", "retry", {"retry_tools": True}),
                ("data", "eval_limit", 0),
            ):
                settings = yaml.safe_load(CONFIG.read_text())
                # 本测试验证 GiGPO 专属冲突；正式共享配置默认运行 GRPO。
                settings["algorithm"] = "gigpo"
                settings[section][key] = value
                path = Path(directory) / "bad.yaml"
                path.write_text(yaml.safe_dump(settings))
                with (
                    self.subTest(section=section, key=key),
                    self.assertRaises(ValueError),
                ):
                    load_experiment(path)


class NativeWebShopTests(unittest.TestCase):
    """验证官方构造和 loader 契约，不依赖 Java、网络或原始仓库 checkout。"""

    def test_backend_uses_native_api_seed_and_separate_browser_history(self):
        class Browser:
            def __init__(self, server):
                self.server = server

        class NativeShop:
            def __init__(
                self,
                *,
                observation_mode,
                file_path,
                num_products,
                human_goals,
                get_image,
                num_prev_obs,
                num_prev_actions,
            ):
                # 显式签名拒绝分支环境才接受的 attr_path/seed 等参数。
                self.server = SimpleNamespace(
                    goals=[{"price": random.random()} for _ in range(2)],
                    user_sessions={},
                    search_engine=SimpleNamespace(close=lambda: None),
                )
                self.browser = Browser(self.server)
                self.session_prefix = None
                self.reset()

            def reset(self, session=0):
                self.session = f"{self.session_prefix or ''}{session}"
                self.server.user_sessions[self.session] = {
                    "goal": self.server.goals[session]
                }
                self.prev_obs, self.prev_actions = [f"goal-{session}"], []
                return self.prev_obs[0], None

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "webshop"
            module_path = source / "web_agent_site/envs/web_agent_text_env.py"
            module_path.parent.mkdir(parents=True)
            module_path.touch()
            (source / "search_engine/indexes_1k").mkdir(parents=True)
            asset = root / "items.json"
            asset.write_text("[]")
            files = {"items_shuffle_1000.json": asset}
            manifest = source / ".triotune-webshop.json"
            manifest.write_text(json.dumps(asset_manifest(files)))
            module = SimpleNamespace(
                __file__=str(module_path), WebAgentTextEnv=NativeShop
            )
            state = random.getstate()
            with (
                patch(
                    "examples.gigpo.environment.subprocess.check_output",
                    return_value=WEBSHOP_REVISION,
                ),
                patch(
                    "examples.gigpo.environment.importlib.import_module",
                    return_value=module,
                ),
            ):
                backend = WebShopBackend(source, files, 42)
                repeated = WebShopBackend(source, files, 42)
                self.assertEqual(random.getstate(), state)
                self.assertEqual(backend.goals, repeated.goals)
                self.assertEqual(backend.template.server.user_sessions, {})
                first, _ = backend.create(1, "candidate-a")
                second, _ = backend.create(1, "candidate-b")
                self.assertIs(first.server, second.server)
                self.assertIsNot(first.browser, second.browser)
                self.assertIsNot(first.prev_actions, second.prev_actions)
                self.assertIsNot(first.prev_obs, second.prev_obs)
                self.assertNotEqual(first.session, second.session)
                backend.release(first)
                self.assertIn(second.session, backend.template.server.user_sessions)
                backend.release(second)
                self.assertEqual(backend.template.server.user_sessions, {})
                backend.close()
                repeated.close()
                manifest.write_text("{}")
                with self.assertRaisesRegex(ValueError, "assets do not match"):
                    WebShopBackend(source, files, 42)
            with (
                patch(
                    "examples.gigpo.environment.subprocess.check_output",
                    return_value="wrong-revision",
                ),
                self.assertRaisesRegex(ValueError, "source revision"),
            ):
                WebShopBackend(source, files, 42)

    def test_prepare_calls_native_loader_without_attrpath_and_reuses_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "webshop"
            engine_path = source / "web_agent_site/engine/engine.py"
            engine_path.parent.mkdir(parents=True)
            engine_path.touch()
            (source / "search_engine").mkdir()
            asset = root / "items.json"
            asset.write_text("[]")
            files = {"items_shuffle_1000.json": asset}
            experiment = SimpleNamespace(
                source_dir=source, data=DataConfig(), environment_seed=42
            )

            def load_products(*, filepath, num_products, human_goals):
                self.assertEqual(num_products, 1000)
                self.assertFalse(human_goals)
                random.random()
                return (
                    [
                        {
                            "asin": "ITEM",
                            "Title": "Shoes",
                            "Description": "Comfortable",
                            "BulletPoints": ["Blue"],
                        }
                    ],
                )

            def build_index(command, *, check):
                self.assertTrue(check)
                Path(command[command.index("--index") + 1]).mkdir()

            module = SimpleNamespace(
                __file__=str(engine_path), load_products=load_products
            )
            state = random.getstate()
            with (
                patch("examples.gigpo.data_prepare.cached_files", return_value=files),
                patch(
                    "examples.gigpo.data_prepare.subprocess.check_output",
                    return_value=WEBSHOP_REVISION,
                ),
                patch(
                    "examples.gigpo.data_prepare.subprocess.run",
                    side_effect=build_index,
                ) as build,
                patch(
                    "examples.gigpo.data_prepare.importlib.import_module",
                    return_value=module,
                ),
            ):
                expected = prepare(experiment)
                self.assertEqual(random.getstate(), state)
                self.assertEqual(prepare(experiment), expected)
                build.assert_called_once()
                (source / ".triotune-webshop.json").write_text("{}")
                with self.assertRaisesRegex(FileExistsError, "unknown/different data"):
                    prepare(experiment)


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = FakeBackend()
        self.pool = SessionPool(self.backend)

    async def asyncTearDown(self):
        await self.pool.close()

    async def test_same_goal_candidates_have_independent_state_and_terminal_guard(self):
        await asyncio.gather(
            self.pool.open("candidate-a", 500), self.pool.open("candidate-b", 500)
        )
        await self.pool.step("candidate-a", "search[shoes]")
        self.assertEqual(
            (await self.pool.observe("candidate-b"))["observation"], "initial"
        )
        for invalid in (
            None,
            "search[]",
            "search[<r>]",
            "click[hidden-answer]",
            "search[x]\nclick[y]",
        ):
            with self.assertRaises(ToolActionError):
                await self.pool.step("candidate-b", invalid)
        await self.pool.step("candidate-a", "click[a]")
        await self.pool.step("candidate-a", "click[buy now]")
        self.assertEqual(await self.pool.outcome("candidate-a"), (True, 1.0))
        with self.assertRaises(ToolActionError):
            await self.pool.step("candidate-a", "click[buy now]")

    async def collect(self, *, reward=None, process_reward_fn=None, **sampler_options):
        policy = snapshot(**sampler_options)
        workflow = build_workflow(
            agent_config(),
            policy.tokenizer,
            self.pool,
            reward or RewardConfig(),
            process_reward_fn=process_reward_fn,
        )
        self.addAsyncCleanup(workflow.close)
        return await workflow.collect(
            [{"task_id": "train-500", "goal_index": 500}],
            policy,
            n_samples=2,
            strict=False,
        )

    async def test_real_runner_exports_trusted_anchors_and_gigpo_masked_advantages(
        self,
    ):
        batch = await self.collect()
        self.assertEqual([e.score.reward for e in batch.episodes], [10, 0])
        self.assertEqual(
            [e.score.components["task_score"] for e in batch.episodes], [1, 0.5]
        )
        self.assertEqual(len(self.pool.sessions), 0)
        self.assertEqual(len(self.backend.released), 2)
        self.assertEqual(
            [t.anchor_observations[0] for t in batch.trajectories],
            [
                {
                    "observation": "initial",
                    "available_actions": {"has_search_bar": True, "clickables": []},
                    "done": False,
                }
            ]
            * 2,
        )
        advantages = GiGPOAdvantageProvider().advantages(batch.trajectories)
        for trajectory, values in zip(
            batch.trajectories, advantages.values, strict=True
        ):
            self.assertEqual(trajectory.anchor_observations[-1]["done"], True)
            self.assertTrue(any(not active for active in trajectory.loss_mask))
            self.assertTrue(any(value != 0 for value in values))
            self.assertTrue(
                all(
                    active or value == 0
                    for active, value in zip(trajectory.loss_mask, values, strict=True)
                )
            )

    async def test_anchor_snapshot_and_zero_state_weight_ablation(self):
        """后续页面变更不能改写动作前锚点；关闭状态项后保留 episode 信号。"""
        await self.pool.open("anchor", 500)
        anchor = await self.pool.observe("anchor")
        await self.pool.step("anchor", "search[shoes]")
        self.assertEqual(anchor["observation"], "initial")
        self.assertEqual(anchor["available_actions"]["clickables"], [])
        await self.pool.release("anchor")
        batch = await self.collect()
        provider = GiGPOAdvantageProvider(GiGPOConfig(step_advantage_weight=0))
        ablated = provider.advantages(batch.trajectories)
        full = GiGPOAdvantageProvider().advantages(batch.trajectories)
        self.assertNotEqual(ablated.values, full.values)
        for trajectory, values in zip(batch.trajectories, ablated.values, strict=True):
            active = [
                v for v, mask in zip(values, trajectory.loss_mask, strict=True) if mask
            ]
            self.assertEqual(set(active), {trajectory.reward - 5.0})

    async def test_process_reward_hook_keeps_outcome_and_token_alignment(self):
        """研究钩子沿用真实 token 导出；过程信号不能篡改环境成功率。"""

        def process_reward(row, episode):
            self.assertEqual(row["goal_index"], 500)
            tokens = episode.turns[0].generation.tokens
            return {0: (0.0,) * (len(tokens) - 1) + (0.25,)}

        batch = await self.collect(process_reward_fn=process_reward)
        self.assertEqual([e.score.reward for e in batch.episodes], [10, 0])
        for trajectory in batch.trajectories:
            self.assertAlmostEqual(sum(trajectory.token_rewards), 0.25)
            self.assertTrue(
                all(
                    value == 0
                    for value, active in zip(
                        trajectory.token_rewards, trajectory.loss_mask, strict=True
                    )
                    if not active
                )
            )

    async def test_dense_reward_and_model_claims_are_not_environment_outcomes(self):
        batch = await self.collect(reward=RewardConfig(mode="dense"))
        self.assertEqual([e.score.reward for e in batch.episodes], [1, 0.5])
        batch = await self.collect(final_only=True)
        self.assertTrue(
            all(e.score.reward == 0 and not e.score.success for e in batch.episodes)
        )

    async def test_sampling_failure_and_cancellation_release_all_sessions(self):
        batch = await self.collect(failure=ConnectionError("sampler unavailable"))
        self.assertTrue(
            all(
                e.status == "sampling_failure" and e.score is None
                for e in batch.episodes
            )
        )
        self.assertEqual(self.pool.sessions, {})
        policy = snapshot()
        entered = asyncio.Event()

        async def wait_forever(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        policy.sampler.generate = wait_forever
        workflow = build_workflow(
            agent_config(), policy.tokenizer, self.pool, RewardConfig()
        )
        try:
            task = asyncio.create_task(
                workflow.collect([{"task_id": "cancel", "goal_index": 500}], policy)
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            # observe/release 使用相同执行队列；这个屏障等待取消后的清理完成。
            await self.pool._call(lambda: None)
            self.assertEqual(self.pool.sessions, {})
        finally:
            await workflow.close()

    async def test_training_uses_real_torch_loss_and_evaluation_does_not_optimize(self):
        client = AgentTrainingClient()
        policy = snapshot()
        config = PPOConfig.for_algorithm(
            "grpo",
            model={"name": "fake"},
            batch_size=1,
            max_steps=1,
            num_generations=2,
            reward_norm=None,
            adv_norm=None,
            eval_before_train=False,
            eval_after_train=False,
        )
        trainer = PPOTrainer(
            args=config,
            training_client=client,
            train_dataset=[{"task_id": "train-500", "goal_index": 500}],
            eval_dataset=[{"task_id": "eval-0", "goal_index": 0}],
        )
        workflow = build_workflow(
            agent_config(), policy.tokenizer, self.pool, RewardConfig()
        )

        async def fixed_snapshot(policy=policy):
            return policy

        loop = AgentTrainingLoop(
            trainer,
            workflow,
            snapshot_factory=fixed_snapshot,
            config=agent_config(),
            batch_advantage_provider=GiGPOAdvantageProvider(),
        )
        with patch("triotune.agentrl.training.loop.tqdm"):
            state = await loop.train()
            self.assertEqual(state.optimizer_steps, 1)
            self.assertTrue(any(call[0] == "custom" for call in client.calls))
            before = client.trace.count("submit:optim")
            try:
                metrics = await loop.evaluate()
            finally:
                await workflow.close()
            self.assertEqual(client.trace.count("submit:optim"), before)
        self.assertEqual(metrics["eval/agent/success_rate"], 1)
        self.assertEqual(
            [created[0] for created in self.backend.created], [500, 500, 0]
        )
        self.assertEqual(self.pool.sessions, {})

    async def test_grpo_and_ppo_use_native_advantage_paths_and_independent_critic(self):
        """同一购物轨迹可进入 GRPO 或 PPO；PPO 的 critic 使用独立客户端更新。"""
        for algorithm in ("grpo", "ppo"):
            with self.subTest(algorithm=algorithm):
                experiment = load_experiment(CONFIG, algorithm=algorithm, steps=1)
                config = replace(
                    experiment.trainer,
                    training=replace(experiment.trainer.training, batch_size=1),
                    rollout=replace(experiment.trainer.rollout, num_generations=2),
                    actor=replace(experiment.trainer.actor, reference_kl_coef=0, ppo_n_minibatches=1),
                    evaluation=replace(
                        experiment.trainer.evaluation,
                        before_train=False,
                        after_train=False,
                    ),
                    # 离线测试不初始化正式配置中的 SwanLab 在线上报。
                    reporting=replace(
                        experiment.trainer.reporting, log_dir=None, report_to=None
                    ),
                )
                actor = AgentTrainingClient()
                critic = AgentTrainingClient() if algorithm == "ppo" else None
                trainer = PPOTrainer(
                    args=config,
                    training_client=actor,
                    critic_training_client=critic,
                    train_dataset=[{"task_id": "train-500", "goal_index": 500}],
                )
                policy = snapshot()
                workflow = build_workflow(
                    agent_config(), policy.tokenizer, self.pool, RewardConfig()
                )

                async def fixed_snapshot(policy=policy):
                    return policy

                loop = AgentTrainingLoop(
                    trainer,
                    workflow,
                    snapshot_factory=fixed_snapshot,
                    config=agent_config(),
                )
                with patch("triotune.agentrl.training.loop.tqdm"):
                    state = await loop.train()
                self.assertEqual(state.optimizer_steps, 1)
                self.assertEqual(state.critic_optimizer_steps, int(algorithm == "ppo"))
                if critic is not None:
                    self.assertIn("submit:optim", critic.trace)
                    self.assertIsNot(actor, critic)


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_continuation_uses_trainer_owned_optimizer_restore(self):
        experiment = load_experiment(
            CONFIG.with_name("gigpo-webshop-resume-configs.yaml")
        )
        config = replace(
            experiment.trainer,
            reporting=replace(
                experiment.trainer.reporting, report_to=None, log_dir=None
            ),
        )
        client = AgentTrainingClient()
        service = SimpleNamespace(
            create_training_client_from_state_with_optimizer_async=AsyncMock(
                return_value=client
            )
        )
        trainer = PPOTrainer(
            args=config,
            service_client=service,
            train_dataset=[{"task_id": "train", "goal_index": 1500}],
            eval_dataset=[],
        )
        trainer._checkpoint.bootstrap_remote_optimizer(
            experiment.continuation.checkpoint
        )
        try:
            await trainer.initialize()
        finally:
            await trainer.close()
        service.create_training_client_from_state_with_optimizer_async.assert_awaited_once_with(
            path="trio://run_i5vpybxfnqef/weights/ppo-step-00000050"
        )

    def test_webshop_score_uses_partial_outcomes_and_counts_unscored_failures(self):
        """Score 使用原始部分匹配分，不受稀疏成功奖励或过程奖励影响。"""
        episodes = [
            EpisodeResult(
                str(index),
                "goal",
                "policy",
                status=status,
                score=EpisodeScore(
                    10.0 if value == 1 else 0.0,
                    value == 1,
                    {"task_score": value},
                    process_rewards={0: (99.0,)},
                ),
            )
            for index, (value, status) in enumerate(
                ((1.0, "completed"), (0.5, "max_turns"), (0.0, "invalid_action"))
            )
        ]
        metrics = webshop_score_metrics(episodes)
        self.assertEqual(metrics["eval/agent/task_score_mean"], 0.5)
        self.assertEqual(metrics["eval/agent/score"], 50.0)
        self.assertEqual(metrics["eval/agent/task_score_count"], 3)
        self.assertEqual(metrics["eval/agent/unscored_episodes"], 0)
        episodes.append(
            EpisodeResult("failure", "goal", "policy", status="sampling_failure")
        )
        metrics = webshop_score_metrics(episodes)
        self.assertEqual(metrics["eval/agent/score"], 37.5)
        self.assertEqual(metrics["eval/agent/task_score_count"], 3)
        self.assertEqual(metrics["eval/agent/unscored_episodes"], 1)
        self.assertEqual(webshop_score_metrics(episodes[-1:])["eval/agent/score"], 0)

    def test_webshop_score_rejects_empty_missing_and_invalid_raw_scores(self):
        """缺少原始分值的已评分记录是契约错误，不能静默计零或产生 NaN。"""
        with self.assertRaisesRegex(ValueError, "at least one"):
            webshop_score_metrics([])
        for value in (None, True, "0.5", -0.1, 1.1, float("nan"), float("inf")):
            episode = SimpleNamespace(
                episode_id="bad-score",
                score=SimpleNamespace(components={"task_score": value}),
            )
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "task_score"),
            ):
                webshop_score_metrics([episode])
        episode.score.components = {}
        with self.assertRaisesRegex(ValueError, "task_score"):
            webshop_score_metrics([episode])

    async def test_algorithm_presets_and_cloud_lora_loading(self):
        """三种算法保留各自优势路径；PPO 测试只加载 actor，不创建 critic。"""
        for algorithm in ("gigpo", "grpo", "ppo"):
            experiment = load_experiment(CONFIG, algorithm=algorithm)
            self.assertEqual(experiment.trainer.use_critic, algorithm == "ppo")
            self.assertEqual(
                advantage_provider(experiment) is not None, algorithm == "gigpo"
            )
            service = SimpleNamespace(
                create_training_client_from_state_async=AsyncMock(
                    return_value="loaded-actor"
                )
            )
            config, client = await load_actor_config(
                experiment, service, "trio://actor-lora", evaluation=True
            )
            self.assertFalse(config.use_critic)
            self.assertEqual(client, "loaded-actor")
            service.create_training_client_from_state_async.assert_awaited_once_with(
                path="trio://actor-lora"
            )
            service.create_training_client_from_state_async.reset_mock()
            _, client = await load_actor_config(
                experiment, service, None, evaluation=True
            )
            self.assertIsNone(client)
            service.create_training_client_from_state_async.assert_not_awaited()
            service.create_training_client_from_state_async.side_effect = RuntimeError(
                "missing LoRA"
            )
            with self.assertRaisesRegex(RuntimeError, "missing LoRA"):
                await load_actor_config(
                    experiment, service, "trio://missing", evaluation=True
                )
        with self.assertRaisesRegex(ValueError, "requires algorithm=gigpo"):
            load_experiment(CONFIG, algorithm="grpo", step_advantage_weight=0)
        self.assertEqual(
            load_experiment(
                CONFIG, algorithm="gigpo", step_advantage_weight=0
            ).gigpo.step_advantage_weight,
            0,
        )

    async def test_empty_checkpoint_is_rejected_and_output_paths_are_isolated(self):
        experiment = load_experiment(CONFIG, output_dir=Path("output/isolated-test"))
        self.assertEqual(
            experiment.trainer.reporting.log_dir, str(experiment.output_dir / "metrics")
        )
        self.assertEqual(
            experiment.agent.log_path, str(experiment.output_dir / "episodes.jsonl")
        )
        service = SimpleNamespace(create_training_client_from_state_async=AsyncMock())
        with self.assertRaisesRegex(ValueError, "nonempty"):
            await load_actor_config(experiment, service, "", evaluation=True)
        service.create_training_client_from_state_async.assert_not_awaited()

    async def test_evaluation_reports_weight_source_and_detects_optimizer_updates(self):
        state = SimpleNamespace(optimizer_steps=3, critic_optimizer_steps=2)
        publish = AsyncMock()
        loop = SimpleNamespace(
            trainer=SimpleNamespace(state=state, publish_metrics=publish),
            evaluate=AsyncMock(return_value={"success_rate": 0.5}),
            last_eval_batch=SimpleNamespace(
                episodes=[
                    EpisodeResult(
                        "partial",
                        "goal",
                        "policy",
                        score=EpisodeScore(0, False, {"task_score": 0.5}),
                    )
                ]
            ),
        )
        result = await evaluate_policy(loop, checkpoint="trio://actor-lora")
        self.assertEqual(result["policy_source"], "cloud_lora")
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(result["actor_lora_checkpoint"], "trio://actor-lora")
        self.assertEqual(result["metrics"]["eval/agent/score"], 50)
        self.assertEqual(result["metrics"]["eval/agent/task_score_mean"], 0.5)
        publish.assert_awaited_once_with(
            webshop_score_metrics(loop.last_eval_batch.episodes)
        )
        baseline = await evaluate_policy(loop, checkpoint=None)
        self.assertEqual(baseline["policy_source"], "base_model")

        async def unexpected_update():
            state.critic_optimizer_steps += 1
            return {}

        loop.evaluate.side_effect = unexpected_update
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            await evaluate_policy(loop, checkpoint=None)
        args = build_parser(evaluation=True).parse_args(
            ["--algorithm", "ppo", "--checkpoint", "trio://actor-lora"]
        )
        self.assertEqual(args.mode, "evaluate")
        self.assertEqual(args.algorithm, "ppo")


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_entrypoint_exports_loaded_lora_and_never_optimizes(self):
        """端到端独立测试确实从加载的 actor 导出采样，而不是另建基础策略。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = yaml.safe_load(CONFIG.read_text())
            config["data"]["eval_limit"] = 1
            config["trainer"]["reporting"]["report_to"] = None
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            asset = root / "asset.json"
            asset.write_text("[]")
            actor = AgentTrainingClient()
            tokenizer = AgentTokenizer()
            tokenizer.eos_token_id = 1
            actor.get_tokenizer = lambda: tokenizer
            service = SimpleNamespace(
                create_training_client_from_state_async=AsyncMock(return_value=actor)
            )
            args = build_parser(evaluation=True).parse_args(
                [
                    "--config",
                    str(path),
                    "--algorithm",
                    "ppo",
                    "--checkpoint",
                    "trio://actor-lora",
                    "--output-dir",
                    str(root / "output"),
                ]
            )
            with (
                patch(
                    "examples.gigpo.train.cached_files",
                    return_value={"items.json": asset},
                ),
                patch(
                    "examples.gigpo.train.WebShopBackend", return_value=FakeBackend()
                ),
                patch(
                    "examples.gigpo.train.subprocess.check_output",
                    return_value="source-revision",
                ),
                patch(
                    "examples.gigpo.train.pytrio.ServiceClient", return_value=service
                ),
                patch("triotune.agentrl.training.loop.tqdm"),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                report = await run(args)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["policy_source"], "cloud_lora")
            self.assertEqual(report["actor_lora_checkpoint"], "trio://actor-lora")
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(report["metrics"]["eval/agent/score"], 0)
            self.assertEqual(report["metrics"]["eval/agent/task_score_mean"], 0)
            self.assertEqual(report["metrics"]["eval/agent/task_score_count"], 1)
            self.assertEqual(report["metrics"]["eval/agent/unscored_episodes"], 0)
            metric_files = list((root / "output/metrics").glob("*.jsonl"))
            self.assertEqual(len(metric_files), 1)
            history = [
                json.loads(line) for line in metric_files[0].read_text().splitlines()
            ]
            self.assertTrue(
                any("eval/agent/score" in event.get("metrics", {}) for event in history)
            )
            self.assertFalse(report["effective_trainer"]["critic"]["enabled"])
            service.create_training_client_from_state_async.assert_awaited_once_with(
                path="trio://actor-lora"
            )
            self.assertTrue(actor.snapshots)
            self.assertTrue(any(sampler.requests for sampler in actor.snapshots))
            self.assertNotIn("submit:optim", actor.trace)

    async def test_check_writes_actual_splits_and_failures_without_training_service(
        self,
    ):
        """检查入口的成功与缺失依赖路径都保留报告，check 不连接训练服务。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = yaml.safe_load(CONFIG.read_text())
            config["output_dir"] = str(root / "output")
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            asset = root / "asset.json"
            asset.write_text("[]")
            args = argparse.Namespace(
                config=path,
                mode="check",
                model=None,
                steps=None,
                checkpoint=None,
                eval_split="eval",
            )
            backend = FakeBackend()
            with (
                patch(
                    "examples.gigpo.train.cached_files",
                    return_value={"items.json": asset},
                ),
                patch(
                    "examples.gigpo.train.WebShopBackend", return_value=backend
                ) as constructor,
                patch(
                    "examples.gigpo.train.subprocess.check_output",
                    return_value="source-revision\n",
                ),
                patch("examples.gigpo.train.pytrio.ServiceClient") as service,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                report = await run(args)
                service.assert_not_called()
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["eval_goal_indices"], list(range(500, 1500)))
                self.assertEqual(report["goal_ranges"]["train"], [1500, 1601])
                self.assertTrue(backend.closed)
                self.assertEqual(len(backend.released), 1)
                constructor.side_effect = FileNotFoundError(
                    "Missing Java/WebShop dependencies"
                )
                with self.assertRaisesRegex(FileNotFoundError, "Missing Java"):
                    await run(args)
            failure = json.loads((root / "output/check-eval-report.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIn("Missing Java", failure["error"])


if __name__ == "__main__":
    unittest.main()
