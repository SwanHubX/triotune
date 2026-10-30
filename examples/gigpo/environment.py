"""环境层：复用外部 WebShop 模拟器，隔离候选会话，保留真实页面与奖励。

没有重写商品搜索、目标生成、页面跳转或奖励公式。一个 SimServer 共享只读商品
与 Lucene 索引，每个 rollout 有自己的 browser/session。环境操作在单个专用线程
顺序执行，模型采样仍可并发；旧版模拟器的全局对象不需要承担线程安全假设。
"""

from __future__ import annotations

import asyncio
import copy
import functools
import importlib
import json
import math
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triotune.agentrl import (
    ActionParseError,
    AgentContractError,
    EpisodeScore,
    FunctionTool,
    FunctionToolExecutor,
    JsonActionParser,
    ToolActionError,
    current_execution_context,
)

from .data_prepare import WEBSHOP_REVISION, asset_manifest, link_assets


class WebShopBackend:
    """直接加载 Princeton 原始 WebShop 的 WebAgentTextEnv（无需启动 HTTP/Chrome）。"""

    def __init__(self, source_dir: Path, files: dict[str, Path], seed: int):
        """用 source_dir 源码、files 缓存映射和 seed 构造共享环境及有序 goals。"""
        source_dir = source_dir.resolve()
        if not (source_dir / "web_agent_site/envs/web_agent_text_env.py").is_file():
            raise FileNotFoundError(
                f"WebShop source missing: {source_dir}; see examples/gigpo/README.md."
            )
        revision = subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != WEBSHOP_REVISION:
            raise ValueError(
                f"Expected WebShop source revision {WEBSHOP_REVISION}; got {revision}."
            )
        if not (source_dir / "search_engine/indexes_1k").is_dir():
            raise FileNotFoundError(
                "Missing indexes_1k; run python -m examples.gigpo.data_prepare first."
            )
        manifest = source_dir / ".triotune-webshop.json"
        if not manifest.is_file() or json.loads(manifest.read_text()) != asset_manifest(
            files
        ):
            raise ValueError(
                "Prepared WebShop assets do not match the HF files; run prepare in a clean checkout."
            )
        # 原始 loader 的属性路径固定为源码 data/；验证链接内容与缓存资产一致。
        link_assets(source_dir, files)
        # 仅在运行真实环境时导入 WebShop 依赖；离线测试不需要 Java/spaCy。
        sys.path.insert(0, str(source_dir))
        module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
        if not Path(module.__file__).resolve().is_relative_to(source_dir.resolve()):
            raise RuntimeError(
                "Another WebShop checkout is already imported in this process."
            )

        # 原始 WebShop 用 Python random 生成商品价格与目标价格约束，并在
        # SimServer 内固定用 233 打乱 goals。seed 控制此前的价格/目标生成，
        # 保留上游的打乱规则；构造后恢复调用方 RNG，避免影响策略采样。
        random_state = random.getstate()
        try:
            random.seed(seed)
            self.template = module.WebAgentTextEnv(
                observation_mode="text",
                file_path=str(files["items_shuffle_1000.json"]),
                num_products=1000,
                human_goals=False,
                get_image=0,
                num_prev_obs=0,
                num_prev_actions=0,
            )
        finally:
            random.setstate(random_state)
        self.goals = self.template.server.goals
        # 构造时产生的初始会话不会被任何候选使用。
        self.template.server.user_sessions.clear()

    def create(self, goal_index: int, rollout_id: str):
        """共享重型只读资源，但不共享浏览器、历史列表或可变购物会话。

        浅拷贝仅复用模板配置和 SimServer；新的 browser 及 reset 重建的历史列表
        各自独立。session_prefix 使用可信 rollout_id，模型无法指定会话 ID。
        reset 的整数参数仍保留目标索引语义，前缀只负责隔离同目标的多个候选。
        """
        env = copy.copy(self.template)
        env.browser = type(self.template.browser)(self.template.server)
        env.session_prefix = f"{rollout_id}:"
        observation, _ = env.reset(session=goal_index)
        return env, observation

    def release(self, env):
        """删除 env 对应的可变购物会话并关闭浏览器；共享商品/索引继续保留。"""
        self.template.server.user_sessions.pop(env.session, None)
        env.close()

    def close(self):
        """关闭模板环境和共享搜索器；不删除数据文件或索引。"""
        self.template.close()
        searcher = self.template.server.search_engine
        close = getattr(searcher, "close", None)
        if close is not None:
            close()


@dataclass
class ShoppingSession:
    """单条候选的 env、当前 observation、终止标记和原始终局 task_score。"""

    env: Any
    observation: str
    done: bool = False
    task_score: float = 0.0

    def visible_state(self) -> dict:
        """只暴露页面及可用动作，不返回 goal 的 asin/options 等隐藏答案。"""
        return {
            "observation": self.observation,
            "available_actions": {} if self.done else self.env.get_available_actions(),
            "done": self.done,
        }


class SessionPool:
    """执行上下文 → 独立会话；取消/超时后也按提交顺序完成清理。

    run_in_executor 的取消不能杀掉正在运行的 Java/Python 调用。专用单线程确保
    release 排在先前动作之后执行，绝不会把一个迟到结果写入另一条 episode。
    字典只由专用线程访问，避免主事件循环和工作线程同时修改环境。
    """

    def __init__(self, backend):
        """绑定共享 backend；sessions 仅由专用单线程读写，不连接训练服务。"""
        self.backend = backend
        self.sessions: dict[str, ShoppingSession] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webshop")
        self._closed = False

    async def _call(self, function, *args):
        """把同步 function(*args) 放入环境队列，异步返回其结果或原始异常。"""
        if self._closed:
            raise RuntimeError("WebShop session pool is closed.")
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, functools.partial(function, *args)
        )

    async def open(self, rollout_id: str, goal_index: int):
        """为可信 rollout_id 建立 goal_index 的独立会话，返回可见初始状态。"""

        def create():
            """在环境线程校验索引和唯一性，再登记会话；禁止重复覆盖。"""
            if rollout_id in self.sessions:
                raise AgentContractError(
                    "A rollout must own exactly one WebShop session."
                )
            if type(goal_index) is not int or not 0 <= goal_index < len(
                self.backend.goals
            ):
                raise AgentContractError("Invalid trusted goal index.")
            env, observation = self.backend.create(goal_index, rollout_id)
            session = ShoppingSession(env, observation)
            self.sessions[rollout_id] = session
            return session.visible_state()

        return await self._call(create)

    async def observe(self, rollout_id: str):
        """返回 rollout_id 的可见状态深拷贝，后续动作不能修改已经锁定的锚点。"""
        return await self._call(
            lambda: copy.deepcopy(self.sessions[rollout_id].visible_state())
        )

    async def step(self, rollout_id: str, action: str):
        """校验并执行 rollout_id 的原生 action，返回下一页面；奖励留在可信会话中。"""

        def transition():
            """只允许可见动作，更新终局分数；非法动作与模拟器故障分别上报。"""
            session = self.sessions[rollout_id]
            if session.done:
                raise ToolActionError(
                    'Shopping has finished; reply with {"final":"done"}.'
                )
            # 先验证再执行；非法模型动作返回可恢复错误，模拟器/索引错误继续上抛。
            match = (
                re.fullmatch(r"(search|click)\[([^\[\]\n]+)\]", action)
                if isinstance(action, str)
                else None
            )
            if match is None or len(action) > 512:
                raise ToolActionError("Use search[keywords] or click[visible button].")
            kind, argument = match.groups()
            available = session.env.get_available_actions()
            if kind == "search" and (
                not argument.strip() or not available["has_search_bar"]
            ):
                raise ToolActionError(
                    "Search requires a visible search bar and nonempty keywords."
                )
            if kind == "search" and argument.split()[0].startswith("<"):
                # 上游存在 <a>/<q>/<r> 等调试检索入口；不开放给购物策略。
                raise ToolActionError("Use ordinary shopping keywords.")
            if kind == "click" and argument.lower() not in available["clickables"]:
                raise ToolActionError(
                    "Click must match one of available_actions.clickables."
                )
            observation, reward, done, _ = session.env.step(action)
            if not math.isfinite(float(reward)) or not 0 <= reward <= 1:
                raise AgentContractError("WebShop task_score must be in [0,1].")
            session.observation, session.done = observation, bool(done)
            session.task_score = float(reward) if done else 0.0
            return session.visible_state()

        return await self._call(transition)

    async def outcome(self, rollout_id: str):
        """返回 rollout_id 的 (done, task_score)，供评分器读取，绝不读取模型声明。"""
        return await self._call(
            lambda: (
                self.sessions[rollout_id].done,
                self.sessions[rollout_id].task_score,
            )
        )

    async def release(self, rollout_id: str):
        """幂等释放 rollout_id，会排在此前动作之后执行，避免超时动作串扰。"""

        def cleanup():
            """先移出会话表，再交给 backend 清理浏览器。"""
            session = self.sessions.pop(rollout_id, None)
            if session is not None:
                self.backend.release(session.env)

        await self._call(cleanup)

    async def close(self):
        """清空所有候选、关闭共享资源并等待执行线程退出；重复调用无副作用。"""
        if self._closed:
            return

        def cleanup():
            """在单线程中完成最终资源回收，避免与尚未完成的 step 并发。"""
            for session in self.sessions.values():
                self.backend.release(session.env)
            self.sessions.clear()
            self.backend.close()

        try:
            await self._call(cleanup)
        finally:
            self._closed = True
            await asyncio.to_thread(self._executor.shutdown, wait=True)


@dataclass(frozen=True)
class RewardConfig:
    """终局评分：mode 选稀疏成功或部分匹配，success_reward 为稀疏成功分值。"""

    mode: str = "sparse"
    success_reward: float = 10.0

    def __post_init__(self):
        """拒绝未知评分模式和非有限/非正奖励，使 JSON 报告与训练目标有效。"""
        if self.mode not in {"sparse", "dense"}:
            raise ValueError("reward.mode must be sparse or dense.")
        if (
            isinstance(self.success_reward, bool)
            or not isinstance(self.success_reward, (int, float))
            or not math.isfinite(self.success_reward)
            or self.success_reward <= 0
        ):
            raise ValueError("reward.success_reward must be finite and positive.")


INSTRUCTIONS = """You are shopping in WebShop. Follow the shopping instruction on the page.
Reply with JSON only. Perform exactly one environment action per response:
{"tool_calls":[{"name":"webshop","arguments":{"action":"search[keywords]"}}]}
or {"tool_calls":[{"name":"webshop","arguments":{"action":"click[visible button]"}}]}.
Use the page and available_actions to choose searches, products, and options.
Click Buy Now to make the purchase. Once done=true, reply {"final":"done"}.
Never invent a session ID, reward, or product. A final reply alone does not buy anything.
"""

def rollout_id() -> str:
    """会话标识仅来自框架的可信 context，不接受模型工具参数指定。"""
    context = current_execution_context()
    if context is None:
        raise AgentContractError("WebShop requires a framework execution context.")
    return context.rollout_id


class SingleActionParser(JsonActionParser):
    """拒绝一次生成多个工具操作，保证一个 GiGPO decision 对应一个环境动作。"""

    def parse(self, generation):
        """将原始 generation 解析为一个工具动作或 final；多个动作直接拒绝。"""
        action = super().parse(generation)
        if len(action.tool_calls) > 1:
            raise ActionParseError("Perform exactly one WebShop action per response.")
        return action


class WebShopAnchors:
    """通过标准 runner 的动作前观察钩子，锁定 GiGPO 比较所需的环境状态。"""

    def __init__(self, pool: SessionPool):
        """绑定可信会话池 pool；不持有跨策略更新的锚点缓存。"""
        self.pool = pool

    async def observe(self, episode, interaction):
        """在动作采样前读取原始页面；排除候选 ID，使相同页面可跨候选分组。

        使用结构化 observation/available_actions/done，避免把完整对话历史
        当成状态。episode/interaction 是 runner 当前执行记录，身份从可信 context
        获取。返回页面快照后，runner 再按 interaction_id 深拷贝保存，最后才采样
        动作并执行工具。真实上下文仍完整提供给模型；anchor 仅服务 GiGPO 分组。
        """
        return await self.pool.observe(rollout_id())


class WebShopScorer:
    """可信终局评分及可选过程奖励注入；成功率始终由原始环境结果定义。"""

    def __init__(self, pool: SessionPool, config: RewardConfig, process_reward_fn=None):
        """pool 提供结果，config 定义终局分值，process_reward_fn 是可选同步研究钩子。"""
        self.pool, self.config = pool, config
        self.process_reward_fn = process_reward_fn

    async def score(self, row, episode):
        """只信任模拟器结果；模型声称“成功”不会产生奖励。

        购买后的结束回复是 JSON 协议所需的额外回合。若预算耗尽前未结束，
        保留真实购物分数作诊断，但 success=False，不伪报完整 episode 成功。

        row 是可信任务行，episode 含原始回合及工具记录，返回 EpisodeScore。可选
        process_reward_fn(row, episode) 返回 {从零开始的回合索引: token 奖励元组}；
        每个元组必须与该回合原始输出 token 等长。框架负责长度和有限值校验，
        默认不注入过程奖励；过程奖励不改变 task_score 或 success 的定义。
        """
        done, task_score = await self.pool.outcome(rollout_id())
        success = episode.status == "completed" and done and task_score == 1.0
        reward = (
            (self.config.success_reward if success else 0.0)
            if self.config.mode == "sparse"
            else (task_score if done else 0.0)
        )
        process_rewards = (
            self.process_reward_fn(row, episode) if self.process_reward_fn else {}
        )
        return EpisodeScore(
            reward,
            success,
            {"task_score": task_score, "purchased": float(done)},
            process_rewards=process_rewards,
        )


class ShoppingEpisodeExecutor:
    """实现现有 EpisodeExecutor 契约，给标准 runner 加一层环境生命周期。

    open/reset → 构造初始页面 prompt → 标准 EpisodeRunner.run → finally release。
    即使采样失败、非法 JSON、超时或取消，也释放候选环境，避免长期训练泄漏。
    """

    def __init__(self, runner, pool: SessionPool):
        """包装标准 runner 并绑定 pool；转发采样预算、导出器和资源归属。"""
        self.runner, self.pool = runner, pool
        self.limits = runner.limits
        self.trajectory_builder = runner.trajectory_builder
        self.resource_owner = runner.resource_owner
        self.anchor_observer = runner.anchor_observer

    async def start(self):
        """启动标准 runner 所管理的工具资源。"""
        await self.runner.start()

    async def close(self):
        """关闭标准 runner；共享会话池由 train.run 的外层生命周期关闭。"""
        await self.runner.close()

    async def run(self, row, snapshot, **kwargs):
        """用 row 目标和固定 snapshot 执行购物，转发 context/seed 等参数并返回 EpisodeResult。"""
        # collect() 在调用 executor 前就分配 context；无需依赖尚未安装的 contextvar。
        context = kwargs["context"]
        try:
            initial = await self.pool.open(context.rollout_id, row["goal_index"])
            task = dict(row)
            task["messages"] = [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps(initial, ensure_ascii=False)},
            ]
            return await self.runner.run(task, snapshot, **kwargs)
        finally:
            # 屏蔽外层取消，确保清理任务排到单线程队列中；pool.close 会等待队列排空。
            await asyncio.shield(self.pool.release(context.rollout_id))


def build_workflow(
    config,
    tokenizer,
    pool: SessionPool,
    reward: RewardConfig,
    *,
    process_reward_fn=None,
):
    """返回标准工作流，接入单动作协议、可信评分和动作前状态锁定。

    config 是 AgentConfig，tokenizer 来自固定策略快照，pool 管理独立环境，
    reward 控制终局奖励。process_reward_fn 默认为 None，研究步骤级奖励时可
    注入 WebShopScorer 所述的 token 对齐回调；同一实验组应共用该回调及预算。
    """

    async def webshop(action: str):
        """只接收模型 action，可信 context 决定会话，返回下一可见页面。"""
        return await pool.step(rollout_id(), action)

    tools = FunctionToolExecutor(
        [
            FunctionTool(
                name="webshop",
                description="Execute one search[keywords] or click[visible button].",
                parameters={
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                function=webshop,
                retry_safe=False,
            )
        ]
    )
    workflow = config.build_workflow(
        tokenizer=tokenizer,
        tools=tools,
        scorer=WebShopScorer(pool, config=reward, process_reward_fn=process_reward_fn),
        anchor_observer=WebShopAnchors(pool),
    )
    workflow.runner.parser = SingleActionParser(allow_plain_text=False)
    workflow.runner = ShoppingEpisodeExecutor(workflow.runner, pool)
    return workflow
