"""WebShop OPD environment, isolated sessions, and action protocol."""

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
    """Load WebAgentTextEnv directly without an HTTP server or browser."""

    def __init__(self, source_dir: Path, files: dict[str, Path], seed: int):
        """Build the shared environment and ordered goals from pinned assets."""
        source_dir = source_dir.resolve()
        if not (source_dir / "web_agent_site/envs/web_agent_text_env.py").is_file():
            raise FileNotFoundError(
                f"WebShop source missing: {source_dir}; see examples/agent/webshop-opd/README.md."
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
                "Missing indexes_1k; run the WebShop OPD data preparation command first."
            )
        manifest = source_dir / ".triotune-webshop.json"
        if not manifest.is_file() or json.loads(manifest.read_text()) != asset_manifest(
            files
        ):
            raise ValueError(
                "Prepared WebShop assets do not match the HF files; run prepare in a clean checkout."
            )
        # Validate cached assets before linking them into the required data path.
        link_assets(source_dir, files)
        # Import simulator dependencies only when constructing a real backend.
        sys.path.insert(0, str(source_dir))
        module = importlib.import_module("web_agent_site.envs.web_agent_text_env")
        if not Path(module.__file__).resolve().is_relative_to(source_dir.resolve()):
            raise RuntimeError(
                "Another WebShop checkout is already imported in this process."
            )

        # Seed price and goal generation without changing the caller RNG state.
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
        # Discard the constructor session because candidates create their own.
        self.template.server.user_sessions.clear()

    def create(self, goal_index: int, rollout_id: str):
        """Create an isolated browser and mutable session for one rollout.

        The shallow copy shares immutable configuration and server resources.
        A new browser and reset history isolate every candidate. The trusted
        rollout identity scopes sessions without exposing session selection to
        model output.
        """
        env = copy.copy(self.template)
        env.browser = type(self.template.browser)(self.template.server)
        env.session_prefix = f"{rollout_id}:"
        observation, _ = env.reset(session=goal_index)
        return env, observation

    def release(self, env):
        """Remove one mutable session and close its browser."""
        self.template.server.user_sessions.pop(env.session, None)
        env.close()

    def close(self):
        """Close the template environment and shared searcher."""
        self.template.close()
        searcher = self.template.server.search_engine
        close = getattr(searcher, "close", None)
        if close is not None:
            close()


@dataclass
class ShoppingSession:
    """Hold one candidate environment, visible page, and terminal score."""

    env: Any
    observation: str
    done: bool = False
    task_score: float = 0.0

    def visible_state(self) -> dict:
        """Expose the page and available actions without hidden goal answers."""
        return {
            "observation": self.observation,
            "available_actions": {} if self.done else self.env.get_available_actions(),
            "done": self.done,
        }


class SessionPool:
    """Map trusted execution identities to isolated serialized sessions.

    Cancellation cannot stop a synchronous environment call already running in
    an executor. A dedicated single thread keeps cleanup behind earlier actions
    and prevents late results from crossing episode boundaries. Only that thread
    accesses the session mapping.
    """

    def __init__(self, backend):
        """Bind a backend and initialize the single-thread session queue."""
        self.backend = backend
        self.sessions: dict[str, ShoppingSession] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webshop")
        self._closed = False

    async def _call(self, function, *args):
        """Run a synchronous environment operation on the ordered queue."""
        if self._closed:
            raise RuntimeError("WebShop session pool is closed.")
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, functools.partial(function, *args)
        )

    async def open(self, rollout_id: str, goal_index: int):
        """Open an isolated goal session for a trusted rollout identity."""

        def create():
            """Validate identity and goal index before registering a session."""
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
        """Return a detached visible-state snapshot for one rollout."""
        return await self._call(
            lambda: copy.deepcopy(self.sessions[rollout_id].visible_state())
        )

    async def step(self, rollout_id: str, action: str):
        """Validate and execute one action while retaining trusted rewards."""

        def transition():
            """Accept visible actions and update the terminal task score."""
            session = self.sessions[rollout_id]
            if session.done:
                raise ToolActionError(
                    'Shopping has finished; reply with {"final":"done"}.'
                )
            # Validate before execution so model errors remain recoverable while
            # environment failures continue to propagate.
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
                # Block debug search commands from the shopping policy.
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
        """Return the trusted terminal flag and task score for one rollout."""
        return await self._call(
            lambda: (
                self.sessions[rollout_id].done,
                self.sessions[rollout_id].task_score,
            )
        )

    async def release(self, rollout_id: str):
        """Release a rollout idempotently after all preceding actions."""

        def cleanup():
            """Remove the session before releasing its browser resources."""
            session = self.sessions.pop(rollout_id, None)
            if session is not None:
                self.backend.release(session.env)

        await self._call(cleanup)

    async def close(self):
        """Close all sessions, shared resources, and the executor idempotently."""
        if self._closed:
            return

        def cleanup():
            """Finalize resources on the environment thread."""
            for session in self.sessions.values():
                self.backend.release(session.env)
            self.sessions.clear()
            self.backend.close()

        try:
            await self._call(cleanup)
        finally:
            self._closed = True
            await asyncio.to_thread(self._executor.shutdown, wait=True)


INSTRUCTIONS = """You are shopping in WebShop. Follow the shopping instruction on the page.
Reply with JSON only. Perform exactly one environment action per response:
{"tool_calls":[{"name":"webshop","arguments":{"action":"search[keywords]"}}]}
or {"tool_calls":[{"name":"webshop","arguments":{"action":"click[visible button]"}}]}.
Use the page and available_actions to choose searches, products, and options.
Click Buy Now to make the purchase. Once done=true, reply {"final":"done"}.
Never invent a session ID, reward, or product. A final reply alone does not buy anything.
"""


class SingleActionParser(JsonActionParser):
    """Require exactly one environment action per WebShop decision."""

    def parse(self, generation):
        """Parse one tool action or final response from a generation."""
        action = super().parse(generation)
        if len(action.tool_calls) > 1:
            raise ActionParseError("Perform exactly one WebShop action per response.")
        return action


def current_rollout_id() -> str:
    """Select a WebShop session from framework-owned execution identity."""

    context = current_execution_context()
    if context is None:
        raise AgentContractError("WebShop requires an AgentRL execution context.")
    return context.rollout_id


class WebShopScorer:
    """Expose environment outcomes for evaluation without shaping OPD loss."""

    def __init__(
        self,
        pool: SessionPool,
        *,
        exact_success_alpha: float = 0.0,
    ):
        """Configure optional training-only emphasis on exact purchases.

        ``exact_success_alpha`` implements the convex mixture

        ``(1 - alpha) * task_score + alpha * I(task_score == 1)``.

        Alpha zero preserves the native WebShop reward exactly.  Increasing it
        discounts partial-credit purchases while keeping both zero reward and
        a perfect score fixed at zero and one.  The mixture is deliberately
        applied only to training episodes: evaluation must continue reporting
        the benchmark's native task score and strict Pass@1 definition.
        """
        if (
            isinstance(exact_success_alpha, bool)
            or not isinstance(exact_success_alpha, (int, float))
            or not math.isfinite(float(exact_success_alpha))
            or not 0.0 <= float(exact_success_alpha) <= 1.0
        ):
            raise ValueError(
                "exact_success_alpha must be a finite number in [0,1]."
            )
        self.pool = pool
        self.exact_success_alpha = float(exact_success_alpha)

    def training_reward(self, task_score: float) -> float:
        """Blend dense task credit with a strict exact-success indicator.

        The simulator already validates ``task_score`` before it reaches this
        scorer.  Keeping this transformation small and deterministic makes the
        raw score auditable through ``components['task_score']`` while GRPO
        receives the explicitly configured optimization target.
        """
        exact_success = float(task_score == 1.0)
        return (
            (1.0 - self.exact_success_alpha) * task_score
            + self.exact_success_alpha * exact_success
        )

    async def score(self, row, episode) -> EpisodeScore:
        """Trust only the simulator's terminal state and product-match score."""

        done, task_score = await self.pool.outcome(current_rollout_id())
        # A training trajectory that purchases on its final available turn has
        # reached the environment goal but has not completed the conversation's
        # required final reply. Keep the partial outcome for diagnostics while
        # reserving ``success`` for a normally completed episode. Evaluation
        # uses the post-tool terminal hook, so a purchase there sets the episode
        # status to completed immediately and remains a valid success.
        success = episode.status == "completed" and done and task_score == 1.0
        # Vanilla OPD ignores this scalar reward.  The unmodified native score
        # remains in components so training diagnostics can compare the shaped
        # objective with the underlying WebShop outcome.
        native_reward = task_score if done else 0.0
        # Reward shaping is an optimization choice, not an evaluation metric.
        # Evaluation therefore retains the original WebShop score, while GRPO
        # training receives the configured mixture.  Strict success remains
        # independent of this scalar and still requires task_score == 1.
        reward = (
            native_reward
            if episode.is_eval
            else self.training_reward(native_reward)
        )
        return EpisodeScore(
            reward,
            success,
            {"task_score": task_score, "purchased": float(done)},
        )


class ShoppingEpisodeExecutor:
    """Open one isolated WebShop session around the generic episode runner."""

    def __init__(self, runner, pool: SessionPool):
        self.runner = runner
        self.pool = pool
        # AgentRolloutWorkflow inspects these stable runner contracts without
        # needing to know that the environment wraps the generic executor.
        self.limits = runner.limits
        self.trajectory_builder = runner.trajectory_builder
        self.resource_owner = runner.resource_owner
        self.anchor_observer = runner.anchor_observer

    async def start(self):
        await self.runner.start()

    async def close(self):
        # The application owns the shared pool because training and evaluation
        # reuse its product catalog.  Closing the workflow only closes AgentRL.
        await self.runner.close()

    async def run(self, row, snapshot, **kwargs):
        context = kwargs["context"]
        try:
            initial = await self.pool.open(context.rollout_id, row["goal_index"])
            task = dict(row)
            task["messages"] = [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": json.dumps(initial, ensure_ascii=False),
                },
            ]
            return await self.runner.run(task, snapshot, **kwargs)
        finally:
            # Environment steps are serialized.  Shielding cleanup ensures a
            # cancelled model request cannot leak its mutable browser session.
            await asyncio.shield(self.pool.release(context.rollout_id))


def build_workflow(
    config,
    tokenizer,
    pool: SessionPool,
    *,
    exact_success_alpha: float = 0.0,
    scorer=None,
):
    """Compose the WebShop protocol with the copied Triotune AgentRL runtime."""

    if scorer is not None and exact_success_alpha != 0.0:
        raise ValueError(
            "Supply either a custom scorer or exact_success_alpha, not both."
        )
    scorer = scorer or WebShopScorer(
        pool, exact_success_alpha=exact_success_alpha
    )

    async def webshop(action: str):
        return await pool.step(current_rollout_id(), action)

    async def stop_completed_evaluation(episode, turn):
        """End evaluation at purchase while retaining training final replies."""

        if not episode.is_eval:
            return False
        done, _ = await pool.outcome(current_rollout_id())
        return done

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
        scorer=scorer,
        post_tool_stop=stop_completed_evaluation,
    )
    workflow.runner.parser = SingleActionParser(allow_plain_text=False)
    workflow.runner = ShoppingEpisodeExecutor(workflow.runner, pool)
    return workflow
