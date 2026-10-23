"""Run the bounded model-action-observation loop for one agent episode.

The runner coordinates rendering, sampling, parsing, tools, and terminal
scoring through their contracts. It records what happened before deciding
whether the outcome is a model failure or an infrastructure failure. Group
collection, persistent logging, trajectory export, and optimizer updates remain
the responsibility of their respective orchestration layers.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from .contracts import (
    SYSTEM_STATUSES,
    ActionParseError,
    ActionParser,
    AgentContractError,
    AgentTurn,
    ConversationRenderer,
    EpisodeLimits,
    EpisodeResult,
    EpisodeScorer,
    ObservationFormatter,
    ToolExecutor,
    ToolSystemError,
    TrajectoryExporter,
    validate_tokens,
)
from .sampling import SamplingSnapshot
from .trajectory_builder import EpisodeTrajectoryBuilder


class EpisodeRunner:
    """Own the model/tool loop and terminal status, but not optimizer state.

    Scorers and tools may run concurrently for different episodes and must not
    keep unkeyed per-episode mutable state. asyncio cancellation propagates.

    A runner can be reused for training and evaluation. The ``is_eval`` flag
    labels the result; it does not switch to a separate agent implementation.
    """

    def __init__(
        self,
        *,
        parser: ActionParser,
        tools: ToolExecutor,
        scorer: EpisodeScorer,
        observations: ObservationFormatter,
        renderer: ConversationRenderer | None = None,
        limits: EpisodeLimits | None = None,
        prompt_key: str = "messages",
        trajectory_builder: TrajectoryExporter | None = None,
    ):
        """Compose execution adapters and optional model-specific serialization.

        ``prompt_key`` selects the row field containing text or chat messages.
        If no renderer is provided, each run uses the snapshot's tokenizer.
        The exporter validates continuity during execution and is later reused
        by the collector to produce PPO trajectories.
        """
        self.parser, self.tools, self.scorer = parser, tools, scorer
        self.observations, self.renderer = observations, renderer
        self.limits, self.prompt_key = limits or EpisodeLimits(), prompt_key
        self.trajectory_builder = trajectory_builder or EpisodeTrajectoryBuilder()

    async def run(
        self,
        row: Mapping[str, Any],
        snapshot: SamplingSnapshot,
        *,
        group_id: str,
        seed: int | None = None,
        is_eval: bool = False,
    ) -> EpisodeResult:
        """Execute and score one task while preserving an inspectable record.

        Args:
            row: Task input plus optional ground truth used only by the scorer.
            snapshot: Fixed policy and tokenizer for every generated turn.
            group_id: Collector-owned identity shared by attempts of one task.
            seed: First turn's seed; following turns increment this value.
            is_eval: Provenance flag stored in the returned episode.

        Ordinary model and system errors become explicit terminal statuses.
        System failures do not receive a task score. Cancellation is allowed to
        escape so callers can stop collection without creating failed samples.
        """
        from .conversation import ChatTemplateRenderer

        episode = EpisodeResult(uuid4().hex, group_id, snapshot.snapshot_id, is_eval)
        try:
            async with asyncio.timeout(self.limits.episode_timeout):
                # Each attempt gets its own conversation. Tool observations
                # must never leak into another sample or mutate dataset rows.
                prompt = copy.deepcopy(row[self.prompt_key])
                messages = (
                    [{"role": "user", "content": prompt}]
                    if isinstance(prompt, str)
                    else list(prompt)
                )
                if not messages or any(
                    not isinstance(m, Mapping) or "role" not in m for m in messages
                ):
                    raise AgentContractError(
                        "Agent input must be text or nonempty chat messages."
                    )
                renderer = self.renderer or ChatTemplateRenderer(snapshot.tokenizer)
                await self._execute(episode, messages, snapshot, renderer, seed)
                if episode.status not in SYSTEM_STATUSES:
                    # Invalid actions and budget exits can still carry useful
                    # negative task signals. Infrastructure failures cannot.
                    try:
                        episode.score = await self.scorer.score(row, episode)
                        # Validate the scorer boundary even if it returns a custom object.
                        episode.score.__post_init__()
                        if episode.status != "completed" and episode.score.success:
                            raise AgentContractError(
                                "An unfinished episode cannot be scored successful."
                            )
                    except Exception as exc:  # noqa: BLE001 -- isolate arbitrary scorer implementations
                        episode.status, episode.error = "scoring_failure", str(exc)
                        episode.score = None
        except TimeoutError:
            episode.status, episode.error = (
                "timeout",
                "Episode, generation or tool deadline exceeded.",
            )
            episode.score = None
        except AgentContractError as exc:
            episode.status, episode.error = "contract_error", str(exc)
        except ToolSystemError as exc:
            episode.status, episode.error = "tool_failure", str(exc)
        except Exception as exc:  # noqa: BLE001 -- preserve failed adapter episodes for audit
            episode.status, episode.error = (
                "contract_error",
                f"{type(exc).__name__}: {exc}",
            )
        return episode

    async def _execute(self, episode, messages, snapshot, renderer, seed) -> None:
        """Advance the conversation until an answer, budget exit, or failure.

        This helper records generations and tool outcomes but leaves final task
        scoring to run(). Every generation is appended before parsing, so even
        malformed actions retain the tokens needed for a valid failure sample.
        """
        limits = self.limits
        generated_tokens = 0
        for index in range(limits.max_turns):
            # Tokenization and user renderers can be synchronous CPU work.
            prompt_tokens = await asyncio.to_thread(
                renderer.render, messages, self.tools.schemas
            )
            validate_tokens(prompt_tokens, name="rendered prompt")
            self.trajectory_builder.validate_prefix(episode, prompt_tokens)
            # Context capacity includes rendered history and observations,
            # whereas the episode output budget counts model generations only.
            context_budget = limits.max_context_tokens - len(prompt_tokens)
            if context_budget <= 0:
                episode.status = "context_limit"
                return
            remaining = limits.max_generated_tokens - generated_tokens
            if remaining <= 0:
                episode.status = "token_limit"
                return
            budget = min(limits.max_tokens_per_turn, remaining, context_budget)
            try:
                generation = await asyncio.wait_for(
                    snapshot.sampler.generate(
                        prompt_tokens,
                        max_tokens=budget,
                        seed=None if seed is None else seed + index,
                    ),
                    timeout=limits.sampling_timeout,
                )
            except (AgentContractError, TimeoutError):
                raise
            except Exception as exc:  # noqa: BLE001 -- classify errors from arbitrary sampling clients
                episode.status, episode.error = "sampling_failure", str(exc)
                return
            generation.__post_init__()
            if len(generation.tokens) > budget:
                raise AgentContractError(
                    "Sampler exceeded the requested output budget."
                )
            turn = AgentTurn(tuple(prompt_tokens), generation)
            episode.turns.append(turn)
            generated_tokens += len(generation.tokens)
            if generation.stop_reason in {"length", "max_tokens", "max_length"}:
                # An unfinished tool action must not be executed. Its original
                # output remains recorded for truncated-trajectory handling.
                episode.status = "token_limit"
                return
            try:
                turn.action = self.parser.parse(generation)
                turn.action.__post_init__()
            except ActionParseError as exc:
                episode.status, episode.error = "invalid_action", str(exc)
                return
            messages.append(turn.action.assistant_message)
            if turn.action.final_text is not None:
                episode.status, episode.final_text = "completed", turn.action.final_text
                return
            for call in turn.action.tool_calls:
                # Preserve call order within a turn. Cross-episode concurrency
                # is managed by the collector, not by parallelizing tool side
                # effects inside a single agent decision.
                result = await asyncio.wait_for(
                    self.tools.execute(call), limits.tool_timeout
                )
                result.__post_init__()
                if result.call_id != call.call_id or result.name != call.name:
                    raise AgentContractError(
                        "Tool result does not match its requested call."
                    )
                turn.tool_results.append(result)
            messages.extend(self.observations.format(turn.tool_results))
        # Exhaustion after tool feedback is distinct from receiving a final
        # answer, even if a tool happened to produce the correct task result.
        episode.status = "max_turns"
