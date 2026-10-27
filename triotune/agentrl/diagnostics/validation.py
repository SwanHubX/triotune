"""Exercise tokenizer boundaries and trajectory/loss alignment without sampling."""

from __future__ import annotations

from ..types import AgentTurn, EpisodeResult, EpisodeScore, Generation, ToolResult
from ..rollout.trajectory import EpisodeTrajectoryBuilder
from ..runtime.conversation import (
    ChatTemplateRenderer,
    JsonActionParser,
    JsonObservationFormatter,
    TokenAppendRenderer,
)


def probe_tokenizer(
    tokenizer, *, template_kwargs=None, renderer="token_append"
) -> dict:
    """Verify UTF-8, whitespace, two turns, returned/omitted EOS and masks."""
    kwargs = {"enable_thinking": False} if template_kwargs is None else template_kwargs
    renderers = {
        "token_append": TokenAppendRenderer,
        "chat_template": ChatTemplateRenderer,
    }
    if renderer not in renderers:
        raise ValueError("Unsupported probe renderer.")
    renderer = renderers[renderer](tokenizer, advertise_tools=False, **kwargs)
    parser, builder = (
        JsonActionParser(allow_plain_text=False),
        EpisodeTrajectoryBuilder(),
    )
    messages = [
        {"role": "system", "content": "Reply with JSON only."},
        {"role": "user", "content": "计算 2 + 3"},
    ]
    prompt = renderer.render(messages, [])
    text = ' {"tool_calls":[{"name":"calculate","arguments":{"operation":"add","a":2,"b":3}}]} '
    cases = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int:
        raise ValueError("Tokenizer probe requires a scalar eos_token_id.")
    for returned_stop in (False, True):
        ids = list(tokenizer.encode(text, add_special_tokens=False)) + (
            [eos] if returned_stop else []
        )
        generation = Generation(tuple(ids), tuple([-1.0] * len(ids)), text)
        action = parser.parse(generation)
        observation = ToolResult("0", "calculate", "5")
        episode = EpisodeResult(
            "probe",
            "probe",
            "tokenizer",
            turns=[AgentTurn(tuple(prompt), generation, action, [observation])],
        )
        following = (
            messages
            + [action.assistant_message]
            + JsonObservationFormatter().format([observation])
        )
        continuation = (
            renderer.render_continuation(episode, following, [])
            if hasattr(renderer, "render_continuation")
            else renderer.render(following, [])
        )
        builder.validate_prefix(episode, continuation)
        final_ids = list(
            tokenizer.encode('{"final":"5"}', add_special_tokens=False)
        ) + [eos]
        episode.turns.append(
            AgentTurn(
                tuple(continuation),
                Generation(
                    tuple(final_ids), tuple([-0.5] * len(final_ids)), '{"final":"5"}'
                ),
            )
        )
        episode.score = EpisodeScore(
            1, True, process_rewards={0: tuple([0.0] * (len(ids) - 1) + [0.25])}
        )
        trajectory = builder.build(episode)
        from ...trainer.ppo.datum import build_datum

        datum = build_datum(trajectory)
        targets = datum.loss_fn_inputs["target_tokens"].data[len(prompt) - 1 :]
        if any(
            (target == -100) == active
            for target, active in zip(targets, trajectory.loss_mask, strict=True)
        ):
            raise ValueError("Causal target mask does not match action mask.")
        if not any(not active for active in trajectory.loss_mask):
            raise ValueError("Probe did not exercise observation masking.")
        cases.append(
            {
                "returned_stop": returned_stop,
                "action_tokens": sum(trajectory.loss_mask),
                "context_tokens": len(trajectory.loss_mask) - sum(trajectory.loss_mask),
                "prefix_preserved": True,
                "causal_mask_aligned": True,
            }
        )
    return {"tokenizer": type(tokenizer).__name__, "eos_token_id": eos, "cases": cases}
