"""Calculator application shared by public examples and regression tests."""

from __future__ import annotations

import math
import random

from triotune.agentrl import (
    AgentConfig,
    AgentRolloutWorkflow,
    CompositeScorer,
    CostConfig,
    EpisodeLimits,
    EpisodeRunner,
    EpisodeScore,
    FunctionTool,
    FunctionToolExecutor,
    JsonActionParser,
    JsonObservationFormatter,
    RewardTerm,
    ToolActionError,
    ToolCostScorer,
)

CALCULATOR_INSTRUCTIONS = (
    "Use the calculator tool at least once. Reply with JSON only. "
    'To call tools: {"tool_calls":[{"name":"calculate","arguments":'
    '{"operation":"add","a":2,"b":3}}]}. '
    "Operations: add, subtract, multiply, divide. "
    'After receiving tool_results, answer with {"final":"5"}.'
)


def calculate(operation: str, a: float, b: float) -> float:
    """Perform bounded arithmetic; invalid model arguments are recoverable."""
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or abs(value) > 1e12
        for value in (a, b)
    ):
        raise ToolActionError(
            "Operands must be finite numbers with absolute value <= 1e12."
        )
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ToolActionError("Division by zero.")
        return a / b
    raise ToolActionError("Unknown operation.")


def calculator_tools() -> FunctionToolExecutor:
    return FunctionToolExecutor(
        [
            FunctionTool(
                name="calculate",
                description="Calculate an arithmetic operation on two numbers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                    "additionalProperties": False,
                },
                function=calculate,
                retry_safe=True,
            )
        ]
    )


class CalculatorScorer:
    async def score(self, row, episode) -> EpisodeScore:
        try:
            correct = math.isclose(
                float(episode.final_text),
                float(row["answer"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        except (ValueError, TypeError):
            correct = False
        used_tool = any(
            not result.is_error
            for turn in episode.turns
            for result in turn.tool_results
        )
        success = episode.status == "completed" and correct and used_tool
        actions = [
            turn.action
            for turn in episode.turns
            if turn.action and turn.action.tool_calls
        ]
        expected = row.get("first_call")
        components = {"correctness": float(correct), "used_tool": float(used_tool)}
        if expected is not None:
            components["first_tool_correctness"] = float(
                bool(
                    actions
                    and actions[0].tool_calls[0].name == "calculate"
                    and actions[0].tool_calls[0].arguments == expected
                )
            )
        return EpisodeScore(
            float(success),
            success,
            components,
        )


def calculator_rows():
    return [
        {
            "messages": [
                {"role": "system", "content": CALCULATOR_INSTRUCTIONS},
                {"role": "user", "content": question},
            ],
            "answer": answer,
        }
        for question, answer in [
            ("What is 2 + 3?", "5"),
            ("What is 6 * 7?", "42"),
            ("What is 18 / 3?", "6"),
            ("What is 11 - 4?", "7"),
        ]
    ]


def generated_rows(count: int, *, split: str, seed: int = 42):
    """Generate unique tasks; evaluation uses disjoint, larger operands."""
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("count must be an integer in [1, 10000].")
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval.")
    rng = random.Random(seed)
    low, high = (2, 99) if split == "train" else (100, 299)
    rows, seen = [], set()
    while len(rows) < count:
        a, b = rng.randint(low, high), rng.randint(low, high)
        operation = rng.choice(["add", "subtract", "multiply", "divide", "two_step"])
        if operation == "two_step":
            c = rng.randint(2, 9)
            question, answer = f"What is ({a} + {b}) * {c}?", (a + b) * c
            first = {"operation": "add", "a": a, "b": b}
        else:
            if operation == "divide":
                a *= b
            symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[
                operation
            ]
            question, answer = f"What is {a} {symbol} {b}?", calculate(operation, a, b)
            first = {"operation": operation, "a": a, "b": b}
        if question in seen:
            continue
        seen.add(question)
        rows.append(
            {
                "task_id": f"calculator-{split}-{len(rows)}",
                "messages": [
                    {
                        "role": "system",
                        "content": CALCULATOR_INSTRUCTIONS
                        + " For parentheses, compute the inner expression first, then use its result in another call.",
                    },
                    {"role": "user", "content": question},
                ],
                "answer": str(answer),
                "first_call": first,
            }
        )
    return rows


def calculator_workflow(
    *,
    logger=None,
    limits=None,
    renderer=None,
    config: AgentConfig | None = None,
    tokenizer=None,
    costs: CostConfig | None = None,
):
    scorer = CalculatorScorer()
    if costs is not None:
        scorer = CompositeScorer(scorer, [RewardTerm("cost", ToolCostScorer(costs))])
    if config is not None:
        if limits is not None or renderer is not None or tokenizer is None:
            raise ValueError(
                "Configured workflow needs tokenizer; limits/renderer come from AgentConfig."
            )
        return config.build_workflow(
            tokenizer=tokenizer, tools=calculator_tools(), scorer=scorer, logger=logger
        )
    return AgentRolloutWorkflow(
        EpisodeRunner(
            parser=JsonActionParser(),
            tools=calculator_tools(),
            scorer=scorer,
            observations=JsonObservationFormatter(),
            renderer=renderer,
            limits=limits or EpisodeLimits(),
        ),
        logger=logger,
    )
