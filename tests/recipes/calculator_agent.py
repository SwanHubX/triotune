"""Calculator tools, outcome scoring, and workflow used by agent RL tests."""

from __future__ import annotations

import math

from triotune.agent import (
    AgentRolloutWorkflow,
    EpisodeLimits,
    EpisodeRunner,
    EpisodeScore,
    FunctionTool,
    FunctionToolExecutor,
    JsonActionParser,
    JsonObservationFormatter,
    ToolActionError,
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
        return EpisodeScore(
            float(success),
            success,
            {
                "correctness": float(correct),
                "used_tool": float(used_tool),
            },
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


def calculator_workflow(*, logger=None, limits=None, renderer=None):
    return AgentRolloutWorkflow(
        EpisodeRunner(
            parser=JsonActionParser(),
            tools=calculator_tools(),
            scorer=CalculatorScorer(),
            observations=JsonObservationFormatter(),
            renderer=renderer,
            limits=limits or EpisodeLimits(),
        ),
        logger=logger,
    )
