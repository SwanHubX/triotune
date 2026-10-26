"""Derive tool schemas and argument validation from Python annotations."""

from __future__ import annotations

import inspect
import json
import math
import types
from collections.abc import Callable
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from ...runtime.tools import FunctionTool


def _schema(annotation: Any) -> dict[str, Any]:
    """Support JSON scalars, lists, string-keyed dictionaries and unions.

    Unsupported annotations fail at registration. Optional values use JSON
    Schema null branches rather than the OpenAPI ``nullable`` extension.
    """
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (inspect.Signature.empty, Any):
        return {}
    if origin is Annotated:
        return _schema(args[0])
    if origin in (Union, types.UnionType):
        return {"anyOf": [_schema(item) for item in args]}
    if origin is Literal:
        json.dumps(args, allow_nan=False)
        return {"enum": list(args)}
    if annotation is list or origin is list:
        return {"type": "array", "items": _schema(args[0]) if args else {}}
    if annotation is dict or origin is dict:
        if args and args[0] is not str:
            raise TypeError("Tool dictionaries require string keys.")
        return {
            "type": "object",
            "additionalProperties": _schema(args[1]) if args else {},
        }
    scalars = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    if annotation in scalars:
        return {"type": scalars[annotation]}
    raise TypeError(f"Unsupported tool annotation: {annotation!r}.")


def _matches(value: Any, annotation: Any) -> bool:
    """Validate recursively without coercing strings or booleans to numbers."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation in (inspect.Signature.empty, Any):
        return True
    if origin is Annotated:
        return _matches(value, args[0])
    if origin in (Union, types.UnionType):
        return any(_matches(value, item) for item in args)
    if origin is Literal:
        return any(type(value) is type(item) and value == item for item in args)
    if annotation is list or origin is list:
        return isinstance(value, list) and (
            not args or all(_matches(item, args[0]) for item in value)
        )
    if annotation is dict or origin is dict:
        return isinstance(value, dict) and all(
            isinstance(key, str) and (not args or _matches(item, args[1]))
            for key, item in value.items()
        )
    if annotation is float:
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is annotation


def function_tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    retry_safe: bool = False,
):
    """Register a synchronous or asynchronous function using its type hints.

    The first docstring paragraph describes the function. Defaults remain
    optional parameters and are applied by Python at invocation. Positional
    only and variadic parameters cannot represent JSON keyword arguments.
    """

    def decorate(target: Callable[..., Any]) -> FunctionTool:
        signature = inspect.signature(target)
        hints = get_type_hints(target, include_extras=True)
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        for parameter in signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise TypeError("Tool parameters must accept keyword arguments.")
            annotation = hints.get(parameter.name, parameter.annotation)
            schema = _schema(annotation)
            if parameter.default is inspect.Signature.empty:
                parameters["required"].append(parameter.name)
            else:
                json.dumps(parameter.default, allow_nan=False)
                if not _matches(parameter.default, annotation):
                    raise TypeError(
                        f"Invalid default for tool parameter {parameter.name!r}."
                    )
                schema["default"] = parameter.default
            parameters["properties"][parameter.name] = schema

        def validate(arguments: dict[str, Any]) -> dict[str, Any]:
            bound = signature.bind(**arguments)
            json.dumps(arguments, allow_nan=False)
            for key, value in bound.arguments.items():
                if not _matches(
                    value, hints.get(key, signature.parameters[key].annotation)
                ):
                    raise ValueError(f"Invalid type for argument {key!r}.")
            return dict(arguments)

        doc = inspect.getdoc(target) or ""
        return FunctionTool(
            name=name or target.__name__,
            description=description
            if description is not None
            else " ".join(doc.split("\n\n", 1)[0].splitlines()),
            parameters=parameters,
            function=target,
            retry_safe=retry_safe,
            validator=validate,
        )

    return decorate(function) if function is not None else decorate
