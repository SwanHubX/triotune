"""Derive OpenAI function schemas and strict argument validation.

The decorator is the translation boundary between a Python callable and the
JSON object accepted by an OpenAI function call.  Schema generation and
runtime validation intentionally share the same annotation interpretation so
that a value accepted locally is representable on the wire.  No coercion is
performed: a string that looks like a number and a boolean that looks like an
integer remain invalid for numeric annotations.
"""

from __future__ import annotations

import inspect
import json
import math
import types
from collections.abc import Callable
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from ...runtime.tools import FunctionTool


def _schema(annotation: Any) -> dict[str, Any]:
    """Convert one supported Python annotation into a JSON Schema fragment.

    The conversion is deliberately limited to values that can be carried by
    OpenAI's JSON tool protocol.  ``Annotated`` metadata is transparent,
    unions become ``anyOf`` branches, and optional values are represented by a
    JSON ``null`` branch rather than a provider-specific nullable extension.
    Dictionaries must have string keys because JSON object keys are strings.

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
    """Check a runtime value against an annotation without changing it.

    Validation recurses through lists, dictionaries, unions, literals, and
    ``Annotated`` values.  Exact type checks preserve the distinction between
    ``bool`` and ``int``; floats intentionally accept integer values because
    JSON has one numeric representation while still rejecting non-finite
    values.
    """
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
    """Return a ``FunctionTool`` wrapper for a sync or async Python function.

    The first docstring paragraph describes the function. Defaults remain
    optional parameters and are applied by Python at invocation. Positional
    only and variadic parameters cannot represent JSON keyword arguments. The
    returned wrapper retains the original callable and retry-safety declaration
    while adding an OpenAI-compatible schema and a validator used immediately
    before tool execution.
    """

    def decorate(target: Callable[..., Any]) -> FunctionTool:
        """Inspect one callable once and freeze its public tool contract."""
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
            """Validate model-supplied keyword arguments before invocation.

            ``Signature.bind`` enforces missing and unexpected parameters;
            annotation checks then enforce the same types used to build the
            advertised schema.  Returning a fresh dictionary prevents a tool
            implementation from mutating the caller's decoded request.
            """
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
