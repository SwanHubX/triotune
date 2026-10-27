"""Discover native tool grammars without coupling them to model catalog IDs.

Adapters describe a wire grammar, schema checks, and exact-token continuation.
They can be registered by an application or provided by an installed package
through the ``triotune.agentrl.tool_formats`` entry-point group. Discovery never
downloads code, loads model weights, or queries a remote catalog.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from threading import RLock
from typing import Any

from ...runtime.conversation import NativeToolContinuationRenderer
from ...types import AgentContractError
from .native import ModelOutputParser, Qwen35ToolParser


@dataclass(frozen=True, slots=True)
class NativeToolFormat:
    """A model-independent tool grammar and its tokenizer integration.

    ``template_matcher`` must recognize the actual tokenizer template, not a
    model name. The parser must return ModelOutput retaining the Generation
    object. Renderers implement render and render_after_tool with the same
    exact-token contract as NativeToolContinuationRenderer. All factories and
    validators are synchronous, local operations.
    """

    name: str
    template_matcher: Callable[[str], bool]
    parser_factory: Callable[[Any], ModelOutputParser]
    schema_validator: Callable[[dict[str, Any]], None]
    renderer_factory: Callable[..., Any] = NativeToolContinuationRenderer

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or self.name == "auto":
            raise AgentContractError("Tool format requires a nonempty name other than auto.")
        for name in ("template_matcher", "parser_factory", "schema_validator", "renderer_factory"):
            if not callable(getattr(self, name)):
                raise AgentContractError(f"Tool format {name} must be callable.")


def tokenizer_chat_template(tokenizer: Any) -> str | None:
    """Read the tool-use template, including Transformers template dictionaries."""
    getter = getattr(tokenizer, "get_chat_template", None)
    if callable(getter):
        # Transformers selects tool_use over default only when tools are set.
        template = getter(tools=[{
            "type": "function",
            "function": {
                "name": "triotune_format_probe",
                "parameters": {"type": "object", "properties": {}},
            },
        }])
    else:
        template = getattr(tokenizer, "chat_template", None)
        if isinstance(template, Mapping):
            template = template.get("tool_use", template.get("default"))
    if template is not None and not isinstance(template, str):
        raise AgentContractError("Tokenizer tool chat template must be a string.")
    return template


def _qwen_xml_template(template: str) -> bool:
    return all(marker in template for marker in ("<tool_call>", "<function=", "<parameter="))


_FORMATS = {
    "qwen3.5": NativeToolFormat(
        name="qwen3.5",
        template_matcher=_qwen_xml_template,
        parser_factory=Qwen35ToolParser,
        schema_validator=Qwen35ToolParser.check_schema,
    ),
}
_LOCK = RLock()


def register_tool_format(tool_format: NativeToolFormat) -> None:
    """Register an application adapter once, before constructing backends.

    Existing registrations cannot be silently replaced. Backends retain the
    immutable adapter selected at construction, independently of later registry
    additions. Per-candidate parser/renderer instances are never shared.
    """
    if not isinstance(tool_format, NativeToolFormat):
        raise AgentContractError("Expected NativeToolFormat.")
    with _LOCK:
        if tool_format.name in _FORMATS:
            raise AgentContractError(f"Tool format already registered: {tool_format.name}.")
        _FORMATS[tool_format.name] = tool_format


@lru_cache(maxsize=1)
def _plugin_formats() -> tuple[NativeToolFormat, ...]:
    """Load installed adapter objects lazily, once per process."""
    formats = []
    for point in sorted(entry_points(group="triotune.agentrl.tool_formats"), key=lambda p: p.name):
        adapter = point.load()
        if not isinstance(adapter, NativeToolFormat):
            raise AgentContractError(
                f"Tool-format entry point {point.name} must export NativeToolFormat."
            )
        formats.append(adapter)
    return tuple(formats)


def _available_formats() -> dict[str, NativeToolFormat]:
    with _LOCK:
        plugins = _plugin_formats()
        formats = dict(_FORMATS)
    for adapter in plugins:
        if adapter.name in formats:
            raise AgentContractError(f"Duplicate tool format: {adapter.name}.")
        formats[adapter.name] = adapter
    return formats


def resolve_tool_format(
    tokenizer: Any, model_format: str | None = None,
) -> tuple[NativeToolFormat | None, str, str | None]:
    """Resolve explicit selection, tokenizer metadata, then template matching.

    Unknown templates remain text-only. Unknown explicit/metadata names and
    ambiguous template matches fail closed. A model family or architecture name
    is not evidence of a tool grammar; no model-ID heuristic is used here.
    """
    if model_format is not None and (not isinstance(model_format, str) or not model_format):
        raise AgentContractError("model_format must be a nonempty string or None.")
    source = "explicit"
    selected = model_format if model_format != "auto" else None
    if selected is None:
        source = "metadata"
        attributes = getattr(tokenizer, "model_format", None)
        settings = getattr(tokenizer, "init_kwargs", {})
        metadata = settings.get("model_format") if isinstance(settings, Mapping) else None
        values = [value for value in (attributes, metadata) if value is not None]
        if any(not isinstance(value, str) or not value or value == "auto" for value in values):
            raise AgentContractError("Tokenizer model_format metadata must name a tool format.")
        if len(set(values)) > 1:
            raise AgentContractError("Conflicting tokenizer model_format metadata.")
        selected = values[0] if values else None
    template = tokenizer_chat_template(tokenizer)
    formats = _available_formats()

    def matches(adapter):
        if not template:
            return False
        result = adapter.template_matcher(template)
        if type(result) is not bool:
            raise AgentContractError("Tool-format template_matcher must return bool.")
        return result

    if selected is not None:
        if selected not in formats:
            raise AgentContractError(
                f"Unknown tool format {selected!r}; register an adapter or install its plugin."
            )
        adapter = formats[selected]
        if not matches(adapter):
            raise AgentContractError(
                f"Tool format {selected!r} does not match the tokenizer tool template."
            )
        return adapter, source, template
    candidates = [adapter for adapter in formats.values() if matches(adapter)]
    if len(candidates) > 1:
        names = ", ".join(sorted(adapter.name for adapter in candidates))
        raise AgentContractError(
            f"Ambiguous tokenizer tool format ({names}); set model_format explicitly."
        )
    return (candidates[0], "template", template) if candidates else (None, "none", template)
