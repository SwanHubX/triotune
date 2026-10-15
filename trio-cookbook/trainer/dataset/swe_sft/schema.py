"""Configuration and public results for SWE trajectory preparation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from ..examples import SupervisedExample


@dataclass(slots=True)
class SWESFTConfig:
    """Control how agent trajectories become PyTRIO SFT examples.

    ``pair`` mode creates one progressive example for each assistant turn and
    supervises only that final turn. ``trajectory`` mode keeps a complete
    conversation and supervises every accepted assistant turn. Tool results
    never receive loss because the tokenizer's assistant mask excludes them.
    """

    sample_mode: Literal["pair", "trajectory"] = "pair"
    pre_split: bool = False
    max_length: int | None = None
    filter_error_segments: bool = True
    filter_empty_tool_calls: bool = False
    filter_bare_tool_calls: bool = False
    strip_context_thinking: bool = True
    strip_target_thinking: bool = False
    include_tools: bool = True
    parse_tool_call_arguments: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_mode not in {"pair", "trajectory"}:
            raise ValueError("sample_mode must be 'pair' or 'trajectory'.")
        if self.pre_split and self.sample_mode != "pair":
            raise ValueError("pre_split is only supported with sample_mode='pair'.")
        if self.max_length is not None and (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, int)
            or self.max_length < 2
        ):
            raise ValueError("max_length must be an integer of at least 2 or None.")
        for name in (
            "pre_split",
            "filter_error_segments",
            "filter_empty_tool_calls",
            "filter_bare_tool_calls",
            "strip_context_thinking",
            "strip_target_thinking",
            "include_tools",
            "parse_tool_call_arguments",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean.")
        if not isinstance(self.chat_template_kwargs, dict):
            raise TypeError("chat_template_kwargs must be a dictionary.")
        reserved = {
            "tokenize",
            "add_generation_prompt",
            "return_dict",
            "return_assistant_tokens_mask",
            "tools",
        }.intersection(self.chat_template_kwargs)
        if reserved:
            raise ValueError(
                "chat_template_kwargs must not override "
                + ", ".join(sorted(reserved))
                + "."
            )


@dataclass(slots=True)
class SWEConversation:
    """One normalized agent conversation and its optional tool definitions."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class SWESFTReport:
    """Counts explaining how source trajectories became training samples."""

    records: int = 0
    conversations: int = 0
    candidates: int = 0
    examples: int = 0
    error_segments: int = 0
    empty_tool_call_segments: int = 0
    bare_tool_call_segments: int = 0
    filtered_overlong: int = 0
    filtered_without_labels: int = 0


@dataclass(slots=True)
class SWESFTDataset(Sequence[Mapping[str, list[int]]]):
    """Prepared examples exposed as rows accepted directly by ``SFTTrainer``."""

    examples: tuple[SupervisedExample, ...]
    report: SWESFTReport

    def __len__(self) -> int:
        return len(self.examples)

    @overload
    def __getitem__(self, index: int) -> Mapping[str, list[int]]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Mapping[str, list[int]]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, list[int]] | list[Mapping[str, list[int]]]:
        if isinstance(index, slice):
            return [example.as_dict() for example in self.examples[index]]
        return self.examples[index].as_dict()

    def __iter__(self) -> Iterator[Mapping[str, list[int]]]:
        for example in self.examples:
            yield example.as_dict()
