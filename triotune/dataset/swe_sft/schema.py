"""Configuration, intermediate conversations, and prepared SWE dataset results.

These structures describe message-selection policies and expose prepared SFT
rows. They perform no tokenizer calls, batch scheduling, or model execution.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from ..examples import SupervisedExample


# ------------------------------------------------------------------
# Preparation policy and validation
# ------------------------------------------------------------------


@dataclass(slots=True)
class SWESFTConfig:
    """Control how agent trajectories become PyTRIO SFT examples.

    ``pair`` mode creates one progressive example for each assistant turn and
    supervises only that final turn. ``trajectory`` mode keeps a complete
    conversation and supervises every accepted assistant turn. Tool results
    never receive loss because the tokenizer's assistant mask excludes them.

    Attributes:
        sample_mode: 'pair' selects each assistant turn with its context;
            'trajectory' keeps one full conversation with accepted turns labeled.
        pre_split: Treat a record as an already selected pair and supervise only
            its last assistant turn; valid only with sample_mode='pair'.
        max_length: Drop overlong candidates completely instead of truncating tools.
        filter_error_segments: Reject assistant segments followed by is_error=True.
        filter_empty_tool_calls: Reject tool calls with no content or reasoning.
        filter_bare_tool_calls: Reject tool calls with nonempty content but no
            nonempty inline thinking block or separate reasoning_content.
        strip_context_thinking: Remove reasoning from earlier assistant turns.
        strip_target_thinking: Remove reasoning from supervised assistant turns.
        include_tools: Pass conversation tool definitions into the chat template.
        parse_tool_call_arguments: Decode JSON-string arguments for templates
            expecting objects. Malformed JSON raises instead of being discarded.
        chat_template_kwargs: Extra template options, excluding reserved controls.

    Filtering skips an entire candidate in pair mode. In trajectory mode it
    masks a rejected assistant turn while preserving the conversation context.
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
        """Validate policy combinations, token limits, and reserved template keys."""

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
            # Preparation owns these controls to keep tools and masks aligned.
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


# ------------------------------------------------------------------
# Message-level inputs and preparation diagnostics
# ------------------------------------------------------------------


@dataclass(slots=True)
class SWEConversation:
    """An extracted message list and its optional conversation-scoped tools.

    Message content is copied from source records; reasoning normalization and
    segment filtering happen later in trajectory.py before tokenization.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class SWESFTReport:
    """Counts explaining how source records became supervised examples.

    records/conversations count input units. candidates counts selected training
    units before token-length filtering; examples counts retained final samples.
    The three *_segments counters record rejection reasons at assistant-turn
    level. In trajectory mode such a rejection masks a turn rather than removing
    the whole sample. filtered_* counters describe discarded tokenized samples.
    """

    records: int = 0
    conversations: int = 0
    candidates: int = 0
    examples: int = 0
    error_segments: int = 0
    empty_tool_call_segments: int = 0
    bare_tool_call_segments: int = 0
    filtered_overlong: int = 0
    filtered_without_labels: int = 0


# ------------------------------------------------------------------
# Prepared dataset adapter for SFTTrainer and ordinary row consumers
# ------------------------------------------------------------------


@dataclass(slots=True)
class SWESFTDataset(Sequence[Mapping[str, list[int]]]):
    """In-memory examples exposed as reusable pretokenized dataset rows.

    Internal SupervisedExample objects retain validated input_ids/labels.
    Indexing and iteration return fresh dictionaries with input_ids, labels,
    and loss_mask, so SFTTrainer can consume them through its normal row path.
    report exposes preprocessing diagnostics alongside the retained examples.
    """

    examples: tuple[SupervisedExample, ...]
    report: SWESFTReport

    def __len__(self) -> int:
        """Return the number of retained examples, not source trajectories."""

        return len(self.examples)

    @overload
    def __getitem__(self, index: int) -> Mapping[str, list[int]]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Mapping[str, list[int]]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, list[int]] | list[Mapping[str, list[int]]]:
        """Serialize one example or a slice; returned lists are independent copies."""

        if isinstance(index, slice):
            return [example.as_dict() for example in self.examples[index]]
        return self.examples[index].as_dict()

    def __iter__(self) -> Iterator[Mapping[str, list[int]]]:
        """Start a fresh traversal and serialize rows without consuming stored data."""

        for example in self.examples:
            yield example.as_dict()
