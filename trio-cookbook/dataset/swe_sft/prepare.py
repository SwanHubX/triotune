"""Map selected SWE assistant turns onto supervised token spans.

trajectory.py selects messages and identifies assistant turns to exclude.
This module obtains the template's assistant-token mask, checks turn/span
alignment, applies exclusions, and stores unshifted SupervisedExample objects.
The resulting SWESFTDataset supplies pretokenized rows to SFTTrainer.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from ..examples import IGNORE_INDEX, SupervisedExample
from .schema import SWESFTConfig, SWESFTDataset, SWESFTReport
from .trajectory import (
    _Candidate,
    candidates_from_conversation,
    conversations_from_record,
    parse_tool_call_arguments,
    records_from_source,
)


# ------------------------------------------------------------------
# Tokenizer shape normalization and assistant-span tracking
# ------------------------------------------------------------------


def _flat_list(value: Any, *, field: str) -> list[Any]:
    """Copy tokenizer output to a list, removing a single leading batch wrapper.

    Token and mask element values are checked by _tokenize_candidate. This
    helper handles array-like outputs without depending on Torch or NumPy.
    """

    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"The chat template must return a one-dimensional {field}.")
    return list(value)


def _assistant_spans(mask: list[int]) -> list[tuple[int, int]]:
    """Return half-open [start, end) ranges of contiguous enabled mask values."""

    spans = []
    start = None
    # A trailing zero closes an assistant span that reaches the end of the mask.
    for index, enabled in enumerate([*mask, 0]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append((start, index))
            start = None
    return spans


# ------------------------------------------------------------------
# Candidate rendering and message-to-token supervision alignment
# ------------------------------------------------------------------


def _tokenize_candidate(
    candidate: _Candidate,
    tokenizer: Any,
    config: SWESFTConfig,
) -> tuple[list[int], list[int]]:
    """Render one candidate and return equally sized token IDs and loss mask.

    The template must return one contiguous tracked span per assistant message.
    This requirement maps message ordinals in masked_assistant_indices to exact
    token positions. Misaligned templates raise rather than guessing boundaries.
    Tool, user, and system tokens remain context with loss_mask=0.
    """

    messages = candidate.messages
    if config.parse_tool_call_arguments:
        messages = parse_tool_call_arguments(messages)
    kwargs = dict(config.chat_template_kwargs)
    if config.include_tools and candidate.tools:
        kwargs["tools"] = candidate.tools
    try:
        output = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kwargs,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "The PyTRIO tokenizer could not render the SWE conversation with "
            "assistant-token tracking. Check its chat template and tool format."
        ) from error
    if not isinstance(output, Mapping) or "input_ids" not in output:
        raise TypeError("The chat template must return a mapping with input_ids.")
    input_ids = _flat_list(output["input_ids"], field="input_ids")
    assistant_mask = _flat_list(
        output.get("assistant_masks"), field="assistant_masks"
    )
    if len(input_ids) != len(assistant_mask):
        raise ValueError("assistant_masks must contain one value per input token.")
    if any(
        not isinstance(token, Integral) or isinstance(token, bool) or token < 0
        for token in input_ids
    ):
        raise ValueError("input_ids must contain nonnegative integer token IDs.")
    if any(
        not isinstance(bit, Integral) or int(bit) not in (0, 1)
        for bit in assistant_mask
    ):
        raise ValueError("assistant_masks must contain only 0 and 1 values.")

    spans = _assistant_spans([int(bit) for bit in assistant_mask])
    # Candidate exclusions refer to assistant ordinals, not message-list indices.
    assistant_count = sum(
        message.get("role") == "assistant" for message in messages
    )
    if len(spans) != assistant_count:
        raise ValueError(
            "The tokenizer returned "
            f"{len(spans)} assistant spans for {assistant_count} assistant messages."
        )
    loss_mask = [0] * len(input_ids)
    for assistant_index, (start, end) in enumerate(spans):
        if assistant_index not in candidate.masked_assistant_indices:
            loss_mask[start:end] = [1] * (end - start)
    return [int(token) for token in input_ids], loss_mask


# ------------------------------------------------------------------
# Public pipeline: load records, select candidates, tokenize, and report
# ------------------------------------------------------------------


def prepare_swe_sft_dataset(
    source: Any,
    tokenizer: Any,
    config: SWESFTConfig | None = None,
) -> SWESFTDataset:
    """Collect assistant-supervised examples from SWE trajectory records.

    Args:
        source: A single record, JSONL path, or iterable accepted by records_from_source.
        tokenizer: Chat tokenizer supporting tools and assistant-token tracking.
        config: Candidate selection, reasoning, filtering, and length policies.

    Returns:
        An in-memory SWESFTDataset with reusable pretokenized rows and a report.
        Multiple conversations and assistant turns may yield multiple examples
        per source record. The source is consumed fully before returning.

    Configured filters update report counters. Overlong candidates are dropped
    whole to preserve tool-call structure. Malformed input/template alignment
    raises immediately; producing no trainable examples also raises with counts.
    """

    config = config or SWESFTConfig()
    report = SWESFTReport()
    examples = []
    # Read each record once; nested conversations are processed independently.
    for record_index, record in enumerate(records_from_source(source)):
        report.records += 1
        conversations = conversations_from_record(record, record_index=record_index)
        report.conversations += len(conversations)
        for conversation in conversations:
            candidates = candidates_from_conversation(conversation, config, report)
            for candidate in candidates:
                input_ids, loss_mask = _tokenize_candidate(candidate, tokenizer, config)
                if config.max_length is not None and len(input_ids) > config.max_length:
                    # Unlike generic SFT truncation, keep tool trajectories intact.
                    report.filtered_overlong += 1
                    continue
                # Position zero has no preceding model input and can never be
                # a causal target, so require supervision after that position.
                if not any(loss_mask[1:]):
                    report.filtered_without_labels += 1
                    continue
                labels = [
                    token if enabled else IGNORE_INDEX
                    for token, enabled in zip(input_ids, loss_mask, strict=True)
                ]
                examples.append(SupervisedExample(input_ids=input_ids, labels=labels))

    report.examples = len(examples)
    if not examples:
        raise ValueError(
            "SWE preprocessing produced no trainable examples: "
            f"records={report.records}, candidates={report.candidates}, "
            f"errors={report.error_segments}, overlong={report.filtered_overlong}, "
            f"without_labels={report.filtered_without_labels}."
        )
    return SWESFTDataset(tuple(examples), report)
