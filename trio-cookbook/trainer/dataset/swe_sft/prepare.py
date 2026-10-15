"""Convert normalized SWE agent trajectories into PyTRIO SFT examples."""

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


def _flat_list(value: Any, *, field: str) -> list[Any]:
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
    """Return contiguous assistant spans produced by generation tracking."""

    spans = []
    start = None
    for index, enabled in enumerate([*mask, 0]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _tokenize_candidate(
    candidate: _Candidate,
    tokenizer: Any,
    config: SWESFTConfig,
) -> tuple[list[int], list[int]]:
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


def prepare_swe_sft_dataset(
    source: Any,
    tokenizer: Any,
    config: SWESFTConfig | None = None,
) -> SWESFTDataset:
    """Prepare a JSONL path or iterable of SWE trajectory records.

    The returned object is an iterable of pretokenized mappings accepted
    directly by :class:`SFTTrainer`. Intentional filters are counted in
    ``dataset.report``; malformed records or tokenizer alignment failures raise
    immediately because silently training the wrong tokens is unsafe.
    """

    config = config or SWESFTConfig()
    report = SWESFTReport()
    examples = []
    for record_index, record in enumerate(records_from_source(source)):
        report.records += 1
        conversations = conversations_from_record(record, record_index=record_index)
        report.conversations += len(conversations)
        for conversation in conversations:
            candidates = candidates_from_conversation(conversation, config, report)
            for candidate in candidates:
                input_ids, loss_mask = _tokenize_candidate(candidate, tokenizer, config)
                if config.max_length is not None and len(input_ids) > config.max_length:
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
