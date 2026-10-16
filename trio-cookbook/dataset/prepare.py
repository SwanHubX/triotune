"""Convert raw rows into prompt, supervised, or preference example contracts.

Each prepare_* function processes one mapping. prompt.py supplies shared text
and chat rendering; examples.py validates the resulting token structures.
prepare_rows additionally loads a source through load.py and collects all
results. None of these functions schedules batches or creates SDK Datums.

Length policies are format-specific: prompts and preference pairs reject
overlong inputs; SFT truncates input_ids and labels together from the right.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from .examples import IGNORE_INDEX, PreferenceExample, PromptExample, SupervisedExample
from .load import load_rows
from .prompt import _token_list, render_prompt


# ------------------------------------------------------------------
# Shared row, mask, sequence-length, and EOS validation
# ------------------------------------------------------------------


def _row(row: Any, *, index: int | None = None) -> Mapping[str, Any]:
    """Require a mapping and include the optional zero-based row index in errors."""

    if isinstance(row, Mapping):
        return row
    location = "Dataset row" if index is None else f"Dataset row {index}"
    raise TypeError(f"{location} must be a mapping.")


def _binary_mask(value: Any, *, length: int, field: str) -> list[int]:
    """Require exactly one 0/1 value per token, preserving positional alignment."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain one 0/1 value per input token.")
    if any(not isinstance(bit, Integral) or int(bit) not in (0, 1) for bit in value):
        raise ValueError(f"{field} must contain only 0 and 1 values.")
    return [int(bit) for bit in value]


def _max_length(value: int | None) -> int | None:
    """Normalize an optional sequence limit, allowing at least one causal target."""

    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 2
    ):
        raise ValueError("max_length must be an integer of at least 2 or None.")
    return None if value is None else int(value)


def _append_eos(tokens: list[int], tokenizer: Any, *, enabled: bool) -> None:
    """Append EOS in place when requested and not already present."""

    if not enabled:
        return
    eos = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos, Integral) or isinstance(eos, bool) or eos < 0:
        raise ValueError("add_eos_token requires a nonnegative tokenizer.eos_token_id.")
    if tokens[-1] != eos:
        tokens.append(int(eos))


# ------------------------------------------------------------------
# Generation prompts: render the context and reserve completion space
# ------------------------------------------------------------------


def prepare_prompt(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    prompt_key: str = "messages",
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    max_length: int | None = None,
    min_completion_tokens: int = 0,
    row_index: int | None = None,
) -> PromptExample:
    """Prepare a generation context as PromptExample(text, tokens).

    Args:
        row: Raw mapping with a prompt_key field; unrelated fields are untouched.
        tokenizer: Tokenizer used by render_prompt for this prompt's input form.
        prompt_key: Field containing text, token IDs, or text-only chat messages.
        add_generation_prompt: Include the chat template's assistant prefix.
        chat_template_kwargs: Additional template-specific rendering options.
        max_length: Optional budget for prompt plus reserved completion tokens.
        min_completion_tokens: Space to reserve within that budget, commonly 1 for OPD.
        row_index: Optional source index used only to locate invalid rows.

    The prompt is never truncated. Exceeding the budget raises an error so the
    trainer cannot silently change the context used for generation/scoring.
    """

    row = _row(row, index=row_index)
    if not isinstance(prompt_key, str) or not prompt_key.strip():
        raise ValueError("prompt_key must be a non-empty string.")
    if prompt_key not in row:
        location = "Dataset row" if row_index is None else f"Dataset row {row_index}"
        raise KeyError(f"{location} needs {prompt_key!r}.")
    max_length = _max_length(max_length)
    if (
        isinstance(min_completion_tokens, bool)
        or not isinstance(min_completion_tokens, Integral)
        or min_completion_tokens < 0
    ):
        raise ValueError("min_completion_tokens must be a nonnegative integer.")

    rendered = render_prompt(
        row[prompt_key],
        tokenizer,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )
    example = rendered
    if max_length is not None and len(example.input_ids) + min_completion_tokens > max_length:
        suffix = (
            " and leaves no completion space"
            if min_completion_tokens
            else ""
        )
        location = "Prompt" if row_index is None else f"Prompt {row_index}"
        raise ValueError(f"{location} exceeds max_length={max_length}{suffix}.")
    return example


# ------------------------------------------------------------------
# Preference pairs: share one prompt and keep responses separate
# ------------------------------------------------------------------


def _completion_tokens(
    value: Any,
    prompt: Any,
    prompt_tokens: list[int],
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> list[int]:
    """Return completion-only tokens from text, token IDs, or chat responses.

    For chat responses, render prompt + response together and remove the exact
    prompt-token prefix. Reject templates that rewrite the prefix, since a
    guessed boundary would associate loss with the wrong tokens.
    """

    if isinstance(value, str):
        tokens = _token_list(
            tokenizer.encode(value, add_special_tokens=False),
            source="tokenizer.encode()",
        )
    elif isinstance(value, (list, tuple)) and value and all(
        isinstance(item, Integral) and not isinstance(item, bool) for item in value
    ):
        tokens = [int(token) for token in value]
    elif (
        isinstance(prompt, (list, tuple))
        and isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, Mapping) for item in prompt)
        and all(isinstance(item, Mapping) for item in value)
    ):
        full = render_prompt(
            list(prompt) + list(value),
            tokenizer,
            add_generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs,
        )
        if full.tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError("Chat template does not preserve prompt tokens before the completion.")
        tokens = full.tokens[len(prompt_tokens) :]
    else:
        raise TypeError("chosen and rejected must be text, token IDs, or chat completion messages.")
    if not tokens:
        raise ValueError("A preference completion must contain at least one token.")
    if any(token < 0 for token in tokens):
        raise ValueError("Preference completion token IDs must be nonnegative.")
    return tokens


def prepare_preference(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    prompt_key: str = "messages",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    add_eos_token: bool = True,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    max_length: int | None = None,
    row_index: int | None = None,
) -> PreferenceExample:
    """Prepare one ordered preference pair with a shared rendered prompt.

    prompt_key, chosen_key, and rejected_key select the three input fields.
    Responses are tokenized separately, optionally terminated with EOS, and
    checked against max_length including the shared prompt. Overlong pairs
    raise instead of truncating one alternative differently from the other.

    Returns PreferenceExample(prompt, chosen, rejected) with completion-only
    response fields. Prompt labels are masked later by its sequence() method
    or by the DPO trainer when constructing a supervised sequence.
    """

    row = _row(row, index=row_index)
    max_length = _max_length(max_length)
    required = (prompt_key, chosen_key, rejected_key)
    if not all(key in row for key in required):
        location = "Dataset row" if row_index is None else f"Dataset row {row_index}"
        raise KeyError(
            f"{location} needs {prompt_key}, {chosen_key}, and {rejected_key}."
        )
    prompt = prepare_prompt(
        row,
        tokenizer,
        prompt_key=prompt_key,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
        row_index=row_index,
    )
    completions: list[list[int]] = []
    # Preserve chosen/rejected ordering for the downstream pairwise objective.
    for key in (chosen_key, rejected_key):
        tokens = _completion_tokens(
            row[key], row[prompt_key], prompt.input_ids, tokenizer, chat_template_kwargs
        )
        _append_eos(tokens, tokenizer, enabled=add_eos_token)
        if max_length is not None and len(prompt.input_ids) + len(tokens) > max_length:
            location = "Dataset row" if row_index is None else f"Dataset row {row_index}"
            raise ValueError(f"{location} {key} exceeds max_length={max_length}.")
        completions.append(tokens)
    return PreferenceExample(prompt.input_ids, completions[0], completions[1])


# ------------------------------------------------------------------
# Supervised examples: derive unshifted labels and supervision masks
# ------------------------------------------------------------------


def prepare_sft(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    text_field: str = "text",
    max_length: int | None = None,
    completion_only_loss: bool | None = None,
    assistant_only_loss: bool = False,
    add_eos_token: bool = True,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> SupervisedExample:
    """Convert one supported SFT row to equal-length input_ids and labels.

    Args:
        row: Pretokenized IDs, a prompt/completion pair, messages, or plain text.
        tokenizer: Encodes raw content and supplies optional assistant masks.
        text_field: Plain-text fallback column, used after other supported shapes.
        max_length: Right-truncate both inputs and labels to this token limit.
        completion_only_loss: Mask the prompt in paired rows; None enables this
            automatically for pairs and disables it for other raw input forms.
        assistant_only_loss: Require a conversational template with assistant masks.
        add_eos_token: Append EOS only to raw non-conversational sequences if absent.
        chat_template_kwargs: Additional options passed to chat rendering.

    Pretokenized rows retain their IDs and labels; no EOS is appended. They
    can provide labels (-100 masks targets), loss_mask (0/1), or both if the
    masks agree. assistant_only_loss requires raw conversational inputs.
    The returned SupervisedExample remains unshifted and verifies that a
    supervised next-token target survives masking and truncation.
    """

    row = _row(row)
    max_length = _max_length(max_length)
    if not isinstance(text_field, str) or not text_field.strip():
        raise ValueError("text_field must be a non-empty string.")
    conversational = False
    if "input_ids" in row:
        # Cached/pretokenized rows already encode supervision boundaries.
        tokens = _token_list(row["input_ids"], source="input_ids")
        if completion_only_loss and "labels" not in row and "loss_mask" not in row:
            raise ValueError(
                "Tokenized completion-only rows require labels or loss_mask for prompt tokens."
            )
        if "labels" in row:
            labels = _token_list(row["labels"], source="labels")
        elif "loss_mask" in row:
            mask = _binary_mask(row["loss_mask"], length=len(tokens), field="loss_mask")
            labels = [
                token if bit else IGNORE_INDEX
                for token, bit in zip(tokens, mask, strict=True)
            ]
        else:
            labels = list(tokens)
        if "labels" in row and "loss_mask" in row:
            # Reject conflicting supervision descriptions rather than picking one.
            mask = _binary_mask(row["loss_mask"], length=len(tokens), field="loss_mask")
            label_mask = [int(label != IGNORE_INDEX) for label in labels]
            if mask != label_mask:
                raise ValueError("labels and loss_mask describe different supervised tokens.")
    else:
        # Raw-input precedence: complete prompt/completion pair, messages, text.
        paired = "prompt" in row and "completion" in row
        prompt_tokens: list[int] = []
        if paired:
            prompt, completion = row["prompt"], row["completion"]
            conversational = isinstance(prompt, (list, tuple)) and isinstance(
                completion, (list, tuple)
            )
            if conversational:
                prompt_tokens = render_prompt(
                    prompt,
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,
                ).tokens
                value = list(prompt) + list(completion)
            elif isinstance(prompt, str) and isinstance(completion, str):
                prompt_tokens = render_prompt(prompt, tokenizer).tokens
                value = prompt + completion
            else:
                raise TypeError("prompt and completion must both be text or chat messages.")
        elif "messages" in row:
            conversational, value = True, row["messages"]
        elif text_field in row:
            value = row[text_field]
            if not isinstance(value, str):
                raise TypeError("The dataset text field must contain a string.")
        else:
            raise KeyError(
                "Row needs input_ids, messages, prompt/completion, or the dataset text field."
            )

        assistant_mask: list[int] | None
        if conversational and assistant_only_loss:
            # Token spans come from the template; string matching cannot reliably
            # recover boundaries around role markers, reasoning, and special tokens.
            output = tokenizer.apply_chat_template(
                value,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **dict(chat_template_kwargs or {}),
            )
            tokens = _token_list(output, source="chat template")
            if not isinstance(output, Mapping):
                raise TypeError("The chat template must return input_ids and assistant_masks.")
            try:
                assistant_mask = _binary_mask(
                    output.get("assistant_masks"),
                    length=len(tokens),
                    field="assistant_masks",
                )
            except ValueError as error:
                raise ValueError(
                    "The chat template must return a non-empty assistant token mask."
                ) from error
            if not any(assistant_mask):
                raise ValueError("The chat template must return a non-empty assistant token mask.")
        else:
            tokens = render_prompt(
                value,
                tokenizer,
                add_generation_prompt=False,
                chat_template_kwargs=chat_template_kwargs,
            ).tokens
            assistant_mask = None

        if paired and tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError("The tokenized prompt is not a prefix of the full example.")
        if not conversational:
            # Chat templates own their terminators; avoid appending a second one.
            before_eos = len(tokens)
            _append_eos(tokens, tokenizer, enabled=add_eos_token)
            if len(tokens) != before_eos and assistant_mask is not None:
                assistant_mask.append(1)

        completion_only = completion_only_loss
        if completion_only is None:
            completion_only = paired
        if completion_only and not paired:
            raise ValueError("completion_only_loss requires prompt/completion rows.")
        # When both policies apply, train only assistant tokens in the completion.
        labels = [
            token
            if (not completion_only or index >= len(prompt_tokens))
            and (assistant_mask is None or assistant_mask[index])
            else IGNORE_INDEX
            for index, token in enumerate(tokens)
        ]

    if len(tokens) != len(labels):
        raise ValueError("input_ids and labels must have the same length.")
    if assistant_only_loss and not conversational:
        raise ValueError("assistant_only_loss requires conversational data.")
    if max_length is not None:
        # Slice both fields together; SupervisedExample checks remaining targets.
        tokens, labels = tokens[:max_length], labels[:max_length]
    return SupervisedExample(input_ids=tokens, labels=labels)


# ------------------------------------------------------------------
# Dataset convenience entry point: load and collect prepared examples
# ------------------------------------------------------------------


def prepare_rows(
    rows: Any,
    tokenizer: Any,
    *,
    format: str,
    **kwargs: Any,
) -> list[PromptExample | SupervisedExample | PreferenceExample]:
    """Load rows and eagerly prepare them using an explicitly selected format.

    Args:
        rows: One mapping, a JSONL path, or an existing iterable accepted by load_rows.
        tokenizer: Tokenizer forwarded to the selected single-row processor.
        format: 'prompt', 'sft', or 'preference'; no content-based algorithm guessing.
        **kwargs: Format-specific options forwarded to the selected processor.

    Returns:
        An in-memory list of example objects in source order, not runtime batches.
        Use the single-row functions directly when preparing a stream incrementally.

    A messages field can represent either a supervised dialogue or a generation
    prompt, so the caller must specify the intended format. The first invalid
    row stops preparation; no filtering or silent recovery is performed here.
    """

    processors = {
        "prompt": prepare_prompt,
        "sft": prepare_sft,
        "preference": prepare_preference,
    }
    try:
        processor = processors[format]
    except KeyError as error:
        raise ValueError("format must be 'prompt', 'sft', or 'preference'.") from error
    prepared = []
    for index, row in enumerate(load_rows(rows)):
        if format in {"prompt", "preference"}:
            prepared.append(processor(row, tokenizer, row_index=index, **kwargs))
        else:
            prepared.append(processor(row, tokenizer, **kwargs))
    return prepared
