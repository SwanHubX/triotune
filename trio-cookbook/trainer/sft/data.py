"""SFT tokenization, label masking, and truncation."""

from collections.abc import Mapping
from typing import Any

from ..common.prompts import _token_list, render_prompt
from .config import SFTConfig


def tokens_and_labels(
    row: Mapping[str, Any], tokenizer: Any, args: SFTConfig,
) -> tuple[list[int], list[int]]:
    """Build validated tokens and labels before the next-token shift."""
    conversational = False
    if "input_ids" in row:
        tokens = _token_list(row["input_ids"], source="input_ids")
        if args.completion_only_loss and "labels" not in row:
            raise ValueError("Tokenized completion-only rows require labels with prompt tokens set to -100.")
        labels = _token_list(row.get("labels", tokens), source="labels")
    else:
        paired = "prompt" in row and "completion" in row
        if paired:
            prompt, completion = row["prompt"], row["completion"]
            conversational = isinstance(prompt, (list, tuple)) and isinstance(completion, (list, tuple))
            if conversational:
                prompt_tokens = render_prompt(
                    prompt, tokenizer, chat_template_kwargs=args.chat_template_kwargs
                ).tokens
                value = list(prompt) + list(completion)
            elif isinstance(prompt, str) and isinstance(completion, str):
                prompt_tokens = render_prompt(prompt, tokenizer).tokens
                value = prompt + completion
            else:
                raise TypeError("prompt and completion must both be text or chat messages.")
        elif "messages" in row:
            conversational, value = True, row["messages"]
        elif args.dataset_text_field in row:
            value = row[args.dataset_text_field]
            if not isinstance(value, str):
                raise TypeError("The dataset text field must contain a string.")
        else:
            raise KeyError("Row needs input_ids, messages, prompt/completion, or the dataset text field.")

        if conversational and args.assistant_only_loss:
            output = tokenizer.apply_chat_template(
                value, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_assistant_tokens_mask=True,
                **args.chat_template_kwargs,
            )
            tokens = _token_list(output, source="chat template")
            if not isinstance(output, Mapping):
                raise TypeError("The chat template must return input_ids and assistant_masks.")
            assistant_mask = output.get("assistant_masks")
            if hasattr(assistant_mask, "tolist"):
                assistant_mask = assistant_mask.tolist()
            if not isinstance(assistant_mask, (list, tuple)) or (
                len(assistant_mask) != len(tokens)
                or not all(bit in (0, 1) for bit in assistant_mask)
                or not any(assistant_mask)
            ):
                raise ValueError("The chat template must return a non-empty assistant token mask.")
        else:
            tokens = render_prompt(
                value, tokenizer, add_generation_prompt=False,
                chat_template_kwargs=args.chat_template_kwargs,
            ).tokens
            assistant_mask = None
        if paired and tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError("The tokenized prompt is not a prefix of the full example.")
        if not conversational and args.add_eos_token:
            eos = getattr(tokenizer, "eos_token_id", None)
            if not isinstance(eos, int) or isinstance(eos, bool):
                raise ValueError("add_eos_token requires tokenizer.eos_token_id.")
            if tokens[-1] != eos:
                tokens.append(eos)
        completion_only = args.completion_only_loss
        if completion_only is None:
            completion_only = paired
        if completion_only and not paired:
            raise ValueError("completion_only_loss requires prompt/completion rows.")
        labels = [
            token if (not completion_only or index >= len(prompt_tokens))
            and (assistant_mask is None or assistant_mask[index]) else -100
            for index, token in enumerate(tokens)
        ]

    if len(tokens) != len(labels):
        raise ValueError("input_ids and labels must have the same length.")
    if any(token < 0 for token in tokens) or any(label < 0 and label != -100 for label in labels):
        raise ValueError("Token IDs must be nonnegative; only -100 may mask a label.")
    if args.assistant_only_loss and not conversational:
        raise ValueError("assistant_only_loss requires conversational data.")
    if args.max_length is not None:
        tokens, labels = tokens[: args.max_length], labels[: args.max_length]
    if len(tokens) < 2 or all(label == -100 for label in labels[1:]):
        raise ValueError("The example has no next-token labels after masking/truncation.")
    return tokens, labels
