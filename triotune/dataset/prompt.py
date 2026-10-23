"""Render one text-only input into reward-facing text and model token IDs.

prepare.py calls this module for shared input rendering. Accepted inputs are
text, a nonempty token sequence, or chat messages. Chat templates own role
markers and generation prefixes; this module does not add EOS, choose loss
masks, truncate sequences, or assemble batches.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

from .examples import PromptExample


# ``RenderedPrompt`` was the original public name. Both names now reference
# one canonical dataset example instead of maintaining duplicate structures.
RenderedPrompt = PromptExample


# ------------------------------------------------------------------
# Input shape detection and tokenizer-output normalization
# ------------------------------------------------------------------


def _is_token(value: Any) -> bool:
    """Accept integer IDs while excluding bool, which subclasses int."""

    return isinstance(value, Integral) and not isinstance(value, bool)


def _text_content(content: Any, *, location: str) -> str:
    """Require string content and preserve its location in validation errors."""

    if isinstance(content, str):
        return content
    raise TypeError(f"{location} must be a text string.")


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Copy nonempty chat messages, validating role and text-only content.

    Extra message fields are preserved for templates. General prompt handling
    requires string content.
    """

    if not isinstance(messages, (list, tuple)) or not messages:
        raise TypeError("A conversational prompt must contain at least one message.")

    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        location = f"messages[{index}]"
        if not isinstance(message, Mapping):
            raise TypeError(f"{location} must be a mapping with role and content fields.")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise TypeError(f"{location}.role must be a non-empty string.")
        if "content" not in message:
            raise TypeError(f"{location} is missing the content field.")
        normalized_message = dict(message)
        normalized_message["role"] = role
        normalized_message["content"] = _text_content(
            message["content"], location=f"{location}.content"
        )
        normalized.append(normalized_message)
    return normalized


def _conversation(prompt: Any) -> list[dict[str, Any]] | None:
    """Detect a messages wrapper, single message, or list of message mappings.

    Return None for other shapes so render_prompt can try text/token handling.
    Recognized chat inputs are validated immediately rather than guessed again.
    """

    if isinstance(prompt, Mapping):
        if "messages" in prompt:
            return _normalize_messages(prompt["messages"])
        if "role" in prompt:
            return _normalize_messages([prompt])
        return None
    if (
        isinstance(prompt, (list, tuple))
        and prompt
        and all(isinstance(item, Mapping) for item in prompt)
    ):
        return _normalize_messages(prompt)
    return None


def _is_token_sequence(prompt: Any) -> bool:
    """Recognize nonempty flat integer sequences before chat/text dispatch."""

    return (
        isinstance(prompt, (list, tuple))
        and bool(prompt)
        and all(_is_token(token) for token in prompt)
    )


def _token_list(value: Any, *, source: str) -> list[int]:
    """Extract one token list from tokenizer mappings, attributes, or arrays.

    A single leading batch dimension is removed. Other nested shapes are
    rejected. This helper checks integer shape and nonemptiness; example
    contracts perform any additional token/label value validation.
    """

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError(f"{source} returned a mapping without input_ids.")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not all(
        _is_token(token) for token in value
    ):
        raise TypeError(f"{source} must return a one-dimensional integer token sequence.")
    tokens = [int(token) for token in value]
    if not tokens:
        raise ValueError("A rendered prompt must contain at least one token.")
    return tokens


# ------------------------------------------------------------------
# Text extraction and canonical prompt rendering
# ------------------------------------------------------------------


def as_prompt_text(prompt: Any) -> str:
    """Extract plain text without invoking a tokenizer or a chat template.

    Chat content is joined with newlines for text-oriented consumers. This is
    not the model's serialized conversation; render_prompt uses the template
    when role markers matter. Token-only inputs require tokenizer.decode and
    therefore cannot be converted by this helper.
    """

    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (bytes, bytearray)):
        return bytes(prompt).decode("utf-8")

    messages = _conversation(prompt)
    if messages is not None:
        return "\n".join(message["content"] for message in messages)

    if isinstance(prompt, Mapping):
        if isinstance(prompt.get("text"), str):
            return prompt["text"]
        if "content" in prompt:
            return _text_content(prompt["content"], location="prompt.content")

    if _is_token_sequence(prompt):
        raise TypeError("Token prompts require a tokenizer to produce reward-facing text.")
    if isinstance(prompt, (list, tuple)) and not prompt:
        raise ValueError("A prompt must not be empty.")
    raise TypeError(
        "A prompt must be text, a non-empty token sequence, or text-only chat messages."
    )


def render_prompt(
    prompt: Any,
    tokenizer: Any,
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> RenderedPrompt:
    """Return a PromptExample containing matching rendered text and token IDs.

    Args:
        prompt: Text, token IDs, a message/list of messages, or a supported wrapper.
        tokenizer: Supplies encode/decode or apply_chat_template for the input form.
        add_generation_prompt: Append the template's assistant prefix for generation.
        chat_template_kwargs: Template-specific options, excluding reserved controls.

    Token inputs are decoded only for display/rewards. Chat inputs are rendered
    twice with the same options: once to text and once to tokens. Plain text is
    encoded without automatic special tokens. No loss labels are produced here.
    """

    if _is_token_sequence(prompt):
        # Preserve the caller's IDs exactly; re-encoding decoded text could differ.
        tokens = [int(token) for token in prompt]
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("Token prompts require tokenizer.decode().")
        text = decode(tokens, skip_special_tokens=False)
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode() must return a string.")
        return RenderedPrompt(text=text, tokens=tokens)

    messages = _conversation(prompt)
    if messages is not None:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise TypeError("Conversational prompts require tokenizer.apply_chat_template().")
        template_kwargs = dict(chat_template_kwargs or {})
        # These arguments must match between the text and token template calls.
        reserved = {"tokenize", "add_generation_prompt"}.intersection(template_kwargs)
        if reserved:
            raise ValueError(
                "chat_template_kwargs must not override " + ", ".join(sorted(reserved)) + "."
            )
        text = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        if not isinstance(text, str):
            raise TypeError("tokenizer.apply_chat_template(tokenize=False) must return a string.")
        tokens = _token_list(
            apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            ),
            source="tokenizer.apply_chat_template(tokenize=True)",
        )
        return RenderedPrompt(text=text, tokens=tokens)

    text = as_prompt_text(prompt)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("Text prompts require tokenizer.encode().")
    tokens = _token_list(
        encode(text, add_special_tokens=False),
        source="tokenizer.encode()",
    )
    return RenderedPrompt(text=text, tokens=tokens)
