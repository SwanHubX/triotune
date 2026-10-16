"""Canonical preparation results and their token/label invariants.

PromptExample holds a generation input, SupervisedExample a labeled sequence,
and PreferenceExample two responses sharing a prompt. These SDK-independent
structures support inspection and caching without remote clients. Construction
validates their fields; it does not load a dataset or invoke a tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any


# The only allowed negative label: exclude this target from supervised loss.
IGNORE_INDEX = -100


# ------------------------------------------------------------------
# Field normalization shared by the example contracts
# ------------------------------------------------------------------


def _token_ids(value: Any, *, field: str) -> list[int]:
    """Copy a nonempty 1-D token sequence; reject negative IDs and booleans."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a one-dimensional integer token sequence.")
    if not value:
        raise ValueError(f"{field} must contain at least one token.")
    if any(not isinstance(token, Integral) or isinstance(token, bool) for token in value):
        raise TypeError(f"{field} must be a one-dimensional integer token sequence.")
    tokens = [int(token) for token in value]
    if any(token < 0 for token in tokens):
        raise ValueError(f"{field} token IDs must be nonnegative.")
    return tokens


def _labels(value: Any, *, field: str) -> list[int]:
    """Copy token labels or IGNORE_INDEX; the example checks their length."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(label, Integral) or isinstance(label, bool) for label in value
    ):
        raise TypeError(f"{field} must be a one-dimensional integer label sequence.")
    labels = [int(label) for label in value]
    if any(label < 0 and label != IGNORE_INDEX for label in labels):
        raise ValueError(f"{field} may only use {IGNORE_INDEX} as a negative mask value.")
    return labels


# ------------------------------------------------------------------
# Generation prompts, supervised sequences, and ordered preference pairs
# ------------------------------------------------------------------


@dataclass(slots=True)
class PromptExample:
    """A rendered prompt: text for rewards/display and tokens for the model."""

    text: str
    tokens: list[int]

    def __post_init__(self) -> None:
        """Check the text type and copy the validated prompt token sequence."""

        if not isinstance(self.text, str):
            raise TypeError("Prompt text must be a string.")
        self.tokens = _token_ids(self.tokens, field="tokens")

    @property
    def input_ids(self) -> list[int]:
        """Expose the same token list under the standard input_ids field name."""

        return self.tokens


@dataclass(slots=True)
class SupervisedExample:
    """One full sequence with equal-length, unshifted token labels.

    ``labels[i]`` is the desired token at position i; -100 masks that target.
    Causal prediction pairs input_ids[:-1] with labels[1:]. The properties
    below expose these aligned lists without creating an SDK Datum.

    For input_ids=[10, 11, 12] and labels=[-100, -100, 12], the model input is
    [10, 11] and the targets are [-100, 12]. labels[0] has no preceding input
    position and can never contribute to next-token loss.
    """

    input_ids: list[int]
    labels: list[int]

    def __post_init__(self) -> None:
        """Require equal lengths and at least one supervised target after shift."""

        self.input_ids = _token_ids(self.input_ids, field="input_ids")
        self.labels = _labels(self.labels, field="labels")
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have the same length.")
        if len(self.input_ids) < 2 or all(
            label == IGNORE_INDEX for label in self.labels[1:]
        ):
            raise ValueError("The example has no next-token labels after masking/truncation.")

    @property
    def loss_mask(self) -> list[int]:
        """Derive an unshifted 0/1 mask from labels; 1 enables supervision."""

        return [int(label != IGNORE_INDEX) for label in self.labels]

    @property
    def model_input(self) -> list[int]:
        """Drop the final input token to align its logits with labels[1:]."""

        return self.input_ids[:-1]

    @property
    def target_tokens(self) -> list[int]:
        """Drop the first label, retaining -100 masks and matching model_input."""

        return self.labels[1:]

    def as_dict(self) -> dict[str, list[int]]:
        """Copy the cacheable fields accepted by prepare_sft on a later pass."""

        return {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "loss_mask": self.loss_mask,
        }


@dataclass(slots=True)
class PreferenceExample:
    """A shared prompt and ordered, completion-only chosen/rejected tokens.

    Call sequence() to build a full supervised sequence. Reference scoring
    and DPO loss computation belong to the training algorithm.
    """

    prompt: list[int]
    chosen: list[int]
    rejected: list[int]

    def __post_init__(self) -> None:
        """Copy and validate the nonempty prompt and both response sequences."""

        self.prompt = _token_ids(self.prompt, field="prompt")
        self.chosen = _token_ids(self.chosen, field="chosen")
        self.rejected = _token_ids(self.rejected, field="rejected")

    def sequence(self, preference: str) -> SupervisedExample:
        """Select chosen/rejected, prepend the prompt, and mask its labels."""

        if preference == "chosen":
            completion = self.chosen
        elif preference == "rejected":
            completion = self.rejected
        else:
            raise ValueError("preference must be 'chosen' or 'rejected'.")
        input_ids = self.prompt + completion
        labels = [IGNORE_INDEX] * len(self.prompt) + completion
        return SupervisedExample(input_ids=input_ids, labels=labels)

    @property
    def chosen_loss_mask(self) -> list[int]:
        """Return the unshifted prompt + chosen mask, enabling response tokens."""

        return self.sequence("chosen").loss_mask

    @property
    def rejected_loss_mask(self) -> list[int]:
        """Return the unshifted prompt + rejected mask, enabling response tokens."""

        return self.sequence("rejected").loss_mask
