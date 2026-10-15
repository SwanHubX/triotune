"""Canonical examples produced by algorithm-neutral dataset preparation.

The classes in this module describe token alignment before an algorithm builds
its PyTRIO ``Datum``.  Keeping the representation independent from the SDK
makes preprocessing easy to inspect and test without creating remote clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any


IGNORE_INDEX = -100


def _token_ids(value: Any, *, field: str) -> list[int]:
    """Copy and validate one flat sequence of nonnegative token IDs."""

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
    """Copy labels while reserving ``IGNORE_INDEX`` as the only mask value."""

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


@dataclass(slots=True)
class PromptExample:
    """A prompt rendered once into both reward-facing text and model tokens."""

    text: str
    tokens: list[int]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Prompt text must be a string.")
        self.tokens = _token_ids(self.tokens, field="tokens")

    @property
    def input_ids(self) -> list[int]:
        """Expose prompt tokens with the canonical dataset field spelling."""

        return self.tokens


@dataclass(slots=True)
class SupervisedExample:
    """One causal-language-model sequence with unshifted token labels.

    ``labels[i]`` is the desired token at position ``i``; ``-100`` excludes
    that position from loss.  PyTRIO consumes next-token-aligned arrays, so
    ``model_input`` and ``target_tokens`` remove the last and first positions
    respectively.  The first label can never contribute to causal next-token
    loss and is therefore ignored when checking that an example is trainable.
    """

    input_ids: list[int]
    labels: list[int]

    def __post_init__(self) -> None:
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
        """Return a 0/1 mask aligned with ``input_ids``."""

        return [int(label != IGNORE_INDEX) for label in self.labels]

    @property
    def model_input(self) -> list[int]:
        """Return tokens that produce logits for the following token."""

        return self.input_ids[:-1]

    @property
    def target_tokens(self) -> list[int]:
        """Return PyTRIO targets aligned with ``model_input`` logits."""

        return self.labels[1:]

    def as_dict(self) -> dict[str, list[int]]:
        """Return the stable serialized form used by cached datasets."""

        return {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "loss_mask": self.loss_mask,
        }


@dataclass(slots=True)
class PreferenceExample:
    """A shared prompt with ordered chosen and rejected completion tokens."""

    prompt: list[int]
    chosen: list[int]
    rejected: list[int]

    def __post_init__(self) -> None:
        self.prompt = _token_ids(self.prompt, field="prompt")
        self.chosen = _token_ids(self.chosen, field="chosen")
        self.rejected = _token_ids(self.rejected, field="rejected")

    def sequence(self, preference: str) -> SupervisedExample:
        """Build a full sequence whose loss covers only the selected response."""

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
        return self.sequence("chosen").loss_mask

    @property
    def rejected_loss_mask(self) -> list[int]:
        return self.sequence("rejected").loss_mask
