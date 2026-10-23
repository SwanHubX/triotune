"""Public entry points for dataset loading and single-example preparation.

Module responsibilities:
    load: Read raw rows from a mapping, JSONL path, or existing iterable.
    prompt: Render text, token IDs, and chat messages into text and tokens.
    prepare: Validate prompt, SFT, and preference rows and derive labels/masks.
    examples: Define the returned data structures, rather than another stage.

``prepare_rows`` loads a source and collects all prepared examples in memory.
Trainers can also call ``prepare_*`` on individual rows after batching them.
Batch scheduling and PyTRIO Datum materialization belong to the runtime layer.
"""

from .examples import IGNORE_INDEX, PreferenceExample, PromptExample, SupervisedExample
from .load import load_rows
from .prepare import prepare_preference, prepare_prompt, prepare_rows, prepare_sft
from .prompt import RenderedPrompt, render_prompt

__all__ = [
    "IGNORE_INDEX",
    "load_rows",
    "PreferenceExample",
    "PromptExample",
    "RenderedPrompt",
    "SupervisedExample",
    "prepare_preference",
    "prepare_prompt",
    "prepare_rows",
    "prepare_sft",
    "render_prompt",
]
