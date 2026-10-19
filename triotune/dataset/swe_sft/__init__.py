"""Public entry points for software-engineering agent trajectory preparation.

prepare_swe_sft_dataset loads records, extracts conversations, selects training
turns, and derives assistant-token labels. schema.py defines configuration and
results; trajectory.py handles message-level decisions; prepare.py maps those
decisions onto token spans and returns rows directly accepted by SFTTrainer.
"""

from .prepare import prepare_swe_sft_dataset
from .schema import SWEConversation, SWESFTConfig, SWESFTDataset, SWESFTReport

__all__ = [
    "SWEConversation",
    "SWESFTConfig",
    "SWESFTDataset",
    "SWESFTReport",
    "prepare_swe_sft_dataset",
]
