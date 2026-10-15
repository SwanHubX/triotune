"""Software-engineering agent trajectory preprocessing for PyTRIO SFT."""

from .prepare import prepare_swe_sft_dataset
from .schema import SWEConversation, SWESFTConfig, SWESFTDataset, SWESFTReport

__all__ = [
    "SWEConversation",
    "SWESFTConfig",
    "SWESFTDataset",
    "SWESFTReport",
    "prepare_swe_sft_dataset",
]
