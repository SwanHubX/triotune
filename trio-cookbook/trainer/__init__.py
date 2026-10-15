"""Text-only GRPO, DPO, SFT, and OPD trainers backed by PyTRIO."""

from importlib import import_module
import sys

from .dataset import (
    IGNORE_INDEX,
    PreferenceExample,
    PromptExample,
    RenderedPrompt,
    SWEConversation,
    SWESFTConfig,
    SWESFTDataset,
    SWESFTReport,
    SupervisedExample,
    prepare_preference,
    prepare_prompt,
    prepare_rows,
    prepare_sft,
    prepare_swe_sft_dataset,
    render_prompt,
)
from .dpo import DPOConfig, DPOResult, DPOState, DPOTrainer
from .grpo import GRPOBatch, GRPOConfig, GRPODatum, GRPOResult, GRPOTrainer, SequenceSample, TrainerState
from .opd import OPDConfig, OPDResult, OPDState, OPDTrainer
from .sft import SFTConfig, SFTState, SFTTrainer

# Historical flat imports resolve to the actual modules, preserving object
# identity, pickled types, and patches of module attributes such as tqdm.
_LEGACY_MODULES = {
    "grpo_config": "grpo.config",
    "grpo_trainer": "grpo.trainer",
    "dpo_config": "dpo.config",
    "dpo_trainer": "dpo.trainer",
    "sft_config": "sft.config",
    "sft_trainer": "sft.trainer",
    "opd_config": "opd.config",
    "opd_trainer": "opd.trainer",
    "opd_distillation": "opd.distillation",
    "advantages": "grpo.advantages",
    "datum": "grpo.datum",
    "rewards": "grpo.rewards",
    "rollout": "grpo.rollout",
    "types": "grpo.types",
    "prompts": "dataset.prompt",
}
for _old_name, _new_name in _LEGACY_MODULES.items():
    _module = import_module(f".{_new_name}", __name__)
    sys.modules[f"{__name__}.{_old_name}"] = _module
    globals()[_old_name] = _module
del _old_name, _new_name, _module

__all__ = [
    "DPOConfig",
    "DPOResult",
    "DPOState",
    "DPOTrainer",
    "GRPOBatch",
    "GRPOConfig",
    "GRPODatum",
    "GRPOResult",
    "GRPOTrainer",
    "IGNORE_INDEX",
    "OPDConfig",
    "OPDResult",
    "OPDState",
    "OPDTrainer",
    "PreferenceExample",
    "PromptExample",
    "RenderedPrompt",
    "SequenceSample",
    "SFTConfig",
    "SFTState",
    "SFTTrainer",
    "SWEConversation",
    "SWESFTConfig",
    "SWESFTDataset",
    "SWESFTReport",
    "SupervisedExample",
    "TrainerState",
    "prepare_preference",
    "prepare_prompt",
    "prepare_rows",
    "prepare_sft",
    "prepare_swe_sft_dataset",
    "render_prompt",
]
