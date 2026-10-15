"""Text-only GRPO, DPO, SFT, and OPD trainers backed by PyTRIO."""

from importlib import import_module
import sys

from .common.config import (
    BaseTrainerConfig,
    DPOConfig,
    EvaluationConfig,
    GRPOConfig,
    OPDConfig,
    SFTConfig,
)
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
from .dpo.types import DPOResult, DPOState
from .dpo_trainer import DPOTrainer
from .grpo.types import GRPOBatch, GRPODatum, GRPOResult, SequenceSample, TrainerState
from .grpo_trainer import GRPOTrainer
from .opd.types import OPDResult, OPDState
from .opd_trainer import OPDTrainer
from .sft.types import SFTState
from .sft_trainer import SFTTrainer

# Historical flat imports resolve to the actual modules, preserving object
# identity, pickled types, and patches of module attributes such as tqdm.
_LEGACY_MODULES = {
    "grpo_config": "common.config",
    "grpo_trainer": "grpo_trainer",
    "dpo_config": "common.config",
    "dpo_trainer": "dpo_trainer",
    "sft_config": "common.config",
    "sft_trainer": "sft_trainer",
    "opd_config": "common.config",
    "opd_trainer": "opd_trainer",
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

# Algorithm subpackages used to own trainer and config modules. Keep those
# import paths as aliases while the canonical implementations live at the
# trainer root and in common.config.
_config_module = import_module(".common.config", __name__)
for _algorithm in ("grpo", "dpo", "sft", "opd"):
    sys.modules[f"{__name__}.{_algorithm}.config"] = _config_module
    sys.modules[f"{__name__}.{_algorithm}.trainer"] = import_module(
        f".{_algorithm}_trainer", __name__
    )
del _algorithm, _config_module

__all__ = [
    "BaseTrainerConfig",
    "DPOConfig",
    "DPOResult",
    "DPOState",
    "DPOTrainer",
    "EvaluationConfig",
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
