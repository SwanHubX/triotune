"""Text-only PPO, DPO, SFT, and OPD trainers backed by PyTRIO."""

import sys
from importlib import import_module

from ..api.cli_args import (
    BaseTrainerConfig,
    DPOConfig,
    NormConfig,
    OPDConfig,
    PPOConfig,
    SFTConfig,
)
from ..dataset import (
    IGNORE_INDEX,
    PreferenceExample,
    PromptExample,
    RenderedPrompt,
    SupervisedExample,
    SWEConversation,
    SWESFTConfig,
    SWESFTDataset,
    SWESFTReport,
    prepare_preference,
    prepare_prompt,
    prepare_rows,
    prepare_sft,
    prepare_swe_sft_dataset,
    render_prompt,
)
from .dpo.state import DPOResult, DPOState
from .dpo_trainer import DPOTrainer
from .opd.contracts import (
    OPDPrompt,
    OPDResult,
    OPDRollout,
    OPDState,
    OPDTeacherGroup,
    OPDTeacherSpec,
    OPDTeacherTarget,
    OPDTrainingSample,
)
from .opd.datum import build_opd_datum
from .opd.execution import OPDExecutionPlan
from .opd.loss import SampledReverseKLLoss, distillation_advantages
from .opd.metrics import OPDMetricReducer, OPDMetricTotals
from .opd.rollout import OPDRolloutPhase
from .opd.targets import OPDTargetBuilder
from .opd.teacher_phase import OPDTeacherPhase
from .opd_trainer import OPDTrainer
from .sft.state import SFTState
from .sft_trainer import SFTTrainer

# Historical flat imports resolve to the actual modules, preserving object
# identity, pickled types, and patches of module attributes such as tqdm.
_LEGACY_MODULES = {
    "dpo_trainer": "dpo_trainer",
    "sft_trainer": "sft_trainer",
    "opd_trainer": "opd_trainer",
    "opd_distillation": "opd",
}
for _old_name, _new_name in _LEGACY_MODULES.items():
    _module = import_module(f".{_new_name}", __name__)
    sys.modules[f"{__name__}.{_old_name}"] = _module
    globals()[_old_name] = _module
del _old_name, _new_name, _module

# Config contracts now live beside trainer in ``triotune.api``. Preserve
# both the historical flat config modules and algorithm-local config paths.
_config_module = import_module("..api.cli_args", __name__)
for _algorithm in ("dpo", "sft", "opd"):
    sys.modules[f"{__name__}.{_algorithm}_config"] = _config_module
    globals()[f"{_algorithm}_config"] = _config_module
    sys.modules[f"{__name__}.{_algorithm}.config"] = _config_module
    sys.modules[f"{__name__}.{_algorithm}.trainer"] = import_module(
        f".{_algorithm}_trainer", __name__
    )
del _algorithm, _config_module

# ``api``, ``dataset`` and ``utils`` are now siblings of trainer. Map the old
# nested paths to the canonical modules so serialized objects and existing
# integrations keep a single module identity during migration.
_COMPAT_MODULES = {
    "common": "api",
    "common.config": "api.cli_args",
    "common.client": "api.client_api",
    "common.checkpoint": "utils.checkpoint",
    "common.metrics_api": "utils.metrics",
    "common.logging": "utils.metrics",
    "dataset": "dataset",
    "dataset.examples": "dataset.examples",
    "dataset.prepare": "dataset.prepare",
    "dataset.prompt": "dataset.prompt",
    "dataset.swe_sft": "dataset.swe_sft",
    "dataset.swe_sft.prepare": "dataset.swe_sft.prepare",
    "dataset.swe_sft.schema": "dataset.swe_sft.schema",
    "dataset.swe_sft.trajectory": "dataset.swe_sft.trajectory",
}
for _old_name, _new_name in _COMPAT_MODULES.items():
    _module = import_module(f"..{_new_name}", __name__)
    sys.modules[f"{__name__}.{_old_name}"] = _module
globals()["common"] = import_module("..api", __name__)
globals()["dataset"] = import_module("..dataset", __name__)

_prompt_module = import_module("..dataset.prompt", __name__)
sys.modules[f"{__name__}.prompts"] = _prompt_module
globals()["prompts"] = _prompt_module
del _old_name, _new_name, _module, _prompt_module

# These algorithm-local modules were renamed to describe their actual
# responsibility. Keep import aliases for callers that used the old internal
# paths, while the source tree exposes only the clearer canonical names.
_RENAMED_MODULES = {
    "dpo.types": "dpo.state",
    "sft.types": "sft.state",
    "opd.types": "opd.contracts",
}
for _old_name, _new_name in _RENAMED_MODULES.items():
    sys.modules[f"{__name__}.{_old_name}"] = import_module(
        f".{_new_name}", __name__
    )
del _old_name, _new_name


def __getattr__(name):
    """Load the root-level PPO trainer without importing it eagerly."""

    if name != "PPOTrainer":
        raise AttributeError(name)
    module = import_module(".rl_trainer", __name__)
    globals()[name] = module.PPOTrainer
    return module.PPOTrainer

__all__ = [
    "BaseTrainerConfig",
    "DPOConfig",
    "DPOResult",
    "DPOState",
    "DPOTrainer",
    "IGNORE_INDEX",
    "OPDConfig",
    "OPDExecutionPlan",
    "OPDMetricReducer",
    "OPDMetricTotals",
    "OPDPrompt",
    "OPDResult",
    "OPDRollout",
    "OPDRolloutPhase",
    "OPDState",
    "OPDTeacherPhase",
    "OPDTeacherGroup",
    "OPDTeacherSpec",
    "OPDTeacherTarget",
    "OPDTargetBuilder",
    "OPDTrainer",
    "OPDTrainingSample",
    "NormConfig",
    "PPOConfig",
    "PPOTrainer",
    "PreferenceExample",
    "PromptExample",
    "RenderedPrompt",
    "SFTConfig",
    "SFTState",
    "SFTTrainer",
    "SWEConversation",
    "SWESFTConfig",
    "SWESFTDataset",
    "SWESFTReport",
    "SupervisedExample",
    "SampledReverseKLLoss",
    "build_opd_datum",
    "distillation_advantages",
    "prepare_preference",
    "prepare_prompt",
    "prepare_rows",
    "prepare_sft",
    "prepare_swe_sft_dataset",
    "render_prompt",
]
