"""Public OPD configuration, trainer, and algorithm extension points."""

import sys

# Public configuration and algorithm components are collected here so research
# code can import extension points from one stable package boundary.
from ...api.cli_args import OPDConfig
from .contracts import (
    OPDResult,
    OPDRollout,
    OPDState,
    OPDTeacherTarget,
    OPDTrainingSample,
    validate_logprobs,
    validate_tokens,
)
from .datum import build_opd_datum
from .execution import OPDExecutionPlan
from .loss import SampledReverseKLLoss, distillation_advantages
from .metrics import OPDMetricReducer, OPDMetricTotals
from .rollout import OPDRolloutPhase
from .targets import OPDTargetBuilder
from .teacher_phase import OPDTeacherPhase, teacher_completion_logprobs
from ..opd_trainer import OPDTrainer

# ---------------------------------------------------------------------------
# Historical module compatibility
# ---------------------------------------------------------------------------

# ``distillation.py`` and ``state.py`` used to be thin implementation modules.
# Their responsibilities now live in the focused modules above.  Preserve the
# historical import paths as aliases without retaining redundant source files.
sys.modules[f"{__name__}.distillation"] = sys.modules[__name__]
sys.modules[f"{__name__}.state"] = sys.modules[f"{__name__}.contracts"]

# Keep this list limited to supported research and trainer extension points.
# Validation helpers remain attributes for old imports but are intentionally
# excluded from the primary public surface.
__all__ = [
    "OPDConfig",
    "OPDExecutionPlan",
    "OPDMetricReducer",
    "OPDMetricTotals",
    "OPDResult",
    "OPDRollout",
    "OPDRolloutPhase",
    "OPDState",
    "OPDTeacherPhase",
    "OPDTeacherTarget",
    "OPDTargetBuilder",
    "OPDTrainer",
    "OPDTrainingSample",
    "SampledReverseKLLoss",
    "build_opd_datum",
    "distillation_advantages",
]
