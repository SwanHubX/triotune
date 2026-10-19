"""Public OPD configuration, trainer, and algorithm extension points."""

import sys

# Public configuration and algorithm components are collected here so research
# code can import extension points from one stable package boundary.
from ...api.cli_args import OPDConfig
from ..opd_trainer import OPDTrainer
from .contracts import (
    OPDPrompt,
    OPDResult,
    OPDRollout,
    OPDState,
    OPDTeacherGroup,
    OPDTeacherSpec,
    OPDTeacherTarget,
    OPDTrainingSample,
)
from .contracts import (
    validate_logprobs as _validate_logprobs,
)
from .contracts import (
    validate_tokens as _validate_tokens,
)
from .datum import build_opd_datum
from .execution import OPDExecutionPlan
from .loss import SampledReverseKLLoss, distillation_advantages
from .metrics import OPDMetricReducer, OPDMetricTotals
from .rollout import OPDRolloutPhase
from .targets import OPDTargetBuilder
from .teacher_phase import (
    OPDTeacherPhase,
)
from .teacher_phase import (
    teacher_completion_logprobs as _teacher_completion_logprobs,
)

# These helpers remain package attributes solely for historical
# ``opd.distillation`` imports; focused modules are preferred by new code.
validate_logprobs = _validate_logprobs
validate_tokens = _validate_tokens
teacher_completion_logprobs = _teacher_completion_logprobs

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
    "SampledReverseKLLoss",
    "build_opd_datum",
    "distillation_advantages",
]
