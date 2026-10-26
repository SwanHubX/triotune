"""Composable PPO-family building blocks with explicit PyTRIO boundaries.

The training lifecycle is implemented by :class:`trainer.rl_trainer.PPOTrainer`.
This package keeps the numerical, rollout, actor, and critic components grouped
under ``trainer.ppo`` while exposing the historical ``ppo.trainer`` import as a
compatibility alias.
"""

import importlib
import sys

from .actor import PPOActor
from .advantages import (
    AdvantageEstimator,
    LossWeightPolicy,
    Objective,
    PPOAlgorithmPlan,
    RewardNormalizer,
    RewardShaper,
    ReturnEstimator,
    normalize,
    prepare_advantages,
)
from ...api.cli_args import (
    NormConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    PPORolloutConfig,
)
from .trajectory import PPOState, PreparedBatch, Trajectory, validate_policy_versions
from .critic import Critic, PyTRIOCritic, value_loss
from .gae import (
    compute_gae,
    compute_token_level_gae,
    compute_trajectory_level_gae,
    compute_turn_level_gae,
)

# Keep historical submodule imports working as identity aliases during migration.
_advantages_module = importlib.import_module(".advantages", __name__)
_critic_module = importlib.import_module(".critic", __name__)
_config_module = importlib.import_module("...api.cli_args", __name__)
_reporting_module = importlib.import_module("...utils.reporting", __name__)
for _legacy_name, _canonical_module in (
    ("config", _config_module),
    ("normalization", _advantages_module),
    ("critic_loss", _critic_module),
    ("reporting", _reporting_module),
):
    sys.modules[f"{__name__}.{_legacy_name}"] = _canonical_module
del (
    _legacy_name,
    _canonical_module,
    _advantages_module,
    _critic_module,
    _config_module,
    _reporting_module,
)

# ``PPOTrainer`` moved to ``trainer.rl_trainer``. Importing the root trainer
# while this package is being initialized would create a circular import, so
# resolve the class after the local PPO components have loaded. The root module
# installs the same alias when it is the import entry point.
_root_trainer = sys.modules.get(f"{__package__.rsplit('.', 1)[0]}.rl_trainer")
if _root_trainer is None:
    _root_trainer = importlib.import_module("..rl_trainer", __package__)
if hasattr(_root_trainer, "PPOTrainer"):
    PPOTrainer = _root_trainer.PPOTrainer
    sys.modules[f"{__name__}.trainer"] = _root_trainer


def __getattr__(name):
    """Resolve the moved trainer lazily for circular-import-safe compatibility."""

    if name != "PPOTrainer":
        raise AttributeError(name)
    module = importlib.import_module("..rl_trainer", __package__)
    sys.modules[f"{__name__}.trainer"] = module
    return module.PPOTrainer

__all__ = [
    "Critic",
    "AdvantageEstimator",
    "LossWeightPolicy",
    "NormConfig",
    "PPOActor",
    "PPOAlgorithmPlan",
    "PPOActorConfig",
    "PPOConfig",
    "PPOCriticConfig",
    "PPOState",
    "PPOTrainer",
    "PyTRIOCritic",
    "PreparedBatch",
    "PPORolloutConfig",
    "Trajectory",
    "Objective",
    "RewardNormalizer",
    "RewardShaper",
    "ReturnEstimator",
    "validate_policy_versions",
    "compute_gae",
    "compute_token_level_gae",
    "compute_turn_level_gae",
    "compute_trajectory_level_gae",
    "normalize",
    "prepare_advantages",
    "value_loss",
]
