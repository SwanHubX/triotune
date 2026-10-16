"""Local OPD training progress and optimizer-step results."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class OPDState:
    """Mutable progress retained across OPD optimizer steps."""

    global_step: int = 0
    behavior_policy_version: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(slots=True)
class OPDResult:
    """Metrics and policy version produced by one OPD step."""

    step: int
    metrics: dict[str, float]
    num_datums: int
    behavior_policy_version: int
    checkpoint_path: str | None = None
