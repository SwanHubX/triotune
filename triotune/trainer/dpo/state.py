"""Local DPO training progress and optimizer-step results."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DPOState:
    """Mutable progress retained across DPO optimizer steps."""

    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    data_cursor: int = 0
    optimizer_steps: int = 0
    checkpoint_manifest_path: str | None = None


@dataclass(slots=True)
class DPOResult:
    """Metrics and checkpoint metadata produced by one DPO step."""

    step: int
    metrics: dict[str, float]
    num_pairs: int
    checkpoint_path: str | None = None
