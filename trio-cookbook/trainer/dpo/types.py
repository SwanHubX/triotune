"""Local DPO training state and step results."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DPOState:
    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


@dataclass(slots=True)
class DPOResult:
    step: int
    metrics: dict[str, float]
    num_pairs: int
    checkpoint_path: str | None = None
