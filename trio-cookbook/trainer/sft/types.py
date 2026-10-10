"""Local SFT training state."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SFTState:
    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
