"""Local progress retained by the SFT training pipeline."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SFTState:
    """Mutable progress retained across asynchronous SFT batches."""

    global_step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    data_cursor: int = 0
    optimizer_steps: int = 0
    checkpoint_manifest_path: str | None = None
