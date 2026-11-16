from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class GridSpec:
    """Common spatial grid metadata for turbulence fields."""

    nx: int
    ny: int
    lx: float
    ly: float


@runtime_checkable
class TurbulenceModel(Protocol):
    """Minimal interface for a step-able turbulence forward model."""

    @property
    def time(self) -> float:
        """Model time in seconds."""

    @property
    def device(self) -> torch.device:
        """Torch device used by the model state."""

    def step(self, n: int = 1) -> None:
        """Advance the model forward by n internal timesteps."""

    def get_state(self) -> Any:
        """Return the full internal model state (for DA / checkpointing)."""

    def set_state(self, state: Any) -> None:
        """Restore a state produced by get_state()."""

    def fields(self) -> Dict[str, torch.Tensor]:
        """Return physical-space fields (e.g., pv, u, v) for diagnostics/DA."""

    def grid(self) -> Optional[GridSpec]:
        """Return grid metadata when available."""

