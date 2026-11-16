from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Precision = Literal["single", "double"]


@dataclass(frozen=True)
class Lorenz96Params:
    # Model parameters
    K: int = 40
    F: float = 8.0

    # Numerical parameters
    dt: float = 0.01
    precision: Precision = "single"

    # Runtime parameters
    device: str | torch.device = "cpu"
    tstart: float = 0.0

    def validate(self) -> None:
        if self.K <= 0:
            raise ValueError("K must be positive")

        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")

    def to_core_kwargs(self) -> dict:
        return {
            "F": self.F,
            "dt": self.dt,
            "precision": self.precision,
            "tstart": self.tstart,
            "device": self.device,
        }
