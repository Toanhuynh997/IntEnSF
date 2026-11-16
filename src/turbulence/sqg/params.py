from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


Precision = Literal["single", "double"]


@dataclass(frozen=True)
class SQGParams:
    # Physical parameters
    f: float = 1.0e-4
    nsq: float = 1.0e-4
    L: float = 20.0e6
    H: float = 10.0e3
    U: float = 30.0
    r: float = 0.0
    tdiab: float = 10.0 * 86400.0
    symmetric: bool = True

    # Numerical parameters
    dt: float = 300.0
    diff_order: int = 8
    diff_efold: float = 86400.0 / 3.0
    dealias: bool = True
    precision: Precision = "single"

    # Runtime parameters
    device: str | torch.device = "cpu"
    tstart: float = 0.0

    def validate(self, *, N: Optional[int] = None) -> None:
        if N is not None:
            if int(N) <= 0:
                raise ValueError("N must be positive")
            if int(N) % 2 != 0:
                raise ValueError("N must be even")

        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.diff_efold <= 0:
            raise ValueError("diff_efold must be > 0")
        if self.diff_order <= 0:
            raise ValueError("diff_order must be > 0")
        if self.precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")

        if self.L <= 0 or self.H <= 0:
            raise ValueError("L and H must be > 0")
        if self.f == 0:
            raise ValueError("f must be nonzero")

    def to_core_kwargs(self) -> dict:
        # Matches _SQGTorchCore keyword args.
        return {
            "f": self.f,
            "nsq": self.nsq,
            "L": self.L,
            "H": self.H,
            "U": self.U,
            "r": self.r,
            "tdiab": self.tdiab,
            "diff_order": self.diff_order,
            "diff_efold": self.diff_efold,
            "symmetric": self.symmetric,
            "dt": self.dt,
            "dealias": self.dealias,
            "precision": self.precision,
            "tstart": self.tstart,
            "device": self.device,
        }

