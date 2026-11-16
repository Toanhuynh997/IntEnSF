"""
Turbulence simulators and synthetic generators.

Each model should expose a step-able forward simulator suitable for
rollouts (dataset generation) and data assimilation (per-step access).
"""

from .base import TurbulenceModel
from .registry import make, register

__all__ = [
    "TurbulenceModel",
    "make",
    "register",
]

