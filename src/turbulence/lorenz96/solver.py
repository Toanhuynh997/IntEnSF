from __future__ import annotations

import torch

from turbulence.registry import register


class _Lorenz96Core:
    """
    Lorenz 96 ring model:

      dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

    with periodic boundary conditions. Integration uses classical RK4.
    """

    def __init__(
        self,
        x: torch.Tensor,
        *,
        F: float = 8.0,
        dt: float = 0.01,
        precision: str = "single",
        tstart: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        if x.ndim != 1:
            raise ValueError("state x must be 1D with shape (K,)")

        if precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")

        device = torch.device(device)
        self.K = int(x.shape[0])
        self.F = torch.tensor(F, dtype=dtype, device=device)
        self.dt = torch.tensor(dt, dtype=dtype, device=device)
        self.t = torch.tensor(tstart, dtype=dtype, device=device)
        self.x = x.to(device=device, dtype=dtype)

    def _rhs(self, x: torch.Tensor) -> torch.Tensor:
        return (x.roll(-1) - x.roll(2)) * x.roll(1) - x + self.F

    def timestep(self) -> None:
        k1 = self.dt * self._rhs(self.x)
        k2 = self.dt * self._rhs(self.x + 0.5 * k1)
        k3 = self.dt * self._rhs(self.x + 0.5 * k2)
        k4 = self.dt * self._rhs(self.x + k3)
        self.x = self.x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.t = self.t + self.dt


@register("lorenz96")
class Lorenz96Solver:
    """
    Step-able Lorenz 96 forward simulator.

    - `step()` advances exactly one internal timestep (or n timesteps).
    - `get_state()/set_state()` expose the state vector for DA/checkpointing.
    - `fields()` returns the physical state as {"x": (K,)}.
    """

    def __init__(self, x: torch.Tensor, **kwargs):
        self._core = _Lorenz96Core(x, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._core.x.device

    @property
    def time(self) -> float:
        return float(self._core.t.item())

    @property
    def K(self) -> int:
        return self._core.K

    def step(self, n: int = 1) -> None:
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        for _ in range(n):
            self._core.timestep()

    def get_state(self) -> dict[str, torch.Tensor]:
        return {"x": self._core.x.clone(), "t": self._core.t.clone()}

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        x = state["x"]
        t = state.get("t", None)
        if not isinstance(x, torch.Tensor):
            raise TypeError("state['x'] must be a torch.Tensor")
        if x.shape != self._core.x.shape:
            raise ValueError(f"x shape {x.shape} != expected {self._core.x.shape}")
        self._core.x = x.to(device=self.device, dtype=self._core.x.dtype)
        if t is not None:
            self._core.t = t.to(device=self.device, dtype=self._core.t.dtype)

    def fields(self) -> dict[str, torch.Tensor]:
        return {"x": self._core.x}

    def grid(self) -> None:
        return None
