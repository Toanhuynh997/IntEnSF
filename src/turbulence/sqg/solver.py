from __future__ import annotations

import numpy as np
import torch

from turbulence.base import GridSpec
from turbulence.registry import register


def _rfft2(*args, **kwargs):
    return torch.fft.rfft2(*args, **kwargs)


def _irfft2(*args, **kwargs):
    return torch.fft.irfft2(*args, **kwargs)


class _SQGTorchCore:
    """
    Core SQG numerics (ported from scripts/turb_SQG/sqg_torch_lib.py).

    State is advanced in spectral space (rfft2 coefficients) for the two boundaries.
    """

    def __init__(
        self,
        pv: torch.Tensor,
        *,
        f: float = 1.0e-4,
        nsq: float = 1.0e-4,
        L: float = 20.0e6,
        H: float = 10.0e3,
        U: float = 30.0,
        r: float = 0.0,
        tdiab: float = 10.0 * 86400.0,
        diff_order: int = 8,
        diff_efold: float,
        symmetric: bool = True,
        dt: float,
        dealias: bool = True,
        precision: str = "single",
        tstart: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        if pv.shape[0] != 2:
            raise ValueError("1st dim of pv should be 2")
        N = int(pv.shape[1])
        if N % 2:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt is None:
            raise ValueError("must specify time step")
        if diff_efold is None:
            raise ValueError("must specify efolding time scale for diffusion")

        self.N = N
        self.dealias = bool(dealias)
        self.symmetric = bool(symmetric)

        if precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")

        device = torch.device(device)

        self.nsq = torch.tensor(nsq, dtype=dtype, device=device)
        self.f = torch.tensor(f, dtype=dtype, device=device)
        self.H = torch.tensor(H, dtype=dtype, device=device)
        self.U = torch.tensor(U, dtype=dtype, device=device)
        self.L = torch.tensor(L, dtype=dtype, device=device)
        self.dt = torch.tensor(dt, dtype=dtype, device=device)
        self.r = torch.tensor(r, dtype=dtype, device=device)
        self.tdiab = torch.tensor(tdiab, dtype=dtype, device=device)
        self.t = torch.tensor(tstart, dtype=dtype, device=device)

        self.ekman = bool(r >= 1.0e-10)

        # equilibrium PV (thermal relaxation target)
        y = torch.arange(0, float(L), float(L) / N, dtype=dtype, device=device)
        pvbar = torch.zeros(2, N, dtype=dtype, device=device)
        pi = float(torch.pi)
        l0 = 2.0 * pi / float(L)
        mu0 = l0 * float(np.sqrt(float(nsq))) * float(H) / float(f)

        if self.symmetric:
            pvbar[:] = (
                -(mu0 * 0.5 * U / (l0 * H))
                * np.cosh(0.5 * mu0)
                * torch.cos(l0 * y)
                / np.sinh(0.5 * mu0)
            )
        else:
            pvbar[:] = -(mu0 * U / (l0 * H)) * np.cos(l0 * y) / np.sinh(mu0)
            pvbar[1, :] = pvbar[0, :] * np.cosh(mu0)

        pvbar = pvbar[:, :, None] * torch.ones(2, N, N, dtype=dtype, device=device)
        self.pvspec_eq = _rfft2(pvbar)

        pv = pv.to(device=device, dtype=dtype)
        self.pvspec = _rfft2(pv)

        # spectral grids
        l = N * torch.fft.fftfreq(N, dtype=dtype, device=device)
        k = l[0 : (N // 2) + 1]
        k, l = torch.meshgrid(k, l, indexing="xy")
        k = 2.0 * pi * k / self.L
        l = 2.0 * pi * l / self.L
        ksqlsq = k**2 + l**2

        self.ksqlsq = ksqlsq
        self.ik = 1.0j * k
        self.il = 1.0j * l

        if self.dealias:
            l_pad = (3 * N // 2) * torch.fft.fftfreq(3 * N // 2, dtype=dtype, device=device)
            k_pad = l_pad[0 : (3 * N // 4) + 1]
            k_pad, l_pad = torch.meshgrid(k_pad, l_pad, indexing="xy")
            k_pad = 2.0 * pi * k_pad / self.L
            l_pad = 2.0 * pi * l_pad / self.L
            self.ik_pad = 1.0j * k_pad
            self.il_pad = 1.0j * l_pad

        mu = torch.sqrt(ksqlsq) * torch.sqrt(self.nsq) * self.H / self.f
        mu = mu.clip(torch.finfo(mu.dtype).eps)
        self.Hovermu = self.H / mu
        mu64 = mu.to(torch.float64)
        self.tanhmu = torch.tanh(mu64).to(dtype)
        self.sinhmu = torch.sinh(mu64).to(dtype)

        self.diff_order = torch.tensor(diff_order, dtype=dtype, device=device)
        self.diff_efold = torch.tensor(diff_efold, dtype=dtype, device=device)
        ktot = torch.sqrt(ksqlsq)
        ktotcutoff = pi * N / self.L
        self.hyperdiff = torch.exp(
            (-self.dt / self.diff_efold) * (ktot / ktotcutoff) ** self.diff_order
        )

        self.u: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def invert(self, pvspec: torch.Tensor | None = None) -> torch.Tensor:
        if pvspec is None:
            pvspec = self.pvspec
        psispec = torch.empty(
            2, self.N, self.N // 2 + 1, dtype=pvspec.dtype, device=pvspec.device
        )
        psispec[0] = self.Hovermu * ((pvspec[1] / self.sinhmu) - (pvspec[0] / self.tanhmu))
        psispec[1] = self.Hovermu * ((pvspec[1] / self.tanhmu) - (pvspec[0] / self.sinhmu))
        return psispec

    def specpad(self, specarr: torch.Tensor) -> torch.Tensor:
        specarr_pad = torch.zeros(
            2,
            3 * self.N // 2,
            3 * self.N // 4 + 1,
            dtype=specarr.dtype,
            device=specarr.device,
        )
        specarr_pad[:, 0 : self.N // 2, 0 : self.N // 2] = 2.25 * specarr[
            :, 0 : self.N // 2, 0 : self.N // 2
        ]
        specarr_pad[:, -self.N // 2 :, 0 : self.N // 2] = 2.25 * specarr[
            :, -self.N // 2 :, 0 : self.N // 2
        ]
        specarr_pad[:, 0 : self.N // 2, self.N // 2] = torch.conj(
            2.25 * specarr[:, 0 : self.N // 2, -1]
        )
        specarr_pad[:, -self.N // 2 :, self.N // 2] = torch.conj(
            2.25 * specarr[:, -self.N // 2 :, -1]
        )
        return specarr_pad

    def spectrunc(self, specarr: torch.Tensor) -> torch.Tensor:
        specarr_trunc = torch.zeros(
            2, self.N, self.N // 2 + 1, dtype=specarr.dtype, device=specarr.device
        )
        specarr_trunc[:, 0 : self.N // 2, 0 : self.N // 2] = specarr[
            :, 0 : self.N // 2, 0 : self.N // 2
        ]
        specarr_trunc[:, -self.N // 2 :, 0 : self.N // 2] = specarr[
            :, -self.N // 2 :, 0 : self.N // 2
        ]
        return specarr_trunc

    def xyderiv(self, specarr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dealias:
            xderiv = _irfft2(self.ik * specarr)
            yderiv = _irfft2(self.il * specarr)
        else:
            specarr_pad = self.specpad(specarr)
            xderiv = _irfft2(self.ik_pad * specarr_pad)
            yderiv = _irfft2(self.il_pad * specarr_pad)
        return xderiv, yderiv

    def gettend(self, pvspec: torch.Tensor | None = None) -> torch.Tensor:
        if pvspec is None:
            pvspec = self.pvspec
        psispec = self.invert(pvspec)
        psix, psiy = self.xyderiv(psispec)
        pvx, pvy = self.xyderiv(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = _rfft2(jacobian)
        if self.dealias:
            jacobianspec = self.spectrunc(jacobianspec)

        dpvspecdt = (1.0 / self.tdiab) * (self.pvspec_eq - pvspec) - jacobianspec

        if self.ekman:
            dpvspecdt[0] += self.r * self.ksqlsq * psispec[0]
            if self.symmetric:
                dpvspecdt[1] -= self.r * self.ksqlsq * psispec[1]

        self.u = -psiy
        self.v = psix
        return dpvspecdt

    def timestep(self) -> None:
        k1 = self.dt * self.gettend(self.pvspec)
        k2 = self.dt * self.gettend(self.pvspec + 0.5 * k1)
        k3 = self.dt * self.gettend(self.pvspec + 0.5 * k2)
        k4 = self.dt * self.gettend(self.pvspec + k3)
        pvspecnew = self.pvspec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.pvspec = self.hyperdiff * pvspecnew
        self.t = self.t + self.dt

    def pv(self) -> torch.Tensor:
        return _irfft2(self.pvspec)


@register("sqg")
class SQGSolver:
    """
    Step-able SQG forward simulator.

    - `step()` advances exactly one internal timestep (or n timesteps).
    - `get_state()/set_state()` expose the spectral PV coefficients for DA/checkpointing.
    - `fields()` returns physical-space fields for diagnostics/observation operators.
    """

    def __init__(self, pv: torch.Tensor, **kwargs):
        self._core = _SQGTorchCore(pv, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._core.pvspec.device

    @property
    def time(self) -> float:
        return float(self._core.t.item())

    def step(self, n: int = 1) -> None:
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        for _ in range(n):
            self._core.timestep()

    def get_state(self) -> dict[str, torch.Tensor]:
        # Include time so DA can restore the full forward model state.
        return {"pvspec": self._core.pvspec.clone(), "t": self._core.t.clone()}

    def set_state(self, state: dict[str, torch.Tensor]) -> None:
        pvspec = state["pvspec"]
        t = state.get("t", None)
        if not isinstance(pvspec, torch.Tensor):
            raise TypeError("state['pvspec'] must be a torch.Tensor")
        if pvspec.shape != self._core.pvspec.shape:
            raise ValueError(f"pvspec shape {pvspec.shape} != expected {self._core.pvspec.shape}")
        self._core.pvspec = pvspec.to(device=self.device)
        if t is not None:
            self._core.t = t.to(device=self.device, dtype=self._core.t.dtype)

    def fields(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {"pv": self._core.pv()}
        if self._core.u is not None:
            out["u"] = self._core.u
        if self._core.v is not None:
            out["v"] = self._core.v
        return out

    def grid(self) -> GridSpec:
        N = int(self._core.N)
        L = float(self._core.L.item())
        return GridSpec(nx=N, ny=N, lx=L, ly=L)

