from __future__ import annotations

import os
from typing import Iterable

import torch

NORM_STATS_FILENAME = "norm_stats.pt"
METHOD_MINMAX_MINUS1_1 = "minmax_to_minus1_1"
SCOPE_GLOBAL = "global"
SCOPE_PER_CHANNEL = "per_channel"


def list_trajectory_files(
    path_dir: str,
    *,
    ignore_burn_in: bool = True,
) -> list[str]:
    fnames = sorted(
        f for f in os.listdir(path_dir) if f.startswith("path_") and f.endswith(".pt")
    )
    if ignore_burn_in:
        fnames = [f for f in fnames if not f.endswith("_burn_in.pt")]
    if not fnames:
        raise FileNotFoundError(f"No path_*.pt files found in {path_dir}")
    return fnames


def compute_minmax_norm_stats(
    path_dir: str,
    *,
    ignore_burn_in: bool = True,
    per_channel: bool = False,
    file_list: Iterable[str] | None = None,
) -> dict:
    """
    Scan raw trajectory files and compute dataset-wide min/max for affine scaling to [-1, 1].
    """
    path_dir = os.path.abspath(path_dir)
    fnames = list(file_list) if file_list is not None else list_trajectory_files(
        path_dir, ignore_burn_in=ignore_burn_in
    )

    vmin_global = float("inf")
    vmax_global = float("-inf")
    vmin_ch: torch.Tensor | None = None
    vmax_ch: torch.Tensor | None = None
    n_frames = 0

    for fname in fnames:
        traj = torch.load(os.path.join(path_dir, fname), map_location="cpu", weights_only=False)
        if not isinstance(traj, torch.Tensor):
            raise TypeError(f"{fname}: expected torch.Tensor, got {type(traj)}")
        traj = traj.to(torch.float32)
        if traj.ndim != 4:
            raise ValueError(f"{fname}: expected (T,C,H,W), got {tuple(traj.shape)}")

        vmin_global = min(vmin_global, float(traj.min().item()))
        vmax_global = max(vmax_global, float(traj.max().item()))
        n_frames += int(traj.shape[0])

        if per_channel:
            _, C, _, _ = traj.shape
            t_vmin = traj.amin(dim=(0, 2, 3))
            t_vmax = traj.amax(dim=(0, 2, 3))
            if vmin_ch is None:
                vmin_ch = t_vmin.clone()
                vmax_ch = t_vmax.clone()
            else:
                if vmin_ch.shape[0] != C:
                    raise ValueError(f"{fname}: channel count {C} != {vmin_ch.shape[0]}")
                vmin_ch = torch.minimum(vmin_ch, t_vmin)
                vmax_ch = torch.maximum(vmax_ch, t_vmax)

    if vmin_global >= vmax_global:
        raise ValueError(
            f"Degenerate range vmin={vmin_global}, vmax={vmax_global} in {path_dir}"
        )

    stats: dict = {
        "method": METHOD_MINMAX_MINUS1_1,
        "scope": SCOPE_PER_CHANNEL if per_channel else SCOPE_GLOBAL,
        "vmin": vmin_global,
        "vmax": vmax_global,
        "source_dir": path_dir,
        "source_files": list(fnames),
        "n_files": len(fnames),
        "n_frames": n_frames,
    }
    if per_channel:
        if vmin_ch is None or vmax_ch is None:
            raise RuntimeError("per_channel stats missing channel tensors")
        stats["vmin_per_channel"] = vmin_ch
        stats["vmax_per_channel"] = vmax_ch
    return stats


def save_norm_stats(stats: dict, out_path: str) -> str:
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(stats, out_path)
    return out_path


def resolve_norm_stats_path(path_dir: str, norm_stats_path: str | None) -> str:
    if norm_stats_path is None:
        return os.path.join(os.path.abspath(path_dir), NORM_STATS_FILENAME)
    if os.path.isabs(norm_stats_path):
        return norm_stats_path
    from project_utils.paths import DATA_DIR

    return os.path.join(DATA_DIR, norm_stats_path)


def load_norm_stats(path: str) -> dict:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Normalization stats not found: {path}")
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(stats, dict):
        raise TypeError(f"Expected dict in {path}, got {type(stats)}")
    if stats.get("method") != METHOD_MINMAX_MINUS1_1:
        raise ValueError(f"Unsupported normalization method: {stats.get('method')}")
    return stats


def _channel_axis(x: torch.Tensor, n_channels: int) -> int:
    """Index of the channel dimension for (C, ...) or batched (B, C, ...)."""
    if x.ndim >= 4 and x.shape[1] == n_channels:
        return 1
    if x.shape[0] == n_channels:
        return 0
    raise ValueError(
        f"Cannot locate channel dim (n_channels={n_channels}) in shape {tuple(x.shape)}"
    )


def normalize_tensor(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Map physical PV to [-1, 1] using dataset stats."""
    scope = stats.get("scope", SCOPE_GLOBAL)
    if scope == SCOPE_GLOBAL:
        vmin = float(stats["vmin"])
        vmax = float(stats["vmax"])
        denom = vmax - vmin
        if denom <= 0:
            return torch.zeros_like(x)
        return (x - vmin) / denom * 2.0 - 1.0

    if scope == SCOPE_PER_CHANNEL:
        vmin_c = stats["vmin_per_channel"].to(dtype=x.dtype, device=x.device)
        vmax_c = stats["vmax_per_channel"].to(dtype=x.dtype, device=x.device)
        if x.ndim < 1:
            raise ValueError("x must have a channel dimension")
        n_ch = int(vmin_c.shape[0])
        ch_axis = _channel_axis(x, n_ch)
        C = x.shape[ch_axis]
        if C != n_ch:
            raise ValueError(f"channel dim {C} != stats length {n_ch}")
        view_shape = [1] * x.ndim
        view_shape[ch_axis] = C
        vmin = vmin_c.view(view_shape)
        vmax = vmax_c.view(view_shape)
        denom = vmax - vmin
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        out = (x - vmin) / denom * 2.0 - 1.0
        return torch.where(vmax > vmin, out, torch.zeros_like(out))

    raise ValueError(f"Unsupported stats scope: {scope}")


def denormalize_tensor(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Map [-1, 1] model output back to physical PV."""
    scope = stats.get("scope", SCOPE_GLOBAL)
    if scope == SCOPE_GLOBAL:
        vmin = float(stats["vmin"])
        vmax = float(stats["vmax"])
        return (x + 1.0) * 0.5 * (vmax - vmin) + vmin

    if scope == SCOPE_PER_CHANNEL:
        vmin_c = stats["vmin_per_channel"].to(dtype=x.dtype, device=x.device)
        vmax_c = stats["vmax_per_channel"].to(dtype=x.dtype, device=x.device)
        n_ch = int(vmin_c.shape[0])
        ch_axis = _channel_axis(x, n_ch)
        C = x.shape[ch_axis]
        if C != n_ch:
            raise ValueError(f"channel dim {C} != stats length {n_ch}")
        view_shape = [1] * x.ndim
        view_shape[ch_axis] = C
        vmin = vmin_c.view(view_shape)
        vmax = vmax_c.view(view_shape)
        return (x + 1.0) * 0.5 * (vmax - vmin) + vmin

    raise ValueError(f"Unsupported stats scope: {scope}")


def compute_and_save_norm_stats(
    path_dir: str,
    *,
    out_path: str | None = None,
    per_channel: bool = False,
    ignore_burn_in: bool = True,
) -> dict:
    stats = compute_minmax_norm_stats(
        path_dir,
        ignore_burn_in=ignore_burn_in,
        per_channel=per_channel,
    )
    out_path = out_path or os.path.join(os.path.abspath(path_dir), NORM_STATS_FILENAME)
    save_norm_stats(stats, out_path)
    return stats
