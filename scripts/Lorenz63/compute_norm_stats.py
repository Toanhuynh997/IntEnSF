#!/usr/bin/env python3
"""Compute dataset-wide normalization stats for raw Lorenz 63 path_*.pt trajectories."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from project_utils.normalization import (  # noqa: E402
    NORM_STATS_FILENAME,
    compute_and_save_norm_stats,
    load_norm_stats,
)
from project_utils.trajectory_paths import resolve_trajectory_output_dir  # noqa: E402
from project_utils.yaml_simple import load_simple_yaml  # noqa: E402


def _resolve_path_dir(
    path_dir: str | None,
    config_path: str,
    *,
    data_root: str | None = None,
    trajectory_subdir: str | None = None,
) -> str:
    if path_dir is not None:
        return resolve_trajectory_output_dir({}, out_dir=path_dir)
    cfg = load_simple_yaml(config_path)
    gen_cfg = dict(cfg.get("generator", {}))
    return resolve_trajectory_output_dir(
        gen_cfg,
        data_root=data_root,
        trajectory_subdir=trajectory_subdir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute global min/max stats for raw Lorenz 63 trajectories and save norm_stats.pt"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/turbulence/lorenz63_default.yaml",
        help="Used to resolve default path_dir from generator.data_root/trajectory_subdir",
    )
    parser.add_argument(
        "--path_dir",
        type=str,
        default=None,
        help="Directory containing path_*.pt (overrides data_root/trajectory_subdir)",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--trajectory_subdir", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=f"Output stats file (default: <path_dir>/{NORM_STATS_FILENAME})",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Use separate vmin/vmax per channel instead of one global range",
    )
    parser.add_argument(
        "--include_burn_in",
        action="store_true",
        help="Include path_*_burn_in.pt files when computing stats",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    path_dir = _resolve_path_dir(
        args.path_dir,
        config_path,
        data_root=args.data_root,
        trajectory_subdir=args.trajectory_subdir,
    )
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"path_dir does not exist: {path_dir}")

    out_path = args.out
    if out_path is not None and not os.path.isabs(out_path):
        out_path = os.path.join(_REPO_ROOT, out_path) if out_path.startswith("datasets/") else os.path.abspath(out_path)

    stats = compute_and_save_norm_stats(
        path_dir,
        out_path=out_path,
        per_channel=args.per_channel,
        ignore_burn_in=not args.include_burn_in,
    )
    saved = out_path or os.path.join(path_dir, NORM_STATS_FILENAME)
    saved_stats = load_norm_stats(saved)

    print("=" * 60)
    print("Lorenz 63 normalization stats")
    print("=" * 60)
    print(f"path_dir:     {path_dir}")
    print(f"output:       {saved}")
    print(f"method:       {saved_stats['method']}")
    print(f"scope:        {saved_stats['scope']}")
    print(f"n_files:      {saved_stats['n_files']}")
    print(f"n_frames:     {saved_stats['n_frames']}")
    print(f"vmin (global): {saved_stats['vmin']:.6g}")
    print(f"vmax (global): {saved_stats['vmax']:.6g}")
    if saved_stats["scope"] == "per_channel":
        print(f"vmin per ch:  {saved_stats['vmin_per_channel'].tolist()}")
        print(f"vmax per ch:  {saved_stats['vmax_per_channel'].tolist()}")
    print("=" * 60)
    print("Forward:  x_norm = 2*(x - vmin)/(vmax - vmin) - 1")
    print("Inverse:  x_phys = (x_norm + 1)/2 * (vmax - vmin) + vmin")


if __name__ == "__main__":
    main()
