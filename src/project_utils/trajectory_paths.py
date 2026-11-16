"""Resolve trajectory output directories from generator/dataset YAML (no hardcoded dataset names)."""

from __future__ import annotations

import os
import warnings

from project_utils.paths import DATA_DIR, dataset_trajectory_dir


def _cfg_data_root(cfg: dict, data_root_key: str = "data_root") -> str | None:
    """Read data_root from config; accept deprecated sqg_root with a warning."""
    if cfg.get(data_root_key) is not None:
        return cfg.get(data_root_key)
    if cfg.get("sqg_root") is not None:
        warnings.warn(
            "Config key sqg_root is deprecated; use data_root instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return cfg.get("sqg_root")
    return None


def resolve_trajectory_output_dir(
    cfg: dict,
    *,
    out_dir: str | None = None,
    data_root: str | None = None,
    trajectory_subdir: str | None = None,
    data_root_key: str = "data_root",
    trajectory_subdir_key: str = "trajectory_subdir",
) -> str:
    """
    Absolute directory for path_*.pt trajectories.

    Requires ``data_root`` + ``trajectory_subdir`` in the config (or CLI overrides).
    Legacy ``out_subdir`` is supported with a deprecation warning.
    """
    if out_dir is not None:
        return out_dir if os.path.isabs(out_dir) else os.path.join(DATA_DIR, out_dir)

    root = data_root if data_root is not None else _cfg_data_root(cfg, data_root_key)
    subdir = (
        trajectory_subdir
        if trajectory_subdir is not None
        else cfg.get(trajectory_subdir_key)
    )
    if root is not None and subdir is not None:
        return dataset_trajectory_dir(str(root), str(subdir))

    if "out_subdir" in cfg:
        warnings.warn(
            f"generator.out_subdir is deprecated; set {data_root_key} and "
            f"{trajectory_subdir_key} in your YAML instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy = cfg["out_subdir"]
        return legacy if os.path.isabs(legacy) else os.path.join(DATA_DIR, legacy)

    raise ValueError(
        f"Trajectory output path requires {data_root_key} and {trajectory_subdir_key} "
        f"in the config file (e.g. configs/turbulence/sqg_default.yaml)."
    )
