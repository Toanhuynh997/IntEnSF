"""Helpers for resolving dataset paths before constructing Dataset classes."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from project_utils.paths import DATA_DIR, dataset_trajectory_dir


def _is_set(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        return False
    return True


def _get_data_root(mapping) -> str | None:
    """Return data_root from a dict or OmegaConf node; accept deprecated sqg_root."""
    if OmegaConf.is_config(mapping):
        data_root = OmegaConf.select(mapping, "data_root", default=None)
        legacy = OmegaConf.select(mapping, "sqg_root", default=None)
    else:
        data_root = mapping.get("data_root")
        legacy = mapping.get("sqg_root")
    if _is_set(data_root):
        return str(data_root)
    if _is_set(legacy):
        warnings.warn(
            "dataset kwargs sqg_root is deprecated; use data_root instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return str(legacy)
    return None


def resolve_path_cond_data_dir(
    *,
    path_dir: str | None = None,
    data_root: str | None = None,
    trajectory_subdir: str | None = None,
) -> str:
    """
    Resolve absolute directory containing path_*.pt trajectories.

    Priority: explicit path_dir (legacy) > data_root + trajectory_subdir.
    """
    if _is_set(path_dir):
        if os.path.isabs(path_dir):
            return path_dir
        return os.path.join(DATA_DIR, str(path_dir))

    if _is_set(data_root) and _is_set(trajectory_subdir):
        return dataset_trajectory_dir(str(data_root), str(trajectory_subdir))

    raise ValueError(
        "Dataset path requires either path_dir or both data_root and trajectory_subdir"
    )


def dataset_location_label(dataset_kwargs) -> str:
    """Human-readable dataset location for logging."""
    if OmegaConf.is_config(dataset_kwargs):
        path = OmegaConf.select(dataset_kwargs, "path", default=None)
        path_dir = OmegaConf.select(dataset_kwargs, "path_dir", default=None)
        trajectory_subdir = OmegaConf.select(dataset_kwargs, "trajectory_subdir", default=None)
        data_path_full = OmegaConf.select(dataset_kwargs, "data_path_full", default=None)
    else:
        path = dataset_kwargs.get("path")
        path_dir = dataset_kwargs.get("path_dir")
        trajectory_subdir = dataset_kwargs.get("trajectory_subdir")
        data_path_full = dataset_kwargs.get("data_path_full")

    data_root = _get_data_root(dataset_kwargs)

    if _is_set(data_path_full):
        return str(data_path_full)
    try:
        return resolve_path_cond_data_dir(
            path_dir=path_dir,
            data_root=data_root,
            trajectory_subdir=trajectory_subdir,
        )
    except ValueError:
        pass
    if _is_set(path):
        return os.path.join(DATA_DIR, str(path))
    return "<unknown>"


def prepare_dataset_kwargs(dataset_kwargs) -> dict[str, Any]:
    """
    Return a plain dict suitable for dnnlib.util.construct_class_by_name.

    - Legacy `DatasetCond`: set `data_path_full` from `path` when needed.
    - `DatasetPathCond`: resolves data_root/trajectory_subdir or path_dir.
    """
    if OmegaConf.is_config(dataset_kwargs):
        kwargs = OmegaConf.to_container(dataset_kwargs, resolve=True)
    else:
        kwargs = dict(dataset_kwargs)

    path = kwargs.get("path")
    if path is not None and kwargs.get("data_path_full") is None:
        kwargs["data_path_full"] = os.path.join(DATA_DIR, str(Path(path)))

    class_name = kwargs.get("class_name", "")
    if isinstance(class_name, str) and class_name.endswith("DatasetPathCond"):
        data_root = _get_data_root(kwargs)
        kwargs["data_path_full"] = resolve_path_cond_data_dir(
            path_dir=kwargs.get("path_dir"),
            data_root=data_root,
            trajectory_subdir=kwargs.get("trajectory_subdir"),
        )

    return kwargs


def should_validate_dataset(dataset_kwargs) -> bool:
    """True if we have enough info to construct the dataset at startup."""
    if OmegaConf.is_config(dataset_kwargs):
        path = OmegaConf.select(dataset_kwargs, "path", default=None)
        path_dir = OmegaConf.select(dataset_kwargs, "path_dir", default=None)
        trajectory_subdir = OmegaConf.select(dataset_kwargs, "trajectory_subdir", default=None)
    else:
        path = dataset_kwargs.get("path")
        path_dir = dataset_kwargs.get("path_dir")
        trajectory_subdir = dataset_kwargs.get("trajectory_subdir")

    data_root = _get_data_root(dataset_kwargs)

    if path is not None:
        return True
    if _is_set(path_dir):
        return True
    if _is_set(data_root) and _is_set(trajectory_subdir):
        return True
    return False
