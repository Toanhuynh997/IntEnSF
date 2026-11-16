import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from project_utils.normalization import compute_and_save_norm_stats
from project_utils.trajectory_paths import resolve_trajectory_output_dir
from project_utils.yaml_simple import load_simple_yaml
from turbulence.lorenz96 import Lorenz96Params, Lorenz96Solver


def _frame_tensor(x: torch.Tensor, K: int) -> torch.Tensor:
    """Map state (K,) to dataloader layout (1, K, 1)."""
    return x.reshape(1, K, 1)


def _make_initial_condition(K: int, F: float, *, device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.full((K,), float(F), device=device, dtype=dtype)
    x[0] = x[0] + torch.tensor(0.01, device=device, dtype=dtype)
    return x


def _print_run_summary(
    *,
    config_path: str,
    data_root: str,
    trajectory_subdir: str,
    out_dir: str,
    K: int,
    N_path: int,
    T_burn_in: int,
    T_total: int,
    save_burn_in: bool,
    output_interval: float,
    timesteps_per_output: int,
    seed: int,
    compute_norm_stats: bool,
    dry_run: bool,
    params: Lorenz96Params,
) -> None:
    print("=" * 60)
    print("Lorenz 96 trajectory generation")
    print("=" * 60)
    print(f"config:                  {config_path}")
    print(f"data_root:               {data_root}")
    print(f"trajectory_subdir:       {trajectory_subdir}")
    print(f"output directory:        {out_dir}")
    print(f"dry_run:                 {dry_run}")
    print()
    print("[generator]")
    print(f"  K:                     {K}")
    print(f"  N_path:                {N_path}")
    print(f"  T_burn_in:             {T_burn_in}")
    print(f"  T_total:               {T_total}")
    print(f"  save_burn_in:          {save_burn_in}")
    print(f"  output_interval:       {output_interval}")
    print(f"  timesteps_per_output:  {timesteps_per_output}")
    print(f"  seed:                  {seed}")
    print(f"  save format:           raw x (physical units), shape (T, 1, K, 1)")
    print(f"  compute_norm_stats:    {compute_norm_stats}")
    print(f"  expected path shape:   ({T_total + 1}, 1, {K}, 1)")
    if save_burn_in:
        print(f"  expected burn-in shape: ({T_burn_in + 1}, 1, {K}, 1)")
    print()
    print("[lorenz96]")
    for key, value in params.to_core_kwargs().items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()


def _print_tensor_summary(label: str, path: str, tensor: torch.Tensor) -> None:
    print(
        f"{label}: {path}\n"
        f"  shape: {tuple(tensor.shape)}  dtype: {tensor.dtype}  "
        f"min/max: {tensor.min().item():.6g}/{tensor.max().item():.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Lorenz 96 simulation and save paths to datasets/LORENZ96/full_path/path_*.pt"
    )
    parser.add_argument("--config", type=str, default="configs/turbulence/lorenz96_default.yaml")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--output_interval", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--N_path", type=int, default=None)
    parser.add_argument("--T_burn_in", type=int, default=None)
    parser.add_argument("--T_total", type=int, default=None)
    parser.add_argument(
        "--save_burn_in",
        type=int,
        default=None,
        help="0/1 override: save burn-in segment to path_{i}_burn_in.pt",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Parent folder under datasets/ (default: generator.data_root in config)",
    )
    parser.add_argument(
        "--trajectory_subdir",
        type=str,
        default=None,
        help="Subfolder under data_root for path_*.pt (default: generator.trajectory_subdir)",
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--F", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["single", "double"])
    parser.add_argument(
        "--compute_norm_stats",
        action="store_true",
        help="After saving raw paths, run dataset-wide min/max stats -> norm_stats.pt",
    )
    parser.add_argument(
        "--per_channel_stats",
        action="store_true",
        help="With --compute_norm_stats, use per-channel vmin/vmax",
    )
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    gen_cfg = dict(cfg.get("generator", {}))
    l96_cfg = dict(cfg.get("lorenz96", {}))

    K = int(args.K if args.K is not None else gen_cfg.get("K", 40))
    N_path = int(args.N_path if args.N_path is not None else gen_cfg.get("N_path", 1))
    T_burn_in = int(args.T_burn_in if args.T_burn_in is not None else gen_cfg.get("T_burn_in", 500))
    T_total = int(args.T_total if args.T_total is not None else gen_cfg.get("T_total", 5000))
    save_burn_in = bool(
        int(args.save_burn_in)
        if args.save_burn_in is not None
        else gen_cfg.get("save_burn_in", False)
    )
    output_interval = float(
        args.output_interval
        if args.output_interval is not None
        else gen_cfg.get("output_interval", 0.05)
    )
    seed = int(args.seed if args.seed is not None else gen_cfg.get("seed", 0))

    if args.device is not None:
        l96_cfg["device"] = args.device
    if args.dt is not None:
        l96_cfg["dt"] = float(args.dt)
    if args.F is not None:
        l96_cfg["F"] = float(args.F)
    if args.precision is not None:
        l96_cfg["precision"] = args.precision

    params = Lorenz96Params(K=K, **l96_cfg)
    params.validate()

    data_root = args.data_root if args.data_root is not None else gen_cfg.get("data_root")
    trajectory_subdir = (
        args.trajectory_subdir
        if args.trajectory_subdir is not None
        else gen_cfg.get("trajectory_subdir")
    )
    out_dir = resolve_trajectory_output_dir(
        gen_cfg,
        out_dir=args.out_dir,
        data_root=data_root,
        trajectory_subdir=trajectory_subdir,
    )
    summary_root = data_root if data_root is not None else gen_cfg.get("data_root", "(see output directory)")
    summary_subdir = (
        trajectory_subdir
        if trajectory_subdir is not None
        else gen_cfg.get("trajectory_subdir", "(see output directory)")
    )

    os.makedirs(out_dir, exist_ok=True)

    timesteps_per_output = max(1, int(output_interval / params.dt))

    compute_norm_stats = bool(
        args.compute_norm_stats or gen_cfg.get("compute_norm_stats", True)
    )

    _print_run_summary(
        config_path=args.config,
        data_root=str(summary_root),
        trajectory_subdir=str(summary_subdir),
        out_dir=out_dir,
        K=K,
        N_path=N_path,
        T_burn_in=T_burn_in,
        T_total=T_total,
        save_burn_in=save_burn_in,
        output_interval=output_interval,
        timesteps_per_output=timesteps_per_output,
        seed=seed,
        compute_norm_stats=compute_norm_stats,
        dry_run=args.dry_run,
        params=params,
    )

    dtype = torch.float32 if params.precision == "single" else torch.float64

    for path_idx in range(N_path):
        path_seed = seed + path_idx
        torch.manual_seed(path_seed)
        np.random.seed(path_seed)

        x = _make_initial_condition(K, params.F, device=params.device, dtype=dtype)
        model = Lorenz96Solver(x, **params.to_core_kwargs())

        burn_all = None
        if save_burn_in:
            burn_frames = [_frame_tensor(model.fields()["x"].detach().cpu(), K)]

        if T_burn_in > 0:
            for _ in tqdm(
                range(T_burn_in),
                desc=f"Burn-in path {path_idx + 1}/{N_path}",
            ):
                model.step(timesteps_per_output)
                if save_burn_in:
                    burn_frames.append(_frame_tensor(model.fields()["x"].detach().cpu(), K))

        if save_burn_in:
            burn_all = torch.stack(burn_frames, dim=0)

        x_all = [_frame_tensor(model.fields()["x"].detach().cpu(), K)]
        for _ in tqdm(
            range(T_total),
            desc=f"Generating path {path_idx + 1}/{N_path}",
        ):
            t1 = time.time()
            model.step(timesteps_per_output)
            _ = time.time() - t1
            x_frame = _frame_tensor(model.fields()["x"].detach().cpu(), K)
            x_all.append(x_frame)
            if not torch.isfinite(x_frame).all():
                raise RuntimeError(f"Non-finite values encountered (path={path_idx})")

        x_all = torch.stack(x_all, dim=0)

        if args.dry_run:
            _print_tensor_summary(
                f"[dry run] path {path_idx} (not saved)",
                os.path.join(out_dir, f"path_{path_idx}.pt"),
                x_all,
            )
            if save_burn_in and burn_all is not None:
                _print_tensor_summary(
                    f"[dry run] burn-in path {path_idx} (not saved)",
                    os.path.join(out_dir, f"path_{path_idx}_burn_in.pt"),
                    burn_all,
                )
            continue

        if save_burn_in and burn_all is not None:
            burn_path = os.path.join(out_dir, f"path_{path_idx}_burn_in.pt")
            torch.save(burn_all, burn_path)
            _print_tensor_summary(f"Saved burn-in path {path_idx}", burn_path, burn_all)

        out_path = os.path.join(out_dir, f"path_{path_idx}.pt")
        torch.save(x_all, out_path)
        _print_tensor_summary(f"Saved path {path_idx}", out_path, x_all)

    if args.dry_run:
        print("Dry run complete: no files written.")
        return

    if compute_norm_stats:
        stats = compute_and_save_norm_stats(
            out_dir,
            per_channel=args.per_channel_stats,
            ignore_burn_in=True,
        )
        print(
            f"Wrote norm_stats.pt (vmin={stats['vmin']:.6g}, vmax={stats['vmax']:.6g}, "
            f"n_files={stats['n_files']})"
        )
    else:
        print(
            "Next step: python scripts/Lorenz96/compute_norm_stats.py "
            f"--path_dir {out_dir}"
        )


if __name__ == "__main__":
    main()
