import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from project_utils.normalization import compute_and_save_norm_stats
from project_utils.trajectory_paths import resolve_trajectory_output_dir
from project_utils.yaml_simple import load_simple_yaml
from turbulence.sqg import SQGParams, SQGSolver


def _print_run_summary(
    *,
    config_path: str,
    data_root: str,
    trajectory_subdir: str,
    out_dir: str,
    Nx: int,
    N_path: int,
    T_burn_in: int,
    T_total: int,
    save_burn_in: bool,
    output_interval_hours: float,
    timesteps_per_output: int,
    seed: int,
    compute_norm_stats: bool,
    dry_run: bool,
    params: SQGParams,
) -> None:
    print("=" * 60)
    print("SQG field generation")
    print("=" * 60)
    print(f"config:                  {config_path}")
    print(f"data_root:               {data_root}")
    print(f"trajectory_subdir:       {trajectory_subdir}")
    print(f"output directory:        {out_dir}")
    print(f"dry_run:                 {dry_run}")
    print()
    print("[generator]")
    print(f"  Nx:                    {Nx}")
    print(f"  N_path:                {N_path}")
    print(f"  T_burn_in:             {T_burn_in}")
    print(f"  T_total:               {T_total}")
    print(f"  save_burn_in:          {save_burn_in}")
    print(f"  output_interval_hours: {output_interval_hours}")
    print(f"  timesteps_per_output:  {timesteps_per_output}")
    print(f"  seed:                  {seed}")
    print(f"  save format:           raw PV (physical units)")
    print(f"  compute_norm_stats:    {compute_norm_stats}")
    print(f"  expected path shape:   ({T_total + 1}, 2, {Nx}, {Nx})")
    if save_burn_in:
        print(f"  expected burn-in shape: ({T_burn_in + 1}, 2, {Nx}, {Nx})")
    print()
    print("[sqg]")
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
        description="Run SQG turbulence simulation and save paths to datasets/SQG/full_path/path_*.pt"
    )
    parser.add_argument("--config", type=str, default="configs/turbulence/sqg_default.yaml")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--Nx", type=int, default=None)
    parser.add_argument("--output_interval_hours", type=float, default=None)
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

    # Optional overrides (otherwise pulled from config file)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--diff_efold", type=float, default=None)
    parser.add_argument("--diff_order", type=int, default=None)
    parser.add_argument("--dealias", type=int, default=None, help="0/1 override for dealias")
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
    sqg_cfg = dict(cfg.get("sqg", {}))

    Nx = int(args.Nx if args.Nx is not None else gen_cfg.get("Nx", 128))
    N_path = int(args.N_path if args.N_path is not None else gen_cfg.get("N_path", 1))
    T_burn_in = int(args.T_burn_in if args.T_burn_in is not None else gen_cfg.get("T_burn_in", 50))
    T_total = int(args.T_total if args.T_total is not None else gen_cfg.get("T_total", 1000))
    save_burn_in = bool(
        int(args.save_burn_in)
        if args.save_burn_in is not None
        else gen_cfg.get("save_burn_in", False)
    )
    output_interval_hours = float(
        args.output_interval_hours
        if args.output_interval_hours is not None
        else gen_cfg.get("output_interval_hours", 12.0)
    )
    seed = int(args.seed if args.seed is not None else gen_cfg.get("seed", 0))

    # Apply SQG overrides from CLI if provided.
    if args.device is not None:
        sqg_cfg["device"] = args.device
    if args.dt is not None:
        sqg_cfg["dt"] = float(args.dt)
    if args.diff_efold is not None:
        sqg_cfg["diff_efold"] = float(args.diff_efold)
    if args.diff_order is not None:
        sqg_cfg["diff_order"] = int(args.diff_order)
    if args.dealias is not None:
        sqg_cfg["dealias"] = bool(int(args.dealias))
    if args.precision is not None:
        sqg_cfg["precision"] = args.precision

    params = SQGParams(**sqg_cfg)
    N = int(Nx)
    params.validate(N=N)

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

    outputinterval = output_interval_hours * 3600.0
    timesteps_per_output = max(1, int(outputinterval / params.dt))

    compute_norm_stats = bool(
        args.compute_norm_stats or gen_cfg.get("compute_norm_stats", True)
    )

    _print_run_summary(
        config_path=args.config,
        data_root=str(summary_root),
        trajectory_subdir=str(summary_subdir),
        out_dir=out_dir,
        Nx=N,
        N_path=N_path,
        T_burn_in=T_burn_in,
        T_total=T_total,
        save_burn_in=save_burn_in,
        output_interval_hours=output_interval_hours,
        timesteps_per_output=timesteps_per_output,
        seed=seed,
        compute_norm_stats=compute_norm_stats,
        dry_run=args.dry_run,
        params=params,
    )

    for path_idx in range(N_path):
        path_seed = seed + path_idx
        torch.manual_seed(path_seed)
        np.random.seed(path_seed)

        # Initial condition: random PV + blob (same idea as scripts/turb_SQG/sqg_raw_gen.py)
        pv = torch.randn(2, N, N, dtype=torch.float32, device=params.device) * 100.0
        nexp = 20
        x = torch.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / N, device=params.device, dtype=torch.float32)
        y = torch.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / N, device=params.device, dtype=torch.float32)
        x, y = torch.meshgrid(x, y, indexing="xy")
        pv[1] = pv[1] + 2000.0 * (torch.sin(x / 2.0) ** (2 * nexp) * torch.sin(y) ** nexp)
        pv = pv - pv.mean(dim=(1, 2), keepdim=True)

        model = SQGSolver(pv, **params.to_core_kwargs())

        burn_all = None
        if save_burn_in:
            burn_frames = [model.fields()["pv"].detach().cpu()]

        # Burn-in (discarded outputs), but show progress (and optionally save).
        if T_burn_in > 0:
            for _ in tqdm(
                range(T_burn_in),
                desc=f"Burn-in path {path_idx + 1}/{N_path}",
            ):
                model.step(timesteps_per_output)
                if save_burn_in:
                    burn_frames.append(model.fields()["pv"].detach().cpu())

        if save_burn_in:
            burn_all = torch.stack(burn_frames, dim=0)  # (T_burn_in+1, 2, N, N)

        # Saved rollout (kept outputs).
        pv_all = [model.fields()["pv"].detach().cpu()]
        for _ in tqdm(
            range(T_total),
            desc=f"Generating path {path_idx + 1}/{N_path}",
        ):
            t1 = time.time()
            model.step(timesteps_per_output)
            _ = time.time() - t1
            pv_frame = model.fields()["pv"].detach().cpu()
            pv_all.append(pv_frame)
            if not torch.isfinite(pv_frame).all():
                raise RuntimeError(f"Non-finite values encountered (path={path_idx})")

        pv_all = torch.stack(pv_all, dim=0)  # (T_total+1, 2, N, N)

        if args.dry_run:
            _print_tensor_summary(
                f"[dry run] path {path_idx} (not saved)",
                os.path.join(out_dir, f"path_{path_idx}.pt"),
                pv_all,
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
        torch.save(pv_all, out_path)
        _print_tensor_summary(f"Saved path {path_idx}", out_path, pv_all)

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
            "Next step: python scripts/SQG/compute_norm_stats.py "
            f"--path_dir {out_dir}"
        )


if __name__ == "__main__":
    main()

