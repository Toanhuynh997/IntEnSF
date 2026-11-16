import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from project_utils.normalization import compute_and_save_norm_stats
from project_utils.trajectory_paths import resolve_trajectory_output_dir
from project_utils.yaml_simple import load_simple_yaml
from turbulence.lorenz63 import Lorenz63Params, Lorenz63Solver

STATE_DIM = 3


def _frame_tensor(x: torch.Tensor) -> torch.Tensor:
    """Map state (3,) to dataloader layout (1, 3, 1)."""
    return x.reshape(1, STATE_DIM, 1)


def _make_initial_condition(
    x0: list[float] | tuple[float, ...] | torch.Tensor,
    *,
    device,
    dtype: torch.dtype,
    perturb_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Classic L63 IC (default (0,1,0)) with optional per-path Gaussian noise."""
    if isinstance(x0, torch.Tensor):
        x = x0.to(device=device, dtype=dtype).reshape(STATE_DIM).clone()
    else:
        if len(x0) != STATE_DIM:
            raise ValueError(f"x0 must have length {STATE_DIM}, got {len(x0)}")
        x = torch.tensor(list(x0), device=device, dtype=dtype)
    if perturb_std > 0:
        noise = torch.randn(
            STATE_DIM, generator=generator, dtype=dtype, device=device
        ) * float(perturb_std)
        # Generator may live on CPU; move noise if needed.
        x = x + noise.to(device=device, dtype=dtype)
    return x


def _print_run_summary(
    *,
    config_path: str,
    data_root: str,
    trajectory_subdir: str,
    out_dir: str,
    N_path: int,
    T_burn_in: int,
    T_total: int,
    save_burn_in: bool,
    output_interval: float,
    timesteps_per_output: int,
    seed: int,
    x0: list[float],
    ic_perturb_std: float,
    compute_norm_stats: bool,
    dry_run: bool,
    params: Lorenz63Params,
) -> None:
    print("=" * 60)
    print("Lorenz 63 trajectory generation (DA benchmark)")
    print("=" * 60)
    print(f"config:                  {config_path}")
    print(f"data_root:               {data_root}")
    print(f"trajectory_subdir:       {trajectory_subdir}")
    print(f"output directory:        {out_dir}")
    print(f"dry_run:                 {dry_run}")
    print()
    print("[generator]")
    print(f"  STATE_DIM:             {STATE_DIM}")
    print(f"  N_path:                {N_path}")
    print(f"  T_burn_in:             {T_burn_in}")
    print(f"  T_total:               {T_total}")
    print(f"  save_burn_in:          {save_burn_in}")
    print(f"  output_interval:       {output_interval}")
    print(f"  timesteps_per_output:  {timesteps_per_output}")
    print(f"  seed:                  {seed}")
    print(f"  x0:                    {x0}")
    print(f"  ic_perturb_std:        {ic_perturb_std}")
    print(f"  save format:           raw (x,y,z), shape (T, 1, 3, 1)")
    print(f"  compute_norm_stats:    {compute_norm_stats}")
    print(f"  expected path shape:   ({T_total + 1}, 1, {STATE_DIM}, 1)")
    if save_burn_in:
        print(f"  expected burn-in shape: ({T_burn_in + 1}, 1, {STATE_DIM}, 1)")
    print()
    print("[lorenz63]")
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
        description=(
            "Run Lorenz 63 simulation (classic DA benchmark) and save paths to "
            "datasets/LORENZ63/full_path/path_*.pt"
        )
    )
    parser.add_argument("--config", type=str, default="configs/turbulence/lorenz63_default.yaml")
    parser.add_argument("--dry_run", action="store_true")
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
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["single", "double"])
    parser.add_argument(
        "--ic_perturb_std",
        type=float,
        default=None,
        help="Std of Gaussian IC perturbation per path (0 = identical IC)",
    )
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
    l63_cfg = dict(cfg.get("lorenz63", {}))

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
        else gen_cfg.get("output_interval", 0.01)
    )
    seed = int(args.seed if args.seed is not None else gen_cfg.get("seed", 0))
    x0 = list(gen_cfg.get("x0", [0.0, 1.0, 0.0]))
    ic_perturb_std = float(
        args.ic_perturb_std
        if args.ic_perturb_std is not None
        else gen_cfg.get("ic_perturb_std", 0.0)
    )

    if args.device is not None:
        l63_cfg["device"] = args.device
    if args.dt is not None:
        l63_cfg["dt"] = float(args.dt)
    if args.sigma is not None:
        l63_cfg["sigma"] = float(args.sigma)
    if args.rho is not None:
        l63_cfg["rho"] = float(args.rho)
    if args.beta is not None:
        l63_cfg["beta"] = float(args.beta)
    if args.precision is not None:
        l63_cfg["precision"] = args.precision

    params = Lorenz63Params(**l63_cfg)
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

    timesteps_per_output = max(1, int(round(output_interval / params.dt)))

    compute_norm_stats = bool(
        args.compute_norm_stats or gen_cfg.get("compute_norm_stats", True)
    )

    _print_run_summary(
        config_path=args.config,
        data_root=str(summary_root),
        trajectory_subdir=str(summary_subdir),
        out_dir=out_dir,
        N_path=N_path,
        T_burn_in=T_burn_in,
        T_total=T_total,
        save_burn_in=save_burn_in,
        output_interval=output_interval,
        timesteps_per_output=timesteps_per_output,
        seed=seed,
        x0=x0,
        ic_perturb_std=ic_perturb_std,
        compute_norm_stats=compute_norm_stats,
        dry_run=args.dry_run,
        params=params,
    )

    dtype = torch.float32 if params.precision == "single" else torch.float64

    for path_idx in range(N_path):
        path_seed = seed + path_idx
        torch.manual_seed(path_seed)
        np.random.seed(path_seed)
        g = torch.Generator()
        g.manual_seed(path_seed)

        x = _make_initial_condition(
            x0,
            device=params.device,
            dtype=dtype,
            perturb_std=ic_perturb_std,
            generator=g,
        )
        model = Lorenz63Solver(x, **params.to_core_kwargs())

        burn_all = None
        if save_burn_in:
            burn_frames = [_frame_tensor(model.fields()["x"].detach().cpu())]

        if T_burn_in > 0:
            for _ in tqdm(
                range(T_burn_in),
                desc=f"Burn-in path {path_idx + 1}/{N_path}",
            ):
                model.step(timesteps_per_output)
                if save_burn_in:
                    burn_frames.append(_frame_tensor(model.fields()["x"].detach().cpu()))

        if save_burn_in:
            burn_all = torch.stack(burn_frames, dim=0)

        x_all = [_frame_tensor(model.fields()["x"].detach().cpu())]
        for _ in tqdm(
            range(T_total),
            desc=f"Generating path {path_idx + 1}/{N_path}",
        ):
            t1 = time.time()
            model.step(timesteps_per_output)
            _ = time.time() - t1
            x_frame = _frame_tensor(model.fields()["x"].detach().cpu())
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
            "Next step: python scripts/Lorenz63/compute_norm_stats.py "
            f"--path_dir {out_dir}"
        )


if __name__ == "__main__":
    main()
