# SQG data generation (paths)

This folder contains a lightweight SQG forward-model data generator and a notebook to visualize the saved trajectories.

## Setup

Install the repo as an editable package (recommended), so `project_utils` and `turbulence` imports work:

```bash
pip install -e .
```

If you don’t want to install editable, you can also run with:

```bash
PYTHONPATH=src python scripts/SQG/generate_fields.py
```

## Config

Default config: `configs/turbulence/sqg_default.yaml`

Key generator fields:

- `generator.data_root` / `generator.trajectory_subdir`: output under `datasets/<data_root>/<trajectory_subdir>/` (set in YAML; no code defaults)
- `generator.N_path`: number of independent trajectories to generate
- `generator.T_burn_in`: number of output frames to discard (spin-up)
- `generator.T_total`: number of output frames to keep
- `generator.save_burn_in`: if true, also save `path_{i}_burn_in.pt`

## Generate paths (default)

Saves **raw** PV in physical units (`path_*.pt`). Normalization is **not** applied at generation time.

```bash
python scripts/SQG/generate_fields.py
```

## Normalization stats (for training / inference)

By default, `generate_fields.py` also writes `norm_stats.pt` after all paths are saved.
To recompute stats without re-simulating:

```bash
python scripts/SQG/compute_norm_stats.py
```

To skip stats during generation, set `compute_norm_stats: false` in `sqg_default.yaml`.

The dataloader (`DatasetPathCond` with `apply_normalization: true`) loads `norm_stats.pt` from
the trajectory folder and maps samples to `[-1, 1]`. Configure paths with `data_root` +
`trajectory_subdir` in `configs/dataset_kwargs/sqg_path_next.yaml` (or legacy `path_dir`).
Use `project_utils.normalization.denormalize_tensor` to map NN outputs back to physical PV.

Training smoke test:

```bash
PYTHONPATH=src python scripts/code_tests/4_test_dataloader.py runs=sqg_path_next
```

### Dataset layout under `datasets/SQG/`


| Subfolder              | Contents                                                                   |
| ---------------------- | -------------------------------------------------------------------------- |
| `full_path/`           | Full trajectories from `generate_fields.py`: `path_0.pt`, `path_1.pt`, …   |
| *(future)* `pairs_* /` | Preprocessed training pairs (zip or `.pt` samples), e.g. for `DatasetCond` |


Outputs from the generator are saved under `datasets/SQG/full_path/` as:

- `datasets/SQG/full_path/path_0.pt`
- `datasets/SQG/full_path/path_1.pt`
- ...

Each file contains a tensor shaped `(T_total + 1, 2, Nx, Nx)` (includes the initial frame at index 0).

## Common option examples

Small quick run (CPU):

```bash
python scripts/SQG/generate_fields.py \
  --Nx 16 --N_path 1 --T_burn_in 5 --T_total 10 \
  --dt 10 --diff_efold 1000 --output_interval_hours 0.01 \
  --dry_run
```

GPU run(much faster):

```bash
python scripts/SQG/generate_fields.py --device cuda:0
```

Custom output directory:

```bash
python scripts/SQG/generate_fields.py --out_dir datasets/SQG_debug
```

Or use split paths (matches training config):

```bash
python scripts/SQG/generate_fields.py --data_root SQG --trajectory_subdir full_path
python scripts/SQG/compute_norm_stats.py --data_root SQG --trajectory_subdir full_path
```

Use a non-default config:

```bash
python scripts/SQG/generate_fields.py --config configs/turbulence/sqg_default.yaml
```

Save burn-in segment:

```bash
python scripts/SQG/generate_fields.py --save_burn_in 1
```

## Hybrid solver + diffusion path

Open `[scripts/SQG/hybrid_solver_dm_path.ipynb](hybrid_solver_dm_path.ipynb)`: SQG burn-in, then parallel rollouts from the same post-burn-in state (true solver vs conditional diffusion) with error plots and comparison animation.

## Visualize

Open `scripts/SQG/visualize_fields.ipynb` and run the cells. Default load path:

- `datasets/SQG/full_path/path_0.pt`

Update the notebook `path = ...` line if your trajectories live elsewhere.

## Data assimilation demos

| Notebook | Method |
|----------|--------|
| [`da_dps_dm_demo.ipynb`](da_dps_dm_demo.ipynb) | True-solver burn-in; **ensemble DPS** (`NE` in config); sparse observation + batched DM forward |
| [`da_ensf_demo.ipynb`](da_ensf_demo.ipynb) | Minimal **EnSF** on the true SQG forward model |