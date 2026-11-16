# SQG raw-data pipeline

SQG is a 2D application of the generic dense-field dataloader. The explicit SQG
reader validates its physical-scale dictionary and supplies SourceData:
fields [T,C,H,W], recorded times [T], time_unit and metadata.
Native resolution is fields.shape[2:], not a loader constant.

## Clean-clone workflow

After preparing/activating condiff, run individual direct Python commands from
bash_cmd_dataloader_examples. This file is a command reference, not a Bash script:

```text
python -m pip install -e . --no-build-isolation
python scripts/turb_SQG/sqg_raw_gen.py --config configs/turbulence/sqg_generation_smoke.yaml
python scripts/turb_SQG/sqg_inspect_example.py tests/test_dataloader/examples/example_01_state_1to1_lowres/dataset.yaml --index 0 --normalization-mode fit --diagnostic-stats-dir datasets/_dataloader_smoke/inspection
```

Generate once; do not overwrite existing scientific source data by default.
The inspection command explicitly requests fresh fitting in an isolated smoke
statistics directory. The CLI defaults to **read-only frozen**, regardless of the
experiment YAML's `fit` default, and explicit fit requires a diagnostic directory.
Dataset construction then automatically fits/saves enabled transforms; **no preceding
sqg_fit_normalization command is required**. That script remains an optional standalone utility with its explicit
overwrite/dry-run behavior, not the standard prerequisite.

The five small examples cover:
1. STATE + STATE -> STATE, 1->1.
2. STATE + STATE -> STATE, 2->1.
3. STATE + STATE -> DELTA, same sampling as 02.
4. DELTA + DELTA -> DELTA, same sampling as 02.
5. STATE + STATE -> DELTA, four unequal-length trajectories pooled together.

All use periodic random patches and global LR target conditions. Full/grid
remain generic capabilities, not new SQG scientific experiments.

## Configuration

Copy configs/dataset_kwargs/sqg_default.yaml to a complete experiment YAML.
Sections are source, temporal, task, spatial, low_res_target, normalization;
top-level normalization_mode is fit by default or explicit frozen.
See [dataset-config-reference.md](dataset-config-reference.md) for all options.

raw_channels=[0] selects top; [1] bottom; [0,1] both in that order. These are
physical variables, not history/target frames. The generic core does not infer
physical channel meaning when metadata is missing.

## Grouped inspection

The CLI reports candidates, selected n_t, spatial n_x, len(dataset)=n_t,
num_patches=n_t*n_x, shapes, normalization mode/paths and provenance.
For real Run A/B inspection, use `--normalization-mode frozen` to read original
experiment stats, or `--normalization-mode fit --diagnostic-stats-dir
 datasets/_notebook_diagnostics/cli-inspection` to recompute without overwriting
original files. Missing frozen stats raise an error without fallback.
--index selects a sampled temporal group, not raw frame number.
dataset[i] returns all n_x patches; get_patch(i,j) retrieves one footprint.

Fields remain [n_x,time,C,H,W]; DataLoader adds B temporal groups. Coordinates
are integer global-grid locations. Source/frame/physical-time metadata is shared
once; patch origins/wrapping are [n_x,2] tensors.

The old --training-check reports unsupported before constructing/fitting a Dataset.
Training integration, two-tensor compatibility and inference are deferred.
Historical _train YAMLs are not current grouped training entry points.
Network, loss, training loop and the old 4_test_dataloader.py are unchanged.

## Normalization and experiment safety

fit mode always uses complete selected physical states and source-local unique
frame-stride DELTA pairs, before crops/degradation and independent of sampled windows.
STATE/DELTA are independent transforms shared across sources; files are overwritten
at initialization and parameters passed directly into the runtime normalizers.
Retrieval never refits. Disabled transforms do not read/write files.

frozen mode explicitly loads matching original statistics without refitting/writing.
Use it for prior checkpoint evaluation or fixed-training-stat validation. A saved
YAML is not a copy/content fingerprint of its raw files or stats; preserve those
artifacts separately. Do not fit over files belonging to a running experiment.
Each independent constructor has the fitting cost and file side effects; there is
no fit-once process cache or distributed synchronization.

## Visualization responsibilities

- [dataloader_final_check.ipynb](../visualization/dataloader_final_check.ipynb):
  synthetic 2D/3D grouped patches, provenance, global LR alignment and plots.
- [dataloader_normalization_synthetic.ipynb](../visualization/dataloader_normalization_synthetic.ipynb):
  independent two-channel standard/minmax global/per-channel fit/frozen/refit.
- [dataloader_legacy_structured_check.ipynb](../visualization/dataloader_legacy_structured_check.ipynb):
  native structured ten-batch InfiniteSampler checks, no training adapter.
- [sqg_dataloader_output_visualization.ipynb](../visualization/sqg_dataloader_output_visualization.ipynb):
  existing real SQG grouped branches, 2×2 extended periodic footprints,
  source/frame/recorded-time metadata and structured batching.
- [sqg_dataloader_normalization.ipynb](../visualization/sqg_dataloader_normalization.ipynb):
  existing real SQG independent statistics, distributions, round trips and
  fixed/adaptive DELTA magnitude views.

Real notebooks expose RUN A/B, a complete dataset YAML, normalization mode,
temporal/patch/channel indices. Fresh fits are redirected to diagnostic config-copy
paths under datasets/_notebook_diagnostics/, not original training stats.
frozen uses original YAML paths read-only. Missing sources/statistics give clear
prerequisite errors; there is no generation or synthetic preview fallback.

The former mixed sampling/normalization/inference notebook has been removed after
its useful loader diagnostics were separated. Older copies remain in Git history.
None of the active notebooks launches training/inference.

## Shared real data and two 3->2 runs

`configs/turbulence/sqg_generation_real_128.yaml` specifies four independent
128x128 trajectories with 505 saved states each. Both physical SQG channels are
saved; datasets default to top [0]. Output spacing is 3600 s (one hour), internal
integration dt=300 s (12 RK4 steps/output), and burn-in is 50 intervals=50 hours.
The old 12-hour-spacing template's 50 intervals meant 600 hours. This is a
shorter spin-up, not a claim of an equally developed/statistically stationary flow.

The existing generator rejects non-finite fields/times. Four seeds passed a
short CPU check of 50-hour spin-up plus six recorded hours at dt=300. Seed 0
also agrees with dt=150 to relative L2 difference about 1.1e-5 in that window.
This does NOT guarantee stability for all 505 frames or other parameters. If
unstable, reduce integration dt, regenerate to a new directory, and refit stats;
changing saved-frame spacing alone is not a stability fix.

Run A uses STATE history/LR to predict DELTA; B uses DELTA history/LR to predict
DELTA. Both have m=3,n=2,C=1,n_t=200,n_x=3, periodic 32x32 patches and global
128->32->128 LR. Each Dataset has 200 groups and 600 individual patches.
Batched history is [B,3,3,1,32,32], targets [B,3,2,1,32,32].
These are structured Dataset shapes, **not a verified model training interface**.

A fits its own STATE/DELTA files; B disables STATE and independently fits DELTA.
Matching numerical populations do not imply reusing experiment files.
The command reference retains real generation and grouped inspection, not active
long-training commands until training integration is addressed separately.
