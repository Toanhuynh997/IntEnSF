# Dataset configuration reference

Use configs/dataset_kwargs/sqg_default.yaml as a complete SQG template, or
generic_2d.yaml for flat generic file-reader options. There is no implicit master
YAML inheritance when loading a standalone structured YAML.

## Source and reader boundary

SQGDataset2D uses nested source.sources and source.raw_channels (default top [0]).
GenericFileDataset2D/3D uses flat sources, source_format and channel_indices.
Generic npy/npz/pt is an explicit reader choice; generic PT accepts dictionaries
with configurable values_key and optional times_key, not bare tensors.
The separate SQG reader validates fields/times/metadata and physical raw scale.

Each source selection has half-open time_start/time_stop; omission/null stop uses
the source length. Every selected source must provide at least one valid task
window. Homogeneous sources need matching selected channels/spatial geometry.
Optional physical names/units are validated when both provide them; otherwise
consistent physical meaning/order is assumed from data/configuration.

## Temporal task

```yaml
temporal:
  n_hist: 3
  n_target: 2
  frame_stride: 1
  time_policy: require_matching_interval
  sampling:
    num_temporal_samples: 200
    seed: 17
```

Generic equivalents are flat n_hist/n_target/frame_stride/time_policy,
num_temporal_samples and temporal_seed.
n_hist=m and n_target=n are model-facing lengths; DELTA history reads m+1 states.
frame_stride counts source frames, not seconds. Internal prediction_gap is fixed
to zero. No generic n_target<=n_hist restriction is imposed.
Sampling is once without replacement from pooled valid windows, not per-source
quotas or per-epoch draws. Null/missing count retains all candidates; excessive N raises.

recorded allows all sources to lack times, omitting physical-time metadata, or all
to have recorded times (even irregular). Mixed availability is rejected.
require_matching_interval requires recorded regular, mutually matching intervals.
Sampling remains index-based, never driven by requested elapsed physical times.

## Task representation

task.condition_representation and task.target_representation support:
state/state, state/delta and delta/delta. The condition representation controls
both hist_hi_res and target_lo_res; target representation controls target_hi_res.
Coordinates/metadata are provenance, not additional model features.

## Spatial sampling and resolution

```yaml
spatial:
  mode: patch
  patch:
    shape: [64, 64]
    output_shape: null
    resize_mode: nearest
    sampling:
      mode: random
      random:
        num_spatial_samples: 3
        seed: 11
      grid:
        stride: [64, 64]
  boundary: periodic
low_res_target:
  enabled: true
  low_res_shape: [32, 32]
  downsample_mode: area
  upsample_mode: nearest
```

shape defines a physical crop; output_shape=null preserves crop resolution.
A full-field returned resize uses spatial.mode=full and spatial.full.output_shape.
Thus cropping 64x64 and resizing the global field to 64x64 are distinct operations.
resize_mode follows the existing rank-specific interpolation options.

For random patches n_x=num_spatial_samples. Full derives n_x=1; grid derives
n_x from the enumerated geometry. Generic 3D accepts three-axis shapes.
Grid requires valid boundary and divisibility of field_size-patch_size by stride.
Stride smaller/equal/larger than patch size means overlap/adjacency/gaps.
Random temporal/spatial seeds are independent.

low_res_shape controls **global** downsample/upsample back to source geometry,
then same-footprint crop. It does not set patch size or LR temporal length:
target_lo_res always has n_target frames and the same returned spatial shape as HR.

## Normalization

```yaml
normalization_mode: fit
normalization:
  state:
    method: standard
    scope: per_channel
    stats_path: SQG/my_experiment_state_stats_v2.pt
  delta:
    method: standard
    scope: per_channel
    stats_path: SQG/my_experiment_delta_stats_v2.pt
```

normalization_mode is a top-level argument in generic and nested SQG configuration.
fit (default): each constructor recomputes enabled statistics from the current
complete source selections and overwrites files. No separate fitting command is
required; runtime only applies the fitted fixed transforms.
frozen: explicit read-only file loading for existing experiment evaluation; missing
or incompatible files raise. Do not refit historical checkpoint statistics.

None or method:none needs no path and performs no fit/write.
Methods: none, standard (mean/population std), minmax (fitted interval -> [-1,1],
no clipping). Scopes: global (pool selected channels), per_channel (one stat per
selected channel). percentile retains the existing minmax quantile behavior.

STATE fits all selected source states/global cells. DELTA fits every within-source
frame-stride pair once, not target-only windows or patch-weighted populations.
n_t, n_x and their seeds never change the fit population. Selections must explicitly
exclude validation/test ranges when required. STATE and DELTA use separate paths;
Run A/B use independent experiment files even if fitted values happen to match.

Enabled but unused transforms are still fitted when configured: disable STATE for
a DELTA-only task, and disable DELTA for a STATE-only task.
Relative stats/source paths resolve below datasets/; absolute paths are supported.
Individual files are atomically replaced, not jointly transactional or archived.
Multiple constructors/ranks each refit; no caching or DDP coordination is implied.

Version-2 delta files require delta_definition=consecutive,
fit_population=selected_source_deltas and matching temporal_task.frame_stride.
Older or target-window-only files are rejected in frozen mode and must be refitted
using the corresponding task/config. Historical results/checkpoints need no edits.

## Cardinality, items and metadata

```text
num_temporal_candidates    all valid pooled windows before temporal sampling
n_t                       selected temporal windows = len(dataset)
n_x                       patches per selected temporal window
num_patches               derived read-only n_t*n_x
```

num_samples is removed (no redundant alias); old max_size is not part of this family.
dataset[i] returns a dict with:
hist_hi_res [n_x,m,C,*spatial], target_hi_res [n_x,n,C,*spatial],
optional target_lo_res [n_x,n,C,*spatial], coord [n_x,spatial_rank,*spatial],
and shared temporal/per-patch spatial metadata.
DataLoader adds B groups. get_patch(i,j) removes only the n_x axis.
Spatial metadata uses tensors [n_x,spatial_rank]; source/frame identity is shared
once. Recorded times are omitted when absent, never fabricated.

## Canonical configurations and notebooks

Five complete small SQG examples are under tests/test_dataloader/examples/:
01 state 1->1, 02 state 2->1, 03 state->delta 2->1, 04 delta->delta 2->1,
05 pooled four-source state->delta. All enable LR target conditions; 02-04 retain
aligned sampling so representation changes can be compared.

Real structured YAMLs are sqg_real_state_to_delta_3to2.yaml (A) and
sqg_real_delta_to_delta_3to2.yaml (B). Four 505-frame sources produce 2004/2000
candidates respectively; each selects n_t=200 groups, n_x=3, num_patches=600.
Use the complete non-_train YAML with the inspection CLI and real notebooks.

The _train Hydra wrappers and run configs remain historical interface references.
The grouped training adapter is **not integrated this round** and rejects grouped
items. Do not use old training commands to claim current compatibility.
scripts/code_tests/4_test_dataloader.py is unchanged. Native structured
batch inspection with InfiniteSampler is demonstrated separately in
visualization/dataloader_legacy_structured_check.ipynb, without a training adapter.
The inspection CLI defaults to frozen; explicit fit requires
`--normalization-mode fit --diagnostic-stats-dir <directory>`, avoiding accidental
overwrites of original Run A/B statistics. The five smoke-example commands use
their own diagnostic directory, while all scientific YAMLs retain fit as the
Dataset-construction default.

Notebook responsibilities and normalization side effects are documented in
[data-loader-design.md](data-loader-design.md#verification-and-notebooks).
