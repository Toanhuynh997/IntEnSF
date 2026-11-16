# Dataloader design

The generic loaders consume dense regular grids: SourceData.values is [T,C,H,W]
or [T,C,H,W,D]. Application readers translate file formats, not the sampling
or scientific representation mathematics. SQG is one 2D application.

## Definitions and public contract

- m=n_hist, n=n_target: model-facing temporal lengths.
- C: selected physical channels in configured order.
- n_t: selected temporal windows. len(dataset)=n_t.
- n_x: spatial patches per window.
- num_patches: read-only property n_t*n_x, not Dataset length.
- spatial_rank: 2 or 3 spatial axes, not the number of tensor dimensions.

The redundant num_samples attribute is removed. No counter changes meaning in
a downstream view. Model-specific training integration is deferred.

One group = dataset[i]:

```text
2D:
hist_hi_res     [n_x,m,C,H_out,W_out]
target_lo_res   [n_x,n,C,H_out,W_out]    optional
target_hi_res   [n_x,n,C,H_out,W_out]
coord           [n_x,2,H_out,W_out]

3D:
hist_hi_res     [n_x,m,C,H_out,W_out,D_out]
target_lo_res   [n_x,n,C,H_out,W_out,D_out]
target_hi_res   [n_x,n,C,H_out,W_out,D_out]
coord           [n_x,3,H_out,W_out,D_out]
```

DataLoader prepends B to each tensor. **B counts temporal groups**, not patches.
For real 3->2, C=1, n_x=3, 32x32 outputs, batched history is
[B,3,3,1,32,32]; targets are [B,3,2,1,32,32].
No time, spatial-sample or physical-channel axis is merged by the Dataset.

Select a patch using group["hist_hi_res"][j], etc. Alternatively,
dataset.get_patch(i,j) computes only that footprint and removes the n_x axis
from fields and spatial metadata. It does not construct a training Dataset.

## Initialization and retrieval

Initialization validates selected sources/tasks, enumerates temporal candidates,
draws the fixed temporal subset, builds the spatial plan, and prepares normalizers.

Each retrieval:
1. Resolves one source-local temporal reference.
2. Selects required time frames **before** physical channels.
3. Builds physical GLOBAL history and HR target representations once.
4. Builds the optional condition representation and global down/up LR field once.
5. Crops each of the n_x shared footprints across all branches, creating coordinates.
6. Stacks patches and applies fixed representation-specific normalizers.
7. Returns grouped fields and source/time/spatial metadata.

The main flow stays explicit in dataset_2d.py / dataset_3d.py. Geometry and
validation remain in their existing modules; normalization preparation is shared
rather than copied into each rank. No eager full-Dataset materialization or
cross-access cache is introduced.

## Temporal sampling and representation

frame_stride is source-frame spacing, not solver dt or requested physical time.
With first target t and stride s, STATE history is
[t-m*s,...,t-s], targets [t,...,t+(n-1)*s].
DELTA history reads m+1 states and returns m consecutive differences.
Targets use x[t]-x[t-s], followed by consecutive target differences.
prediction_gap remains zero internally and is absent from normal YAMLs.

Enumerate valid windows separately per source, pool them, then sample n_t uniformly
without replacement. Longer trajectories contribute more candidates, not equal
quotas. Missing/null num_temporal_samples preserves all candidates in enumeration
order; excessive counts raise. The temporal draw is made once using temporal_seed.
Spatial seed is independent; access and epochs do not redraw times or footprints.

Supported tasks: STATE+STATE->STATE, STATE+STATE->DELTA, DELTA+DELTA->DELTA.
History and LR target use condition_representation; HR target uses
target_representation. LR always has n_target frames.

## Spatial geometry and resolution

Full mode has n_x=1; random uses num_spatial_samples; grid derives n_x from geometry.
Origins are corners on the original grid. valid keeps crops inside; periodic wraps
indices modulo the domain. Current grid sampling requires valid boundary and
(field_size-patch_size) divisible by stride on each axis.
Stride < patch size overlaps; equality gives adjacent non-overlapping patches;
larger strides leave gaps. These rules are unchanged.

The same footprint controls history, HR target, LR target and coord.
output_shape controls an optional post-crop resize, not the physical footprint.
Coordinates always report integer original-grid indices (nearest mapping if resized).

LR degradation is GLOBAL target condition -> downsample -> upsample back to original
global geometry -> shared crop. It is not independently downsampling each patch,
and is not the same as spatial_mode=full.

## Metadata and collation

Each group records source_index/source_id, temporal_sample_index,
target_start_index, hist_time_indices [m], target_time_indices [n],
channel_indices [C], representation names and boundary.
DELTA history additionally records hist_raw_time_indices [m+1].
SQG adds source_path. Source/time identity is not duplicated n_x times.

Spatial metadata is tensor-valued:
- spatial_sample_index [n_x]
- patch_origin / grid_index / wrapped [n_x,spatial_rank]

DataLoader therefore produces [B,n_x,...] spatial metadata and [B,m/n] frame
indices, rather than axis-major lists of origin tuples. Strings collate as lists.
get_patch(i,j) returns scalar spatial_sample_index and [spatial_rank] origins.

Recorded physical times/unit are included only if the source supplies them.
With time_policy=recorded all selected sources must consistently have or lack times;
frame-index sampling works in both cases, including irregular recorded times.
require_matching_interval requires real timestamps, regularity per source and a
matching interval across sources. Physical-time-driven frame selection is unsupported.
Physical channel meanings/order are an input contract; optional names/units are
checked when both sources provide them, not invented when absent.

## Normalization lifecycle

normalization_mode defaults to **fit** in both generic ranks and the SQG wrapper.
Every Dataset instance refits every enabled state/delta transform from its current
selections, overwrites the configured stats_path, and passes those same fitted
dictionaries directly to Normalizer. No immediate save/reload, cache reuse or
per-item/per-batch fit occurs. None / method:none fits and saves nothing.

The explicit **frozen** mode loads existing files without fitting or writing.
Use it for old checkpoint/validation evaluation where matching training statistics
must remain fixed. It is not a silent auto-reuse policy and does not choose files
or training splits for the user. Missing/incompatible files raise.
Never refit over statistics of a running or historical experiment.

STATE population: all selected raw states x selected channels x global cells.
DELTA population: all unique source-local pairs x[t+s]-x[t] within each selection,
where s=frame_stride, independent of history/target roles, sampled n_t/n_x and seeds.
Selections are explicit training ranges; no implicit train/test split is inferred.
Repeated/overlapping source entries are not automatically deduplicated.

standard uses mean/population std; minmax maps fitted bounds to [-1,1] without
clipping; global pools channels and per_channel separates them. Existing percentile
semantics and version-2 statistics schema are preserved.
Delta files must record consecutive definition, selected_source_deltas population,
and the matching temporal_task.frame_stride; old incompatible files must be refitted.

Statistics are written by atomic replacement of an individual file. STATE/DELTA
need different paths. This is not a two-file transaction or DDP coordination:
each independent constructor may fit again, and the latest complete write wins.
Use separate experiment paths. Files and raw sources referenced by a saved YAML
are not automatically archived or content-fingerprinted. Fit cost scales with full
selected sources; streaming accumulation is retained (trimmed quantiles may retain values).

## Module responsibilities

- types.py / file_sources.py / npz/source.py: canonical sources and generic readers.
- sqg/source.py and sqg/validation.py: explicit SQG schema/raw-scale boundary.
- temporal.py: source-local indices and fixed pooled temporal selection.
- representation.py: STATE/consecutive DELTA math.
- spatial_2d.py / spatial_3d.py: plans, footprints, coordinates and crops.
- resolution.py: global degradation.
- normalization.py: shared preparation, fitting, fixed transforms and persistence.
- validation.py: shared essential configuration/source/stats checks.
- dataset_2d.py / dataset_3d.py: readable orchestration of those operations.
- sqg/config.py / sqg/dataset.py: nested SQG configuration and wrapper.
- training_adapter.py: historical single-patch two-tensor helper; rejects groups.

## Verification and notebooks

[Synthetic grouped output](../visualization/dataloader_final_check.ipynb)
checks 2D/3D temporal grouping, get_patch(i,j), independent STATE/DELTA
and global LR alignment, and exact returned-field plots.
[Synthetic normalization](../visualization/dataloader_normalization_synthetic.ipynb)
checks two-channel standard/minmax × global/per_channel fitting, saved .pt files,
frozen read-only loading, broadcasting and fresh refits, for both spatial ranks.
[Native legacy-style batch check](../visualization/dataloader_legacy_structured_check.ipynb)
reproduces the ten-batch InfiniteSampler and finite/statistical inspection purpose
of unchanged scripts/code_tests/4_test_dataloader.py on the canonical structured
DataLoader output, not the old two-tensor training interface.

[Real SQG output](../visualization/sqg_dataloader_output_visualization.ipynb)
shows exact returned fields, source/time and spatial provenance, 2×2 periodic
extension with red footprint rectangles, coordinates and batched groups.
[Real SQG normalization](../visualization/sqg_dataloader_normalization.ipynb)
checks source statistics, output distributions, round trips, temporal DELTA RMS,
and fixed/adaptive selected DELTA visualizations.

Real notebooks redirect fit-mode statistics into independent diagnostic copies;
`frozen` reads original YAML statistics without fitting or writing. The inspection
CLI additionally defaults to frozen and requires --diagnostic-stats-dir for
explicit fit, protecting existing Run A/B statistics in the command cookbook.
All five notebooks can be run independently. None generates data, trains a
network or performs checkpoint inference. Older copies remain in Git history.

Training adapter integration and the two-tensor compatibility notebook are explicitly
deferred. The legacy helper can convert an explicitly selected get_patch(i,j) result,
but a grouped item raises instead of silently merging n_x into model channels.
Passing the existing tests does not imply a complete grouped training path was tested.
Network, loss, training loop and the old checker are unchanged.
