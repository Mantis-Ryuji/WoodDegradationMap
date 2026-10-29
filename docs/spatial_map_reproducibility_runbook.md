# Runbook for the Spatial Map Reproducibility Supplementary Experiment

Compare 9 maps for all 49 specimens using 3 trained M11 encoders at K=8 and 3 Cosine-KMeans seeds shared across encoders.
For the research conditions, metrics, and interpretation, see the [reproducibility evaluation specification](design/spatial_map_reproducibility_plan.md).

This document describes reevaluation from saved artifacts. The current CLI has no action for training encoders 2 and 3,
so the trained weights and reused maps listed below must be provided separately. These artifacts are not included in Git.

## Inputs and outputs

| Purpose | Default path |
| --- | --- |
| Global fit, encoder 1, and initial map | `outputs/experiments/global_v1/` |
| Trained encoders 2 and 3 and reused maps | `outputs/experiments/global_spatial_reproducibility_v1/` |
| Preprocessed inputs | `data/processed/production_v1/` |
| Specimen metadata | `data/metadata/古材メタデータ.csv` |
| Output | `outputs/experiments/global_spatial_reproducibility_v2/` |
| CLI | `scripts/experiments/global_spatial_reproducibility.py` |

Use `--source-experiment-dir` for the global-fit input location, `--artifacts-dir` for the trained-artifact input location,
and `--experiment-dir` for the output location. Treat input artifacts as read-only and keep their locations separate from the output location.
Keep the input folders because the trained encoders and reused maps are referenced there.

| Replicate | Map used |
| --- | --- |
| `(1,1)` | Global fit: `results/clustering/M11/repeat_1/maps/` |
| `(1,3)` | Global fit: `results/global_k8_3seed_v1/clustering/M11/repeat_3/maps/` |
| `(2,1)`, `(2,3)`, `(3,1)`, `(3,3)` | Trained artifacts: `results/clustering/M11/encoder_{e}/kmeans_{k}/maps/` |
| `(1,2)`, `(2,2)`, `(3,2)` | Generated at `results/clustering/M11/encoder_{e}/kmeans_2/maps/` in the output location |

The first seed is 1960870825, and the third is 2051198256.
The second seed is derived from `_seed("kmeans", "global_spatial_reproducibility", "seed2_v2", 2, 8)` and
shared across the 3 encoders. Its integer value and collision checks are available in the `prepare` output and `protocol/run.json`.
If the inputs above are available, encoder retraining is unnecessary. The global-fit and `global_k8_3seed_v1` aggregations can be used.

## Tests

Run from the repository root. Use the existing Python environment and lock; `--no-sync` prevents dependency updates.

```powershell
uv run --no-sync pytest tests/experiments/test_spatial_reproducibility_metrics.py tests/experiments/test_spatial_reproducibility_training.py tests/experiments/test_spatial_reproducibility_reporting.py tests/experiments/test_spatial_reproducibility_pipeline.py -q
```

Small arrays, synthetic maps, and temporary files check boundary midpoints, distance conversion, empty boundaries, ARI, seeds, input validation,
3 fits, resumption, aggregation, saving, and plotting. The tests do not train on real data, use GPUs, or evaluate the full dataset.
They create pytest cache entries and temporary files and figures.

## From preparation to figure export

When using the default paths, run the following in order.

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py prepare
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py cluster --encoders 1 2 3 --device 0 --chunk-pixels 1024
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py evaluate --include-overlap
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py report --dpi 160
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py check
```

| Stage | Processing and workload |
| --- | --- |
| `prepare` | On CPU, checks input completion records, provenance of weights, centers, and maps, hashes, shapes, and masks; saves seeds, replicate pairs, and evaluation conditions in `protocol/`. Reads models and many maps |
| `cluster` | On GPU 0, extracts representations of fit pixels from each encoder and performs 3 fits with the second seed. Runs inference on all 49 specimens and saves 147 maps and 3 sets of centers |
| `evaluate` | On CPU, evaluates 9 maps and 36 pairs per specimen and aggregates 49 specimens with equal weights. `--include-overlap` saves overlap tables and VI components for all 36 pairs and also exports both components and total VI to tables |
| `summarize` | Updates aggregate tables from saved per-pair JSON. Does not recalculate metrics, training, or clustering; performs input validation, CPU aggregation, and file I/O |
| `report` | On CPU, saves 99 PNGs of 3 types, boundary-frequency arrays for 49 specimens, captions, and a color-matching table |
| `check` | Validates provenance, hashes, target counts, and correspondence between evaluations and figures. Does not judge whether reproducibility passes or fails |

Fit pixels share the existing 401,408 coordinates. Cosine-KMeans uses K=8, `max_iter=500`, `tol=1e-4`, and FP32.
Representations of all pixels are processed in units of `--chunk-pixels`; intermediate representations are not saved to disk.
The centers from each fit are saved and reloaded to check agreement of the centers and probe predictions.

## Updating aggregate tables from saved evaluations

Outputs completed through `evaluate --include-overlap` can be updated using the saved VI components.
Run the following from the repository root. The default target is `global_spatial_reproducibility_v2`.

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py summarize
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py report --dpi 160 --replace
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py check
```

The first command updates the aggregate JSON and CSV files in `results/evaluation/`. It reads saved per-pair JSON,
so it requires neither recalculation of ARI, boundary coverage, or VI components nor encoder inference or Cosine-KMeans fitting.
Weights, centers, original maps, and per-pair JSON remain unchanged. Inputs are read to validate provenance, hashes, shapes, and masks.
Run only the first command if aggregate tables are all that is needed. Run the second and third commands as well
to reflect correspondence with the updated evaluation in the figure completion record.

`report --replace` redraws the same 99 figures of 3 types and replaces the existing `results/report/`.
Figure composition and plotting specifications remain unchanged; no VI figure is added. This involves CPU plotting, boundary-frequency array generation, and saving to temporary storage.
Both reaggregation and replotting complete processing in temporary storage before replacing the output.
Do not pass `--resume` or `--include-overlap` to `summarize`; use `--replace` only with `report`.

The following 2 test files focus on checking aggregation.

```powershell
uv run --no-sync pytest tests/experiments/test_spatial_reproducibility_reporting.py tests/experiments/test_spatial_reproducibility_pipeline.py -q
```

## Resuming after interruption

For generation stages, add `--resume` to the same arguments. Completed stages and replicates are validated and skipped.

```powershell
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py prepare --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py cluster --encoders 1 2 3 --device 0 --chunk-pixels 1024 --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py evaluate --include-overlap --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py report --dpi 160 --resume
uv run --no-sync python scripts/experiments/global_spatial_reproducibility.py check
```

Clustering retries incomplete fits; evaluation and plotting retry incomplete stages.
The target encoder can be restricted, for example with `--encoders 2`. Match `--include-overlap`, DPI, input locations, and other options to the saved settings.
Do not add `--resume` to `check`.

Processing stops if inputs, settings, numerical dependency code, or runtime are inconsistent, or artifacts are corrupted.
Completed preparation, clustering, and per-pair evaluation retain the hashes of the pipeline and aggregation code used at generation time;
a change only to aggregation code does not require retraining or refitting. The code used for reaggregation is recorded in `aggregation` in the evaluation `run.json`.
Do not hand-edit completion records to pass validation; inspect the inputs or outputs identified by the error.
Partial outputs and stages left behind by OS shutdown are not automatically deleted or overwritten.

## Evaluation and aggregation checks

- Separate the 9 within-encoder pairs from the 27 between-encoder pairs. The 8 pairs against the initial `(1,1)` map are a subset of the 36 pairs.
- Calculate ARI from original labels. Calculate boundary coverage in both directions using internal boundary midpoints,
  treating 3×3, 5×5, and 9×9 neighborhoods as Chebyshev radii of 1, 2, and 4 pixels.
- All 9 maps use the same set of valid pixels. Missing maps or changed masks are input errors.
- Coverage from a side with boundaries toward an empty side is 0; it is undefined when the source side is empty.
  Coverage is not set to 1 even when both sides are empty.
- Main aggregation uses complete cases for each comparison-group × metric combination. Boundary coverage uses specimens for which all planned pairs and directions are defined,
  with the same eligible specimens across neighborhood widths. Partial missingness remains in diagnostic tables and is not replaced with the mean of the remaining pairs.
- Report macro means with equal specimen weights and distributions across specimens. Do not treat the 36 pairs as independent samples for tests or confidence intervals.

VI is saved as the following 3 metrics in bits. Each metric is aggregated over 9 within-encoder pairs, 27 between-encoder pairs, and 8 initial-reference pairs.
Average the planned pairs within each specimen, then calculate the macro mean with equal specimen weights. The 8 initial-reference pairs are subsets of the other 2 groups.

| `metric` name | Meaning |
| --- | --- |
| `h_b_given_a_bits` | $`H(B\mid A)`$: split-side disagreement with A as the reference |
| `h_a_given_b_bits` | $`H(A\mid B)`$: merge-side disagreement with A as the reference |
| `vi_bits` | $`H(B\mid A)+H(A\mid B)`$: total VI. 0 indicates identical partitions; smaller values indicate less disagreement |

A and B retain the stored order of replicate pairs. A comes first in lexicographic order of `(encoder, Cosine-KMeans replicate ID)`;
for initial-reference comparisons, A is always `(1,1)`. Swapping A and B swaps the 2 components without changing their sum.
Means of components across all pairs alone do not establish a temporal trend in whether training increases splitting or merging.
VI contains no spatial distance, so read it alongside boundary coverage.

For results saved without `--include-overlap`, leave VI undefined; do not fill it with 0 during reaggregation.
Because `pair_metrics.csv` repeats ARI and VI in each row of 3 scales × 2 directions,
use `sample_summary.csv` and `summary.csv` for specimen-level and overall summaries.
The former's `sample_sd` is the SD across pairs within a specimen; the latter's `sample_sd` is the SD of specimen means across specimens. Neither is a confidence interval.

## Reading the figures

| Figure | What it shows | Purpose |
| --- | --- | --- |
| `summary/01_ari_vs_boundary_coverage.png` | The horizontal axis is each specimen's pair-mean ARI; the vertical axis is pair-mean boundary coverage. Top: within encoders; bottom: between encoders; left to right: neighborhoods 3, 5, and 9 | Numerical comparison of all specimens |
| `samples/{sample_id}/01_nine_maps.png` | Rows are encoders; columns are Cosine-KMeans initializations. Compare initializations within a row and differences including retraining across rows | Map comparison for all 49 specimens |
| `samples/{sample_id}/03_boundary_frequency.png` | Fraction of maps with an internal boundary within 3×3, 5×5, and 9×9 neighborhoods, from left to right | Boundary locations for all 49 specimens |

In the summary figure, blue is the mean of the smaller directional value for each pair, orange is the mean of the larger value, and vertical lines connect the two.
These are not confidence intervals. Points with low ARI and high boundary coverage are candidates for specimens whose boundary locations remain similar despite partition variation.
Higher boundary density increases nearby matches, so read density and occupancy alongside the 3×3 map grid.
Pair means can obscure deviations in particular replicates; inspect individual differences in the 3×3 map grid and `pair_metrics.csv`.

In boundary-frequency figures, yellow (1) indicates locations with a boundary in the neighborhood in all 9 maps; purple (0) indicates none in any map.
Purple does not mean instability. Intermediate colors indicate locations where boundary presence varies across replicates.
Frequency cannot decrease as neighborhood width increases, so more yellow in 9×9 neighborhoods alone does not establish stability.
Frequency describes the 9 maps; it is neither a probability over independent trials nor a measure of agreement in chemical meaning.

For the 3×3 map grid, a one-to-one correspondence with the initial `(1,1)` map is obtained from overlaps pooled over valid pixels in all 49 specimens.
This correspondence is composed with the transformation in `global_v1/results/figures/global_k8_v1/M11/matching.csv`,
using the palette in `report.json` for the same global-fit figures. The same correspondence is used for all specimens, preserving background 0.
Color matching affects display only; original labels, metrics, and boundary frequencies remain unchanged.

Figures and subplots have no titles; maps have no pixel axes, ticks, or borders.
Specimen IDs are placed below the image groups, and terminology is standardized to Cosine-KMeans. Descriptions are saved in `captions.json`.
`--dpi` changes plotting resolution only and does not affect pixel coordinates or tolerance distances.

## Artifacts

Paths are relative to the output location. Trained weights and the 6 reused maps are referenced at their input locations, not copied.

| path | Content |
| --- | --- |
| `protocol/run.json` | Seeds, boundary definitions, replicate pairs, and references to input manifests, models, and maps |
| `protocol/initial_map_audit.json` | Comparison of the initial map with the same-seed regenerated version used by the global-fit figures |
| `results/clustering/M11/encoder_{e}/kmeans_2/` | Generated maps, run records, and completion hashes |
| `checkpoints/clustering/M11/encoder_{e}/kmeans_2/centers_k8.npz` | Cosine-KMeans centers |
| `results/evaluation/map_sources.json` | Paths, hashes, shapes, and replicate IDs for 49×9 maps |
| `results/evaluation/map_diagnostics.csv` | Valid-pixel counts, adjacency-edge counts, boundary counts and densities, and occupancy |
| `results/evaluation/pair_metrics.csv` | Boundary coverage for 49×36 pairs×3 scales×2 directions, numerators, denominators, reasons for undefined values, ARI, both VI components, and total VI |
| `results/evaluation/initial_reference_metrics.csv` | 8 pairs against the initial map |
| `results/evaluation/samples/{sample_id}.json` | Per-pair values, degeneracy flags, overlap tables, and VI |
| `results/evaluation/summary.json` | Specimen-level summaries, complete-pair counts, exclusion reasons, macro distributions, and group differences using common specimens |
| `results/evaluation/sample_summary.csv` | Means, minima, maxima, and SDs for specimens × 3 comparison groups × 10 metrics; planned, observed, and defined pair counts and reasons for missingness |
| `results/evaluation/summary.csv` | Equally weighted macro means, minima/maxima/SDs across specimens, and eligible specimen counts for 3 comparison groups × 10 metrics. The 3 groups are within-encoder, between-encoder, and initial-reference |
| `results/report/` | 99 PNGs of 3 types, boundary-frequency arrays for 49 specimens, plotting conditions, and hashes |
| `results/report/color_matching.json` | Correspondence and overlap tables against the initial map, conversion to global-fit display IDs, composed correspondence, and palette |
| `results/report/captions.json` | Figure roles, panel meanings, specimen counts, and display notes |
| `attempts/` | Execution stages, completion/failure/interruption status, exception types, elapsed time, and resume settings |

Undefined JSON values are `null`; CSV uses empty cells and reasons. Invalid pixels in boundary-frequency arrays are NaN, distinct from 0 at valid pixels.
Trained payloads such as centers follow the existing ignore rules for `checkpoints/`.
