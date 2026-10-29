# Experiment Runbook

## 1. Purpose

This document describes execution of the fixed design with `production_v1` for CV and `global_v1` for global fitting. For the definitions of conditions, seeds, splits, and evaluation methods,
see the [research design](design/README.md).

The clustering, evaluation, and OOF commands in this document are for the main Cosine-KMeans experiment and the supplementary mask-rate experiment.
The supplementary mask-rate experiment separates [CV and OOF](#mask-rate-sweep) from the [global-fit comparison for the appendix](#global-mask-rate).
For the current Bacteria-ID Table 5, see the [HPO runbook](bacteria_id_hpo_runbook.md); for existing pretraining and Table 4, see the [dedicated runbook](bacteria_id_runbook.md).

Run from the repository root in PowerShell with the ChemoMAE v0.2.2 environment.
Production examples use `uv run --no-sync` to separate environment setup and updates from run startup.

Use the commands in this document to distinguish new execution from checking saved artifacts.
For locations of figures, numerical results, and provenance, see the [artifact inventory](results.md).

- [Input checks](#input-preparation) and [manifests](#3-production-manifests)
- [One neural-network run and resumption](#neural-run) and [PCA](#5-b1-pca-fitting-and-baseline-transform-validation)
- [Clean test maps](#6-clean-test-map) and [evaluation](#7-evaluation)
- [OOF aggregation](#oof-aggregation), [OOF sanity figures](#oof-sanity), and [main-condition OOF figures and tables](#oof-reporting)
- [Global-fit preparation and batch training](#global-fit-pipeline), [single PCA figure](#global-pca), and [storage conventions](#artifact-records)
- [Mask-rate CV and OOF figures and tables](#mask-rate-sweep) and [mask-rate global fit and 2×7 figure](#global-mask-rate)
- [NIR spectral interpretation figures](#nir-interpretation)

After each CLI finishes, check `$LASTEXITCODE -eq 0`; do not proceed if it is nonzero. Check the JSON status
in addition to the exit code.

## 2. Fixed paths

| Purpose | Path |
| --- | --- |
| Production inputs | `data/processed/production_v1/` |
| Preprocessing check figures | `outputs/preprocessing/production_v1/` |
| preflight | `outputs/experiments/preflight_v1/` |
| Production experiment | `outputs/experiments/production_v1/` |
| metadata | `data/metadata/古材メタデータ.csv` |
| B0, B1, A0, and M00 OOF sanity | `outputs/sanity_checks/a0_m00_oof_visualization/` |
| Main-condition OOF snapshot | `outputs/experiments/production_v1/results/oof/main_oof_v1/` |
| Main-condition OOF figures and tables | `outputs/experiments/production_v1/results/figures/main_oof_v1/` |
| Mask-rate OOF snapshots, figures, and tables | `outputs/experiments/production_v1/results/{oof,figures}/mask_rate_oof_v1/` |
| Global fit | `outputs/experiments/global_v1/` |
| Appendix figures and tables for the mask-rate global fit | `outputs/experiments/global_v1/results/figures/global_mask_rate_k8_v1/` |

Do not copy smoke or preflight artifacts into the production root. Once production starts, do not recreate manifests;
treat `outputs/experiments/production_v1/manifests/` as the fixed input for the same experimental series.

<a id="input-preparation"></a>

### 2.1 Preprocessing and input checks

Run preprocessing according to the [fixed preprocessing specification](design/preprocessing.md)
only when both output locations are new or empty.

```powershell
uv run --no-sync python scripts/preprocess/run_production_preprocessing.py
```

To update only the check figures for existing results, run the following. The first command overwrites 4 preprocessing PNGs
from the saved quality tables without reading or writing `reflectance_l2_norm/`. The second updates 2 example figures and
the visualization record in `summary.json` in the existing augmentation output, and removes the noise and shift boxplot PNGs.
The second reads only the train pixels specified by the saved `selection.csv` (8 specimens × 128 pixels) from HDF5,
selecting one observed pixel per specimen whose RMS distance to the specimen's wavelength-wise median is smallest. From these 8 candidates,
it selects the combination of 3 specimens that maximizes the minimum RMS distance between pixels. Ties are resolved by the sum of distances, then specimen order.
Noise and shift share the same 3 pixels and example order; the plotted clean values are not smoothed, averaged, or restandardized.
The selection method, specimen IDs, HDF5 rows, and pixel coordinates are retained in the visualization record in `summary.json`.
Preprocessed data, the train-pixel `selection.csv` and `metrics.csv`, and numerical summaries are not regenerated.

```powershell
.venv\Scripts\python.exe scripts/preprocess/redraw_production_reports.py
.venv\Scripts\python.exe scripts/experiments/sanity_check_augmentation_strengths.py --plots-only
```

Check exit code 0 and the [shared vertical-axis conventions](design/visualization_and_interpretation.md#figure-style).
Check that the cutoff is shown only for the SNR proxy and that candidate-pixel figures use separate SNV panels.
Check that shift examples use 3 rows × 2 columns for Example 1 (+), Example 2 (−), and Example 3 (+), with the same dimensions as the noise figure.

Use the following to check from existing quality tables whether 8192 pixels can be sampled without replacement from each specimen.
It reads neither spectra nor HDF5 and changes neither data nor manifests.

```powershell
uv run --no-sync python scripts/preprocess/check_sampling_pixels.py --q 8192
```

It displays saved valid-pixel counts, sampling fractions, and shortfalls per specimen, and overall minimum, median, and maximum values.
The exit code is 0 if all specimens support sampling and 1 if any are short. Actual sampling and split creation are performed in the next section.

## 3. Production manifests

At production startup, create and validate new manifests only if the output location does not exist.
For an existing `production_v1`, do not run `create` below; use `check` on the saved manifests.

```powershell
uv run --no-sync python scripts/experiments/prepare_manifests.py create --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'manifest creation failed' }
uv run --no-sync python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
if ($LASTEXITCODE -ne 0) { throw 'manifest check failed' }
```

Checking existing manifests:

```powershell
uv run --no-sync python scripts/experiments/prepare_manifests.py check --experiment-id production_v1
```

<a id="neural-run"></a>

## 4. One neural-network run

The following example uses supplementary mask-rate condition M11-25, fold 1, and replicate 1.
Do not retrain completed runs; use each stage's `check` to inspect saved artifacts.

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
$repeat = 1

uv run --no-sync python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir
```

Even when running unstarted replicates of the same fold consecutively, invoke the CLI once per run.
The following executes M11-25, fold 1, replicates 1–3 sequentially without parallelization.
At execution time, list only unstarted runs in `$repeats`; exclude completed and interrupted runs.

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
$repeats = 1..3

foreach ($repeat in $repeats) {
    uv run --no-sync python scripts/experiments/train_neural.py train `
        --condition $condition `
        --fold $fold `
        --repeat $repeat `
        --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) {
        throw "$condition fold $fold repeat $repeat training failed"
    }
}
```

A nonzero exit stops the loop. After interruption, do not rerun completed replicates;
explicitly resume only the affected replicate using the procedure below.

The neural conditions are `A0`, `A1`, `M00`, `M10`, `M01`, `M11`, `M11-25`, and `M11-75`.
Each run has 800 epochs, with 249,600 planned batch attempts for folds 1–4 and 256,000 for fold 5.

After successful termination, inspect `completion.json` as follows.

```powershell
$completionPath = Join-Path $experimentDir "results/neural/$condition/fold_$fold/repeat_$repeat/completion.json"
$completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
$completion |
    Select-Object status, completed_epochs, attempted_updates, optimizer_updates, nonzero_lr_updates, amp_skips, training_seconds

if (-not (Test-Path -LiteralPath $completion.weights_file)) {
    throw 'weights file does not exist'
}
$actualWeightsHash = (Get-FileHash -LiteralPath $completion.weights_file -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualWeightsHash -ne $completion.weights_sha256) {
    throw 'weights hash does not match the completion record'
}
```

Completion requires the following.

- `status` is `training_completed`
- `completed_epochs` is 800
- `attempted_updates` matches the planned count for the fold
- `optimizer_updates + amp_skips == attempted_updates`
- `weights_file` exists and its actual SHA-256 matches `weights_sha256`

Save `amp_skips` as an observed value; do not assume 0. Do not rerun the same `train` command for a successfully completed run.
`training_seconds` is the sum of epoch processing times, not the wall time of the entire CLI. Even for a resumed run,
it sums each of the 800 completed epochs once.

### Resuming after interruption

Resume only the same interrupted or failed run from an explicitly specified `last.pt`. Do not change the manifest, condition, fold, replicate,
or configuration.

```powershell
$resumePath = Join-Path $experimentDir "checkpoints/neural/$condition/fold_$fold/repeat_$repeat/checkpoints/last.pt"

uv run --no-sync python scripts/experiments/train_neural.py train `
    --condition $condition `
    --fold $fold `
    --repeat $repeat `
    --experiment-dir $experimentDir `
    --resume $resumePath
```

If a checkpoint is missing or the source hash, configuration, or run identity differs, investigate the cause before treating it
as a separate run. Do not bypass agreement checks to continue.
Do not resume a run from an old checkpoint after changing its fixed configuration. For an incomplete run explicitly judged invalid,
remove only its results and checkpoints after verifying their target paths, then start a new run.

### When checkpoint record replacement fails on Windows

If only replacement of `checkpoint.json.tmp` with `checkpoint.json` at the end of an epoch raises `PermissionError`,
do not delete, move, or overwrite existing files. First check read-only that the latest `last.pt`, temporary record, and training history
represent the same completed epoch.

```powershell
$resultDir = Join-Path $experimentDir "results/neural/$condition/fold_$fold/repeat_$repeat"
$resumePath = Join-Path $experimentDir "checkpoints/neural/$condition/fold_$fold/repeat_$repeat/checkpoints/last.pt"
$pendingPath = Join-Path $resultDir 'checkpoint.json.tmp'

$pending = Get-Content -LiteralPath $pendingPath -Raw | ConvertFrom-Json
$history = @(Get-Content -LiteralPath (Join-Path $resultDir 'training_history.json') -Raw | ConvertFrom-Json)
$actualCheckpointHash = (Get-FileHash -LiteralPath $resumePath -Algorithm SHA256).Hash.ToLowerInvariant()

if ($actualCheckpointHash -ne $pending.checkpoint_sha256 -or
    $history.Count -ne $pending.completed_epochs -or
    $history[-1].epoch -ne $pending.completed_epochs) {
    throw 'checkpoint, temporary record, and training history do not agree'
}
```

Only if they agree, resume the same run with `--resume $resumePath` as above. Do not omit the resume loader's validation of internal state,
run identity, configuration, manifest, and source hash. If the same replacement error recurs,
close editors or previews displaying the JSON and investigate the cause before resuming.

## 5. B1 PCA fitting and baseline transform validation

Fit B1 PCA once on the train pixels of each fold, for 5 fits in total. Share PCA across replicates 1–3 of the same fold;
fit KMeans separately for each replicate. `fit_baselines.py fit` also saves the parameter-free B0 transformation specification and
performs probe validation. Do not rerun `fit` for a completed fold.

```powershell
$experimentDir = 'outputs/experiments/production_v1'

foreach ($fold in 1..5) {
    uv run --no-sync python scripts/experiments/fit_baselines.py fit `
        --fold $fold --repeat 1 --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) { throw "B1 PCA fit failed: fold $fold" }

    uv run --no-sync python scripts/experiments/fit_baselines.py check `
        --fold $fold --repeat 1 --experiment-dir $experimentDir
    if ($LASTEXITCODE -ne 0) { throw "baseline check failed: fold $fold" }
}
```

For each fold, check `status=fitted_and_roundtrip_checked` and `pca_reusable_across_repeats=true` for `fit`,
and `status=validated_existing_baselines` for `check`. For B1 replicates 2 and 3 sharing PCA,
explicitly pass `--pca-repeat 1` to `cluster_representations.py run`.

`results/baselines/fold_<fold>/repeat_1/b0.json` holds the B0 transformation specification; `fit.json` in the same location primarily holds
B1 PCA fit provenance and B0/B1 probe diagnostics. Actual PCA parameters are saved in
`checkpoints/baselines/fold_<fold>/repeat_1/pca.npz`.

## 6. clean test map

For each condition, fold, and replicate whose neural training or required fold-specific B1 PCA fit is complete,
create KMeans fits and clean test maps for every prespecified K, then validate saved artifacts with CPU `check`.
B0 has no preceding fit. The following uses the same M11-25, fold 1, replicate 1 example as Section 4.

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
$repeat = 1

uv run --no-sync python scripts/experiments/cluster_representations.py run `
    --condition $condition --fold $fold --repeat $repeat --experiment-dir $experimentDir
if ($LASTEXITCODE -ne 0) { throw 'clustering failed' }
uv run --no-sync python scripts/experiments/cluster_representations.py check `
    --condition $condition --fold $fold --repeat $repeat --experiment-dir $experimentDir
```

Example of reusing replicate 1's PCA for B1:

```powershell
uv run --no-sync python scripts/experiments/cluster_representations.py run `
    --condition B1 --fold 1 --repeat 2 --pca-repeat 1 `
    --experiment-dir outputs/experiments/production_v1
```

Check `status=clean_test_maps_completed` and `checks_passed=true` for run, and
`status=validated_existing_clustering` for check. Also verify test specimen and pixel counts and coverage of all 7 K values.

## 7. Evaluation

Evaluate combinations with complete clean test maps. `run` uses a GPU; `check` validates saved results on CPU.
Conditions and replicates can be supplied together, but do not respecify and overwrite combinations that have already been evaluated.

```powershell
$experimentDir = 'outputs/experiments/production_v1'
$condition = 'M11-25'
$fold = 1
$repeat = 1

uv run --no-sync python scripts/experiments/evaluate_representations.py run `
    --conditions $condition --fold $fold --repeats $repeat `
    --experiment-dir $experimentDir
if ($LASTEXITCODE -ne 0) { throw 'evaluation failed' }
uv run --no-sync python scripts/experiments/evaluate_representations.py check `
    --conditions $condition --fold $fold --repeats $repeat `
    --experiment-dir $experimentDir
```

To combine multiple conditions and replicates, list them separated by spaces in `--conditions` and `--repeats`.
First verify that every combination in the specified Cartesian product has complete clean test maps and remains unevaluated.

For each combination, check `status=full_test_evaluation_completed` and `checks_passed=true` for run, and
`status=validated_existing_evaluation` for check. Treat undefined metrics as `null` with a reason;
do not replace them with 0.

## 8. Execution matrix

| Category | Conditions | Required combinations |
| --- | --- | ---: |
| baseline | B0, B1 | 5 folds × 3 replicates each |
| Main-experiment training | A0, A1, M00, M10, M01, M11 | 90 runs |
| Supplementary mask-rate training | M11-25, M11-75 | 30 runs |

Complete clean maps and evaluation for each combination. Neural training totals 120 runs; B0/B1 fitting,
KMeans, and evaluation are not included in this count. Do not use the 3 replicates to select seeds; include all of them in OOF aggregation.

<a id="mask-rate-sweep"></a>

### 8.1 Supplementary mask-rate experiment: CV and OOF

Compare M11 mask rates of 25%, 50%, and 75%.
The 50% condition is shared with main-condition M11; train M11-25 and M11-75 over 5 folds × 3 replicates each, for 30 runs total.
The 800 epochs, splits, sampled pixels, seeds, augmentation strengths, all 7 K values, and evaluation metrics inherit the main experiment's fixed conditions.
Report this as sensitivity analysis rather than selection of the best mask rate, and do not replace main-condition M11 after the fact.
OOF aggregation covers 49 specimens and 45 source runs.

1. Run M11-25 and M11-75 using the training and resume procedures in Section 4, and check each run.
2. Following Sections 6–7, complete Cosine-KMeans, prediction for all test pixels, evaluation, and checks for each run.
3. Aggregate the 3 conditions M11-25, M11, and M11-75 into `mask_rate_oof_v1` through [OOF aggregation](#oof-aggregation). Retain the main-condition snapshot.
4. Use the dedicated CLI below to export LLA, LFR, ARI, Cosine-Silhouette, occupancy, and dependence on K.

<a id="mask-rate-reporting"></a>

#### Regenerating mask-rate OOF figures and tables

Use `export_mask_rate_oof.py`, separately from the main-condition CLI. The following regenerates existing figures and tables.

```powershell
uv run --no-sync python scripts/experiments/export_mask_rate_oof.py `
    --experiment-dir outputs/experiments/production_v1 `
    --snapshot mask_rate_oof_v1 --with-figure --overwrite
if ($LASTEXITCODE -ne 0) { throw 'Mask-rate OOF reporting failed' }
```

This reads saved OOF values on CPU without rerunning training, inference, or evaluation. The default output location is
`outputs/experiments/production_v1/results/figures/mask_rate_oof_v1/`, with default 240 dpi.
`--output-dir` changes the output location; `--overwrite` replaces only files managed by this report.
Omitting `--with-figure` generates only CSV files and provenance and completion records.

The output comprises 5 files: 1 PNG, `metrics_all_k.csv`, `metrics_k8.csv`, `report.json`, and `completion.json`.
The PNG uses a 2×2 layout (top: LLA 3 and 5; bottom: LLA 9 and LFR(TGN+FS)), with the 50% condition labeled `M11(-50)`.
Solid lines show macro means; dashed, dotted, and dash-dotted lines show training replicates 1, 2, and 3. Tables retain 11 metrics for all 7 K values and K8.
Check exit code 0, `status=mask_rate_report_completed`, `checks_passed=true`, text, legends, and curves.
This export checks the consumed snapshot's hash and specimen/fold/replicate correspondence but does not rerun the complete OOF check.
To revalidate saved OOF results, use `aggregate_oof.py check` in [Section 9](#oof-aggregation).

For conditions, see the [experimental protocol](design/experiment_protocol.md#mask-rate-sweep);
for files and provenance, see the [artifact inventory](results.md#mask-rate-handoff).

<a id="global-fit-pipeline"></a>

### 8.2 Global fitting and interpretation

Following the [global visualization design](design/visualization_and_interpretation.md), fit
B1 PCA and A0, A1, M00, and M11 on common sampled pixels from all 49 specimens. Including B0, perform 6 Cosine-KMeans fits at $`K_0=8`$ using representations from 6 conditions.
The 4 global-training runs and global clustering are separate from the 120 planned CV training runs including supplementary conditions, and are excluded from OOF aggregation.

`global_fit.py` handles manifests, B0/PCA, batch training, resumption, and checks for A0/A1/M00/M11.
The existing `train_neural.py` is for CV, requires `--fold`, and is not used for global fitting.

Sample 8,192 pixels from each of all 49 specimens, totaling 401,408 pixels shared across the 6 conditions.
Use a dedicated manifest with ROOT_SEED=20260905, fold position `global`, and replicate ID 1, separate from CV manifests.
After preparation, PCA, and neural training, proceed to [clustering and visualization](#global-post-fit).
Global-fit maps are descriptive analyses and are distinguished from OOF evaluation on unseen specimens.

#### Preparation, PCA, and GPU smoke checks

Run in PowerShell from the repository root. The following creates a new output location for the first time; do not rerun it on a completed `global_v1`.
`create` reads coordinates and masks; PCA fits on the common 401,408 SNV rows on CPU (392 MiB for the FP32 matrix alone).
`smoke` uses full-size models and batch size 1024, checking 2 epochs × 2 batches and resumption of the second epoch for A0 → A1 → M00 → M11.
It executes 24 batches total across the 4 conditions on GPU and compares raw-weight save/load, all-visible 16-dimensional representations, and resumed inputs, LR, AMP decisions, and weights.
Smoke weights are not carried into production.

```powershell
foreach ($step in @("create", "check", "baseline-fit", "baseline-check", "smoke")) {
    uv run --no-sync python scripts/experiments/global_fit.py $step
    if ($LASTEXITCODE -ne 0) { throw "Global fit preparation failed: $step" }
}
```

After initial creation, use `check` to validate existing manifests and `baseline-check` to validate PCA.
`create` and `baseline-fit` do not overwrite existing outputs. Smoke checks use separate timestamped directories.
After a failure, inspect the failed stage without repeating successful create/fit stages. Partial outputs without completion records are not automatically reused.
PCA restoration preserves the C/F array layout used at saving. Changing the layout changes FP32 multiply-add rounding,
so compare using the same layout while keeping the save/load tolerance at `1e-6`. Restoration checks run in a temporary directory before passing outputs are placed in the production location.

#### Batch training of 4 conditions and completion checks

Batch resumption and `training-check` apply to runs trained with the same current configuration.
Some completed runs in the saved experiments have a mismatch between current records and contracts inside optimizer checkpoints;
do not resume those runs. Use `global_cluster.py check` to check saved final weights and labels.
For a new output location, execute this only after all preparation stages above succeed, training A0 → A1 → M00 → M11 sequentially for 800 epochs each on 1 GPU.
Planned counts are 392 batches/epoch, 313,600 attempted updates/run, and 1,254,400 attempted updates in total.
Record AMP-overflow skips and actual optimizer updates separately; do not assume actual updates equal the planned count.

```powershell
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A0 A1 M00 M11
if ($LASTEXITCODE -ne 0) { throw "Global training failed; inspect the checkpoint before resuming" }
uv run --no-sync python scripts/experiments/global_fit.py training-check
if ($LASTEXITCODE -ne 0) { throw "Global training completion check failed" }
```

Batch execution stops at the first failure. After interruption, the single command below validates and skips completed conditions,
resumes incomplete conditions from the same run's `last.pt`, and starts unstarted conditions. An interrupted epoch is rerun from the last saved boundary.
Processing stops if run/config/manifest/code/runtime differ, or an existing run has no checkpoint.
To change the GPU, specify `--device` consistently from the first invocation (default 0).

```powershell
uv run --no-sync python scripts/experiments/global_fit.py train --conditions A0 A1 M00 M11 --resume
if ($LASTEXITCODE -ne 0) { throw "Global training resume failed" }
uv run --no-sync python scripts/experiments/global_fit.py training-check
if ($LASTEXITCODE -ne 0) { throw "Global training completion check failed" }
```

#### Global-fit artifacts

All are under `outputs/experiments/global_v1/`. Do not add global-fit artifacts to CV OOF snapshots.

| Location | Content |
| --- | --- |
| `config/experiment.json`, `config/seeds.json` | Global-fit contract and 55 seeds, including those for subsequent clustering |
| `manifests/samples.parquet`, `manifests/fit_pixels.parquet` | All 49 specimens and common 401,408 coordinates, without a fold column |
| `manifests/inputs.json`, `manifests/complete.json` | Original-input fingerprints, manifest hashes, and planned update counts |
| `results/baselines/b0.json`, `completion.json` | B0 transformation contract, PCA provenance, solver, restoration probes, and hashes |
| `checkpoints/baselines/pca.npz` | PCA coefficients without pickle |
| `results/neural/{condition}/repeat_1/` | `run.json`, history, checkpoint records, attempts, and `completion.json` |
| `checkpoints/neural/{condition}/repeat_1/last_model.pt` | Final raw weights at epoch 800 |
| `checkpoints/neural/{condition}/repeat_1/checkpoints/last.pt` | Resume state including optimizer, AMP scaler, each RNG, and epoch boundary |
| `results/neural_smoke/{timestamp}/{condition}/repeat_1/smoke.json` | Pass/fail results and resume errors for short GPU checks; corresponding checkpoints are in `checkpoints/neural_smoke/` |

`training-check` reads saved checkpoints and checks epochs, update counts, manifest/config/code/runtime, weight hashes,
and agreement between checkpoint weights and final raw weights. Proceed to global Cosine-KMeans after confirming completion of all 4 conditions.

#### Global-fit completion records

The authoritative records for B0/PCA and A0, A1, M00, and M11 update counts, AMP skips, and save/load results are
the condition-specific `completion.json` files and run records under `outputs/experiments/global_v1/results/`.
The following sections cover post-training K8 maps, figures and tables, and PCA operations.

<a id="global-post-fit"></a>

#### Work after training completes

`global_cluster.py` handles **representation extraction, Cosine-KMeans, all-pixel prediction, and checks** using saved models;
`visualize_global.py` handles **matching, observed-spectrum aggregation, PNG/CSV generation, and checks**.
`global_fit.py` continues to handle preparation, baselines, smoke checks, NN training, and checks.
The existing `cluster_representations.py` is for CV and requires `--fold`; do not run it directly on `global_v1`.

1. Read saved PCA, final NN weights, and the common global manifest to extract B0, B1, A0, A1, M00, and M11 representations.
   B0 uses 256-dimensional SNV; B1 and NNs use 16 dimensions with the specified L2 normalization and all-visible extraction.
   Without retraining or refitting PCA, fit Cosine-KMeans once per condition on representations of the common 401,408 pixels, with fixed seeds and $`K_0=8`$.
2. Apply each condition's fixed model and centers to all 3,902,250 valid pixels in all 49 specimens, and save labels associated with pixel coordinates.
   Check save/load consistency, target counts, label ranges, and provenance. Do not refit clustering per specimen.
3. Align labels to M00 + Cosine-KMeans using Hungarian matching of cosine similarities between observed SNV representative curves, formed by equally averaging within-specimen cluster means across specimens.
   Export cluster maps, occupancy, matching tables, and representative curves and variation for reflectance, SNV, and SG second derivatives of pseudo-absorbance as PNG and CSV.
4. Use the [PCA figure](#global-pca) for supplementary display of representations and the [NIR interpretation figures](#nir-interpretation) for within-specimen spectral comparisons.

For mask-rate comparisons, see [CV and OOF](#mask-rate-sweep) and [global fit](#global-mask-rate).

##### Commands: clustering and visualization

The following procedure is for a new global-clustering output location. The existing `global_v1` has already been clustered,
so when regenerating visualizations, run only the second command, `visualize_global.py run`.
For a new output location, run the following in order in PowerShell from the repository root.
Clustering processes B0 → B1 → A0 → A1 → M00 → M11 on 1 GPU and stops at the first failure.
After each condition completes, it checks saved centers, all maps, and provenance. Visualization is also checked after saving, so the 2 commands below perform steps 1–3.

```powershell
uv run --no-sync python scripts/experiments/global_cluster.py run
if ($LASTEXITCODE -ne 0) { throw "Global clustering failed" }

uv run --no-sync python scripts/experiments/visualize_global.py run
if ($LASTEXITCODE -ne 0) { throw "Global visualization failed" }
```

`global_cluster.py run` requires CUDA (default `--device 0`); the default chunk for representation extraction and all-pixel prediction is 1,024 pixels.
Fitting uses all representations of the common 401,408 pixels at once, with no refitting per chunk.
The FP32 fit array alone occupies 392 MiB for B0 and 24.5 MiB for the 16-dimensional B1/NN arrays; KMeans workspace and GPU transfer buffers require additional memory.
NNs use the final raw weights at epoch 800 with all inputs visible in FP32. Existing PCA/NN fits are not repeated, and no additional restarts are performed.
Representation arrays are held only during processing; saved outputs are centers, labels, provenance, and diagnostic records.

`visualize_global.py run` reads saved labels and observed spectra for all 6 conditions on CPU, without KMeans fitting or NN inference.
Observed spectra are read in chunks of 2,048 pixels by default and aggregated across all 49 specimens. PNGs default to 240 dpi (changeable with `--dpi`).
Complete clustering for all 6 conditions first. There is no setting to generate visualization for only a subset of conditions.

To revalidate only saved artifacts after completion, use the following. No GPU is required; it validates provenance, hashes, valid masks, label ranges, artifact counts, and related details.

```powershell
uv run --no-sync python scripts/experiments/global_cluster.py check
if ($LASTEXITCODE -ne 0) { throw "Global clustering check failed" }
uv run --no-sync python scripts/experiments/visualize_global.py check
if ($LASTEXITCODE -ne 0) { throw "Global visualization check failed" }
```

After interruption, use `global_cluster.py run --resume`. It checks conditions with outputs before skipping them, then executes unstarted conditions.
For ordinary processing exceptions, working directories are cleaned up so the affected condition can restart from the beginning. It does not resume from an intermediate KMeans iteration.
If forced termination leaves only one side of the final output locations, or existing-output provenance or hashes disagree, processing stops without automatic deletion or reuse.
`visualize_global.py run --resume` checks and skips completed figures and tables; it does not redraw or resume aggregation partway through.
Normal visualization `run` regenerates figures and tables and replaces the specified `global_k8_v1` in full after completion.
Old PNGs, CSVs, and display-label NPZs are not retained, avoiding mixed old and new layouts. Failure partway through preserves the original visualization.
The independently managed `pca-latent-2d/` is carried forward; PCA coordinates, NN artifacts, and clustering artifacts are excluded from replacement.
Clustering `run` does not overwrite existing artifacts.

<a id="representative-sample-update"></a>

##### Regenerating figures for the 7 representative specimens

For target IDs, order, and the role of this selection, see [selection of display examples](design/visualization_and_interpretation.md#representative-samples).
To regenerate representative figures, run the following in PowerShell from the repository root.

For the representative 1×7 raw BMP and reflectance L2 norm figures, the following single command reads saved images on CPU and overwrites both under the same filenames.
It leaves the existing 7×7 figures of all 49 specimens unchanged and performs no preprocessing or spectral calculations.

```powershell
uv run --no-sync python scripts/preprocess/create_sample_overviews.py --representatives-only --overwrite
if ($LASTEXITCODE -ne 0) { throw "Representative sample overviews failed" }
```

The targets are `outputs/sample_overviews/raw_bmp_representatives_1x7.png` and
`outputs/sample_overviews/reflectance_l2_norm_representatives_1x7.png`.

The following single command updates both the condition-specific representative 1×7 global-fit figures and the 6-condition × 7-specimen figure.
It reads existing labels and observed spectra for all 49 specimens on CPU, regenerates the full figure/table set (67 PNGs, 44 CSVs, and associated records),
and replaces the same `results/figures/global_k8_v1/`, so it performs more processing than compositing the 2 representative figures alone.
It does not rerun training or clustering and retains the independent `pca-latent-2d/`.

```powershell
uv run --no-sync python scripts/experiments/visualize_global.py run
if ($LASTEXITCODE -ne 0) { throw "Global representative visualization failed" }
```

The representative targets are `{B0,B1,A0,A1,M00,M11}/labels/07_representative_samples_1x7.png` and
`01_representative_samples_6x7.png` at the root. `representative_sample_ids` in `report.json` and
`captions.csv` also reflect the specified 7 specimens and order. Do not add `--resume`, which only validates existing figures.
Check exit code 0, figure IDs and column order, representative IDs in `report.json`, and `checks_passed=true` at the end of execution.
Existing OOF sanity figures are not updated by these 2 commands.

##### Saved artifacts and checks

All paths below are relative to `outputs/experiments/global_v1/`.

| Location | Content |
| --- | --- |
| `checkpoints/clustering/{condition}/repeat_1/centers_k8.npz` | Fixed K=8 centers, global seed, fit-pixel count, settings, and fit diagnostics |
| `results/clustering/{condition}/repeat_1/maps/{sample_id}.npz` | Original labels for all valid pixels: `labels_k8`, background 0, clusters 1–8 |
| `results/clustering/{condition}/repeat_1/run.json`, `completion.json` | Representation provenance, code/runtime/hashes, save/load checks, per-specimen pixel counts and occupancy, and completion records |
| `results/figures/global_k8_v1/` | PNGs and CSVs below, plus `report.json` and `completion.json` |

The figure/table directory is divided into `B0/`, `B1/`, `A0/`, `A1/`, `M00/`, and `M11/`. Each condition has 11 PNGs.
Including `01_representative_samples_6x7.png` at the root gives 67 in total.
This comparison figure uses the fixed 7 representative specimens as columns and B0, B1, A0, A1, M00, and M11 as rows from top to bottom. Specimen IDs appear only below the bottom row.
The following paths are relative to each condition directory.

| PNG | Content |
| --- | --- |
| `labels/00_samples_01-07_1x7.png`–`06_samples_43-49_1x7.png` | 7 specimens per image in ascending specimen-ID order; 49 specimens across 7 images |
| `labels/07_representative_samples_1x7.png` | The fixed 7 representative specimens, in the same order as the root 6-condition × 7-specimen figure |
| `labels/08_all_samples_7x7.png` | All 49 specimens in the same order, 7×7 |
| `01_representative_spectra.png` | Top: SNV and reflectance; bottom: full-width SG second derivatives. Each panel compares the 8 cluster colors |
| `02_snv_cosine_matrix.png` | 8×8 SNV cosine similarities, with M00 vertically and the current condition horizontally, in display-ID order and range −1 to 1 |

Maps match the margins and large bold specimen IDs in `outputs/sample_overviews`. Nearest-neighbor interpolation preserves colors; the colorbar is `Cluster ID`.
Spectra share cluster colors with maps. Bands around mean curves are between-specimen IQRs, not confidence intervals.
Grid lines follow major wavelength ticks at 100 nm and minor ticks at 50 nm. Endpoint labels preferentially use 2 decimal places; adjacent labels less than 30 nm apart are omitted.
SNV uses a vertical range of −2 to 2, reflectance 0 to 1, and second derivatives an automatic range shared by all 6 conditions.
Each condition's `label_maps.npz` contains aligned labels keyed by specimen ID. Original labels and centers remain unchanged.
The root `spectral_summary.npz` stores wavelengths, representative curves, IQRs, and original-to-display ID mappings for all conditions.

There are 44 CSVs: 7 per condition—`sample_spectra.csv`, `spectrum_counts.csv`, `representative_spectra.csv`,
`difference_spectra.csv`, `matching.csv`, `matching_matrices.csv`, and `occupancy.csv`—plus root `wavelengths.csv` and `captions.csv`.
Differences are representative curves for each condition minus M00; they are not paired differences because contributing specimens may differ. M00's difference table contains only a header.
Matching tables retain `snv_cosine_similarity`, the similarity margin over other candidates for the same reference cluster, matching-pixel counts, and original and display IDs.
`iou` is also saved as a supplementary value. IoU pools all common pixels rather than weighting specimens equally, and is not used for SNV assignment.
The specimen-level spectral columns `band_000`–`band_255` map to nm through `wavelengths.csv`.
Representative curves use within-specimen means followed by equally weighted means across specimens. Records retain the all-band positivity check and exclusion counts for pseudo-absorbance, and contributing specimen counts, IDs, and pixel counts for each curve.
Undefined curves are NaN in NPZ and blank in CSV. Matching stops if any cluster is empty across all specimens, or an SNV representative curve is nonfinite or has zero norm.
Existing figures and tables use neither spatial smoothing nor band integration.
For individual-specimen interpretation, consult `sample_spectra.csv` and the contribution/exclusion-count CSVs, distinguishing them from macro means across all specimens.

Outputs cover 49 specimens, 67 PNGs, and 44 CSVs. Check artifact consistency using
`report.json`, `completion.json`, and `visualize_global.py check`.
Completion of figure/table generation is distinct from completion of physicochemical interpretation of regional differences.

<a id="global-pca"></a>

### 8.3 Latent-space PCA: one 2-row, 6-column figure

Perform visualization PCA and PNG plotting in the existing host environment.
Columns are B0, B1, A0, A1, M00, and M11 from left to right. The top row shows pixel-count hexbins (turbo with a shared logarithmic color range);
the bottom row shows Cluster IDs after SNV alignment and centroids defined as arithmetic means of the assigned pixels' PC1 and PC2 coordinates.
Condition names (22 pt) appear above the top panels, with PC names, explained variance ratios, and a small number of numerical ticks.
Axis and colorbar labels are 18 pt, tick labels 15 pt, and centroid numbers 14 pt. Layout and definitions are also saved in `captions.csv`.

Share the existing global-fit 49 specimens × 8,192 = 401,408 pixels across all conditions, and independently fit 2-dimensional PCA
to the L2-normalized clustering representations of the 5 conditions other than B1.
Center the inputs; do not additionally standardize columns, whiten, or L2-normalize after projection.
Use FP64 `PCA(n_components=2, svd_solver="covariance_eigh", whiten=False)` and
save coordinates, means, principal-component coefficients, eigenvalues, explained variance ratios, runtime, and input hashes.
For B1, select the top 2 components in descending order of the saved baseline PCA singular values and directly display scores before L2 normalization.
Use the saved explained variance ratios relative to total original SNV variance. Do not rerun PCA for B1, retrain the baseline, or redo clustering.
Because coordinate scales differ across conditions, use condition-specific display ranges that match between the top and bottom rows of each condition. Hexbin color ranges are shared across all conditions.
For details, see the [design specification](design/visualization_and_interpretation.md#latent-spectral-maps).

Coordinate generation and figure plotting are separate stages. Coordinates in `results/pca/global_k8_v1/` do not guarantee
that PNGs and CSVs exist or that plotting is complete. Before using a figure, perform the plotting and checks in this section.
Small CPU tests for code changes are as follows.

```powershell
uv run --no-sync pytest tests/experiments/test_global_pca.py -q
if ($LASTEXITCODE -ne 0) { throw 'PCA CPU tests failed' }
uv run --no-sync pytest tests/experiments/test_global_analysis.py -q -k "pca_export or report_png_csv or publish_report"
if ($LASTEXITCODE -ne 0) { throw 'PCA export/report tests failed' }
```

The output `global_k8_v1/pca-latent-2d/` is checked independently of its parent cluster figures and tables.
Regeneration of parent figures carries forward the existing `pca-latent-2d/`, excluding it from the parent's counts of 67 PNGs and 44 CSVs.
Input preparation requires parent figures and matching results consistent with the current code.
For saved coordinates, use the coordinate check and plotting commands at the end of this section. Check figures after plotting.
When preparing new inputs and coordinates, first generate the parent figures, then run the following in PowerShell from the repository root.

```powershell
& {
    # Extract representations of common pixels from saved models (host CUDA), then fit PCA on CPU.
    foreach ($step in @("prepare", "fit")) {
        uv run --no-sync python scripts/experiments/global_pca.py $step --resume
        if ($LASTEXITCODE -ne 0) { throw "PCA failed: $step" }
    }

    # Plot and check one PNG from the fixed coordinates.
    uv run --no-sync python scripts/experiments/visualize_global_pca.py run
    if ($LASTEXITCODE -ne 0) { throw 'PCA plotting failed' }
}
```

Only NN representation extraction in `prepare` uses CUDA (default device 0, chunk 1,024).
`fit` and PNG plotting run on CPU. Saved input arrays occupy approximately 517 MB plus a pixel CSV;
converting B0 to FP64 alone requires approximately 822 MB, so allow RAM for inputs and workspace during PCA.
PCA uses all common pixels once for each of the 5 conditions other than B1; B1 retrieves existing PCs. Plotting does not refit.
`--resume` validates and skips completed inputs and conditions.
To regenerate inputs and projections at the same v1 location, use `--overwrite` instead of `--resume` for both `prepare` and `fit`.
PCA fitting uses no random seed; only scatter overlap order is fixed with ROOT_SEED.

| Location (relative to `outputs/experiments/global_v1/`) | Content |
| --- | --- |
| `checkpoints/pca_projection/global_k8_v1/inputs/` | Per-condition FP32 arrays (B1 uses unnormalized scores of the top 2 components); `B1_projection.npz` contains existing coefficients, explained variance ratios, and original component indices; pixel CSV and provenance |
| `results/pca/global_k8_v1/{condition}/` | `projection.npz`, `coordinates.npy`, `coordinates.csv`, `components.csv`, `explained_variance.csv`, and execution/completion records |
| `results/figures/global_k8_v1/pca-latent-2d/01_pca_density_clusters.png` | One PNG with the specified 2-row, 6-column layout |
| CSVs in `results/figures/global_k8_v1/pca-latent-2d/` | `pixels.csv`, `hexbin_counts.csv`, `centroids.csv`, `explained_variance.csv`, and `captions.csv` |

Use the following to regenerate and check the figure from saved coordinates. `pca-latent-2d/` is replaced only after the new figure is complete.
For plotting-only changes, use `visualize_global_pca.py run`; omit the `run` line when only checking an existing figure.

```powershell
uv run --no-sync python scripts/experiments/global_pca.py check
if ($LASTEXITCODE -ne 0) { throw 'PCA coordinate check failed' }
uv run --no-sync python scripts/experiments/visualize_global_pca.py run
if ($LASTEXITCODE -ne 0) { throw 'PCA plotting failed' }
uv run --no-sync python scripts/experiments/visualize_global_pca.py check
if ($LASTEXITCODE -ne 0) { throw 'PCA figure check failed' }
```

<a id="global-mask-rate"></a>

### 8.4 Supplementary mask-rate comparison with global fitting

Train M11-75 once using M11's sampled pixels, settings, and seed, then fit and evaluate K8 Cosine-KMeans
with the same 3 seeds on the fixed representation. The 2×7 figure for the first fit places `M11(-50)` on top and `M11-75` below,
retaining the column order of the 7 previously selected specimens. Scope, metrics, replicates, and matching follow
[Section 1.4 of the visualization design](design/visualization_and_interpretation.md#global-mask-rate).

The default root is `outputs/experiments/global_v1/`. Existing 50% training, 3-seed diagnostics, and main figures and tables are referenced as sources,
so retain those artifacts and manifests. Use the following to check existing artifacts.

```powershell
uv run --no-sync python scripts/experiments/global_mask_rate.py check
if ($LASTEXITCODE -ne 0) { throw 'Global mask-rate report check failed' }
```

This validates inputs, provenance, saved labels, contracts, and artifact hashes on CPU without training or all-pixel inference.
Check exit code 0, `status=mask_rate_report_completed`, and `checks_passed=true`.
Execute unstarted or interrupted stages in the following order.

```powershell
& {
    foreach ($step in @('train', 'cluster', 'evaluate', 'report')) {
        uv run --no-sync python scripts/experiments/global_mask_rate.py $step --resume
        if ($LASTEXITCODE -ne 0) { throw "Global mask-rate step failed: $step" }
    }
}
```

`train` trains only M11-75; `--resume` resumes from the same run's checkpoint.
`cluster` performs 3 fits with fixed weights, `evaluate` calculates 6 metrics for all 49 specimens, and `report` creates comparison tables and figures from saved results.
`train`, `training-check`, `cluster`, and `evaluate` require CUDA (default `--device 0`);
`report` and `check` run on CPU. Training is a long GPU operation; refer to condition-specific training completion records for elapsed times and update counts.
The default `--chunk-pixels` for all-pixel processing is 1,024, and the default figure `--dpi` is 240.
Each stage validates its output after writing; `--resume` validates and skips completed outputs. `report --resume` does not redraw existing figures.
Outputs with inconsistent contracts, code, or runtime, and incomplete outputs, are not automatically replaced. This CLI has no `--overwrite`.

| Location (under global_v1) | Content |
| --- | --- |
| `results/neural/M11-75/repeat_1/` | Run, history, attempt, and completion records for 1 training run |
| `checkpoints/neural/M11-75/repeat_1/` | `last_model.pt` and resume checkpoint `checkpoints/last.pt` |
| `results/global_mask_rate_k8_v1/clustering/M11-75/repeat_{1,2,3}/` | K8 maps for 3 seeds, provenance, and completion records; centers are under the corresponding `checkpoints/global_mask_rate_k8_v1/` |
| `results/global_mask_rate_k8_v1/conditions/M11-75/` | Specimen-level metrics, SNV means, and evaluation provenance/completion records |
| `results/figures/global_mask_rate_k8_v1/` | 8 files comprising a 2×7 PNG, aggregate table, matching, captions, provenance, and completion record |

For the role of each file and interpretation of statistics, see the [artifact inventory](results.md#mask-rate-handoff).

<a id="nir-interpretation"></a>

### 8.5 NIR spectral interpretation figures

From saved within-specimen cluster means for global-fit M11 at K8, plot equally specimen-weighted representative spectra and
specimen-level second-derivative figures on CPU. This does not read all raw data or refit models.
It requires `report.json` and the wavelength, pixel-count, and spectral CSVs in `global_k8_v1` from the preceding sections.

```powershell
uv run --no-sync python scripts/experiments/visualize_nir_interpretation.py --scope both
```

Display clusters and reference curves for each specimen are read by default from
[`nir_plot_selection.yaml`](../scripts/experiments/nir_plot_selection.yaml).
Use `--report-dir` and `--output-dir` for input/output locations and `--sample-ids` for target specimens.
Add `--overwrite` only when existing PNGs need to be replaced.

The default output location is `outputs/nir_interpretation/`; the representative figure is `M11_equal_sample.png`,
and specimen-level figures are `each_nir_spectra/{sample_id}.png`.
The horizontal axis is converted to wavenumber, but the vertical axis remains the second derivative with respect to wavelength.
Check saved SNV, pseudo-absorbance, and second-derivative definitions, displayed clusters, specimen IDs, and legends.
The 18 numbered bands in the figures are literature-based reference positions; this CLI does not generate band-intensity bar charts or aggregate tables.

<a id="oof-aggregation"></a>

## 9. OOF aggregation

Run only after evaluations for 5 folds × 3 replicates are available for every specified condition.
Use `check` to inspect an existing snapshot and explicit `--overwrite` to reaggregate at the same location.

`main_oof_v1` aggregates 49 specimens, 120 source runs, and 82,320 score records for the 8 main conditions.
The following regenerates the same v1; run only `check` to inspect a saved snapshot.

```powershell
uv run --no-sync python scripts/experiments/aggregate_oof.py run `
    --conditions B0 B1 A0 A1 M00 M10 M01 M11 `
    --snapshot main_oof_v1 --overwrite `
    --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF aggregation failed' }
uv run --no-sync python scripts/experiments/aggregate_oof.py check `
    --snapshot main_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

For the supplementary mask-rate experiment, retain the main-condition snapshot and aggregate 49 specimens and 45 source runs from M11-25, M11, and M11-75
into the separate `mask_rate_oof_v1`.
The following is a reaggregation example; run only `check` to recheck an existing snapshot.

```powershell
uv run --no-sync python scripts/experiments/aggregate_oof.py run `
    --conditions M11-25 M11 M11-75 `
    --snapshot mask_rate_oof_v1 --overwrite `
    --experiment-dir outputs/experiments/production_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF aggregation failed' }
uv run --no-sync python scripts/experiments/aggregate_oof.py check `
    --snapshot mask_rate_oof_v1 `
    --experiment-dir outputs/experiments/production_v1
```

Check `status=oof_aggregation_completed` and `checks_passed=true` for run, and
`status=validated_existing_oof` for check. Do not ignore missing runs, failed runs, or incomplete specimen/pixel correspondence
when aggregating.

<a id="oof-sanity"></a>

### 9.1 B0, B1, A0, and M00 OOF sanity visualization

Read completed clustering and evaluation artifacts for all 5 folds × 3 replicates of B0, B1, A0, and M00 on CPU,
and generate 5 PNGs and 3 CSVs according to the [dedicated display specification](design/oof_sanity_visualization.md).
This requires neither an OOF snapshot covering all main conditions nor model refitting.

For saved figures, see `outputs/sanity_checks/a0_m00_oof_visualization/`.
When regenerating, specify the 4 conditions and a new output location explicitly. Replace the placeholder below with an output location that does not yet exist.

```powershell
uv run --no-sync python scripts/experiments/visualize_b0_b1_oof.py `
    --conditions B0 B1 A0 M00 `
    --experiment-dir outputs/experiments/production_v1 `
    --output-dir '<new-output-location>'
```

Check exit code 0 and the saved artifacts. Check that KYOw names appear below each specimen in the condition-specific representative 7-specimen figures,
and that silhouette has no lower subplot. Read uncorrected LLA, LFR, occupancy, and reasons for undefined values from CSVs.
Do not add logs or completion JSON to sanity outputs.

<a id="oof-reporting"></a>

### 9.2 Main-condition OOF figures and tables

Using `main_oof_v1` as the source, generate 3 PNGs and 11 CSVs for the 8 main conditions, all 49 specimens, and all 7 K values.
Representative metrics appear in the order LLA (saved keys `adjusted_lla_*`), LFR(TGN+FS), ARI, Cosine-Silhouette, and Cluster Occupancy.
Show LLA windows 3, 5, and 9 separately; keep Occupancy and interactions in CSV only.
For LFR, CSVs retain TGN+FS, TGN alone, and FS alone, while PNGs show only LFR(TGN+FS).
`source_metric` retains the original saved keys `lfr_both`, `lfr_noise`, and `lfr_shift`.
The 3 LLA panels in each PNG share a vertical-axis range; all panels in 01 and 02 use vertical-axis tick spacing of 0.1.
LLA tick spacing in 03 is set by the renderer, and vertical-axis ranges are set separately for different metrics.

```powershell
uv run --no-sync python scripts/experiments/visualize_main_oof.py `
    --experiment-dir outputs/experiments/production_v1 `
    --snapshot main_oof_v1
if ($LASTEXITCODE -ne 0) { throw 'OOF reporting failed' }
```

The default output location is `outputs/experiments/production_v1/results/figures/main_oof_v1/`.
Use `--output-dir` to change it; the default `--dpi` is 240.
Regeneration overwrites outputs managed by this renderer and deletes managed old-format PNGs, Occupancy figures, paired-distribution figures, and interaction figures.
Original OOF snapshots, clustering/evaluation artifacts, and unrelated files at the output location are retained.
Failures return a nonzero exit code. After input validation, the old `completion.json` is removed when output overwriting begins,
so a failure or interruption during overwriting cannot be treated as completed based on the previous success record.

| File | Content |
| --- | --- |
| `01_main_metrics_k_sweep.png` | Single-column-width 2×2 layout. Top: LLA 3 and 5; bottom: LLA 9 and LFR(TGN+FS). Curves over all K values for the 8 main conditions |
| `02_k8_distributions.png` | 2×3 layout. Top: LLA 3, 5, and 9; bottom: LFR(TGN+FS), ARI, and Cosine-Silhouette. $`K_0=8`$; black lines show macro means, and `n` is the common eligible count |
| `03_paired_k_sweep.png` | The same 2×3 layout as the distribution figure, showing differences M11−B0, M11−B1, M11−M00, A1−A0, and M11−A1 |
| `metrics_all_k.csv`, `metrics_k8.csv` | Representative metrics, 3 LFR variants, clean-test Occupancy, means, between-specimen SDs, between-replicate SDs, and eligible counts |
| `paired_all_k.csv`, `paired_k8.csv` | 12 comparisons: paired differences in LLA, 3 LFR variants, ARI, and Cosine-Silhouette |
| `interaction_all_k.csv`, `interaction_k8.csv` | 2×2 interactions for LLA 3, 5, and 9 and 3 LFR variants |
| `sample_values.csv` | Specimen-level and replicate values for conditions, paired comparisons, and interactions; inclusion in the common eligible set |
| `availability.csv` | Defined counts per replicate and undefined specimens with reasons |
| `ari_pairs.csv` | Values for the 3 ARI replicate pairs, reasons for undefined values, numbers of used clusters, and degeneracy flags |
| `occupancy_samples.csv`, `occupancy_folds.csv` | Clean-test specimen-level distributions, pixel-pooled distributions by fold and replicate, and specimen-macro distributions |
| `report.json` | Source hashes, generation-code hashes, targets, metric definitions, figure layouts, and information needed for captions |
| `completion.json` | `status=oof_figures_completed` and hashes of managed outputs |

For LLA, LFR(TGN+FS), and Cosine-Silhouette curves, solid lines show macro means; dashed, dotted, and dash-dotted lines show replicates 1, 2, and 3.
ARI averages the 3 replicate pairs within each specimen, then takes a specimen macro mean; paired ARI aggregates condition differences in these specimen means over common eligible specimens.
ARI and its differences have neither replicate-specific curves nor between-replicate SDs. For details, see the [reporting conventions](design/evaluation_metrics.md#reporting).

Corrected LLA interactions and paired ARI are computed from saved scores in the reporting pipeline.
It does not read raw spectra, weights, or maps, and does not rerun training, inference, or CV evaluation.
It checks metadata hashes and target correspondence for consumed snapshots and source artifacts, but does not replace a complete OOF `check`.
Check exit code 0, PNGs 01–03, 11 CSVs, completion status, and the display of text, legends, and eligible counts.

<a id="artifact-records"></a>

## 10. Storage and Git

Save configurations, manifests, numerical results, figures, and completion records with experiment outputs.
All of `outputs/`, `data/`, `*.pt`, `*.pth`, and `*.safetensors` are
excluded from Git tracking by `.gitignore` and are absent from the public repository.
For reanalysis, provide the corresponding inputs, weights, and execution records separately.

Save the following records per run. Schema and provenance handling follow [Section 11.2 of the experimental protocol](design/experiment_protocol.md#execution-records).

- Condition, fold, replicate, seed, and manifest/code/configuration hashes
- Status, epoch, attempted/optimizer updates, and AMP skips
- Total training epoch time, clustering/evaluation wall time and GPU peak allocated/reserved memory, and storage volume
- Source path and consistency checks when resuming
- Clustering/evaluation completion records and check results

Production training completion records do not store GPU peaks. Do not estimate training values after the fact; distinguish preflight measurements
as engineering reference values.
These are recording conventions for production experiments and are separate from OOF sanity outputs, which save only PNGs and CSVs.

Production CV fixes the code, design conditions, input data, and manifests.
If changes are needed, first confirm their scope and treatment of the experimental series, then record the changes and validation results.
