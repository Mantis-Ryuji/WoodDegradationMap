# Bacteria-ID Table 4 Runbook

This document covers input preparation and Table 4 pretraining and clustering with an 80/20 split.
For Table 5 pretraining and classification, see the [dedicated runbook](bacteria_id_hpo_runbook.md).
For the condition definitions and limitations of the comparisons, see the [experimental specification](design/bacteria_id_experiment_plan.md).
Run the commands below in PowerShell from the repository root, using the configured environment.
Do not proceed to the next stage if a command returns a nonzero exit code.

## Inputs and preparation

The required original files in `data/raw/bacteria_id/` are `X_reference.npy`, `y_reference.npy`,
`X_finetune.npy`, `y_finetune.npy`, `X_test.npy`, and `y_test.npy`.
If `wavenumbers.npy` is available, record the axis shape, monotonicity, and range.
The wavenumber axis of the adopted input descends from 1792.40 to 381.98 cm⁻¹; do not reverse the channel order.
Verify that the original float64 labels are finite integers before converting them to integer IDs.

The inputs were obtained through the [public mirror instructions in bacteria-SANet](https://github.com/DenglinGo/bacteria-SANet).
The download URL, archive, and extracted files are recorded in `data/raw/bacteria_id/acquisition.json`.
Byte-for-byte identity with the official distribution has not been verified. Do not modify the originals.

Prepare the SNV inputs and splits in an output location that does not yet exist.

```powershell
uv run --no-sync python scripts/experiments/bacteria_id.py prepare
```

The preparation destination is `data/processed/bacteria_id_v1/`. Do not regenerate existing inputs; pass their manifest and
`splits.npz` to the Table 4 split preparation. For the split definition and provenance,
see the [experimental specification](design/bacteria_id_experiment_plan.md#table4-split).

## Table 4 split preparation

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py prepare
```

Combine the saved train and validation indices, preserving the test indices and their order, to create
an 80/20 split in `outputs/experiments/bacteria_id_table4_80_20_v1/`.
The originals, SNV data, and input splits remain unchanged. Reads primarily involve indices and labels; the full spectra are not preprocessed again.
The command stops if the output location already exists.

Check that `split_sizes` reports train 6,400/test 1,600 for Bacteria-4, train 9,600/test 2,400 for Bacteria-6,
train 1,600/test 400 per class, and that `planned_runs` is 20.
The preparation code verifies split disjointness, coverage of all eligible entries, and exact agreement of test indices.
`manifest.json` stores the input manifest and split hashes, counts, run identities, and seeds for each purpose.

## Pretraining and clustering

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py all --device 0
```

Run the 20 combinations of Bacteria-4/6 × M00/M11 × seeds 0–4 sequentially on a single CUDA GPU.
Each run trains from initialization on train data only for 800 epochs with batch size 1024.
With `drop_last=True`, Bacteria-4 has 6 batches/epoch and 4,800 attempted updates;
Bacteria-6 has 9 batches/epoch and 7,200 attempted updates. See `completion.json` for actual updates after AMP skips.
This is a long GPU operation that saves checkpoints and evaluation results to the output location.

`all` validates and skips completed runs, and resumes interrupted runs from the same run's `last.pt`.
Known incomplete clustering outputs (an empty results directory and only `centroids.npz`) are removed before
clustering is rerun; other incomplete outputs stop processing and require the cause to be checked.
Table 5 checkpoints are not loaded. Once all 20 evaluations are available, the command creates the main table, per-seed CSV, statistics JSON, and per-seed plot.

Example of separate execution:

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py pretrain --corpus bacteria4 --condition M00 --seed 0 --device 0
if ($LASTEXITCODE -ne 0) { throw 'Table 4 pretraining failed' }
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py cluster --corpus bacteria4 --condition M00 --seed 0 --device 0
```

Example of resuming the same interrupted pretraining run:

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py pretrain --corpus bacteria4 --condition M00 --seed 0 --device 0 --resume outputs/experiments/bacteria_id_table4_80_20_v1/checkpoints/pretrain/bacteria4/M00/seed_0/checkpoints/last.pt
```

## Aggregation and checks

Use the following after completing all 20 evaluations separately, or to produce tables from saved evaluations.

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_table4_80.py report
```

Compare `table4.md` in the output location against all 20 rows of `table4_seed_scores.csv` and
the means and sample SDs (`ddof=1`) in `table4_statistics.json`.
`table4_seed_scores.png` shows each seed's value and the mean ± sample SD.
`outputs/` and checkpoints are excluded from Git tracking.

The following small test applies when input, split, or evaluation processing is changed. It uses no real data or GPU.

```powershell
uv run --no-sync python -m pytest tests/experiments/test_bacteria_table4_80.py
```
