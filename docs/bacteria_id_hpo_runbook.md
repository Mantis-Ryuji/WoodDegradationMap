# Bacteria-ID Table 5 Runbook

This document describes reference30 pretraining, configuration selection by CV, and final training and evaluation
under the [classification protocol](design/bacteria_id_hpo_protocol.md). For input preparation, see the [Table 4 runbook](bacteria_id_runbook.md#inputs-and-preparation).
Run all commands in PowerShell from the repository root, using the configured environment.
Do not proceed to the next stage if a command returns a nonzero exit code.

## Inputs and output locations

| Purpose | Path |
| --- | --- |
| SNV inputs and splits | `data/processed/bacteria_id_v1/` |
| reference30 pretraining | `outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/` |
| HPO, final training, and test evaluation | `outputs/experiments/bacteria_id_table5_hpo_cv_v1/` |

Pretraining and final training use a single CUDA GPU. HPO trains 5 folds for 50 epochs each per configuration,
so it is a long GPU operation. Outputs include trained weights, run records, the Optuna study, and evaluation values.
`outputs/` and weights are excluded from Git tracking; the artifacts must be placed separately to inspect saved experiments.
If absolute paths in saved JSON need to be mapped to the current location, see `relocation_manifest.json` in each output location.

## reference30 pretraining

For each of M00/M11 and seeds 0–4, train for 800 epochs using only the 60,000 reference entries.
Do not use Table 4 checkpoints. The following is an example of one run.

```powershell
uv run --no-sync python scripts/experiments/bacteria_id.py pretrain `
    --corpus reference30 --condition M11 --seed 0 --device 0 `
    --output-dir outputs/experiments/bacteria_id_table5_reference30_pretrain_v1
```

Pass only new runs to `pretrain`. Resume the same interrupted run by explicitly specifying its
`checkpoints/last.pt` with `--resume`. Do not retrain or overwrite completed runs.

## Configuration selection by CV

The selection target, search space, seeds, epochs, and scope of records used for selection follow the
[classification protocol](design/bacteria_id_hpo_protocol.md#configuration-selection).
Use the following command to check and fix the selected configuration from the saved study.

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py select --completed-trials 60
```

`select` adds no CV training. It evaluates the first 60 completed configurations in trial-number order and
saves the result to `selected.json`. It stops if this differs from an existing selection.
Trial 38 is the selected configuration; the following settings are shared by M00/M11.

| Item | Selected value |
| --- | --- |
| Number of final blocks to update $`k`$ | 8 |
| Head learning rate | $`3\times10^{-4}`$ |
| Encoder block learning rate | $`8\times10^{-5}`$ |
| batch size | 8 |
| TGN / FS | off / off |
| weight decay | $`10^{-4}`$ |

Use `bacteria_id_hpo.py search` to search a new study; the implementation limit is 100 configurations.
This command does not check the saved selection from 60 configurations. `search` stops at an output location
where selection by the cutoff has been finalized. For reproduction that includes the random search history and execution environment, refer to the saved study and run records.

## Final training, test evaluation, and aggregation

Adapt M00/M11 × 5 seeds to all 3,000 finetune entries using the selected configuration.
Use only the final weights at 50 epochs, and score the 3,000 distributed test entries once per run.

```powershell
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py train --device 0
if ($LASTEXITCODE -ne 0) { throw 'Table 5 final training failed' }
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py evaluate --device 0
if ($LASTEXITCODE -ne 0) { throw 'Table 5 evaluation failed' }
uv run --no-sync python scripts/experiments/bacteria_id_hpo.py tables
if ($LASTEXITCODE -ne 0) { throw 'Table 5 reporting failed' }
```

Whether training or evaluation can resume is determined from each run's contract and completion record. Do not alter output locations or selected settings to reuse them.
Use `tables` for aggregation only. Do not score unselected configurations on test data.

## Artifacts and checks

Cross-reference the following records in the HPO output location.

| Saved artifact | Content |
| --- | --- |
| `selected.json` | Selected trial, selection target, per-fold and mean validation Accuracy, and configuration |
| `cv_splits.json` | Train/validation indices for each fold |
| `final/` | Classification training records and weights for M00/M11 × 5 seeds |
| `evaluation/` | Test evaluation for each seed |
| `table5.md` | Main table showing published values alongside this study's values |
| `table5_statistics.json` | Per-seed values, means and 95% confidence intervals, and search results |

Check each seed's Accuracy and the t confidence interval with 4 degrees of freedom computed from the sample SD of 5 seeds.
For differences from published conditions and the meaning of the shared configuration selected with M11, see the [interpretation of results](bacteria_id_interpretation_plan.md).
