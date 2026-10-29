# Output guide

This guide lists output locations for each analysis and explains the scope of the results. Paths are relative to the repository root.
`outputs/` is not tracked by Git; the files below are generated locally.
See the [experiment runbook](experiment_runbook.md) for execution and the [evaluation specification](design/evaluation_metrics.md) for metric definitions.

## Preprocessing and input checks

| Output location | Contents |
| --- | --- |
| `outputs/preprocessing/production_v1/` | Wavelength cutoff, bandwise reflectance and SNV distributions, and quality-control figures |
| `outputs/experiments/production_v1/manifests/` | CV specimen splits, common sampled pixels, conditions, and seeds |
| `outputs/sanity_checks/augmentation_strengths_train_fold1/` | TGN/FS transformation examples and strength summaries using training pixels |

Candidate-pixel figures from preprocessing are for quality control, not ground-truth degradation or anomaly labels.
Perturbation examples are based on observations of training pixels and are distinguished from results optimized against CV metrics.

## Main experiment: OOF evaluation

Aggregated results are stored in `outputs/experiments/production_v1/results/oof/main_oof_v1/`,
and figures and tables in `outputs/experiments/production_v1/results/figures/main_oof_v1/`.

| File or content | Interpretation |
| --- | --- |
| `01_main_metrics_k_sweep.png` | LLA-3, LLA-5, LLA-9, LFR(TGN+FS), and their dependence on K for the eight main-experiment conditions |
| `02_k8_distributions.png` | Specimen-level distributions and macro means at K=8 |
| `03_paired_k_sweep.png` | Paired differences between conditions for the same specimen, K, and replicate |
| CSV tables | Condition-specific metrics, paired differences, perturbation interactions, occupancy, counts, and undefined-value information |
| `report.json` | Metric and display definitions and the sources used for aggregation |

The analysis covers 49 specimens, five folds, three training replicates, and K=2/4/6/8/10/12/14.
LLA denotes the chance-corrected value stored as `adjusted_lla`; the uncorrected value is stored as `lla`.
Undefined values are not imputed, and the numbers of specimens contributing to condition summaries and differences are reported.

`outputs/sanity_checks/a0_m00_oof_visualization/` contains diagnostic figures for B0, B1, A0, and M00.
Use the main experiment aggregates for evaluation of all eight conditions.

## Global-fit K8 maps and spectra

The output directory is `outputs/experiments/global_v1/results/figures/global_k8_v1/`.

| File or content | Purpose |
| --- | --- |
| `01_representative_samples_6x7.png` | Comparison of the six main conditions on seven specimens selected for display |
| `M11/labels/` | M11 cluster maps for all 49 specimens and the seven display specimens |
| `M11/01_representative_spectra.png` | SNV, reflectance, and second derivatives of pseudo-absorbance; means with equal specimen weights and inter-specimen IQR |
| `M11/sample_spectra.csv` | Mean observed spectra by specimen and display cluster |
| `M11/spectrum_counts.csv` | Pixel counts and exclusions contributing to spectral summaries |

These maps are descriptive analyses fitted on all 49 specimens, not OOF predictions.
Each of the six main representations is trained or fitted once, and KMeans is run with three initializations.
The first initialization is used for display. Colors across conditions are aligned to M00 by matching representative SNV curves.
The same color does not guarantee the same chemical state.

The seven display specimens appear in this column order: KYOw02752, KYOw02772, KYOw02777, KYOw02787, KYOw02790, KYOw16744, KYOw16750.
They are examples selected from each tree species and are not guaranteed to be typical of all specimens or chemical states.
See [global-fit PCA](experiment_runbook.md#global-pca) for coordinates and plotting procedures.

## NIR spectral interpretation

`scripts/experiments/visualize_nir_interpretation.py` reads saved CSV files of M11 within-specimen means
and writes `outputs/nir_interpretation/M11_equal_sample.png` and `each_nir_spectra/<sample_id>.png`.
The clusters and reference curves compared for each specimen are specified in the [selection configuration](../scripts/experiments/nir_plot_selection.yaml).
Reference curves from other specimens identify their specimen and cluster IDs and are not included in the target specimen's aggregation.

The horizontal axis is converted to wavenumber, but the second derivatives on the vertical axis remain derivatives with respect to wavelength.
The [NIR analysis specification](design/visualization_and_interpretation.md#nir-interpretation) defines pixel exclusions for pseudo-absorbance,
aggregation with equal specimen weights, 18 bands used for assignments, and nine displayed bands.
The NIR plotting CLI in this repository outputs spectral curves. Code for the paper's band-value aggregation and baseline bar charts is not included.

<a id="mask-rate-handoff"></a>

## Mask-ratio sensitivity analysis

| Analysis | Output location and main files | Evaluation scope |
| --- | --- | --- |
| CV, 25%/50%/75% | `outputs/experiments/production_v1/results/figures/mask_rate_oof_v1/`: `01_mask_rate_metrics_k_sweep.png`, `metrics_all_k.csv`, `metrics_k8.csv`, `report.json`, `completion.json` | Five folds × three training replicates with the same splits and sampled pixels; all 49 specimens and all seven K values |
| Global fit, 50%/75% | `outputs/experiments/global_v1/results/figures/global_mask_rate_k8_v1/`: `01_representative_samples_2x7.png`, `summary.csv/json`, `sample_metrics.csv`, `matching.csv`, `captions.csv`, `run.json`, `completion.json` | Descriptive comparison using three K8 KMeans initializations after training each representation once |

The 50% condition reuses the main M11 condition and is displayed as `M11(-50)`.
The global-fit 2×7 figure shows assignments from the first initialization, with 50% in the upper row and 75% in the lower row.
Inter-specimen SD and inter-replicate SD are distinguished; the latter in global fitting does not represent variability from neural-network retraining.

## Spatial map reproducibility

The output directory is `outputs/experiments/global_spatial_reproducibility_v2/`.
For M11/K8, nine maps from 3 encoders × 3 KMeans initializations are compared over all 36 pairs using the same pixel sets.
Read the pairwise, specimen-level, and overall aggregates in `results/evaluation/` alongside the maps and boundary figures in `results/report/`.
The [dedicated runbook](spatial_map_reproducibility_runbook.md) describes the input trained weights and maps and the relationships among reused artifacts.
The current CLI reevaluates saved artifacts and has no training action for encoders 2 and 3.
Execution requires trained weights and reused maps that are not included in Git.

## Raman Bacteria-ID

| Analysis | Output location | Main outputs |
| --- | --- | --- |
| Bacteria-4/6 clustering | `outputs/experiments/bacteria_id_table4_80_20_v1/` | `table4.md`, `table4_seed_scores.csv`, `table4_statistics.json`, `table4_seed_scores.png` |
| Classification of 30 isolates and HPO | `outputs/experiments/bacteria_id_table5_hpo_cv_v1/` | `table5.md` and records of HPO, selected configurations, and per-seed evaluation |
| Pretraining for classification | `outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/` | M00/M11 reference30 pretraining records |

The artifact names `table4` and `table5` come from the comparison tables in the referenced SMAE paper.
Quoted literature values are distinguished from results rerun in this repository.
See the [Bacteria-ID specification](design/bacteria_id_experiment_plan.md) and [HPO specification](design/bacteria_id_hpo_protocol.md)
for differences in splits, preprocessing, and evaluation conditions, and for the scope of configuration selection.
