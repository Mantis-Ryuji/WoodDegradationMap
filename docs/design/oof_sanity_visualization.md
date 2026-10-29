# B0, B1, A0, and M00 OOF Sanity Visualization

After all CV runs for B0, B1, A0, and M00 are complete, exploratory diagnostic figures are generated
from the existing clean OOF maps and evaluation artifacts. PCA/KMeans fitting, representation extraction,
and evaluation are not rerun. See the [runbook](../experiment_runbook.md#oof-sanity) for execution
and [interpretation of results](../interpretation_notes.md) for guidance on reading the outputs.

## 1. Saved Artifacts

The output directory is `outputs/sanity_checks/a0_m00_oof_visualization/`.
Only the following five PNGs and three CSVs are saved; no sanity-specific logs or run/completion/failure JSON files are created.

| File | Contents |
| --- | --- |
| `labels/B0_representatives_k8_repeat1.png` | B0 label maps for the seven representative specimens in a single row |
| `labels/B1_representatives_k8_repeat1.png` | B1 label maps for the same seven specimens in the same order |
| `labels/A0_representatives_k8_repeat1.png` | A0 label maps for the same seven specimens in the same order |
| `labels/M00_representatives_k8_repeat1.png` | M00 label maps for the same seven specimens in the same order |
| `silhouette_k_sweep.png` | K-sweep on a single axes; no lower subplot |
| `occupancy.csv` | Pixel counts per cluster, number of used clusters, maximum occupancy, and single-cluster flag by condition, specimen, fold, replicate, and K |
| `matching.csv` | Mapping from raw IDs to B0 display IDs, contingency, and overlap for each fold and comparison condition |
| `metrics_summary.csv` | Aggregates for silhouette, uncorrected LLA-3/5/9, LLA-3/5/9, and LFR for noise, shift, and both |

Uncorrected LLA, LFR, and occupancy are inspected in the CSVs. Boundary maps and representative spectra are outside the scope of this visualization.

## 2. Label Maps of Representative Specimens

At $`K_0=8`$, replicate 1, the [fixed seven specimens](visualization_and_interpretation.md#representative-samples)
are arranged in ascending specimen-number order (hinoki cypress, pine, zelkova, hemlock, chestnut, Japanese cedar, and fir),
with one figure saved for each of B0, B1, A0, and M00. Specimen IDs in the saved figures are checked against the representative-specimen definition.

Each specimen is labeled with its `KYOw...` ID below the map, with a shared legend for background 0 and clusters 1–8.
The [shared plotting conventions](visualization_and_interpretation.md#figure-style) apply.

## 3. Within-Fold Hungarian Matching

For all test specimens in each fold, contingency counts over valid pixels shared by B0 and each comparison condition
are summed, and a one-to-one correspondence maximizing the number of matching pixels is obtained.
B1, A0, and M00 are each aligned directly to B0, using the same correspondence for every specimen within the same fold and comparison condition.
Matching is not restricted to the seven representative specimens, and a separate correspondence is not fitted for each specimen.

The correspondence is applied to copies of the maps used for display. Saved CV labels and metrics remain unchanged;
per-cluster pixel counts in occupancy use raw ID order. The same color across folds does not necessarily represent the same state.
Correspondences with zero or weak overlap are inspected in the CSVs.
The scope and objective differ from the [M00 + Cosine-KMeans reference matching based on cosine similarity of observed SNV representative spectra](visualization_and_interpretation.md#matching-reference) used after global fitting.

## 4. Numerical Aggregation and Silhouette

All 49 specimens, five folds, three replicates, and all seven K values are included, following the
[evaluation aggregation conventions](evaluation_metrics.md#reporting). For each condition, K, and metric,
specimens with defined values in all three replicates are aggregated. The mean, between-specimen SD,
between-replicate SD, replicate-specific means, valid counts, excluded specimens, and reasons for undefined values
are saved to CSV. Undefined values are not replaced with zero.

In the silhouette figure, the thick line is the specimen macro mean over three replicates, thin lines are the
macro means for individual replicates, and the band is the between-replicate SD (`ddof=1`).
The specimens included in the mean and reasons for undefined values are listed in the CSVs.
This figure is not used to select a condition, K, or best run.

## 5. Input and Output Handling

Input completion records, scores, OOF assignments, and map correspondence are validated; existing CV artifacts are not modified.
An existing output directory is rejected. Regeneration uses an explicitly specified new output directory.
These artifacts are separate from the interpretation figures after global fitting and the manuscript figures and tables after all CV runs are complete.
