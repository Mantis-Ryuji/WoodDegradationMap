# Visualization and Interpretation after Global Fitting

## 1. Scope

This document defines descriptive analysis after fitting on all 49 specimens.
[OOF sanity visualization](oof_sanity_visualization.md) is a separate stage that reads existing predictions from fold models.
Execution records follow the [protocol](experiment_protocol.md#execution-records).
The observation granularity is $`K_0=8`$; Cosine-KMeans, maps, representative spectra, and PCA are compared across six conditions.
Chemical interpretation of M11 uses NIR spectral features and spatial distributions together with literature assignments,
wood anatomy, environmental history, and specimen information.
See [results and figures](../results.md) for artifacts and the [runbook](../experiment_runbook.md#global-fit-pipeline) for execution and verification.
Distinguish the fitting scope and replicates of the mask-ratio sweep's [supplementary CV experiment](experiment_protocol.md#mask-rate-sweep)
from those of the descriptive comparison after global fitting.

- [Global fitting conditions](#global-fit)
- [Supplementary mask-ratio comparison after global fitting](#global-mask-rate)
- [Matching to M00](#matching-reference)
- [Shared plotting conventions](#figure-style) and [representative specimens](#representative-samples)
- [PCA](#latent-spectral-maps) and [NIR spectral analysis](#nir-interpretation)
- [Evidence used for interpretation](#evidence-triangulation)

### 1.1 Conditions and Role

Global training includes the following six conditions. Their definitions and the role of each comparison follow the
[protocol](experiment_protocol.md#conditions).

| ID | Condition | Role |
| --- | --- | --- |
| B0 | raw SNV | Input-space baseline |
| B1 | PCA | Linear baseline |
| A0 | AE 0% | Baseline with no mask and full-spectrum reconstruction |
| A1 | Denoising AE 0% | Full-spectrum reconstruction with the same TGN + shift as M11 |
| M00 | MAE 50% | Standard MAE |
| M11 | Aug-MAE 50% | Proposed condition |

Apply Cosine-KMeans to each of the six conditions at the representative display value $`K_0=8`$.
This analysis describes state distributions across all specimens. Coherence and stability on unseen specimens are assessed by CV;
correspondence with degradation is discussed using the evidence in Section 5.

<a id="global-fit"></a>

### 1.2 Inputs and Representations for Global Fitting

Uniformly sample $`q=8192`$ pixels without replacement from each of the 49 specimens, as in the experiment protocol.
All conditions share the same sampled coordinates. PCA and ChemoMAE are fitted or trained on the pooled sampled pixels from all specimens.
Cosine-KMeans is then fitted to each condition's representation.
After fitting, fix the models and infer over all valid pixels of every specimen to produce maps and spectral summaries.
Clustering is not refitted separately for each specimen.
Save the sampling seed and coordinate manifest for global training.
Although all specimens contribute to fitting, distinguish fitting pixels from displayed pixels.
Representations, all-visible extraction, and precision follow the [same specification as CV](experiment_protocol.md#representations).
Two-dimensional projections are supplementary displays after clustering; they do not replace the original representations as clustering inputs.

### 1.3 Replicates, Seeds, and Clustering

For the initial fit used for display, fit each transformer and clustering model once using the prespecified seed for replicate ID 1.
Use ROOT_SEED=20260905 and the existing SHA-256 scheme, replacing the fold position in seed derivation with the string `global`.
Purposes remain the same: `(sampling, global, sample_id)` for sampling,
`(purpose, global, 1)` for initialization, pixel order, masks, augmentation, and PCA,
and `(kmeans, global, 1, 8)` for clustering.
Condition and method names are excluded from seeds so they are shared; save the list of 55 seeds to `config/seeds.json`.
The storage root is `outputs/experiments/global_v1/`. Config and manifest schemas are version 1;
NN run records use schema 2 of the existing contract/execution scheme.
Shared sampled coordinates are saved to `manifests/fit_pixels.parquet` without a fold column.
CV splits, seeds, and artifacts remain unchanged.
Train A0, A1, M00, and M11 once each, four runs in total, using the same mask, loss, augmentation, and 800-epoch recipe as CV.
Fit Cosine-KMeans once for each of the six representations, six fits in total.
Record these separately from the 120 planned CV training runs including supplementary conditions.
Seeds are not replaced with ones that performed well in CV; reproducibility across training replicates is reported using CV ARI.

In `global_k8_3seed_v1`, fix each condition's representation and repeat only KMeans initialization with three seeds.
Replicate 1 retains the original fit; replicates 2 and 3 are derived from `(kmeans, global, repeat, 8)`.
NNs are not retrained, and displayed figures also use replicate 1.
The between-replicate SD in this diagnostic reflects only KMeans initialization variability.

Keep global-fit interpretation artifacts separate from CV evaluation and OOF aggregation.
Condition-specific fit records retain correspondence with the source representations, sampled coordinates, settings,
and source hashes, together with completion/failure status.

<a id="global-mask-rate"></a>

### 1.4 Supplementary Mask-Ratio Comparison after Global Fitting (Appendix)

Train M11-75 once on all 49 specimens and compare it with the existing 50% M11 condition.
Reuse the existing shared 49×8,192 = 401,408 sampled coordinates, model architecture, 800-epoch recipe,
and prespecified seeds for initialization, pixel order, masks, and augmentation.
Only the mask ratio changes, from eight to 12 masked patches out of 16.
Reuse the trained weights and evaluation for 50%.

Fix the final raw weights at epoch 800 for each condition, and perform three K=8 Cosine-KMeans replicates on the
16-dimensional representations extracted with all patches visible.
Use the same replicate 1, 2, and 3 seeds as the existing global-fit diagnostic (1960870825, 322674335, and 2051198256);
do not select favorable seeds.
After fitting, evaluate all 3,902,250 valid pixels across all 49 specimens.
The compared metrics are corrected LLA at widths 3, 5, and 9, Cosine-Silhouette,
per-specimen used cluster count, and maximum occupancy; LFR and ARI are not added to this supplementary comparison.
Silhouette is calculated over the pooled valid pixels of all specimens using the same cosine distance and FP64 aggregation as before.
Each metric is aggregated with equal weights over specimens for which it is defined in both conditions and all three replicates;
paired differences M11-75 − M11(-50) are also saved.
`sample_sd` is the sample SD of the three-replicate mean for each specimen;
`repeat_sd` is the sample SD of the three specimen macro means (both use ddof=1).
The latter reflects only KMeans initialization variability, not variability from NN retraining.

The figure uses two rows and seven columns from the initial fit (replicate 1), with `M11(-50)` above and `M11-75` below;
columns retain the order of the [seven selected specimens](#representative-samples).
Saved condition IDs remain `M11` and `M11-75`.
The 50% condition reuses the display labels from the existing 6×7 figure;
the 75% condition is matched directly to M00's observed SNV representative spectra by Hungarian matching.
Color correspondence is used only for display; metrics are calculated on the original labels.
A 75% cluster absent across all specimens is excluded from matching; remaining display IDs are assigned in raw-ID order,
with similarity recorded as undefined. No pixels are imputed for absent clusters.

This is an appendix supplementary comparison.
Partitions along growth rings, peripheral/interior divisions, fine intermixing, and region continuity are examined descriptively.
All specimens contribute to fitting, so the figure is not treated as OOF evaluation on unseen specimens or a formal visual evaluation.
It is not used to select an optimal mask ratio or replace the main M11 condition.
See the [runbook](../experiment_runbook.md#global-mask-rate) for storage locations and verification commands,
and [results and figures](../results.md#mask-rate-handoff) for the artifact list.

## 2. Label Conventions

- Background is 0.
- Clusters are numbered from 1 to $`K_0`$.
- Cluster numbers do not encode order or degradation severity.
- Label alignment does not change cluster membership.

## 3. Hungarian Matching Using Observed SNV Representative Spectra

<a id="matching-reference"></a>

### 3.1 Reference Condition

The partition obtained by applying Cosine-KMeans to M00 representations (standard MAE without additional augmentation)
is the shared reference for display IDs across all six conditions.
Match B0, B1, A0, A1, and M11 independently and directly to this reference.

Standard MAE is the starting point of the comparison, providing a display reference for reading changes in region partitioning from M00 to M11.
This choice does not rely on M00 having the best verified stability, nor does it guarantee degradation-detection performance,
interpretability, or an ordering of clusters.
The [within-fold B0 reference for OOF sanity plots](oof_sanity_visualization.md#3-within-fold-hungarian-matching) is a separate display convention.

Do not match separately for each specimen or chain matching through another condition.
Apply the same convention to all six conditions; do not use a separate matching-pixel-count method only for B0 versus B1.

### 3.2 Cosine Similarity of Observed SNV Representative Spectra

Use only wood pixels valid for all compared conditions across all specimens, sharing the same coordinates across the six conditions.
Matching uses cosine similarity between observed SNV representative spectra.
Pixels are identified by specimen ID and coordinates, with hard-label membership fixed.
Arithmetically average saved pixelwise SNV spectra within each specimen and cluster,
then average those specimen means with equal weights over specimens in which the cluster is present.
These representative spectra are the same as those in the SNV plots in Section 4.1;
SNV is not reapplied to mean reflectance.
Let the representative spectrum of reference cluster i be $`\boldsymbol{s}_i`$ and that of comparison cluster j be $`\boldsymbol{t}_j`$.
Construct an 8×8 matrix using all 256 bands.

```math
S_{ij}=\frac{\boldsymbol{s}_i^{\mathsf T}\boldsymbol{t}_j}
{\lVert\boldsymbol{s}_i\rVert_2\lVert\boldsymbol{t}_j\rVert_2}
```

L2-normalize representative spectra and take their dot products, retaining signed similarity (−1 to 1).
Do not weight specimens by pixel count, restandardize representative spectra along wavelength, or select bands.
Background 0 is excluded from the matrix and assignment.
For the main six conditions, if a cluster is empty across all specimens or a representative spectrum is nonfinite or has zero norm,
stop and report the condition and raw cluster ID.
Absent clusters in the 75% supplementary comparison follow [Section 1.4](#global-mask-rate).
Curves for clusters absent within a specimen are excluded from averaging, not filled with zero.
Latent vectors, reflectance, pseudo-absorbance, second derivatives, metadata, and IoU are not included in the assignment objective.

Hungarian assignment maximizes the sum of SNV cosine similarities.

```math
\max_{\pi}\sum_{i=1}^{K_0}S_{i,\pi(i)}
```

Obtain this one-to-one correspondence $`\pi`$ for each comparison condition, replacing its cluster $`\pi(i)`$ with display ID $`i`$.
Cluster membership, trained models, and clustering centers are unchanged.

### 3.3 Contingency and Overlap for Inspecting Correspondence

Also compute a contingency matrix from reference and comparison hard labels over the same shared valid pixels.

```math
C_{ij}
=\sum_p\mathbf{1}[y_p^{(\mathrm{ref})}=i\land y_p^{(c)}=j]
```

Save the full SNV cosine similarity matrix, the selected correspondence and its similarity,
and similarity differences from other candidates for the same reference cluster.
Save the IoU matrix, contingency matrix, and matching pixel counts/overlap after correspondence as supplementary CSV values.
IoU is calculated as $`C_{ij}/(\sum_\ell C_{i\ell}+\sum_\ell C_{\ell j}-C_{ij})`$.
IoU and matching pixel counts are not used for assignment; they inspect the spatial overlap of spectrally matched clusters.
Overall overlap is $`\sum_i C_{i,\pi(i)}/\sum_{i,j}C_{ij}`$.
State that it is based on all shared pixels and therefore gives larger specimens greater weight.

Identify low-similarity correspondences and those with closely competing candidate similarities;
the same color alone does not imply a strong correspondence.
Equally optimal assignments are not treated as evidence of a unique correspondence.
One-to-one assignment can still be obtained when SNV similarity is low.
Inspect the full matrix for differences in cluster splitting and merging; identical display IDs do not guarantee chemical identity.
SNV proximity is the matching criterion itself, not independent chemical validation.
Read reflectance, pseudo-absorbance, and second derivatives alongside it.
Use the correspondence table to relate raw cluster IDs to display IDs.

### 3.4 Scope of Application

The resulting correspondence is shared across all specimens and applied consistently to:

- Label-map colors and IDs.
- Cluster sizes and occupancy.
- Representative and difference spectra.
- Per-cluster supplementary statistics.

Save the mapping between raw cluster IDs and display IDs.
Matching supports display and inspection of correspondence; it does not guarantee identical meanings for clusters from different conditions.
Report the similarity matrix together with contingency and overlap, and do not interpret weakly matched clusters by shared color alone.
Use matching as display-label alignment, not for calculating CV metrics or measuring accuracy against the shared reference as ground truth.

Comparisons across conditions use only pixels verified as identical through `pixel_row_col` in the production preprocessing HDF5.
Do not change pixel sets by condition or spatially correct label maps.

## 4. Visualization and Interpretation Outputs

<a id="figure-style"></a>

The following plotting conventions apply to all visualizations in the repository, including preprocessing diagnostics.

- Do not add figure or axes titles, except for condition names in the top row of the PCA comparison figure in Section 4.3.
- Manage figure meaning, condition names, and panel descriptions through captions or filenames.
- Display the specimen ID (the name beginning with KYOw) below each specimen in cluster maps.
- Show axis labels, ticks, legends, and colorbar labels when needed to interpret values.
- Ranks, specimen IDs, pixel coordinates, and example IDs used to trace individual spectra may appear as panel annotations.
- Fix the main spectrum y-axis to `[0, 1]` with tick spacing `0.2` for reflectance, and `[-2, 2]` with spacing `0.5` for SNV.
  Apply the same convention to preprocessing distribution plots and augmentation examples.
  Display limits must not clip saved values or exclude pixels.
- For augmentation diagnostic residuals (perturbed − clean), fix the y-axis to `[-0.5, 0.5]` with tick spacing `0.1` for both noise and shift.
- For preprocessing and augmentation diagnostic plots with wavelength on the x-axis, including residuals,
  use the first and last plotted wavelengths as the axis endpoints, with no horizontal data-range padding.
  Add ticks at both endpoints, displayed to two decimal places.
  Interior ticks are the values among `1000, 1200, 1400, ...` nm that lie within the displayed range, shown as integers.
  Post-interpolation plots use `913.10`–`2305.59` nm; cutoff plots use both ends of the original measured grid, including excluded bands.
  Rotate x-axis tick labels by 45 degrees in small individual-candidate panels to avoid overlap.
- In these wavelength plots, make both major and minor ticks on both axes point outward.
  Show tick marks on the bottom and left edges, not the top and right.
  Leave space between tick labels and axis lines so endpoint numbers remain legible.
  Do not add horizontal padding outside the data range.

The following outputs are saved after global fitting.
OOF sanity artifacts follow the [dedicated specification](oof_sanity_visualization.md).

| Output | Purpose |
| --- | --- |
| Label map | Compare cluster distributions on specimen surfaces across conditions |
| SNV cosine similarity matrix | Inspect assignment to the M00 reference and the strength or ambiguity of representative-SNV correspondence |
| IoU and contingency/overlap (CSV) | Inspect spatial overlap, splitting/merging, and matching pixel counts for SNV-matched clusters |
| Representative spectra | Inspect each cluster's NIR spectral shape |
| Difference spectra | Identify wavelength regions with large cluster differences |
| Cluster size distribution / occupancy | Inspect collapse, undersized clusters, and imbalance |
| PCA (PC1 and PC2) | Observe pixel distributions and cluster membership in representation space |
| Specimen-specific NIR spectra and band values | Relate map regions to observed spectral features |

Use the same specimen order, label palette, and display ranges for B0, B1, A0, A1, M00, and M11.

Cosine-KMeans figures and tables are saved under `results/figures/global_k8_v1/`.
Separate conditions into `B0/`, `B1/`, `A0/`, `A1/`, `M00/`, and `M11/`.
Each `labels/` directory contains seven 1×7 maps (00–06) with seven specimens each in ascending ID order,
one 1×7 map of the fixed seven representative specimens (07), and one 7×7 map of all 49 specimens (08).
The representative 1×7 maps share their specimens and order with the root-level six-condition × seven-specimen figure.
Use image widths, margins, and large bold specimen IDs matching `outputs/sample_overviews`;
preserve original orientation, aspect ratio, and field of view with nearest-neighbor interpolation that does not mix colors.
The colorbar label is `Cluster ID`.
The root-level `01_representative_samples_6x7.png` displays the seven fixed representative specimens from Section 4.2
in that section's order as seven columns, with B0, B1, A0, A1, M00, and M11 as six rows from top to bottom.
Specimen IDs appear only below the bottom row, condition names to the left of each row,
and a shared `Cluster ID` colorbar below.
Representative spectra for each condition are combined into one figure:
SNV at top left, reflectance at top right, and SG second derivatives across the bottom.
Each panel shows mean curves and between-specimen IQRs for eight clusters, using the same colors as the maps.
The second-derivative y-axis range is shared across all six conditions.
Major wavelength ticks are spaced 100 nm apart and minor ticks 50 nm apart, with gridlines for both inside the plot.
Display endpoints to two decimal places; for major ticks less than 30 nm from an endpoint,
retain the line but omit the number to prevent label overlap.
Save one 8×8 SNV cosine similarity figure for each condition, using IDs aligned to M00 on both axes.
M00's own figure has 1 on the diagonal; off-diagonal entries show SNV similarity between different M00 clusters.
Occupancy and difference spectra are saved only as CSVs.
Differences subtract the M00 representative spectrum of the same display cluster from each condition's representative spectrum;
they are not treated as paired differences.
Save CSVs for specimen-specific, representative, and difference spectra; contributing and excluded counts;
matching, SNV similarities, supplementary IoU, contingency, and occupancy under each condition.
Save wavelength, caption, and provenance records and the all-condition spectral-summary NPZ at the shared root;
aligned labels are saved to each condition's `label_maps.npz`.
`run` regenerates the figures and tables and replaces the visualization directory on success.
Original cluster labels and centers are retained.
See the [runbook](../experiment_runbook.md#global-post-fit) for output locations and execution.

<a id="representative-spectra"></a>

### 4.1 Spectral Display and Summaries

Representative spectra are aggregated as within-specimen means followed by equal-weight means across specimens.
Compare spectra using display IDs after the SNV cosine matching in Section 3.
Only SNV representative spectra are used for matching.
The three types of representative curves compared are reflectance, SNV, and the second derivative of pseudo-absorbance.

All-specimen representative spectra support an overview across conditions and matching.
For interpretation of regional differences within an individual specimen, refer to saved specimen-specific cluster means,
pixel counts, and exclusion counts.
Distinguish the possibility that all-specimen macro means dilute local differences from differences in the specimens contributing to each cluster.
Between-specimen IQRs must not be reused as variability within a single specimen.
Specimen-specific analysis and band values for M11 at K8 follow [Section 4.4](#nir-interpretation).

#### Inputs and Transformation Order

- Fix cluster membership given by each condition's hard labels, starting from all valid pixels in the same production preprocessing data.
  Representative spectra come from observed spectra of member pixels, not latent cluster centers, inverse PCA transforms, or decoder reconstructions.
- Use the 256 interpolated bands for reflectance and existing pixelwise SNV for SNV spectra.
- Calculate pseudo-absorbance from each pixel's reflectance $`R_p(\lambda)`$ as $`A_p(\lambda)=-\log_{10}R_p(\lambda)`$.
  Do not log-transform SNV or substitute SNV/log transformation of mean reflectance.
- Aggregation of pseudo-absorbance and its second derivative includes only pixels with positive, finite reflectance in all 256 bands.
  If even one band fails this condition, exclude the entire spectrum from that aggregation.
  Do not replace values with small positive numbers or clip them.
  This exclusion applies only to pseudo-absorbance analysis; it does not change reflectance/SNV aggregation,
  SNV matching, training, CV, or cluster membership.
  Apply the same reflectance-validity rule across conditions and methods.

#### Two-Stage Means and Variability

- For reflectance, pixelwise SNV, and pixelwise pseudo-absorbance, calculate bandwise arithmetic means over eligible member pixels within each specimen and cluster.
- Then calculate the bandwise equal-weight arithmetic mean of specimen means with remaining eligible pixels to obtain the overall representative spectrum.
  Do not pool all pixels directly; this limits bias from differences in pixel counts among specimens.
- Exclude specimens where a cluster is absent, and specimen–cluster combinations with no eligible pixels after pseudo-absorbance filtering,
  from aggregation of the corresponding representative curve.
  Do not fill with zero or another spectrum; a representative spectrum without contributing specimens is undefined.
- Show between-specimen variability as the bandwise interquartile range of specimen-specific mean spectra.
  For second derivatives, apply the SG procedure below to each specimen's mean pseudo-absorbance before calculating the between-specimen IQR.
  Do not substitute derivatives of the quantile curves.
- Report contributing specimen counts, specimen IDs, and pixel counts for each representative curve.
  Use post-exclusion counts for pseudo-absorbance and second derivatives, and record excluded pixel counts.
  State that contributing specimens/pixels can differ by curve type and cluster.
- If only one specimen contributes, state that between-specimen variability cannot be assessed.
  A mean representative curve is not the spectrum of an actual individual pixel or an estimate of component concentration.

#### SG Second Derivative of Pseudo-Absorbance

Apply SciPy's `scipy.signal.savgol_filter` to the overall mean pseudo-absorbance curve to calculate its second derivative with respect to wavelength (nm).
Use the following settings, also shared when obtaining IQRs from specimen-specific mean curves.

| Argument | Setting |
| --- | --- |
| `window_length` | `7` |
| `polyorder` | `2` |
| `deriv` | `2` |
| `delta` | Adjacent spacing (nm) in the saved equally spaced wavelength grid; do not replace it with the default 1 |
| `axis` | Wavelength is the last axis; default `-1` |
| `mode` | Default `"interp"` |
| Other arguments | SciPy defaults |

At endpoints, derivatives are obtained from a quadratic polynomial fitted to the first/last seven points.
Second-derivative units are $`\mathrm{nm}^{-2}`$.
Because averaging and SG processing with fixed settings are linear, differentiating specimen-specific means before averaging across specimens
gives the same representative curve, apart from rounding error, when targets, weights, wavelength grid, and endpoint handling are identical.
Do not interchange SNV or log transformation with averaging.
SG processing is used for the second derivative of pseudo-absorbance; reflectance and SNV are compared using the mean representative curves defined above.

#### Comparison and Interpretation

- Compare all three types of representative curves across conditions and clusters to identify wavelength regions with differences.
  Hungarian matching to M00's Cosine-KMeans uses the SNV representative-spectrum cosine similarity in Section 3.
  Do not manually change correspondence based on second derivatives or selected bands.
- Difference spectra subtract representative curves of the same type; state the conditions/clusters and subtraction direction.
  Differences between representative curves formed from different contributing specimens are not interpreted as paired chemical changes within the same specimens.
- SNV removes each pixel's offset and scale, so differences in SNV amplitude are not used as direct comparisons of absorption or component amounts.

<a id="representative-samples"></a>

### 4.2 Selection of Display Examples

Save maps for all six Cosine-KMeans conditions for all 49 specimens.
Do not base the study's overall evidence solely on examples that look favorable.
Record selection criteria, specimen IDs, and the timing of changes.
Characteristic or failure examples selected after inspecting results must be identified as such.
Changing exploratory examples does not change the main conditions, K, seeds, or evaluation targets.

The seven representative specimens are display examples selected from each wood species;
their selection does not establish validated diversity of chemical states.
Selection is restricted to the 49 adopted specimens in `production_v1`;
use the following seven in **ascending specimen-number order**.
Species names correspond to `樹種名` in `data/metadata/古材メタデータ.csv`.
Saved valid-pixel counts are shown for reference.

| Wood species | Count among adopted specimens | Representative specimen in the main text | Saved valid-pixel count |
| --- | ---: | --- | ---: |
| Hinoki cypress | 14 | KYOw02752 | 117,193 |
| Pine | 23 | KYOw02772 | 73,211 |
| Zelkova | 4 | KYOw02777 | 121,687 |
| Hemlock | 1 | KYOw02787 | 106,684 |
| Chestnut | 3 | KYOw02790 | 26,249 |
| Japanese cedar | 3 | KYOw16744 | 108,863 |
| Fir | 1 | KYOw16750 | 58,739 |

The representative 1×7 raw BMP and reflectance L2-norm figures, per-condition representative 1×7 full-fit maps,
and six-condition × seven-specimen figure share this selection.
OOF sanity plots also use these specimens and this order.

Pixel-count source: `outputs/experiments/preflight_v1/manifests/inputs.json`.

This selection is for display in the main text; it does not guarantee typicality within a species or representativeness of degradation states.
The 49 CV specimens, splits, training-pixel sets, and evaluation targets remain unchanged.
Do not automatically reselect if inputs change; record the reason and the updated specification.

<a id="latent-spectral-maps"></a>

### 4.3 PCA Display of Representation Space

Display representations of the same fitting pixels in two dimensions for B0, B1, A0, A1, M00, and M11.
For B0, A0, A1, M00, and M11, project the L2-normalized clustering representations with centered PCA.
For B1, use the scores before L2 normalization for the first two components of the saved baseline PCA.
PCA axes, orientations, and coordinate values are not assumed to correspond across conditions;
separation in the plots alone is not used to rank conditions.
Two-dimensional coordinates are not used for clustering or CV metrics.

| Item | Setting |
| --- | --- |
| Fitting/display pixels | 49 specimens × 8,192 = 401,408 pixels from the global manifest, shared across all six conditions |
| Input | FP32, L2-normalized representations: 256 dimensions for B0 and 16 for A0, A1, M00, and M11; FP32 unnormalized scores of the first two components for B1 |
| Projection | B1 selects saved PCs in descending singular-value order; the other five use PC1/PC2 of centered PCA, without column standardization, whitening, or post-projection normalization |
| Computation | The other five use scikit-learn on CPU in FP64 with `n_components=2, svd_solver="covariance_eigh", whiten=False`; B1 scores are saved in FP64 |
| Randomness | PCA fitting uses no random draws; scatter-point drawing order is fixed by ROOT_SEED |
| Saved outputs | Coordinates, means, principal-component coefficients, eigenvalues, explained-variance ratios, runtime, input pixels, and hashes; numerical tables in CSV |

For B1, select the two components with the largest singular values from `checkpoints/baselines/pca.npz`;
preserve saved order for ties.
Do not refit, recenter, or rotate axes; save original component indices and provenance.
B1's explained-variance ratios are saved values relative to total SNV variance, not renormalized within 16 or two components.
Cluster membership comes from the 16-dimensional L2-normalized representation.
Each condition's explained-variance ratios use the total variance of its own input representation as the denominator;
they are not a shared measure of chemical information across conditions.

The output figure is `results/figures/global_k8_v1/pca-latent-2d/01_pca_density_clusters.png`.
It has two rows and six columns: B0, B1, A0, A1, M00, and M11 as columns,
pixel-count hexbins on the top row, and display Cluster IDs with centroids on the bottom.
Hexbins use a grid size of 70 and a shared logarithmic `turbo` color scale,
with a lower bound of 1 and an upper bound equal to the maximum bin count across all conditions (at least 2).
Each condition uses a square viewing region containing all points, with identical ranges in the top and bottom panels.
Axis ranges and hexagon areas are not equalized across conditions.
Colors encode pixel counts per bin, not density per shared coordinate area.
Centroids are arithmetic means of the displayed pixels' two-dimensional coordinates, distinct from KMeans centers in the original space.
Do not invent positions for clusters without member fitting pixels; record a pixel count of 0 in the CSV.
Top-row condition names use 22 pt, axis/colorbar labels 18 pt, ticks 15 pt, and centroid numbers 14 pt.

Fitting is handled by `global_pca.py`, and rendering by `visualize_global_pca.py`.
PCA outputs are validated and retained independently of the parent clustering figures and tables.
See the [runbook](../experiment_runbook.md#global-pca) for execution.

<a id="nir-interpretation"></a>
<a id="spectral-band-selection"></a>

### 4.4 NIR Spectral Analysis

For all 49 specimens, use the initial cluster assignment from global-fit M11 at K=8.
Display IDs and colors follow Section 3; spectra start from the within-specimen cluster means in Section 4.1.
Pseudo-absorbance aggregates only pixels with positive, finite reflectance in all 256 bands,
so contributing pixels can differ between SNV and pseudo-absorbance.
Save included and excluded counts; do not impute missing values.

Calculate the SG second derivative from each specimen's mean pseudo-absorbance with a seven-point window,
polynomial order 2, derivative order 2, and `mode="interp"`.
Even when the horizontal axis is converted to wavenumber (cm$`^{-1}`$),
the derivative remains with respect to wavelength, $`d^2A/d\lambda^2`$ (nm$`^{-2}`$).
In the aggregate figure, average contributing specimen means with equal weights, not weights based on pixel counts.

For specimen-specific comparisons, inspect differences between regions near externally exposed surfaces and the interior,
and characteristic local distributions, then select clusters for comparison.
Save the selection in the [configuration](../../scripts/experiments/nir_plot_selection.yaml).
The `note` field annotates interpretation and hypotheses; it is not a target label or a confirmed chemical state.
Reference curves from other specimens display their specimen and cluster IDs and are not included in the target specimen's band aggregation.

The [visualization CLI](../../scripts/experiments/visualize_nir_interpretation.py) generates an M11 aggregate figure
and specimen-specific spectral PNGs from saved within-specimen means.
Input CSVs are those saved by the global-fit visualization pipeline.
Band annotations use the 18 bands defined in the script's `BANDS`.
Assignment positions indicate correspondences from the literature, distinct from peaks detected or components identified in this dataset.

Band values and baseline comparisons in the paper use the following definitions.

- Use piecewise linear interpolation on the wavenumber axis. For a single position, take the interpolated value; for an interval, take the minimum including both endpoints.
- Of the 18 bands, display nine (3, 5, 7, 9, 10, 12, 13, 14, and 17) in bar plots to contrast cellulose, polysaccharides, lignin, and water/hydrogen-bonding environments.
- Plot $`-d^2A/d\lambda^2`$ so that deeper negative troughs appear higher. This is a curvature measure, not a direct estimate of component amount.
- Construct the reference baseline by equally averaging band values of selected clusters 1, 6, and 7 within each specimen. Define it when at least one such cluster is present; do not impute missing values.
- Bars for each state are equal-weight means across contributing specimens, and error bars are between-specimen sample standard deviations.
  Within-specimen differences use the shared specimens containing every state being compared.

The NIR CLI in this repository generates spectral figures from saved numerical CSVs;
code generating the paper's band-value aggregates and baseline bar plots is not included.
See [results and figures](../results.md) for figure correspondence and the scope of reproduction.

## 5. Interpretation Limitations

Interpretation is exploratory, relating maps, NIR spectra, literature assignments, wood anatomy,
environmental history, and specimen information.
Chemical interpretation and guidance on reading the metrics are summarized in [interpretation of results](../interpretation_notes.md).

- Cluster numbers do not represent degradation classes, severity, or a temporal transition order; representative spectra do not represent pure components.
- A0 versus M00 and A1 versus M11 differ in both masking and the loss target. M00 versus M11 is interpreted as the difference in combined TGN/FS use.
- Consistency with absorption bands provides observational support for chemical hypotheses, but may include contributions from species, wood anatomy, moisture, and the measured surface.
  It does not uniquely identify degradation mechanisms or component amounts, or establish causality.
- The baseline is a comparison reference formed from regions judged relatively sound within the observations, not an undegraded state or the initial state before use.
- Do not reselect main conditions, K, or seeds based on two-dimensional PCA projections or map appearance.

<a id="evidence-triangulation"></a>

### 5.1 Evidence Used for Interpretation

| Evidence | What is examined | Scope of interpretation |
| --- | --- | --- |
| Appearance, wood anatomy, and member information | Positions of observed regions relative to growth rings, externally exposed surfaces, local damage, and other features | Not pixel-level ground-truth labels or degradation-detection accuracy |
| LLA, LFR, ARI, occupancy, and silhouette | Spatial coherence, stability to specified perturbations, reproducibility across replicates, geometric separation, and degeneration | Do not directly guarantee correct boundaries or chemical identity |
| NIR spectra and literature assignments | Consistency between spectral differences across clusters and chemical interpretation | Not unique identification of chemical states or degradation mechanisms across all regions |

When different evidence shows different trends, keep the observations distinct rather than combining them into a composite score.
Exploratory visual interpretation is not treated as a blind evaluation or a formal assessment using a predefined rating scale.
