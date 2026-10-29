# Evaluation Metrics

## 1. Scope

This document defines metric computation, undefined values, aggregation, and reporting. For interpretation and its limits,
see [Interpreting the Results](../interpretation_notes.md); for runtime checks and records, see [Experimental Protocol, Section 11.2](experiment_protocol.md#execution-records).

The current CV has neither pixel-level ground-truth degradation labels nor independent degradation measurements, so supervised classification accuracy and IoU are not evaluated.
Exploratory interpretation of chemical correspondence is covered in the [NIR analysis](visualization_and_interpretation.md#nir-interpretation).

In the documentation, “uncorrected LLA” denotes the label agreement rate among valid neighboring pairs, and “LLA” denotes the value corrected for chance agreement due to occupancy.
In equations, these are written as $`\mathrm{LLA}^{\mathrm{raw}}`$ and $`\mathrm{LLA}`$, respectively.
In the implementation and stored data, `lla` corresponds to uncorrected LLA, and `adjusted_lla` corresponds to LLA.

Report the principal metrics in this priority order:
**Corrected LLA → LFR(TGN+FS) → ARI → Cosine-Silhouette → Cluster Occupancy**.
This reporting order was determined after OOF aggregation, using the saved CV evaluations and OOF snapshot as the sources.

| Role | Metric | Property evaluated |
| --- | --- | --- |
| Primary evaluation and principal display | LLA-3/5/9 | Spatial agreement beyond random placement preserving occupancy |
| Primary evaluation and principal display | LFR(TGN+FS): label flip rate for noise + shift | Stability under the specified combined perturbation |
| Supplementary diagnostic | ARI across 3 replicates | Reproducibility of partitions on the same test pixels |
| Geometric diagnostic | Cosine-Silhouette in each representation space | Cluster separation in that representation under cosine distance |
| Required accompanying report | Specimen- and fold-level occupancy and number of occupied clusters | Concentration in a single cluster, imbalance, and metric degeneracy |
| Accompanying CSV | LFR(TGN), LFR(FS) | Stability under TGN alone and Fractional Shift alone; save condition-specific, paired, interaction, and specimen-level values to CSV |
| Supplementary, saved evaluation | Uncorrected LLA-3/5/9 | Agreement before correction, retained in the original evaluations and OOF snapshot |

Do not combine the metrics into a single score or select an overall “best representation.”
Improvements in the primary metrics support only the properties listed in the table; they do not guarantee degradation detection, chemical correspondence, or boundary accuracy.

## 2. Shared Conventions

### 2.1 Labels

- The background label is 0.
- Cluster labels range from 1 to K.
- Add 1 to the labels from 0 to K-1 returned by Cosine-KMeans when storing them in label maps.
- Exclude background pixels from all cluster evaluations.
- The evaluation set $`\Omega_m`$ is the set of valid pixels corresponding to `valid_spectrum_mask` from production preprocessing,
  with pixels matched using `pixel_row_col`. Pixels excluded by quality filtering are also excluded from neighborhoods and denominators.
- Use the same evaluation pixels across all conditions, K values, and replicates. Do not drop pixels separately by condition when representation extraction fails.

### 2.2 Evaluation Unit

The aggregation unit is the specimen identified by `KYOw...`, not the pixel. Follow Section 3 of
[experiment_protocol.md](experiment_protocol.md) for the treatment of independence between specimens.

1. Obtain a specimen score for each specimen, condition, K, and training replicate according to the metric-specific definition.
2. For LFR, average evaluation perturbation replicates within each specimen. Distinguish these from the 3 training replicates.
3. For each training replicate, combine the out-of-fold specimens from all 5 folds and calculate a macro average with equal specimen weights.
4. Aggregate the 3 replicates, reporting between-specimen variation separately from variation across training and clustering replicates.

Do not use a pixel-weighted average dominated by specimens with more pixels as the principal value for condition comparisons.
Treat condition differences as paired values for the same specimen, fold, K, and replicate ID.
Do not inflate the independent specimen count by treating pixels, perturbation replicates, or training replicates as independent specimens. Follow Section 8 for the specific aggregation rules.

### 2.3 K

Use the prespecified representative display value $`K_0=8`$ and the common set $`\mathcal{K}=\{2,4,6,8,10,12,14\}`$.
Follow [experiment_protocol.md](experiment_protocol.md) for the selection policy.
Use $`K_0`$ for the main table and curves over $`K\in\mathcal{K}`$ for dependence on K. Do not base conclusions solely on $`K_0`$.

### 2.4 Undefined Values and Common Comparison Sets

- Record an empty valid-pixel set, non-finite representations, or zero norms that make cosine distance undefined as input or run failures,
  including the cause. Do not handle them through silent condition-specific exclusions.
- If a metric is undefined despite valid input, record the undefined value and its reason. Do not fill it with a favorable value or 0.
- For each metric, report both the number of specimens with defined values and the total number of eligible specimens. Do not replace failed runs with favorable ones.
- Calculate condition differences only for the same specimens and replicates with defined values in both compared conditions, and report the size of this common set.
  Do not judge conditions with frequent missing values or single-cluster assignments as superior based only on averages from different specimen sets.

<a id="silhouette"></a>

## 3. Cosine-Silhouette

Follow the [numerical settings in the protocol](experiment_protocol.md#evaluation-precision) for representation extraction and evaluation.

### 3.1 Definition

Use each condition's test representations $`z_i`$ and labels $`c_i`$ assigned by the fixed training centroids.
Follow the [protocol](experiment_protocol.md#representations) for representations and transformation order, and define cosine distance as

```math
d_{\cos}(z_i,z_j)
=1-\frac{z_i^\top z_j}{\lVert z_i\rVert_2\lVert z_j\rVert_2}
```

Let $`a_i`$ be the mean distance to other members of the same cluster, excluding the pixel itself, and $`b_i`$ the mean distance to the nearest other cluster. Calculate

```math
s_i=\frac{b_i-a_i}{\max(a_i,b_i)}
```

Higher values indicate greater proximity within the assigned cluster and greater separation from other clusters.

### 3.2 Computation

- Pool all test wood pixels in the fold to calculate pixel-wise silhouette scores.
- Split the resulting $`s_i`$ values by specimen boundaries, calculate within-specimen means, and then take the macro average.
- Exclude background 0 before passing data to the calculation.
- Assign a score of 0 to pixels in singleton clusters.
- If the number of occupied clusters in the entire test set is 1 or equals the total number of pixels, record the entire silhouette evaluation as undefined.
- Assign a score of 0 to pixels with $`a_i=b_i=0`$.
- Distinguish single-cluster assignment in the test set from training collapse. Do not infer training collapse from an undefined silhouette alone
  without checking training and test occupancy.

Use `silhouette_samples_cosine_gpu` from ChemoMAE v0.2.2.
This function calculates exact silhouette scores using cluster statistics and chunked processing,
without constructing an $`N\times N`$ matrix of cosine distances.
Treat chunk size as an implementation parameter that does not change the result.

### 3.3 Scope of Interpretation

Because the representation and distance structure differ by condition, this value diagnoses separation within each particular space.
Do not rank representation utility or correspondence with degradation solely by numerical differences between conditions.
Re-evaluating all conditions in raw SNV space could favor partitions based on raw SNV, so do not add it as an alternative primary evaluation.

Since $`a_i`$ and $`b_i`$ are calculated from all test pixels in a fold, specimens with more pixels contribute more
at the distance-averaging stage even if the final aggregation is a specimen macro average. Do not describe this as an equal-specimen-weighted distance calculation.
Inspect specimen-level occupancy and label maps together to distinguish separation due to differences between specimens or tree species from within-specimen regional structure.

<a id="lla"></a>

## 4. Uncorrected LLA and LLA: Local Label Agreement

### 4.1 Definition of Uncorrected LLA

Uncorrected LLA is the proportion of valid neighboring pairs within a specimen that have matching labels.
Let $`\Omega_m`$ be the valid-pixel set of specimen $`m`$, and $`y_p\in\{1,\ldots,K\}`$ the label of valid pixel $`p`$.
Identify pixel $`p`$ with its integer coordinates in the original image, and define the cluster-$`k`$ indicator $`B_{m,k}`$ and valid-pixel mask $`M_m`$ as

```math
B_{m,k}(p)=
\begin{cases}
1, & p\in\Omega_m,\ y_p=k,\\
0, & \text{otherwise},
\end{cases}
\qquad
M_m(p)=\sum_{k=1}^{K}B_{m,k}(p)
```

Set all these functions to 0 for background, pixels excluded by quality filtering, and positions outside the image.
Represent a square neighborhood of width $`r\in\{3,5,9\}`$ using the kernel that excludes its center:

```math
h_r(\boldsymbol{u})
=\mathop{\mathrm{Ind}}\nolimits\!\left[0<\|\boldsymbol{u}\|_\infty\le\frac{r-1}{2}\right],
\qquad \boldsymbol{u}\in\mathbb{Z}^{2}
```

Here, $`\mathop{\mathrm{Ind}}\nolimits[\cdot]`$ is an indicator returning 1 when the condition holds and 0 otherwise;
$`\boldsymbol{u}`$ is a difference between pixel coordinates, and $`\|\boldsymbol{u}\|_\infty=\max(|u_1|,|u_2|)`$.
The condition $`h_r(\boldsymbol{0})=0`$ excludes self-agreement.

Define two-dimensional discrete convolution and the inner product over pixels as

```math
(h_r*f)(p)=\sum_{q\in\mathbb{Z}^{2}}h_r(p-q)f(q),
\qquad
\langle f,g\rangle=\sum_{p\in\mathbb{Z}^{2}}f(p)g(p)
```

where $`f,g`$ are real-valued functions that are zero outside a finite region. Define uncorrected LLA-r for specimen $`m`$ as

```math
\mathrm{LLA}^{\mathrm{raw}}_{r,m}
=
\frac{
\displaystyle\sum_{k=1}^{K}
\left\langle B_{m,k},\,h_r*B_{m,k}\right\rangle
}{
\left\langle M_m,\,h_r*M_m\right\rangle
}
```

The numerator counts valid neighboring pairs with matching labels, and the denominator counts all valid neighboring pairs.
Both include $`(p,q)`$ and $`(q,p)`$, matching the definition that enumerates neighbors for each center pixel.

Although valid-neighbor counts vary near boundaries and masks, zero extension and $`M_m`$ exclude ineligible pixels from both numerator and denominator.
Sum the numerator and denominator separately within each specimen before taking their ratio; do not take the simple mean of the local agreement rates at individual center pixels.
The metric is undefined when the denominator is 0, meaning that no valid neighboring pairs exist.

| Metric | Maximum neighbors excluding the center | Main structure examined |
| --- | ---: | --- |
| Uncorrected LLA-3 | 8 | Local salt-and-pepper variation |
| Uncorrected LLA-5 | 24 | Small- to medium-scale coherence |
| Uncorrected LLA-9 | 80 | Broader areal or stripe-like structures |

Higher uncorrected LLA indicates more local agreement, but occupancy imbalance can also raise it, so report occupancy alongside it.
Report uncorrected LLA-3/5/9 separately rather than averaging them.

The values 3, 5, and 9 are widths in pixel coordinates; do not assume that they represent equal scales in physical length.
Record physical pixel dimensions and scan-direction anisotropy where available.
Uncorrected LLA does not directly evaluate boundary correctness or preservation of small regions.

### 4.2 LLA: Correction Based on Random Placement Preserving Occupancy

Let $`N_m=\sum_p M_m(p)`$ be the number of valid pixels in specimen $`m`$, and $`n_{m,k}=\sum_p B_{m,k}(p)`$ the number in cluster $`k`$.
Consider a null arrangement that uniformly randomly permutes labels over valid-pixel positions while preserving
the mask shape and each cluster's pixel count. The probability that two distinct pixels have matching labels is

```math
P_m=\frac{\sum_{k=1}^{K}n_{m,k}(n_{m,k}-1)}{N_m(N_m-1)}
```

When valid neighboring pairs exist, the expected uncorrected LLA under this null arrangement is $`P_m`$ for every $`r`$.
Calculate this expectation analytically rather than generating large numbers of random label maps.

Report

```math
\mathrm{LLA}_{r,m}
=\frac{\mathrm{LLA}^{\mathrm{raw}}_{r,m}-P_m}{1-P_m}
```

as the principal LLA value. Zero indicates the expectation under the null arrangement; positive values indicate more agreement,
and negative values less agreement. Do not clip negative values to 0.
This is a correction relative to the null arrangement specified in this study, not agreement with ground-truth degradation or correct spatial structure.

The metric is undefined when $`N_m<2`$, when no valid neighboring pairs exist, or for a single-cluster specimen with $`P_m=1`$.
In particular, do not assign a corrected value of 1 to single-cluster specimens. As $`P_m`$ approaches 1, the corrected value becomes sensitive to small changes,
so report uncorrected LLA, $`P_m`$, the number of occupied clusters, and occupancy together.
Do not assume that LLA completely resolves the problem of excessive homogenization.

LLA-3/5/9 uses the same neighborhoods as uncorrected LLA and is also reported separately rather than averaged.

<a id="lfr"></a>

## 5. Label Flip Rate

### 5.1 Definition

Let $`y_p`$ be the label of test pixel $`p`$ under clean input, and $`y_p^{(a)}`$ its label under perturbed input.
Define the label flip rate for specimen $`m`$ and perturbation $`a`$ as

```math
\mathrm{LFR}_{m}^{(a)}
=\frac{1}{|\Omega_m|}
\sum_{p\in\Omega_m}
\mathbf{1}[y_p\ne y_p^{(a)}]
```

Lower values indicate greater stability under the perturbation.

### 5.2 Fixed Settings

- Use the same test pixels in the same order for clean and perturbed inputs.
- For B0, perturb the same SNV input; for B1, pass it through the fixed training PCA; for ChemoMAE conditions, pass it through the fixed encoder.
  Then apply row-wise L2 normalization as for clean input.
- Keep PCA, the encoder, and the training centroids fixed. Use inference mode for the encoder so that dropout or random masking
  does not introduce variation into the evaluation perturbations.
- Follow the [protocol](experiment_protocol.md#representations) for the full-visibility extraction API and mask specification.
- Do not retrain or recluster after perturbation.
- Set the application probability of each evaluated augmentation to 1.
- Explicitly switch ChemoMAE's `SpectraAugmenter` to training mode only for generating evaluation perturbations.
- Fix perturbation seeds and replicate counts before evaluation, and use identical perturbation sequences for all conditions.
- Apply perturbations to the SNV spectra before representation transformation. Share the input for each specimen, pixel, and perturbation replicate
  across all conditions, K values, and training replicates. Wavelength shifts perturb the spectral axis; they do not move spatial coordinates.

Report the following 3 settings separately rather than mixing them:

| Perturbation | Evaluation settings |
| --- | --- |
| Noise | `noise_prob=1`, `shift_prob=0` |
| Shift | `noise_prob=0`, `shift_prob=1` |
| Noise + shift | `noise_prob=1`, `shift_prob=1` |

Use the same strength distributions, interpolation, endpoint handling, operation order, recentering, and norm restoration as
[Protocol Section 4.1.3](experiment_protocol.md#augmentation-clustering).
Only the application probabilities in the table change at evaluation; do not perform a strength sweep tailored to LFR.

### 5.3 Perturbation Replicates and Scope of Interpretation

Fix the number of evaluation perturbation replicates at $`R=5`$ and report the mean of the 5 LFR values for each specimen and training replicate.
Also save the value from each of the 5 draws. These replicates are separate from the 3 training replicates.
Generate each of the 3 perturbation types 5 times for the same specimen, sharing all 15 perturbed inputs across conditions, K values, and training replicates.
Reuse a given model's perturbed representations across K values rather than repeating encoder inference for each K.

This is a prespecified setting to limit computational cost, not an assumption that 5 draws provide convergence.
Show variation across draws; do not count them as independent specimens or change their number separately by condition after inspecting results.

This evaluation examines stability when perturbations are always applied using the same types and strength distributions as the training augmentations.
It retains evaluation without retraining and comparison on common inputs, but is distinct from robustness to unseen perturbations,
differences between instruments, or actual repeated acquisitions. Single-cluster outputs can also have low LFR,
so report occupancy and the number of occupied clusters for both clean and perturbed inputs.

<a id="ari"></a>

## 6. ARI Across Training and Clustering Replicates

Compare the clean test labels produced by the 3 replicates for the same fold, condition, and K.
For each replicate pair $`(1,2)`$, $`(1,3)`$, and $`(2,3)`$, calculate the Adjusted Rand Index (ARI)
on the same valid pixels of the same specimen. ARI is invariant to label permutations, so do not use Hungarian matching.

Let $`n_{ij}`$ be the contingency table for two partitions, $`a_i`$ and $`b_j`$ their marginal sums, and $`N`$ the specimen's pixel count. Define

```math
T=\binom{N}{2},\qquad
A=\sum_i\binom{a_i}{2},\qquad
B=\sum_j\binom{b_j}{2},\qquad
C=\sum_{i,j}\binom{n_{ij}}{2}
```

```math
\mathrm{ARI}=\frac{C-AB/T}{(A+B)/2-AB/T}
```

A value of 1 indicates identical partitions, values near 0 indicate agreement near the chance baseline with fixed marginal sums, and negative values are possible.
Average ARI over the 3 replicate pairs within each specimen, then take the specimen macro average.
Do not treat the replicate pairs as 3 independent observations.

ARI is undefined for $`N<2`$. For degenerate exact agreement where both partitions contain a single cluster or both consist entirely of singleton pixels,
use 1 according to the standard ARI convention, and always save a degeneracy flag and the number of occupied clusters in each partition.
Do not treat ARI=1 for these specimens as evidence of useful segmentation.

Specimen-level ARI evaluates reproducibility of the partition within each specimen; it does not guarantee semantic correspondence of clusters across specimens.
Both the model and KMeans vary across replicates, so do not describe ARI as encoder stability alone.
Whereas LFR evaluates input perturbations to a fixed model, ARI evaluates variation in partitions when fitting is repeated.

<a id="occupancy"></a>

## 7. Occupancy and Other Diagnostics

Use the following to inspect cluster collapse or label-map abnormalities, not to select conditions in CV.
Occupancy is required for interpreting the primary metrics and must be available in the main table or an accompanying diagnostic table.

| Diagnostic | Purpose |
| --- | --- |
| Cluster occupancy | Check for empty clusters, concentration in a single cluster, and extreme imbalance |
| Cluster size distribution | Inspect label distributions across conditions and specimens |
| Label map | Visually inspect spatial structure |

For specimen $`m`$, save $`n_{m,k}/N_m`$ as occupancy, and report the number of occupied clusters, maximum occupancy,
and number of single-cluster specimens by condition, K, and replicate. Distinguish training fit pixels, clean test pixels, and each perturbed input.
For fold-level occupancy, distinguish the distribution over pooled pixels from the macro average of specimen-level occupancies.
Since cluster numbers differ by fold and replicate, do not average occupancy by unaligned cluster number across folds or replicates.

A single cluster within a specimen may be appropriate for a homogeneous specimen. Distinguish a single-cluster specimen,
a single-cluster fold, and empty clusters during training fitting. Do not require every specimen to contain K clusters,
select single-cluster specimens as favorable examples, or unconditionally exclude them as failures.

<a id="reporting"></a>

## 8. Reporting Results and Uncertainty

### 8.1 Separating Specimen Variation from Replicate Variation

For a fixed condition and K, let $`s_{m,t}`$ be the score of specimen $`m`$ in training replicate $`t\in\{1,2,3\}`$.
For LFR, this is the value after averaging the $`R`$ evaluation perturbations within the specimen.
When values are defined for all $`M`$ specimens, calculate

```math
\bar{s}_m=\frac{1}{3}\sum_{t=1}^{3}s_{m,t},\qquad
\mu_t=\frac{1}{M}\sum_{m=1}^{M}s_{m,t},\qquad
\bar{\mu}=\frac{1}{3}\sum_{t=1}^{3}\mu_t
```

In the main table, distinguish $`\bar{\mu}`$, the between-specimen SD of $`\bar{s}_m`$, and the SD of $`\mu_t`$ across the 3 replicates.
Use sample standard deviations (`ddof=1`), and report the 3 individual $`\mu_t`$ values alongside them or in a supplementary table.
Do not present the between-replicate SD as a standard error or 95% confidence interval.

For metrics with undefined values, apply these equations to the common set of specimens with defined values in all 3 replicates and report its size.
Separately save the number of available specimens and the reasons for undefined values for each replicate.
If the common set is empty, the aggregate is undefined; if it contains 1 specimen, the between-specimen SD is undefined.
If the eligible set shrinks for LLA or silhouette, do not treat the result as representing the same population as results for all specimens.

Because ARI compares replicate pairs, report the macro average and between-specimen SD of the values averaged over 3 pairs within each specimen.
Do not interpret the SD over the 3 pairs as uncertainty across independent training replicates.

### 8.2 Planned Comparisons

For metrics defined for each replicate, including the primary metrics, save the difference between condition $`c`$ and reference condition $`b`$
for the same specimen, K, and replicate as

```math
\Delta_{m,t}=s^{(c)}_{m,t}-s^{(b)}_{m,t}
```

Use the common set of specimens with defined values for both conditions in all 3 replicates. Report specimen-level differences after averaging over replicates,
the macro average of those differences, and the macro difference for each replicate.
Positive LLA differences and negative LFR differences indicate improvement in the respective metrics.
In the main table and corresponding paired plot, show whether improvement or deterioration varies by specimen, in addition to the mean.
Define paired ARI comparisons as differences between each condition's specimen-level average over the 3 replicate pairs.
Use common specimens for which this specimen average is defined in both conditions, and save the macro difference, between-specimen SD, eligible count, and reasons for undefined values.
Do not treat replicate pairs as independent replicates or create replicate-specific curves or between-replicate SDs for ARI differences.

The current reporting pipeline calculates 2×2 interactions for LLA-3/5/9 and LFR(TGN+FS), LFR(TGN), and LFR(FS)
from saved scores for the same specimen, K, and replicate. Use common specimens with defined values in all 4 conditions and all 3 replicates.
`main_oof_v1/interaction/` stores uncorrected LLA and the 3 LFR variants.
The reporting pipeline calculates interactions for corrected LLA and saves them to CSV, distinguishing them from uncorrected LLA.
The pipeline also calculates paired ARI while retaining scores, aggregates, and provenance records from the original OOF snapshot.

This plan does not use significance tests or p-value-based acceptance decisions. Describe effect sizes, specimen-level differences,
dependence on K, and trends across the 3 replicates.
The training sets overlap across the 5 folds, and silhouette distance calculations are shared across test specimens.
Do not add tests or confidence intervals that unconditionally treat folds or OOF specimen scores as independent.
Do not interpret a simple specimen bootstrap as uncertainty in the entire training process. If interval estimation is added,
separately design the target uncertainty, resampling unit, and whether training is repeated.

### 8.3 Required Tables and Figures

This section applies to reporting figures and tables after CV and OOF aggregation have been completed for all folds and replicates of the target conditions.
Generate the main 8-condition figures and tables from the OOF snapshot.
Use a dedicated pipeline for mask-rate OOF figures and tables for M11-25, M11, and M11-75.
[OOF sanity visualizations](oof_sanity_visualization.md) follow a separate output specification limited to
label-map and silhouette PNGs and numerical CSVs.

The following specifies the figures, tables, and reporting conventions for the main 8 conditions. The mask-rate comparison layout is given at the end of this section.

- Save tables as CSVs in the priority order of the principal metrics. For $`K_0=8`$ and all 7 K values, distinguish the mean, between-specimen SD,
  between-replicate SD, eligible count, and reasons for undefined values. Do not attach a between-replicate SD to ARI or its paired differences.
  Save all 3 LFR variants: TGN+FS, TGN alone, and FS alone. Show only TGN+FS in the figures.
  Display the saved keys `lfr_both`, `lfr_noise`, and `lfr_shift` as LFR(TGN+FS), LFR(TGN), and LFR(FS), respectively.
- Use a single-column-width 2×2 layout for the primary-metric summary, with LLA 3 and 5 in the top row and LLA 9 and LFR(TGN+FS) in the bottom row.
  Show all 7 K values on the horizontal axis. In each panel, show the macro average for all 8 main conditions and replicate-specific curves from the same common specimen set.
- Use a 2×3 layout for specimen-level distributions at $`K_0=8`$: LLA 3, 5, and 9 in the top row, and LFR(TGN+FS), ARI, and Cosine-Silhouette in the bottom row.
  Each point represents the mean over 3 training replicates for one specimen (the mean over 3 replicate pairs for ARI);
  the black line is the macro average, and `n` is the number of common specimens with defined values.
- Use one K-dependence figure for paired comparisons, with the same 2×3 layout as the specimen-level distribution figure.
  Overlay the 3 primary comparisons M11−B0, M11−B1, and M11−M00, and the additional ablations A1−A0 and M11−A1.
  Use the specimen-level definition in Section 8.2 for ARI differences.
- Within each figure, use a common vertical range for LLA 3, 5, and 9 that includes all points, replicate curves, and padding, without truncating values.
  Use vertical-axis ticks at intervals of 0.1 in all panels of the primary-metric summary and specimen-level distribution figures.
  Use separate vertical ranges for different metrics. For paired LLA plots, use a common tick spacing that preserves the first panel's existing spacing.
- Save Cluster Occupancy and 2×2 interactions only as CSV tables, without standalone PNGs or distribution panels.
  The current occupancy table covers clean test data and retains specimen-level occupied-cluster counts, maximum occupancy, and single-cluster status,
  as well as distributions by fold and replicate. Do not average unaligned cluster numbers across folds or replicates.
- Save the remaining planned comparisons to CSV as well. Apply the metrics, notation, and aggregation conventions in this section to the mask-rate comparison.
- When metrics support different conclusions, report the trade-offs rather than reducing them to an overall ranking.
- Use Hungarian matching to align display labels, not to calculate CV metrics.
  [OOF sanity](oof_sanity_visualization.md) maximizes matching pixel counts against B0 within each fold.
  [After global fitting](visualization_and_interpretation.md#matching-reference), use M00 Cosine-KMeans as the display reference and maximize the sum of cosine similarities between observed-SNV representative spectra with equal specimen weights.
  Also save global-fit IoU, contingency tables, and matching pixel counts for inspection.

The main-condition artifacts comprise 3 PNGs and 11 CSVs. See the [runbook](../experiment_runbook.md#oof-reporting)
for filenames, output locations, and regeneration steps. Number the PNGs consecutively from 01 to 03.

Summarize the mask-rate comparison in 1 PNG showing LLA 3, 5, and 9 and LFR(TGN+FS) for the 3 conditions in the same 2×2 layout,
and 2 CSVs for all 7 K values and K8. Solid lines show macro averages; dashed, dotted, and dash-dot lines show training replicates 1, 2, and 3.
The tables contain 11 metrics: 3 LLA windows, 3 LFR variants, ARI, Cosine-Silhouette, occupied-cluster count, maximum occupancy, and single-cluster rate.
Use `M11(-50)` as the display name for 50%, while retaining `M11` as the stored condition ID.
See the [mask-rate OOF runbook](../experiment_runbook.md#mask-rate-reporting) for provenance, completion records, and regeneration steps.
Keep the numerical comparison across 3 seeds with fixed representations from global fitting separate from OOF aggregation across 3 training replicates,
following the [definition of the appendix comparison](visualization_and_interpretation.md#global-mask-rate).

## 9. Methodological References

- [Yerbury et al., On the Use of Relative Validity Indices for Comparing Clustering Approaches](https://doi.org/10.1145/3748726):
  Limitations of using internal cluster indices to select between conditions with different representations, normalization, or distances.
  This plan limits silhouette to a geometric diagnostic within each representation space.
- [Bengio and Grandvalet, No Unbiased Estimator of the Variance of K-Fold Cross-Validation](https://www.jmlr.org/papers/v5/grandvalet04a.html):
  Dependence and variance-estimation issues in CV. This is the basis for not treating folds as independent replicates.
