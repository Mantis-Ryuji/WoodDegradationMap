# Spatial Map Reproducibility Evaluation

This document defines a supplementary experiment assessing the stability of spatial maps that connect
NIR spectral features to positions on a specimen when training and clustering are repeated.
It refers to the existing [global fitting and visualization](visualization_and_interpretation.md),
[experiment protocol](experiment_protocol.md), and [evaluation metrics](evaluation_metrics.md),
while recording this experiment separately from the CV and global fitting conditions and artifacts.
Commands for execution, resumption, and verification are collected in the [runbook](../spatial_map_reproducibility_runbook.md).

## 1. Purpose and Role

This study relates regions in NIR-HSI maps to features of the observed spectra.
It examines how much the positions and extents of regions used for interpretation vary with the randomness of training and clustering.

This supplementary experiment evaluates reproducibility of maps of the same measured surface from two perspectives.

| Evaluation perspective | Question | Method |
| --- | --- | --- |
| Reproducibility of the cluster partition | Do the same groups of pixels belong to the same clusters? | ARI for pairs of replicates |
| Reproducibility of spatial structure on the measured surface | Do internal boundaries separating regions appear near the same positions? | Bidirectional boundary agreement within 3×3, 5×5, and 9×9 neighborhoods, and visualizations preserving spatial positions |

The question is whether partition variability is concentrated mainly near boundaries or extends far enough to change the arrangement of regions.
Boundary variation is not assumed to be an error or failure; its magnitude and location are described.

- No pass/fail threshold, threshold for the number of specimens judged stable, or composite metric score is defined.
- Constructing decision criteria from paired schematic examples of acceptable and unacceptable maps is not a prerequisite.
- The same saved NIR-HSI observations are used. Reproducibility under repeat imaging or measurement is outside the scope.

## 2. Experimental Configuration

The main target is M11 at K=8, as used in the global-fit interpretation. No additional comparison is performed to rank methods or select K.

| Item | Specification |
| --- | --- |
| Specimens | The same 49 specimens as in the existing global fit |
| Encoder replicates | $`e\in\{1,2,3\}`$; use $`e=1`$ from the global fit and the trained encoders $`e=2,3`$ |
| Cosine-KMeans replicates | Three initializations, $`k\in\{1,2,3\}`$, for each encoder |
| Number of maps | Nine maps per specimen, 49×9 maps in total |
| Fitting pixels | The existing 8,192 pixels per specimen, 401,408 coordinates in total, shared across all replicates |
| Evaluation pixels | All valid pixels of each specimen, including pixels not used for fitting; the same set in every replicate |
| Computation performed | Three Cosine-KMeans fits, one for the second seed of each encoder; generation of 147 maps; evaluation across nine maps per specimen |

Fitting pools the shared sampled pixels from all 49 specimens; Cosine-KMeans is not refitted separately for each specimen.
The trained encoders all use M11 with the same architecture, mask ratio, augmentation, loss, training budget, and stopping conditions.
The six sets of maps from the first and third seeds are reused, and no encoder training is performed.
This is a descriptive analysis after global fitting, distinct from estimating generalization to unseen specimens.

The three existing OOF replicates can provide reference information on unseen specimens after repeated training and Cosine-KMeans fitting,
but are not mixed into the nine maps in this experiment. Differences between OOF and global-fit ARI are not subtracted to estimate an encoder contribution.

## 3. Fixed Conditions and Randomness

### 3.1 Inputs, Training, and Inference

- Use the existing preprocessing in `data/processed/production_v1/` and the sampled coordinates in
  `outputs/experiments/global_v1/manifests/fit_pixels.parquet`. Do not resample, repeat preprocessing, or add perturbations to evaluation inputs.
- Use M11 trained for 800 epochs and its final raw weights. Checkpoints are not selected by map appearance or evaluation scores.
- Training follows the existing single-GPU, FP16 AMP recipe with FP32 model parameters.
  Inference uses all patches visible, evaluation mode, FP32, and TF32 disabled, normalizing each pixel's 16-dimensional representation according to the existing specification.
- Cosine-KMeans follows the [existing specification](experiment_protocol.md#augmentation-clustering).
  Retain K=8, `max_iter=500`, and `tol=1e-4`; do not introduce specimen-specific centers or a new initialization selection procedure.
- Preserve the same pixel coordinates, valid mask, and original image shape. No smoothing, morphological processing,
  small-region removal, cluster merging, or image registration is performed before evaluation.

### 3.2 Encoder and Cosine-KMeans Replicate IDs

Treat `encoder_repeat` and `kmeans_repeat` as separate IDs, distinct from the existing single `repeat` field.

- Use the weights and randomness records for encoder $`e=1`$. Training of $`e=2,3`$ uses the existing ROOT_SEED and SHA-256 scheme,
  deriving purpose-specific seeds for initialization, pixel order, masks, and training augmentation from `(purpose, global, e)`.
  These are described as replicates of training randomness encompassing all these sources, not initialization alone.
- Cosine-KMeans uses seeds shared across all three encoders. $`k=1`$ is 1960870825, $`k=3`$ is 2051198256,
  and $`k=2`$ is derived by `_seed("kmeans", "global_spatial_reproducibility", "seed2_v2", 2, 8)`.
  $`k`$ denotes the Cosine-KMeans replicate ID; seeds do not change with encoder ID.
- `prepare` saves the seed list for all purposes and collision checks.
  The pixel-sampling seed and saved coordinates are also shared across encoders.
- The same Cosine-KMeans seed is not assumed to yield the same centers or clusters in different encoder spaces.
  Between-encoder comparisons represent “variation including retraining and clustering,” not a pure encoder effect.

Record the GPU, versions of Python, PyTorch, ChemoMAE, and other software, code, config, manifests, weight provenance,
actual precision and determinism settings, and execution time according to the
[execution record conventions](experiment_protocol.md#execution-records).
GPU nondeterminism is not assumed to have been fully eliminated.

## 4. Reusing Existing Maps and Defining Comparisons

### 4.1 Source of the Initial Map

To trace the initial map used for NIR spectral interpretation, $`(e,k)=(1,1)`$ uses the labels from the original global fit
as the authoritative source. The following paths are relative to `outputs/experiments/global_v1/`.

| Replicate | Reused source |
| --- | --- |
| $`(1,1)`$ | `results/clustering/M11/repeat_1/maps/{sample_id}.npz` |
| $`(1,3)`$ | `results/global_k8_3seed_v1/clustering/M11/repeat_3/maps/{sample_id}.npz` |
| Encoder $`e=1`$ | `checkpoints/neural/M11/repeat_1/last_model.pt` |

Weights for encoders 2 and 3 and their four sets of maps from the first and third seeds are read from
`outputs/experiments/global_spatial_reproducibility_v1/`.
Maps for the second seed of each encoder are generated under `results/clustering/M11/encoder_{e}/kmeans_2/maps/` in the output directory.

Labels use `labels_k8`, with background 0 and clusters 1–8. The correspondence among run records, completion records,
centroids, checkpoints, and fitting pixels is checked for each reused source at runtime.
Separate saved artifacts are not assumed identical merely because their seeds match.
Replicate 1 of `global_k8_3seed_v1` is checked against the original displayed map; it is not counted as a tenth map.
Even if they differ, the initial-map source is retained and their correspondence is checked.
Metrics for the replicate pairs defined here are aggregated without mixing in three-seed summary values based on different targets.

### 4.2 Replicate Pairs

Let the map for specimen $`m`$ be $`y_m^{e,k}`$. Comparisons use the same valid pixel set $`\Omega_m`$ for the same specimen.
$`\Omega_m`$ follows the existing `valid_spectrum_mask`, and positions are matched using `pixel_row_col`.
Missing pixels or changed masks between replicates are treated as input inconsistencies;
comparisons are not restricted to a newly computed intersection.

| Comparison group | Replicate pairs | Pairs per specimen | Interpretation |
| --- | --- | ---: | --- |
| Within encoder | Fixed $`e`$, $`k<k'`$ | Three encoders × three pairs = nine pairs | Variation from Cosine-KMeans initialization with the representation fixed |
| Between encoders | $`e<e'`$, all $`k,k'`$ | Three encoder pairs × nine pairs = 27 pairs | Variation including encoder retraining and Cosine-KMeans |

All 36 pairs are saved, and the two comparison groups are reported separately.
Comparisons from the initial map $`(1,1)`$ to the other eight maps are also identified.
These eight pairs are a subset of the 36, not additional independent replicates.

## 5. Evaluation Metrics and Computation

The boundary definitions, proximity matching, and aggregation in this section apply to all specimens and replicate pairs.

### 5.1 ARI

Use the same definition as the [existing ARI](evaluation_metrics.md#ari) to compare partitions over all valid pixels for each replicate pair.
ARI is invariant to label permutations, so Hungarian matching is unnecessary before calculation.
Degenerate and undefined cases follow the existing conventions, with used cluster counts and occupancy reported alongside.

### 5.2 Internal Boundaries

To avoid double-thick boundaries and library-default rendering behavior, count each shared edge between horizontally
or vertically adjacent valid pixels once and use its midpoint as the boundary position.
For pixel coordinates $`p=(\mathrm{row},\mathrm{col})`$, define

```math
E_m=\{\{p,q\}:p,q\in\Omega_m,\ \|p-q\|_1=1\},
\qquad
B(y)=\left\{\frac{p+q}{2}:\{p,q\}\in E_m,\ y_p\ne y_q\right\}
```

Only edges with two valid endpoints are used; the specimen outline and edges adjoining background,
quality-excluded pixels, or holes are not boundaries.
The mask is not eroded in a way that removes genuine internal boundaries near the outline.
Boundary sets contain half-integer coordinates.
Boundary orientation and the cluster IDs on either side are not criteria for proximity matching.

### 5.3 Three Scales Corresponding to LLA

The LLA values $`r=3,5,9`$ denote neighborhood widths, not tolerance distances.
This experiment also uses square neighborhoods in image coordinates, with radius $`\rho_r=(r-1)/2`$.

| Displayed neighborhood width $`r`$ | Tolerance from a boundary midpoint in each vertical/horizontal direction, $`\rho_r`$ |
| --- | ---: |
| 3×3 pixels | 1 pixel |
| 5×5 pixels | 2 pixels |
| 9×9 pixels | 4 pixels |

All three are reported.
The shared feature with LLA is the observation scale; the metrics do not measure the same property.
Distances are in pixel coordinates. If physical pixel dimensions and scan-direction anisotropy are established, they are noted;
distances are not automatically interpreted as an isotropic millimeter scale.

### 5.4 Bidirectional Boundary Agreement

For boundary sets $`B_A,B_B`$ from replicates A and B, compute directional proximity coverage using Chebyshev distance.

```math
C_{A\to B}(r)=
\frac{\sum_{b\in B_A}\mathbf{1}\left[\min_{c\in B_B}\|b-c\|_\infty\le\rho_r\right]}{|B_A|}
```

Compute $`C_{B\to A}(r)`$ analogously, and save and display both directions.
For example, if B retains A's existing boundaries but adds internal boundaries, agreement from A to B can remain high
while agreement from B to A decreases. Reproducibility is not assessed in only one direction.

This definition is a coverage rate that permits many-to-one proximity; it does not guarantee one-to-one boundary matching
or semantic correspondence between regions.
It is not described as the same implementation or metric as the boundary F-score of an existing benchmark.
Combining directions into an F-score is excluded from the main evaluation; directional values are retained.

- If the target has no boundaries but the source does, the value in that direction is 0.
- If the source has no boundaries, the direction is undefined. If both have no boundaries, agreement is not filled with 1.
- If there are no valid adjacent edges, the specimen is outside the scope of boundary evaluation, and the reason is recorded.
- Single-cluster specimens are not automatically excluded as failures; they are described separately alongside ARI and occupancy.

Because proximity can occur by chance more readily in maps with dense boundaries, always report the boundary count $`|B(y)|`$,
boundary density $`|B(y)|/|E_m|`$, number of used clusters, and maximum occupancy for each replicate.
Maps with different densities are not ranked by boundary agreement alone.

### 5.5 Supplementary Diagnostics of Splitting and Merging

When partition variability requires explanation, use the overlap table $`n_{ij}=|A_i\cap B_j|`$ over the same valid pixels,
with row fractions $`n_{ij}/|A_i|`$ and column fractions $`n_{ij}/|B_j|`$. Fractions for absent clusters are undefined.

When overlap diagnostics are enabled, report the two VI components $`H(B\mid A)`$ and $`H(A\mid B)`$ and
their sum $`\mathrm{VI}(A,B)=H(B\mid A)+H(A\mid B)`$ in bits in pair-level, specimen-level, and overall summary tables.
$`H(B\mid A)`$ describes splitting-side disagreement relative to A; $`H(A\mid B)`$ describes merging-side disagreement.
Neither map is treated as ground truth.
If one component is zero, one partition is a refinement or merging of the other; positive values for both do not establish structural collapse.
Total VI is zero for identical partitions, and smaller values indicate less partition disagreement.
VI does not account for spatial distance and is not used for pass/fail decisions.
A and B are fixed in lexicographic order of replicate IDs $`(e,k)`$; A is $`(1,1)`$ for the eight initial-map comparisons.
Swapping A and B exchanges the components, so group-level component means are not interpreted as temporal tendencies to split or merge.
VI is undefined for pairs without saved overlap diagnostics and is not filled with zero.

Splitting is not evaluated solely by one-to-one IoU, and clusters are not freely merged to increase agreement.
Treating disconnected parts of a cluster as one pixel group is distinct from treating them as spatially connected regions.
Decomposition into connected components or their merging is not used as evaluation preprocessing.

## 6. Visualization Preserving Spatial Positions

All 49 specimens are evaluated numerically. Standard visualization is limited to one overall figure and two types of figures per specimen,
for a total of 99 figures. Save the following for each specimen.

1. A 3×3 map grid, with encoders as rows and Cosine-KMeans initializations as columns, using the same orientation, scale, and valid mask.
2. Maps showing, for each neighborhood width, the fraction of replicates with a boundary within that neighborhood.

Retain all original replicate maps, evaluation values for all 36 pairs and the eight initial-map comparisons,
specimen-level distributions, boundary densities, and occupancies.
Individual diagnostics can be traced through the saved original maps and pair-level evaluation values.

For visualization 2, at valid pixel position $`p`$, use the fraction of the nine replicates having a boundary within radius $`\rho_r`$:

```math
F_r(p)=\frac{1}{9}\sum_{e=1}^{3}\sum_{k=1}^{3}
\mathbf{1}\left[\min_{b\in B(y_m^{e,k})}\|p-b\|_\infty\le\rho_r\right]
```

The indicator is 0 for a replicate with an empty boundary set.
A map missing because of a training failure or similar issue is distinguished from a map with no boundaries.
$`F_r`$ is not calculated for a specimen without all nine maps; missing replicates are not filled with 0.
This is a descriptive frequency over nine maps sharing models, not a probability or confidence estimate from nine independent trials.
A value of 0 indicates that no replicate has an internal boundary in the neighborhood, but does not guarantee agreement in chemical meaning at that position.
A value of 1 indicates that every replicate has a boundary within the neighborhood, not necessarily at exactly the same position.
Intermediate values indicate locations where the presence of a nearby boundary varies across replicates.
Frequency cannot decrease as the neighborhood widens, so a high value at 9×9 alone is not evidence of stability.
Use the 3×3 map grid and pair-level evaluation values to identify which replicates change position.

The 3×3 grid uses the initial map $`(1,1)`$ as the color reference.
For each replicate, an overlap table with the initial map is pooled over all valid pixels of all 49 specimens,
and a one-to-one Hungarian assignment maximizing total overlapping pixels is obtained.
Each replicate's correspondence is shared across all specimens; there is no specimen-specific optimization
or chained matching through other replicates.
This correspondence is composed with the raw-to-display ID mapping saved in `M11/matching.csv` of the existing global-fit figures,
and rendered using the palette saved in their `report.json`.
The initial map also follows the global-fit display IDs and colors.
Background 0 is retained; split or merged clusters are not combined into the same display ID.
The initial input remains the original map and is not replaced by the same-seed regenerated version used in the global-fit figures.
Their label comparison is available in `protocol/initial_map_audit.json`.
Absent clusters are assigned unused display IDs in ascending order.
Zero-overlap assignments are explicitly marked in the correspondence table as having no pixels supporting the correspondence.
Matching uses ascending raw IDs, and the SciPy version and actual correspondence are saved.
Original labels and metric calculations are unchanged.
The matching method, full overlap tables, raw-to-display mappings, and overlap pixel counts of assignments are saved to
`results/report/color_matching.json`.
Correspondence to initial raw IDs, conversion to global-fit display IDs, the composed mapping, and palette provenance are distinguished;
captions state that matching is for display only.
Color agreement is not used as numerical evidence of reproducibility. Boundary-frequency maps do not depend on color matching.
No figure or subplot titles are added. Pixel axes, ticks, and frames are hidden in map and boundary-frequency figures;
the specimen ID appears once in bold below the image group. Clustering is labeled Cosine-KMeans.
Panel letters and necessary numerical axes, legends, and colorbars are retained;
replicate descriptions, neighborhood widths, and longer explanations are placed in captions.
Figure descriptions and sample counts are saved to `results/report/captions.json`.

The summary is one figure relating ARI to boundary agreement for all 49 specimens.
It uses a 2×3 layout: within-encoder comparisons on the top row, between-encoder comparisons on the bottom row,
and 3×3, 5×5, and 9×9 neighborhoods as columns.
The smaller and larger directional values for each specimen are shown separately;
the line connecting them is not interpreted as a confidence interval.
Moving right indicates greater partition agreement; moving up indicates greater boundary proximity coverage.
Pair-averaged figures can obscure shifts in individual replicates, so the 3×3 maps and pair-level values are read alongside them.
The 3×3 maps and boundary-frequency figures are saved for all 49 specimens;
the previously selected seven representative specimens provide shared display examples.
Stable examples alone are not used to represent the full dataset; individual examples state their selection rationale.

## 7. Aggregation and Interpretation

- Save values by replicate pair, direction, and neighborhood width, together with numerator and denominator counts.
- Within each specimen, distinguish the nine within-encoder pairs from the 27 between-encoder pairs.
  Save the smaller and larger directional agreement values only when both directions are defined for a pair;
  a high value in one direction must not conceal a low value in the other.
- Main summaries are within-specimen means over replicate pairs, minima and maxima across pairs,
  equal-specimen-weight macro means, and between-specimen distributions.
  Aggregate smaller and larger directional values separately, and do not average the three neighborhood widths.
  Retain the original directional data.
- Aggregate both VI components and their sum over the same specimens.
  For the eight initial-map comparisons, report ARI, boundary agreement, and VI in the same specimen-level and overall table formats.
  Do not add those eight pairs again to the 36-pair aggregate.
- The main aggregate for each comparison group × metric includes specimens with defined values for every planned pair.
  Smaller and larger directional boundary agreement require both directions for all nine within-encoder pairs,
  all 27 between-encoder pairs, or all eight initial-map pairs, respectively.
  If no specimen qualifies, the main aggregate is undefined.
  Defined pairs for partially missing specimens, directional zeros, and counts of replicates with empty boundaries are retained in separate tables;
  means over only the remaining pairs do not substitute for the main aggregate.
- Report the numbers of defined pairs and included specimens for each metric, with reasons for undefined values in one or both directions.
  Summaries across neighborhood widths use the same comparison set, without filling missing values with 0 or 1.
  Summaries of differences between comparison groups are also restricted to shared specimens.
- Plot specimen-level relationships between ARI and boundary agreement. Keep the eight initial-map comparisons separately traceable.
- Do not conflate variability across specimens, encoders, Cosine-KMeans initializations, and replicate pairs.
  Do not use 36 pairs or 49×36 as independent sample counts for tests or CIs.
  There are three encoder replicates; no high-probability guarantee for future retraining in general is made.

Describe numerically and spatially how much boundary agreement occurs at each scale and where disagreement is located.
If ARI is low and boundary agreement is high, inspect boundary density and maps before interpreting the result as, for example,
“cluster partitions vary, while boundary arrangements on the measured surface share common features.”
Stable boundaries do not imply that membership relationships between distant regions or correspondence with chemical features are unchanged.

Whether region interiors and surrounding boundaries in the initial map persist across replicates indicates the stability
of the basis for interpreting spatial distributions.
Chemical meaning is examined by relating observed NIR spectra to literature assignments, wood anatomy,
environmental history, and specimen information.
The fact that observed spectra at fixed positions remain unchanged is not itself counted as evidence of model reproducibility.

## 8. Artifacts and Storage

The output directory is `outputs/experiments/global_spatial_reproducibility_v2/`.
`global_spatial_reproducibility_v1/` supplies trained encoders and reusable maps.
Existing weights, manifests, maps, and figures/tables in `global_v1` and `production_v1` are not overwritten.
Reused artifacts retain their source paths and existing provenance records, associated with the two replicate IDs for encoder and Cosine-KMeans.

| Artifact | Saved contents |
| --- | --- |
| Settings and provenance | Specification version, boundary definition, neighborhood widths, seeds, and correspondence among specimens, pixels, checkpoints, and existing maps |
| Clustering records | Config, run identity, centers, environment, referenced trained weights, and completion/failure status |
| All replicate maps | `labels_k8`, specimen ID, encoder ID, Cosine-KMeans ID, and correspondence with original image shape and mask |
| Per-map diagnostics | Valid pixel count, internal-boundary count and density, used cluster count, and occupancy |
| Pair-level evaluation | Comparison group, two run IDs, ARI, two VI components and their sum, directional agreement with numerators/denominators for each neighborhood width, and reasons for undefined values |
| Specimen-level and overall summaries | Replicate-pair summaries of ARI, boundary agreement, and VI; specimen macro means and distributions; counts; and initial-map comparisons |
| Figures | One ARI–boundary agreement figure, plus 3×3 maps and boundary-frequency figures for all 49 specimens: 99 figures in total |
| Supplementary overlap diagnostics | Targets and values when overlap tables, correspondence fractions, and VI are calculated; table values are undefined when VI is not calculated |

Specific CLI options, schemas, and filenames are documented in the [dedicated runbook](../spatial_map_reproducibility_runbook.md).
Trained payloads such as weights and centers are placed under `checkpoints/` or `weights/`, following the
[existing ignore rules](../../.gitignore).
Large intermediate representations are not saved; fitting-pixel representations are extracted for each encoder.
Boundary comparisons primarily use small fixed neighborhoods and specimen-level processing, without constructing an all-boundary distance matrix.

## 9. Procedure and Completion Criteria

1. Validate boundary extraction, distances, background exclusion, empty boundaries, aggregation, and storage with small arrays.
2. Use `prepare` to check consistency of weights, sampled coordinates, and reused maps, and save seeds, replicate pairs, and evaluation settings.
3. Use `cluster` to fit the second seed for each of the three encoders and generate maps for all 49 specimens.
4. Use `evaluate` to assess all 36 pairs among the nine maps per specimen and aggregate within-encoder, between-encoder, and initial-map comparisons.
5. Use `report` to export figures and tables, then `check` to verify counts, provenance, missingness, and correspondence between metrics and figures.

To update tables from saved pair-level evaluations, use `summarize`.
It generates aggregate JSON/CSV files from evaluation JSON without recomputing models, clustering, or pair-level metrics.
Then redraw the same figures with `report --replace`, and use `check` to verify correspondence between the updated evaluation and figures.

Completion means that artifacts and provenance for the planned replicates, numerical results, figures/tables,
and explanations of undefined values or failures are present.
High reproducibility is not a completion criterion.

## 10. References

- [Existing LLA, ARI, and evaluation-set definitions](evaluation_metrics.md)
- [Global fitting, visualization, and NIR spectral analysis](visualization_and_interpretation.md)
- [Execution and storage conventions](../experiment_runbook.md)
- [Berkeley Segmentation boundary and region evaluation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html): A reference for evaluating boundaries and regions separately; its implementation is distinct from the proximity coverage defined here.
- [Meilă, 2007: Comparing clusterings—an information based distance](https://doi.org/10.1016/j.jmva.2006.11.013): A reference for VI.
- [scikit-image VI definition](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.variation_of_information): Interpretation of splitting-side and merging-side disagreement through conditional entropy.
