# Reading and interpreting results

This document explains what maps, metrics, and reconstruction loss can establish, and their limitations.
Metric definitions and aggregation follow the [evaluation specification](design/evaluation_metrics.md); comparison conditions follow the [experiment protocol](design/experiment_protocol.md).
Chemical interpretation follows the [global visualization design](design/visualization_and_interpretation.md#evidence-triangulation).

## 1. Maps and metrics

The encoder takes one pixel spectrum as input and receives no coordinates or neighboring pixels.
Spatially coherent regions may arise when coherent anatomy or chemical states are reflected in similar neighboring spectra and assignments.
This study distinguishes the assumption that imaging conditions such as illumination were controlled as far as possible from variation due to the specimens' own surface properties.
Because differences involving surfaces, anatomy, and scattering are also interpreted, "physicochemical features" here are not restricted to chemical composition.
Controlling conditions does not by itself eliminate measurement-related variation, and label agreement does not identify its causes.

| Metric | What it indicates | Limits of interpretation |
| --- | --- | --- |
| LLA | Agreement beyond random placement that preserves occupancy | Does not directly measure isolated pixels, connectivity, or the correctness of small regions |
| LFR(TGN+FS) | Assignment stability after specified noise and shift | Does not directly measure latent displacement, chemical information retention, or robustness to measurement errors in general |
| ARI | Reproducibility of partitions across replicates on the same test pixels | Not agreement with ground truth or stability across different splits |
| Cosine-Silhouette | Relative within- and between-cluster distances in a representation | Not a common measure of chemical validity or the information content of a latent representation |
| Cluster Occupancy | Concentration in clusters, number of occupied clusters, and imbalance | A single-cluster specimen is not judged a success or failure on that basis alone |
| Uncorrected LLA (supplementary) | Apparent local label agreement | Does not directly measure correspondence among latent distances, physical distances, and chemical states, or boundary correctness |

The representative display for the main conditions follows the order of the first five metrics above; LLA denotes the corrected value.
Occupancy and 2×2 interactions are reported in CSV tables. LFR(TGN+FS), LFR(TGN), and LFR(FS) are saved in the report CSVs.
Uncorrected LLA and the original LFR evaluation values are also retained in OOF snapshots.

Low silhouette, high uncorrected LLA, and low LFR can coexist. Partitioning continuous variation at a fixed K can produce coherent surface regions
despite weak separation, but this alone is not evidence of a continuous chemical gradient.
Uncorrected LLA can also increase with occupancy imbalance or collapse to a single cluster; read it alongside occupancy, the number of occupied clusters, and maps.

Latent representations may capture overlapping variation due to degradation, tree species, anatomy, moisture content, and the measured surface.
Clustering quantizes a continuous representation, and representative spectra help explore regional differences.
The number of clusters is not the number of states in nature, and representative curves are not pure components, degradation stages, or concentration scales.

LLA measures agreement within specified local neighborhoods and does not require a monotonic relationship between physical distance and spectral or latent distance.
Repeated structures such as growth rings can place similar states far apart. Local coherence should not be confused
with an assumption that greater distance always means greater difference throughout a specimen.

### 1.1 Relating spectral features to spatial distributions

The study aims to make it easier to explore differences in specimen surface states and candidate chemical interpretations
by summarizing relationships across wavelengths in a representation and grouping pixels by clustering.
"Supporting exploration of chemical states" is the research objective; it does not mean that chemical states have already been identified.

This use is shared by direct clustering of B0 and PCA in B1. For ChemoMAE, the focus is on using a representation
that learns predictive relationships between bands, and comparing the regional and spectral differences it organizes.
Using multiple wavelengths is not unique to ChemoMAE; interpretive usefulness beyond the baselines must be assessed from results.
The architecture and learning objective are described in the [ChemoMAE overview](chemomae_positioning.md#1-positioning-of-the-architecture).

Chemical interpretation uses the first global-fit M11/K8 assignments. Within-specimen cluster means of SNV spectra and
second derivatives of pseudo-absorbance are related to cluster locations relative to growth rings, externally exposed surfaces, and insect holes.
Band assignments are based on the literature, together with information on tree species, timber members, and environmental history.
Sharing a clustering reference across specimens allows comparison of similarities and differences in spatial distributions and spectra.

Second-derivative band values describe curvature in absorption shapes, not component quantities.
The depth of a negative trough depends on band width, overlap, and the hydrogen-bonding environment, as well as component contributions.
The involvement of moisture history or degradation is inferred from spectra, distributions, and timber-member information; that history was not measured directly.
Internal and surface degradation models are hypotheses based on observations and are distinguished from causal demonstrations of transport and reaction processes.

### 1.2 Scope of aggregation and exploratory interpretation

CV compares spatial consistency, stability under specified perturbations, and reproducibility across replicates on unseen specimens.
Global-fit maps and spectra from all 49 specimens are used to describe regional differences and explore their interpretation.
Clusters compared within each specimen were selected by visually examining their distributions; this is not chemical validation on unused data.
Detailed interpretation of M11 does not imply superiority on every metric.

Macro-average curves across specimens provide an overview of conditions but can dilute local differences within individual specimens.
Read within-specimen cluster means in reflectance, SNV, and second derivatives of pseudo-absorbance together, and check membership and exclusion counts.
A difference visible in SNV is not rendered meaningless if it is small in the second derivative.
Pseudo-absorbance uses only pixels with positive, finite reflectance in every band, so its contributing pixels can differ from those in SNV summaries.

Band-value comparisons construct a common reference baseline within each specimen from clusters judged relatively sound.
This baseline is not undegraded wood confirmed by chemical analysis. Bars for each state are equally weighted means across contributing specimens,
and error bars are inter-specimen sample standard deviations. Within-specimen differences are compared using the common specimens for which all required states are available.
Missing values are not imputed. See the [visualization and interpretation specification](design/visualization_and_interpretation.md) for selection and aggregation definitions.

PCA PC1 and PC2 are supplementary projections; separation or grouping in these plots alone does not establish superiority of the M conditions.
Use LFR and ARI to assess assignment stability under specified perturbations and reproducibility across replicates, rather than the appearance of exploratory plots.

<a id="pca-snv-geometry-note"></a>

## 2. PCA and SNV geometry

The stages of B1 have the following effects.

| Operation | Mathematical effect |
| --- | --- |
| Centering by the training mean | Changes the origin for angles. A zero bandwise mean within each pixel does not imply a zero mean vector across training pixels |
| Projection onto 16 components | Can change both direction and norm; constant input norm does not carry over to the scores |
| L2 normalization of scores | Compares directions without using information in the score norms downstream |

For constant-norm inputs, squared Euclidean distance is proportional to cosine dissimilarity; spherical geometry is not itself incompatible with PCA.
Maps alone cannot separate the contributions of centering, compression, and normalization or attribute differences solely to L2 normalization.
Compression to 16 dimensions and separation of chemically useful regions are distinct properties. Neural networks likewise provide no guarantee of preserving input angles or chemical information.

## 3. Comparing noise and shift

TGN is specified by an angle, whereas Fractional Shift is specified by a width in channels on the wavelength grid.
Their strengths are not matched on a common scale, so differences in LFR do not rank the general severity of the perturbations.
Shift deforms the wavelength axis; it is neither a displacement of image coordinates nor an adversarial perturbation.

M01 versus M00 and M11 versus M10 examine the effect of adding shift.
A0 versus M00 differs in both input masking and the loss region; it does not isolate additional TGN/shift or mask ratio alone.
A1 is a denoising AE that adds the same TGN+shift as M11 to A0's 0% mask and full-region reconstruction.
A1 versus M11 also differs in both masking and the loss region. See the [experiment protocol](design/experiment_protocol.md#planned-comparisons) for the roles of these comparisons.

<a id="decoder-residual-discussion"></a>

## 4. Reconstruction loss and latent representations

Training loss is the epoch mean of batch MSE during training, not a reevaluation of the fixed final model or performance on unseen specimens.
A0 uses all 256 channels, whereas M11 uses the 128 invisible channels with additional perturbations and masking; their loss ratio does not compare performance on the same task.
RMSE is error in SNV coordinates, not a relative reflectance error or a rate of chemical information retention.

A small loss is consistent with reconstructing prominent, shared spectral shapes, but does not guarantee retention of subtle differences related to wood anatomy or degradation.

The relationship between training AMP and FP32 loss is described in the [numerical implementation notes](numerical_implementation_notes.md#training-precision).

## 5. Reporting comparisons

- Report the representative $`K_0=8`$ and all seven fixed K values, including reversals in trends and specimen-specific exceptions. The same representations and specimens are reused across K, so these are not independent replications.
- Distinguish results from individual folds from the full OOF aggregation. Formal comparisons use paired differences on common specimens, variability across specimens and replicates, ARI, and occupancy.
- Passing artifact checks establishes input/output consistency, not chemical validity or superiority of a condition.
- Examine chemical meaning by relating maps, NIR spectra, published band assignments, and specimen information. Improvements in unsupervised metrics alone do not establish degradation detection accuracy.
