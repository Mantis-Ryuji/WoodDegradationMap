# Research overview

This study addresses self-supervised representation learning for materials whose state categories and supervised labels are difficult to define.
It relates spectral features to spatial distributions and compares findings across specimens using a common reference.
The data comprise NIR-HSI of 49 historic wood specimens from seven tree species, originating from timber members of historic buildings.

<a id="research-focus"></a>

## Research question

We examine whether learning relationships between spectral bands without ground-truth chemical-state labels can yield representations useful for exploring and interpreting state differences.
Maps are evaluated using spatial information excluded from learning, under the hypothesis that shared spectral features associated with state differences
will also produce spatial continuity in partitions of unseen specimens.

Spectra of historic wood may combine effects of aging, external exposure, moisture history, tree species, wood anatomy, moisture content, surface properties, and scattering.
Clusters are treated as partitions for investigating continuous, overlapping state differences, rather than predefined degradation classes.

## Methods

1. Extract specimen regions, convert to reflectance, select the wavelength range, interpolate to 256 equally spaced points, and apply SNV.
2. Use ChemoMAE to learn a 16-dimensional unit latent vector from each pixel spectrum. Apply TGN and FS, which preserve the SNV mean and norm, and reconstruct the original, unperturbed values of masked bands from visible bands.
3. Pass the spectra to the trained encoder with all bands visible and no additional perturbations, then map Cosine-KMeans assignments back to image coordinates.
4. Compare map properties using specimen-level CV, and interpret regional differences using global-fit maps and observed NIR spectra.

The encoder receives only a single pixel spectrum, without coordinates or neighboring pixels.
The decoder is an affine map that reconstructs all bands from one latent vector; the latent coordinates are not constrained to represent chemical concentrations.
See [positioning of ChemoMAE](chemomae_positioning.md) for the architectural rationale and [research design](design/README.md) for computational settings.

## Evaluation and analyses

| Analysis | Purpose and evidence |
| --- | --- |
| Five-fold CV with three replicates | Compare partitions from direct input, PCA, AE, denoising AE, and ChemoMAE on specimens excluded from training |
| LLA | Measure spatial consistency of neighboring labels after correcting for chance agreement under random placement that preserves cluster occupancy |
| LFR | Measure the fraction of assignments that change under specified TGN/FS perturbations with the model and centers fixed |
| Supplementary diagnostics | Assess partition reproducibility across replicates with ARI, separation with cosine silhouette, and degeneracy with the number of occupied clusters and maximum occupancy |
| Global fit | Describe all 49 specimens using common representations and cluster centers, and compare regional distributions across the six main conditions at K8 |
| Spatial map reproducibility | Examine agreement of regional partitions and boundary locations across 3 M11 encoders × 3 KMeans initializations |
| NIR interpretation | Interpret observed spectra and second derivatives of pseudo-absorbance in light of published band assignments, spatial distributions, and specimen information |
| Raman supplementary experiments | Train separately on Bacteria-ID to examine clustering of held-out spectra and isolate classification through supervised fine-tuning |

Mask ratios of 25%, 50%, and 75% are a sensitivity analysis and are not used to reselect the main 50% condition.
Training replicates in CV are distinguished from KMeans initialization replicates with fixed representations in global fitting.
See the [metric definitions](design/evaluation_metrics.md) and [output guide](results.md) for details.

## Paper findings and scope of interpretation

In OOF evaluation of the 49 historic wood specimens, ChemoMAE partitions showed greater spatial consistency than direct input or PCA.
The improvement from replacing full-band reconstruction with masked-band prediction was more consistent than the additional gains from perturbations.
Some conditions improved assignment stability under the specified perturbations without increasing LLA, so these are evaluated as distinct properties.

Global fitting characterized distributions corresponding to wood anatomy and externally exposed surfaces, and repeated runs showed common regional and boundary patterns.
Relating NIR features and spatial distributions to previous studies and timber-member information led to two hypotheses:
an internal degradation model involving aging and water, and a surface degradation model involving photodegradation, erosion, leaching, inward water transport, and secondary reactions.
These are exploratory interpretations based on observations, not demonstrations of reaction processes or causal relationships.

The Raman experiments show that useful discriminative representations can also be learned separately from spectra other than NIR.
They do not transfer the model trained on historic wood to Raman data, nor do they directly validate the chemical assignments made for wood.

LLA, LFR, and ARI do not measure agreement with ground-truth chemical states. Global fitting is not a performance evaluation on unseen specimens.
Because common origin from an original timber member and relative positions during use have not been established for all specimens,
CV should be interpreted within the scope of specimen-ID-level splitting.
Further limitations are described in [interpreting results](interpretation_notes.md).
