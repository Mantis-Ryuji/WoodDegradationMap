# Research design

This directory defines the research conditions, calculations, and data contracts corresponding to the paper.
See the [research overview](../research_overview.md) for the study's scope and the [experiment runbook](../experiment_runbook.md) for operational procedures.

## Specifications

| Document | Definitions |
| --- | --- |
| [Preprocessing](preprocessing.md) | 200 Hz input, specimen masks, reflectance, pixel exclusion, interpolation to 256 points, SNV, and storage schema |
| [Experiment protocol](experiment_protocol.md) | Comparison conditions, splits, seeds, common pixels, models, training, and clustering |
| [Evaluation metrics](evaluation_metrics.md) | LLA, LFR, ARI, silhouette, occupancy, undefined values, aggregation, and paired differences |
| [OOF diagnostic figures](oof_sanity_visualization.md) | OOF maps for B0, B1, A0, and M00, with display alignment to B0 within each fold |
| [Global-fit visualization and interpretation](visualization_and_interpretation.md) | Display alignment to M00, representative specimens, observed spectra, PCA, and NIR interpretation |
| [Spatial map reproducibility](spatial_map_reproducibility_plan.md) | M11/K8 encoder and KMeans replicates, and comparisons of regions and boundaries |
| [Bacteria-ID supplementary experiments](bacteria_id_experiment_plan.md) | Raman input, pretraining, clustering, and conditions for comparison with the reference paper |
| [Bacteria-ID classification and HPO](bacteria_id_hpo_protocol.md) | Classification of 30 isolates, configuration selection by five-fold CV, and final test evaluation |

## Analysis structure

| Analysis | Conditions and scope | Replicates and evaluation scope |
| --- | --- | --- |
| Main experiment | Six main conditions B0, B1, A0, A1, M00, and M11, plus perturbation ablations M10 and M01 | Specimen-level five-fold CV with three replicates; K=2/4/6/8/10/12/14; OOF evaluation on unseen specimens |
| Mask-ratio sensitivity, CV | M11 at 25%, 50%, and 75% | Same splits and pixel sets, with three training replicates; 50% reuses M11 from the main experiment |
| Global fit | Six main conditions, all 49 specimens | Each representation is trained or fitted once, followed by three K8 clustering initializations; maps use the first initialization |
| Mask-ratio sensitivity, global fit | M11 at 50% and 75% | K8 with three initializations for each fixed representation; a descriptive comparison of the specimens used for fitting |
| Spatial map reproducibility | M11, all 49 specimens, K8 | Comparison of nine maps from 3 encoders × 3 KMeans initializations |
| Raman clustering | Bacteria-4/6, M00 and M11 | 80% train / 20% test, five seeds, K=4/6 |
| Raman classification | 30 isolates, M00 and M11 | A common configuration selected by five-fold CV on M11, followed by final training and evaluation with five seeds per condition |

Seeds and input scope for each analysis are defined in the corresponding specification. The roles of comparisons and the provenance of configuration selection
are documented in the [main experiment protocol](experiment_protocol.md#planned-comparisons) and the Raman specifications.

## Conditions and limits of interpretation

- Data splitting and sampling: [CV and seeds](experiment_protocol.md#cv-design), [common pixels](experiment_protocol.md#pixel-sampling).
- Models: [comparison conditions](experiment_protocol.md#conditions), [representations and PCA](experiment_protocol.md#representations), [architecture](experiment_protocol.md#model-architecture), [training settings](experiment_protocol.md#training-recipe).
- Operations and evaluation: [perturbations and clustering](experiment_protocol.md#augmentation-clustering), [numerical precision](experiment_protocol.md#evaluation-precision), [K](experiment_protocol.md#cluster-counts), [metrics and aggregation](evaluation_metrics.md).

CV evaluates spatial consistency and stability under specified perturbations on specimens excluded from training.
Global fitting and NIR analysis explore and interpret chemical states within the observed specimen set.
Cluster labels are not treated as degradation severity or chemical concentrations, and degradation models are hypotheses without causal validation.
In the Raman experiments, "conditions specified in this study" fill in details absent from the reference paper and are distinguished from the original study's actual settings.
