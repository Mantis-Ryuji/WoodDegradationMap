# Documentation index

Research objectives, analysis specifications, reproduction guides, and guidance on interpreting results.

## Research and methods

| Document | Contents |
| --- | --- |
| [Research overview](research_overview.md) | Research questions, analysis structure, and scope of results and claims |
| [Positioning of ChemoMAE](chemomae_positioning.md) | Roles of masked denoising, unit latent vectors, and the affine decoder |
| [Related work](related_work.md) | Connections to prior methods and references |
| [Interpreting results](interpretation_notes.md) | Limits of maps, unsupervised metrics, and spectral interpretation |
| [Mathematics of SNV and spectral perturbations](mathematical_notes.md) | Constraint preservation by SNV, TGN, and Fractional Shift |
| [Numerical implementation notes](numerical_implementation_notes.md) | Mixed precision, training loss, and numerical handling in clustering |

## Specifications and reproduction guides

| Analysis | Specification | Execution and outputs |
| --- | --- | --- |
| Historic wood NIR-HSI | [Research design index](design/README.md) | [Experiment runbook](experiment_runbook.md), [output guide](results.md) |
| Mask-ratio sensitivity | [Experimental conditions](design/experiment_protocol.md), [global-fit conditions](design/visualization_and_interpretation.md#global-mask-rate) | [CV procedure](experiment_runbook.md#mask-rate-sweep), [global-fit procedure](experiment_runbook.md#global-mask-rate) |
| Spatial map reproducibility | [Evaluation specification](design/spatial_map_reproducibility_plan.md) | [Dedicated runbook](spatial_map_reproducibility_runbook.md) |
| Raman clustering | [Bacteria-ID specification](design/bacteria_id_experiment_plan.md) | [Runbook](bacteria_id_runbook.md) |
| Raman classification | [HPO specification](design/bacteria_id_hpo_protocol.md) | [Runbook](bacteria_id_hpo_runbook.md), [interpretation](bacteria_id_interpretation_plan.md) |

Research conditions, calculations, and data contracts are defined in `docs/design/`; operational procedures are described in the runbooks.
Measured values and execution provenance are linked to each output's configuration, manifests, run records, and completion records.
Paths under `data/` and `outputs/` are not tracked by Git and refer to local inputs and outputs.

## Terminology

| Term | Meaning in this repository |
| --- | --- |
| raw SNV / B0 | Baseline using the preprocessed 256-dimensional SNV input directly; not raw sensor intensity |
| Specimen | Unit of splitting and aggregation identified by `KYOw...`. Correspondence to the same original timber member and relative positions during use have not been established |
| CV repeat | Three training and clustering replicates using the same split and pixel sets |
| Global-fit KMeans repeat | Three K8 initializations with each condition's representation fixed; these do not include variability from retraining the neural network |
| draw | An evaluation perturbation replicate; each perturbation is generated five times for LFR |
| OOF | Predictions and evaluations from the fold model whose training set excludes the specimen |
| Global fit | Analysis fitted or trained on common pixel samples from all 49 specimens to describe the entire specimen set |
| matching | Mapping display labels without changing cluster membership; it does not establish a degradation order or a shared chemical meaning |
| clean | The observation before additional noise and shift; not a ground truth free of measurement noise |

Alignment to B0 within each fold in OOF figures differs from alignment to M00 in global-fit figures.
They are defined in the [OOF diagnostic figure specification](design/oof_sanity_visualization.md) and the [global-fit visualization specification](design/visualization_and_interpretation.md), respectively.
