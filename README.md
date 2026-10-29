<h1 align="center">WoodDegradationMap</h1>

[![Python](https://img.shields.io/badge/python-3.11.15-blue)](./.python-version)
[![chemomae](https://img.shields.io/badge/chemomae-0.2.2-orange)](https://github.com/Mantis-Ryuji/ChemoMAE)
[![CI](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml/badge.svg)](https://github.com/Mantis-Ryuji/WoodDegradationMap/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code for exploring differences in chemical state in historic wood using near-infrared hyperspectral imaging (NIR-HSI).
ChemoMAE, a self-supervised representation learning method, compresses each pixel spectrum, and clustering produces maps of regions on specimen surfaces.

Associated paper: *Self-Supervised Spectral Learning Reveals Chemical Heterogeneity in Historic Wood through Near-Infrared Hyperspectral Imaging*

## Research overview

ChemoMAE applies Tangent Gaussian Noise (TGN) and Fractional Shift (FS), which preserve the mean and norm after SNV transformation,
and reconstructs the original, unperturbed values of masked bands from visible bands. Cosine-KMeans partitions the learned
16-dimensional representations, and cluster assignments are mapped back to the image. Coordinates and neighboring pixels are not inputs to learning or clustering.

The analyses cover 49 historic wood specimens:

- **Evaluation on unseen specimens:** Specimen-level five-fold cross-validation with three replicates compares direct input, PCA, AE, denoising AE, and ChemoMAE. The main metrics are LLA, which corrects local label agreement for chance, and LFR, which measures assignment changes under specified perturbations.
- **Mapping with a common reference:** Representations and cluster centers are fitted on all 49 specimens to compare the distributions of the six main conditions at K=8. For M11, repeated training and clustering also assess the reproducibility of regions and boundaries.
- **Interpretation using NIR spectra:** Spatial cluster distributions and observed spectra are related to published band assignments, wood anatomy, environmental history, and specimen information. Models of degradation processes are treated as hypotheses based on observations.
- **Supplementary experiments:** These include mask-ratio sensitivity analysis and clustering and supervised classification on Raman Bacteria-ID.

The metrics characterize maps; they do not measure pixel-level degradation detection accuracy or amounts of chemical components.
The evidence and scope of interpretation are described in the [research overview](docs/research_overview.md).

## Documentation

| Purpose | Reference |
| --- | --- |
| Understand the research objectives, methods, and scope of results | [Research overview](docs/research_overview.md) |
| Check preprocessing, conditions, and metric definitions | [Research design](docs/design/README.md) |
| Reproduce experiments and figures | [Experiment runbook](docs/experiment_runbook.md) |
| Locate outputs for each analysis | [Output guide](docs/results.md) |
| Read mathematical and implementation notes and related work | [Documentation index](docs/README.md) |

## Environment

The project uses Python 3.11.15 and `uv`. Dependencies are pinned in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).
PyTorch is configured to use the CUDA 13.0 distribution index.

```powershell
uv sync --locked --dev --group bacteria-hpo
uv run --no-sync pytest
```

Run commands from the repository root. Training uses a GPU.
See the [experiment runbook](docs/experiment_runbook.md) for input layout, preprocessing, training, evaluation, and resumption conditions.
The Raman experiments have separate guides for [clustering](docs/bacteria_id_runbook.md) and [classification and HPO](docs/bacteria_id_hpo_runbook.md).

## Directories and data

```text
src/wood_degradation_map/        Preprocessing, training, clustering, and evaluation
scripts/                        Experiment and figure-generation CLIs
tests/                          Tests
docs/                           Research specifications, reproduction guides, and technical notes
data/                           Original and preprocessed data (not tracked by Git)
outputs/                        Experiment records, numerical results, and figures (not tracked by Git)
```

This repository contains code, configuration, and documentation. Historic wood data, experiment outputs, and trained weights are not included.
Rerunning the wood analyses requires the inputs described in the [preprocessing specification](docs/design/preprocessing.md).
See the [supplementary experiment specification](docs/design/bacteria_id_experiment_plan.md) for the source and input layout of Raman Bacteria-ID.
Paths under `data/` and `outputs/` in the documentation refer to files supplied or generated locally.

## License

The code in this repository is released under the [MIT License](LICENSE).
The underlying [ChemoMAE](https://github.com/Mantis-Ryuji/ChemoMAE) library and third-party data are subject to their respective licenses and terms of use.
