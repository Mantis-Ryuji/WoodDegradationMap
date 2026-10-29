# Related work and references

This document summarizes prior work and references relevant to the [rationale for ChemoMAE](chemomae_positioning.md).
It identifies the version and scope of each source and distinguishes them from this study's evaluation on historic wood NIR-HSI.

## 1. Connections to related work

Prior work is related to this study in terms of architecture, learning task, and use of representations.
None validates the same combination of preprocessing, architecture, and evaluation used here; reported performance is not extrapolated to historic wood NIR data.

| Reference | Connection to this architecture | Differences and scope of citation |
| --- | --- | --- |
| [Vincent et al. (2010), Denoising AE](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) | Uses reconstruction from corruption as a task for learning useful representations; motivates adding noise and shift | Corruption, architecture, and evaluation differ. It does not establish validity for chemical states in historic wood or weaker assumptions than other SSL methods |
| [He et al. (2022), MAE](https://arxiv.org/html/2111.06377v3) | Passes only visible patches to the encoder and trains with masked MSE | Uses a Transformer decoder with visible and mask tokens; does not restrict full reconstruction to a single 16-dimensional CLS representation |
| [Ren et al. (2025), Raman SMAE](https://arxiv.org/html/2504.16130v1) | Reconstructs masked spectra and compares clustering of the learned representations with PCA and other methods | Uses a Transformer decoder and separately evaluates supervised fine-tuning |
| [Jensen et al. (2024), Raman denoising VAE](https://www.nature.com/articles/s41598-024-56788-7) | Learns representations by reconstructing original observations from spectra with added noise and axis variations | Builds on a VAE that treats intensity and the frequency axis separately, unlike this architecture with SNV-preserving TGN and masking |
| [Ji et al. (2026), LeafVAE](https://doi.org/10.1016/j.compag.2026.111971) | Unsupervised compression of pixel spectra, latent clustering, and spatial mapping; applies a trained model and fixed cluster centers to new data | Uses a two-dimensional VAE and nonlinear decoder, without masked reconstruction. Leaf diagnosis uses supervised random forests with cluster proportions as inputs and SHAP |

The data splits and training conditions in SMAE's Raman clustering comparisons are not necessarily identical to those in this study.
See the [Bacteria-ID protocol](design/bacteria_id_experiment_plan.md) for the supplementary experiment's comparison conditions.

<a id="references"></a>

## 2. References

### 2.1 Representation learning through denoising

1. Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y. & Manzagol, P.-A. (2010).
    [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://jmlr.org/papers/v11/vincent10a.html).
    *JMLR*, 11, 3371–3408. [Section 3 of the public text](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
    is cited for representation learning through denoising and the choice of corruption. The original paper's manifold interpretation
    does not establish this study's SNV constraints or identify a manifold of chemical states.

### 2.2 Masked modeling and contrastive learning

2. He, K. et al. (2022).
    [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
    *CVPR*. See [Section 3 of the public text](https://arxiv.org/html/2111.06377v3) for the architecture.
3. Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020).
    [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html).
    *ICML*, PMLR 119, 1597–1607. Cited for the role of augmentation in SimCLR.

### 2.3 Spectral representation learning and spatial mapping

4. Ren, P., Zhou, R.-G. & Li, Y. (2025).
    [A Self-supervised Learning Method for Raman Spectroscopy based on Masked Autoencoders](https://arxiv.org/abs/2504.16130).
    The descriptions of its architecture and experiments here are based on the [public v1 manuscript](https://arxiv.org/html/2504.16130v1).
    This version is distinguished from the [published version](https://doi.org/10.1016/j.eswa.2025.128576) whose reported values are used in the supplementary experiments.
5. Jensen, M. N. et al. (2024).
    [Identification of extracellular vesicles from their Raman spectra via self-supervised learning](https://www.nature.com/articles/s41598-024-56788-7).
    *Scientific Reports*, 14, 6791. Cited for the Introduction's description of corruption, reconstruction targets, and the VAE,
    as a precedent for denoising representation learning with noise and axis variations.
6. Ji, K. et al. (2026).
    [Variational autoencoder enables unsupervised leaf diagnosis via hyperspectral imaging](https://doi.org/10.1016/j.compag.2026.111971).
    *Computers and Electronics in Agriculture*, 251, 111971.
    Sections 2.3 and 3–4 and Figures 1–4 are cited for VAE representations of pixel spectra, clustering,
    application to new data with fixed centers, and spatial mapping.
    Unsupervised representation learning is distinguished from downstream diagnosis and SHAP explanations that use labels.

### 2.4 SNV geometry, angular comparisons, and augmentation

7. Fearn, T., Riccioli, C., Garrido-Varo, A. & Guerrero-Ginel, J. E. (2009).
    [On the geometry of SNV and MSC](https://doi.org/10.1016/j.chemolab.2008.11.006).
    *Chemometrics and Intelligent Laboratory Systems*, 96(1), 22–26.
    Cited for SNV geometry and curved structures in score plots described in the publisher's public abstract and Introduction.
    The [dimension and radius of the input sphere](chemomae_positioning.md#snv-geometry) are derived from this study's SNV definition.
8. Banerjee, A., Dhillon, I. S., Ghosh, J. & Sra, S. (2005).
    [Clustering on the Unit Hypersphere using von Mises-Fisher Distributions](https://jmlr.org/papers/v6/banerjee05a.html).
    *JMLR*, 6, 1345–1382.

### 2.5 Measurement noise

9. Hamamatsu Photonics.
    [Image sensors product selection](https://hub.hamamatsu.com/us/en/technical-notes/image-sensors/image-sensors-product-selection.html).
    Manufacturer's technical documentation. Sections 1.2–1.3 are cited for noise sources and the definition of SNR.
    This source does not measure the noise distribution of the instrument used in this study.
