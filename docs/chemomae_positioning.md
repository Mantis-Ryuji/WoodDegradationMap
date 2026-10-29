# ChemoMAE and its role in chemometrics

This document describes the rationale for using ChemoMAE v0.2.2, the model and perturbation definitions, and the scope of evaluation.
See the [experiment protocol](design/experiment_protocol.md) for fixed conditions, [mathematics of SNV and spectral perturbations](mathematical_notes.md) for derivations,
[related work](related_work.md) for connections to prior studies, and [interpreting results](interpretation_notes.md) for guidance on reading the results.

- [Rationale for MAE and additional corruption](#1-positioning-of-the-architecture)
- [Model and loss](#2-implemented-architecture)
- [PCA and L2 normalization](#pca-comparison)
- [TGN and shift](#spectral-augmentation), [interpretation through VRM](#augmentation-vicinity)
- [Scope of evaluation and interpretation](#claims)

## 1. Positioning of the architecture

For materials whose state categories and ground-truth labels are difficult to define in advance, ChemoMAE compresses spectra
through predictive relationships between bands and learns a coordinate system for exploring state differences.
Historic wood NIR-HSI is the empirical application: cluster maps from fixed encoders are compared by CV,
and regional differences are interpreted by relating NIR spectral features and spatial distributions to published band assignments,
wood anatomy, environmental history, and specimen information.
The relationship between the research questions and evidence is described in the [research overview](research_overview.md#research-focus).

**As with PCA, low-dimensional coordinates obtained without labels are used for visualization and clustering.**
Using MAE representations for clustering also has a precedent in Raman analysis.
([Ren et al., 2025, §3.2](https://arxiv.org/html/2504.16130v1))

This architecture has a nonlinear encoder, an affine decoder, and unit latent vectors; it does not mathematically subsume PCA.
It is positioned as **"nonlinear spectral representation learning with a linear reconstruction constraint, serving the role of PCA."**
The procedure of fixing the encoder after training is distinguished from the hypothesis that additional corruption improves representations.

### 1.1 Why MAE was chosen

In historic wood, variation related to degradation, tree species, and surface properties overlaps. It is not obvious which information
should be retained as state differences or which variations a representation should be invariant to.
Reconstruction-based learning therefore uses observed spectra as the learning reference.
With values before additional perturbations as targets, predicting hidden bands from visible bands learns dependencies between bands.

### 1.2 Why noise and shift were added: denoising as a representation learning task

**Random masking is itself a denoising corruption. TGN and shift add corruption to missingness.**
Keeping the observation before additional perturbations as the target and predicting invisible bands from deformed visible bands
is intended to learn inter-band relationships useful for exploring chemical states. Noise-removal performance or resistance to the specified perturbations is not the primary objective.

Using reconstruction from corruption to learn intermediate representations follows the principle of denoising autoencoders.
Vincent and colleagues' classification experiments and other results support this principle, but do not demonstrate extraction of chemical states from historic wood NIR data.
([Vincent et al., 2010, §3](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf))

This task provides an explicit reconstruction reference but assumes that the original observation should be recovered from the specified corruption.
Here, clean means the observation before additional perturbations, not a ground truth free of measurement noise.

The research question is **how corruption conditions in the MAE group change the properties of maps obtained from fully visible representations,
and how their regional differences can support exploration and interpretation of state differences**.
CLS is a token that aggregates one pixel's information into a single bottleneck; it does not represent a chemical component or supervised label.
"Structural inference" likewise refers to estimating relationships between bands, not identifying molecular structure.
The reconstruction loss does not directly require clusters to form or separate.

This is examined using the six main conditions—direct input B0, PCA B1, AE A0, denoising AE A1, M00, and M11—
and the 2×2 ablation of M00, M10, M01, and M11.
Effects of additional corruption in MAE are evaluated using corrected LLA, LFR, and the specified diagnostics.

## 2. Implemented architecture

### 2.1 Compressing the full spectrum into one latent vector

| Component | Configuration |
| --- | --- |
| Input | A 256-channel SNV spectrum from one pixel; no coordinates or neighboring pixels |
| Patches and masking | 16 patches of 16 consecutive channels; the main MAE conditions have 8 visible patches plus CLS, giving 9 tokens |
| Encoder | Width 256, 8 layers, 8 heads, FFN width 1024, GELU, pre-norm, dropout 0, and learnable positional embeddings |
| Bottleneck | Linear projection of the final CLS to 16 dimensions, followed by L2 normalization |
| Decoder | `Linear(16, 256)` with bias |
| Loss | MSE against SNV before additional perturbations; invisible channels for MAE, all channels for A0 and A1 |
| Use after training | All patches visible, no augmentation, 17 tokens; the fixed encoder supplies unit latent vectors to Cosine-KMeans |

The only input-derived information passed to the decoder is a single $`z`$; patch-specific outputs, skip connections, and decoder mask tokens are not used.
This describes the study's `decoder_num_layers=1` configuration, not every configuration supported by the library.
Sources are the [fixed configuration](../src/wood_degradation_map/experiments/config.py),
[mask generation and extraction](../src/wood_degradation_map/experiments/neural.py), [training code](../src/wood_degradation_map/experiments/training.py),
and [reference model](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/models/chemo_mae.py).

### 2.2 Learned mapping and loss

Let $`x_i\in\mathbb{R}^{256}`$ be the observation before additional perturbations, $`g`$ the additional corruption, $`V_i`$ the set of visible patches,
and $`f_\theta`$ the encoder including the CLS projection. In the usual nonzero-norm case,

```math
u_i=f_\theta(g(x_i);V_i),\qquad z_i=\frac{u_i}{\lVert u_i\rVert_2},\qquad
\hat{x}_i=Wz_i+b,\quad W\in\mathbb{R}^{256\times16},\quad b\in\mathbb{R}^{256}.
```

L2 normalization includes epsilon protection, and extraction checks for nonfinite values and zero or very small norms before normalization.
The equations describe the case in which that protection is inactive. The set of hidden channels $`M_i`$ contains 128 elements for the main MAE comparison.
For batch size $`B`$,

```math
\mathcal{L}_{\mathrm{MAE}}
=\frac{1}{B}\sum_{i=1}^{B}\frac{1}{128}\sum_{j\in M_i}
\left([Wz_i+b]_j-x_{ij}\right)^2.
```

For M00, $`g`$ is the identity; M10, M01, and M11 use TGN, shift, and both, respectively. Target values and wavelength positions are unchanged.
Application probabilities and strengths follow the [experiment protocol, §4.1.3](design/experiment_protocol.md).
A0 has no additional perturbations and full visibility; the inner sum and denominator are changed to all 256 channels.
A1 is a denoising AE with full visibility and loss over all 256 channels, applying the same TGN+shift as M11 to its input.

At each MAE step, loss covers only invisible bands, but random masks across pixels and steps make all bands reconstruction targets.
Because predictions themselves depend on the mask, however, the mask-averaged objective is not identical to fully visible, full-band loss.
The masked training task and fully visible representations used after training are evaluated separately.

The loss contains no class labels, KMeans assignments, spatial terms, or contrastive learning terms. It is also distinguished from AdamW weight decay.
The implementation corresponds to `Trainer._compute_loss` with `loss_type="mse"`, `reduction="mean"`, and a condition-specific `loss_region`.
([Reference Trainer](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/training/trainer.py);
[reference loss](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/models/losses.py))

## 3. Similarities to PCA and mathematical differences

<a id="pca-comparison"></a>

### 3.1 Comparison with the PCA baseline

| Aspect | PCA baseline B1 | ChemoMAE in this study |
| --- | --- | --- |
| Coordinate estimation | Linear projection after centering by the training mean | Nonlinear mapping dependent on visible patches |
| Learning objective | Variance maximization corresponding to full-band squared error | Full-band MSE for A0/A1; masked MSE for MAE conditions |
| Axes | Orthogonal and ordered by variance | Neither orthogonality nor variance ordering is imposed |
| L2 normalization | Applied after fitting to scores used for clustering | Applied within the bottleneck during reconstruction learning |

Both use 16-dimensional unit vectors for clustering.
PCA fitting and score normalization follow the [experiment protocol](design/experiment_protocol.md).

<a id="snv-geometry"></a>

### 3.2 SNV and latent L2 normalization

The design choice is to **remove norm as a latent degree of freedom and concentrate information in direction,
motivated by the constant norm of SNV inputs**. For nonconstant spectra, SNV using the sample standard deviation gives

```math
\mathcal{M}_{\mathrm{SNV}}
=\{x\in\mathbb{R}^{C}:\mathbf{1}^{\mathsf T}x=0,\ \lVert x\rVert_2=\sqrt{C-1}\}.
```

At $`C=256`$, this is a sphere of radius $`\sqrt{255}`$ and intrinsic dimension 254 within the zero-mean hyperplane.
SNV inputs can be recovered from their directions by a common constant factor, but this does not imply that the norm after nonlinear compression is necessarily dispensable.
Latent normalization is an intentional constraint; it does not imply lossless compression or normalization.
No zero-mean constraint is imposed on the latent vectors.
([Preprocessing specification, §5.6](design/preprocessing.md))

SNV-induced curved structures in score plots have been reported previously; SNV geometry itself is not proposed as new.
([Fearn et al., 2009](https://doi.org/10.1016/j.chemolab.2008.11.006))

Comparisons based on cosine similarity are made in both input and latent spaces.
Normalization does not guarantee preservation of input angles or chemical information.
See [interpreting results](interpretation_notes.md#pca-snv-geometry-note) for the effects of centering, compression, and normalization in PCA.

## 4. Using SNV geometry to design augmentation

### 4.1 Augmentation assumptions: geometric constraints and actual spectral variation

The corruption used here is **designed around SNV's zero-mean and constant-norm constraints**; it is not derived
from instrument error distributions or physicochemical generative processes. The TGN angle and shift width are fixed in advance.

Preserving SNV constraints does not guarantee reproduction of measured errors or preservation of chemical states.
The effects of reconstruction from the specified corruption are evaluated through the defined comparisons.

<a id="spectral-augmentation"></a>

### 4.2 Proposed augmentations: Tangent Gaussian Noise and Fractional Shift

The proposal consists of defining SNV-preserving TGN and Fractional Shift and incorporating them into masked denoising.
Settings correspond to the [experiment protocol](design/experiment_protocol.md) and ChemoMAE v0.2.2's `SpectraAugmenter`.

In ideal arithmetic, let $`C=256`$, $`r=\sqrt{C-1}`$, $`x`$ be a zero-mean input with norm $`r`$, and $`u=x/r`$.
TGN removes the mean and radial directions from a Gaussian direction and rotates toward the resulting tangent direction.

```math
P=I-\frac{\mathbf{1}\mathbf{1}^{\mathsf T}}{C},\qquad
\epsilon\sim\mathcal{N}(0,I),\qquad
v=P\epsilon-u\bigl(u^{\mathsf T}P\epsilon\bigr),\qquad q=\frac{v}{\lVert v\rVert_2}.
```

```math
T_{\mathrm{TGN}}(x)=r\bigl(\cos\theta\,u+\sin\theta\,q\bigr),\qquad
\theta\sim U(0,\pi/36).
```

If $`\lVert v\rVert_2>0`$, then $`q`$ has zero mean and unit norm and is orthogonal to the input; the output also preserves the SNV constraints.
Its angle to the input is $`\theta`$. Gaussian refers to the distribution of candidate directions, not the angle or final residual.

Let $`S_\delta`$ denote linear interpolation with endpoint-value extension on the equally spaced wavelength grid. Fractional Shift is

```math
y=P S_\delta(x),\qquad
T_{\mathrm{FS}}(x)=r\frac{y}{\lVert y\rVert_2},\qquad \delta\sim U(-2,2).
```

If $`\lVert y\rVert_2>0`$, the output returns to the SNV constraint set, but the angular change for a given shift width depends on the input shape.
The implementation protects degenerate candidates with `eps=1e-8` and retains the original input. The equalities describe ideal arithmetic.

M11 applies each operation with probability 0.5 per pixel and randomizes their order per batch.
It does not always apply both operations in a fixed order. See [mathematics of SNV and spectral perturbations](mathematical_notes.md)
for derivations of angle control and constraint preservation.
There are no comparisons with additive noise or shifts without reprojection, so the advantage of geometry preservation itself is not independently tested.

<a id="augmentation-vicinity"></a>

### 4.3 Positioning through VRM

Vicinal Risk Minimization (VRM) places a vicinity distribution around each training example and minimizes expected loss
under their mixture. This study treats placing those distributions within the set preserving SNV's zero mean and constant norm
as the design of an inductive bias. Reconstruction targets are the observations before additional perturbations.

<a id="claims"></a>

## 5. Scope of evaluation and interpretation

| Claim or question | Evidence and limits |
| --- | --- |
| Proposed design | Incorporates SNV-preserving TGN and shift into masked denoising and compresses spectra into a single 16-dimensional unit latent vector |
| Use of fixed representations | In wood CV, the trained encoder and centers fitted on training data are applied to test data without downstream fine-tuning |
| Effect of additional corruption | Evaluates the MAE group's 2×2 comparisons and interactions using corrected LLA, LFR, and specified diagnostics. More stable representation of chemical states remains a hypothesis, not a guarantee from metrics or reconstruction loss |
| Stability under specified perturbations | Uses LFR to measure assignment retention under artificial perturbations of the same types and strengths as in training; does not extrapolate to all measurement errors or chemical information retention |
| Chemical correspondence | Relates NIR SNV spectra, second derivatives of pseudo-absorbance, and spatial distributions to published band assignments and specimen information; distinct from quantifying components or causally identifying degradation mechanisms |
| Architectural optimality | Does not establish optimal layer counts, latent dimension, linear decoder, or normalization, or superiority to SSL methods not compared |

Five-fold map evaluation with three replicates follows the [evaluation metrics](design/evaluation_metrics.md).
Read LLA and LFR alongside occupancy, inter-replicate ARI, and other diagnostics to distinguish degeneracy such as single-cluster collapse. Uncorrected LLA is supplementary.
Cosine silhouette is a geometric diagnostic within each representation, not a common measure of chemical validity.
See the [visualization and interpretation specification](design/visualization_and_interpretation.md) for the chemical interpretation procedure.

The following distinctions apply to comparisons and interpretation:

- **Training and use:** MAE has 8 visible patches during training and 16 during use. A fully visible mask is supplied explicitly; `eval()` alone does not switch visibility.
- **Extrapolation to missing bands:** Masking follows full-band SNV, so applicability to unmeasured bands for which the same SNV cannot be computed is not demonstrated.
- **Comparison scope:** M00 versus A0 differs in both input masking and the loss region. B1 versus neural networks also differs in nonlinearity, the stage of normalization, and other factors. These comparisons do not isolate a single element.

## 6. Connections to related work

[Related work](related_work.md) covers the learning principle of denoising autoencoders,
masking and clustering in Raman SMAE, and spatial mapping in LeafVAE. Their performance is not extrapolated to historic wood NIR data.

Denoising, spectral MAE, and SNV geometry have precedents. This study connects known SNV geometry
to perturbation and masked-denoising design, specimen-level evaluation on historic wood, and interpretation of spatial distributions.
Bibliographic details and source versions appear in the [references](related_work.md#references);
implementation details are provided by the code links here and the referenced ChemoMAE v0.2.2 source.
