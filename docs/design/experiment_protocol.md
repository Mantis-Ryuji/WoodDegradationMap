# Experimental Protocol

## 1. Scope

This document defines the research conditions and execution records for the main experiment and the supplementary mask-rate experiment.

Follow the [preprocessing specification](preprocessing.md) for preprocessing and the definition of raw SNV.

Use ChemoMAE v0.2.2. Record the configuration, environment, and source hashes for each run in its artifacts.
See the [runbook](../experiment_runbook.md) for artifact inspection procedures.

- [CV, seeds, and pixel sampling](#cv-design)
- [Conditions, representations, and training settings](#conditions), and [planned comparisons](#planned-comparisons)
- [Supplementary mask-rate experiment](#mask-rate-sweep) and [K policy](#cluster-counts)
- [Records to save during execution](#execution-records)

## 2. Research Objectives

Using historical-wood NIR-HSI as the empirical application, investigate whether adding corruption that preserves SNV constraints to masked reconstruction
can produce spectral representations useful for exploring and interpreting differences in material state.
See the [research objectives](../research_overview.md) for the overall research questions and the role of each type of evidence,
and [the positioning of ChemoMAE](../chemomae_positioning.md) for the rationale for adopting MAE and augmentation.

CV under this protocol addresses three questions:

1. How does the proposed Aug-MAE change spatial consistency and stability under specified perturbations relative to raw SNV, PCA, and standard MAE?
   Are there trade-offs between metrics?
2. How does standard MAE's masked reconstruction differ from an AE with no masking and reconstruction over all channels?
3. What effects do noise and wavelength-axis shifts have separately and in combination?

Treat mask rate as a sensitivity analysis of the proposed condition, not as the main optimization target.
CV metrics do not directly measure chemical representation quality or degradation-detection accuracy.
Correspondence with degradation is examined through [exploratory interpretation of NIR spectra](visualization_and_interpretation.md#nir-interpretation)
using maps, representative/difference spectra, specimen information, and literature assignments.
Do not change CV conditions, metrics, or splits to match the chemical interpretation.

<a id="cv-design"></a>

## 3. Experimental Unit and Cross-Validation

### 3.1 Specimen Unit

- The split and aggregation unit is the specimen identified by `KYOw...` in the filename.
- Always assign images, measurement conditions, and pixels belonging to the same `KYOw` to the same fold.
- Do not use pixels as independent split units.
- The metadata source is `data/metadata/古材メタデータ.csv`.

Split the 49 specimens processed by production preprocessing at the KYOw level.
It is unknown whether different KYOw specimens originate from the same original timber, so independence of provenance is not assumed.
Verify that no KYOw appears in both training and test sets. If higher-level dependencies are identified, decide how to handle them before creating splits.
Report the verified composition by tree species, provenance, and measurement conditions; do not claim generalization to unseen species, unseen origins, or different instruments.

### 3.2 Five-Fold CV

- Randomly divide specimens into 5 folds.
- Each fold has approximately an 8:2 training-to-test ratio, and each specimen appears in the test set exactly once.
- Do not stratify by tree species.
- Explicitly fix the split-generation seed and record it with the split manifest.
- Use identical folds for every comparison condition.

Within each fold, perform representation learning, PCA fitting, and Cosine-KMeans fitting using only training specimens.
Evaluate test specimens with the transformations, encoder, and cluster centers obtained from training held fixed.

<a id="seed-plan"></a>

### 3.3 Three Replicates and Random-Number Management

- Keep the same 5-fold split fixed and run 3 replicates each of the main comparison and supplementary mask-rate experiment. Do not recreate the splits.
- Use replicate IDs 1, 2, and 3, fixing the corresponding seed list before CV begins. Do not search for favorable seeds.
- Match replicate IDs across all conditions, and separately record random-number streams for model initialization, pixel order, masking,
  training augmentation, PCA, KMeans initialization, and evaluation perturbations.
- Separate random-number streams so that differences in the order of random-number consumption between conditions do not change shared comparison inputs or evaluation perturbations.
- Keep the training-pixel set and test-pixel set fixed across replicates. Replicates assess stochastic variation in training and clustering
  for fixed splits and pixel sets; they do not evaluate uncertainty from data splitting.
- B0 performs no representation learning. If B1's PCA settings are deterministic, reuse the fitted PCA for each fold; if they are stochastic,
  fit PCA for each replicate and record its seed. Fit KMeans in every replicate, including B0/B1.

The 6 training conditions in the main comparison require $`6\times5\times3=90`$ training runs, and the 2 additional mask-rate conditions require
$`2\times5\times3=30`$, for a total of 120 neural-network training runs in CV.
Do not repeat representation learning for each K; share representations across all K values within the same fold, condition, and replicate.
PCA, KMeans, and training for global visualization are not included in these 120 runs.

The base value for the CV seed plan is `20260905`.
Convert `["wood-degradation-map-seeds-v1", base_value, purpose, ...context]` to whitespace-free ASCII JSON,
then interpret the first 4 bytes of its SHA-256 digest as a big-endian unsigned 32-bit integer.
If seed values collide across purposes, stop rather than silently choosing a different seed.

| Purpose | Context included in the seed | Shared across |
| --- | --- | --- |
| Split | None | All conditions, K values, and replicates |
| Training-pixel sampling | Fold and KYOw | All conditions, K values, replicates, and epochs |
| Model construction, pixel order, masking, training augmentation, and PCA | Fold and training replicate for each purpose | All conditions and K values |
| KMeans | Fold, training replicate, and K | All conditions |
| Evaluation perturbations | KYOw, perturbation type, and perturbation replicate 1..5 | All conditions, K values, and training replicates |

Generate splits by randomly permuting sorted specimen IDs using NumPy PCG64, then dividing them into 5 approximately equal groups.
For pixel sampling, select HDF5 row indices within each specimen using uniform PCG64 `choice` without replacement, and sort them by row index before saving.
Save the software versions and the complete list of numerical seeds with the manifest.
On reload, use the saved sets rather than recreating splits or pixel sets because of differences in library versions or call order.
The seed plan is a contract for random-number management; it does not replace verification of subsequent training or shared evaluation perturbations.

<a id="pixel-sampling"></a>

### 3.4 Equal Training-Pixel Counts Across Specimens

- In each fold, select the same number $`q=8192`$ of valid pixels from each training specimen by uniform random sampling without replacement within that specimen.
- Fix $`q`$ and the sampling method across all folds.
- Within a fold, share the same set of pixel coordinates across all conditions, K values, and replicates.
- Use this same training-pixel set for ChemoMAE training, PCA fitting, and Cosine-KMeans fitting.
- Pixel order and training augmentation may be stochastic within each replicate, but do not let the original specimen pixel counts create unequal contributions.
- If any training specimen has fewer than $`q`$ valid pixels, revise $`q`$ before execution rather than silently compensating
  by sampling with replacement or changing the pixel count by specimen.
- Evaluate all valid pixels of each test specimen without sampling. Follow production preprocessing for background and quality-based exclusions.

All 49 production-input specimens support sampling $`q=8192`$ pixels without replacement.
Quality criteria follow the [preprocessing specification](preprocessing.md).
There are 39 or 40 training specimens, giving 319,488 or 327,680 pixels/fold, 312 or 320 batches/epoch,
and 249,600 or 256,000 scheduled updates/run over 800 epochs.
These pixel counts are multiples of 1024, so `drop_last=True` does not discard a partial batch at this $`q`$.

This $`q`$ was fixed in advance considering the computational cost of 800 epochs and the available pixel counts per specimen;
it was not optimized by evaluating accuracy or rare-region coverage. More pixels do not increase the number of independent specimens.
If the specimen set or preprocessed data changes, recheck sampling feasibility using `saved_pixel_count` in `sample_quality.parquet`.
Continue to use all valid pixels for test evaluation.

Check available counts using the existing quality tables.
The [runbook input checks](../experiment_runbook.md#input-preparation) provide the command and exit codes for `check_sampling_pixels.py`.
This checks pixel counts; it does not perform sampling, create splits, or evaluate spectral coverage.

Save the sampling seed, $`q`$, specimen IDs, and sampled coordinates corresponding to `pixel_row_col` in the manifest.
Do not calculate uncorrected LLA, which uses spatial neighborhoods, on sampled training pixels.
Using the same number from each specimen equalizes specimen weights; it does not balance the proportions of tree species or origins.

<a id="conditions"></a>

## 4. Main Comparison Conditions

Fix the mask rate at 50% for MAE conditions in the main comparison and 0% for A0/A1.
Do not apply augmentation to normal inference inputs. During training, set each enabled augmentation's application probability to 0.5.
For LFR evaluation, explicitly generate evaluation perturbations separately for the fixed models.

| ID | Representation | Mask rate | Gaussian noise | Shift | Role |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | raw SNV | - | - | - | Input-space baseline |
| B1 | PCA | - | - | - | Linear dimensionality-reduction baseline |
| A0 | AE | 0% | No | No | Unmasked reconstruction baseline |
| A1 | Denoising AE | 0% | Yes | Yes | TGN + shift added to unmasked, full-channel reconstruction |
| M00 | MAE | 50% | No | No | Standard MAE |
| M10 | Noise-MAE | 50% | Yes | No | Effect of Gaussian noise alone |
| M01 | Shift-MAE | 50% | No | Yes | Effect of shift alone |
| M11 | Aug-MAE | 50% | Yes | Yes | Proposed condition |

A0/A1 use the full-region reconstruction loss in ChemoMAE v0.2.2.
Combine `ChemoMAE(n_mask=0)` with `TrainerConfig(loss_region="all")`,
pass all patches to the encoder, and calculate reconstruction loss over the entire clean target spectrum.
A1 applies the same TGN + shift as M11 to the input, with identical strengths and an application probability of 0.5 for each operation.
The target is clean SNV before the added perturbations.
Comparisons involving A1 are ablations added after results for the existing 7 conditions were obtained, and are distinguished from prespecified confirmatory comparisons.
MAE conditions, including M00, use `loss_region="masked"`.
Here, “standard MAE” M00 means the unaugmented control using the common ChemoMAE architecture in this study,
not a reproduction of the image architecture in the original MAE paper.

### 4.1 Settings Held Fixed Across Conditions

- Encoder, decoder, latent dimension, and training budget for ChemoMAE-based conditions
- Training data, folds, preprocessing, and pixel masks
- Cosine-KMeans implementation, initialization policy, and common cluster counts
- Specimens and pixels used for evaluation
- Augmentation strengths and seed policy

The adopted augmentation values and the rationale for fixing them are given in Section 4.1.3.

<a id="representations"></a>

#### 4.1.1 Representation Dimension and L2 Normalization

Fix B1's PCA component count and `latent_dim` for A0, A1, and all MAE conditions at 16.
Use the same 16 dimensions for the supplementary mask-rate experiment and global training. B0 is the 256-dimensional input-space baseline.

| Condition | Representation passed to clustering and evaluation |
| --- | --- |
| B0 | 256-dimensional SNV → row-wise L2 normalization |
| B1 | 16-dimensional scores from PCA fitted on training data → row-wise L2 normalization |
| A0/A1/MAE conditions | CLS output from the fully visible encoder → 16 dimensions via `to_latent` → L2 normalization |

Let $`h`$ be the representation before normalization. For each pixel, use $`z=h/\lVert h\rVert_2`$.
For PCA, normalize **after dimensionality reduction**; normalizing the PCA input alone is not a substitute.
Restrict fitting of the PCA mean, basis, and other parameters to the training-pixel set in Section 3.4.
L2 normalization itself is a pixel-wise operation and estimates no statistics from the test set.
Use the same transformation order for clean, perturbed, training, and test inputs.
Use `sklearn.decomposition.PCA(n_components=16)`, with default settings except for the component count.
Following [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
center each wavelength using its training-pixel mean, with `whiten=False`, `svd_solver="auto"`, and `copy=True`.
Do not add autoscaling by wavelength-specific standard deviation after SNV.
Within-pixel centering in SNV and wavelength-wise centering over the PCA training-pixel set are separate operations.
Whitening equalizes component variances and is distinct from subsequent L2 normalization.

Use covariance-based PCA on mean-centered spectra as the conventional linear baseline.
Spectral variables often share the same units, making mean centering a common choice that avoids amplifying small variations or noise through autoscaling
([Eigenvector chemometrics guide](https://eigenvector.com/wp-content/uploads/2022/12/Chemometrics_Crash_Course_Part_2.pdf)).
Treat L2 normalization of PCA scores as postprocessing that connects them to cosine clustering in this study.

Keep the remaining arguments at their defaults: `tol=0.0`, `iterated_power="auto"`, `n_oversamples=10`,
`power_iteration_normalizer="auto"`, and `random_state=None`. Save the software version and the solver actually selected.
`auto` selects a computational method based on input shape and library version, not on CV metrics.
If the selected solver uses randomness, control NumPy's random state with the PCA seed according to Section 3.3 during fitting,
separately from other random-number streams. Do not interpret `random_state=None` as eliminating the need to manage randomness.

For ChemoMAE, explicitly set `latent_normalize=True` and pass normalized 16-dimensional latent vectors to the decoder during training as well.
The primary rationale is the design choice to concentrate information in direction without retaining a norm degree of freedom after compression,
given the constant norm of full-band SNV spectra. Consistency between the representations used during training and cosine clustering
is a resulting benefit. This is neither a mathematical necessity nor a guarantee of preserving input angles.
See [SNV and Latent L2 Normalization](../chemomae_positioning.md#snv-geometry) for the rationale.

Extract the latent vector obtained by passing the final CLS output through `to_latent`; do not substitute the 256-dimensional CLS output or a patch average.
For full-visibility extraction, use `Extractor`, or explicitly set `model.eval()` and an all-True `visible_mask`
before calling `model.encoder(x, visible_mask)`.
`model.eval()` alone does not disable random mask generation in `ChemoMAE.forward()`.
Use `augmenter=None` for normal extraction and generate shared LFR perturbations separately.

Cosine-KMeans also L2-normalizes its fit/predict inputs and updated centers (Section 4.1.3).
Record non-finite values, zero norms, and representations that cannot be normalized before or after normalization according to
[Evaluation Conventions, Section 2.4](evaluation_metrics.md).
Follow Section 4.1.4 for precision and epsilon handling.

Even with the same 16 dimensions and unit norm, PCA and ChemoMAE produce different inter-pixel angles and distributions.
Continue to treat silhouette as a diagnostic within each representation space.
Distinguish SNV inputs, reconstruction targets, and stored preprocessed data from this downstream unit normalization.
Unit normalization of B0 likewise does not change the cosine distances defined on the original SNV spectra.

<a id="model-architecture"></a>

#### 4.1.2 ChemoMAE Architecture and Initialization

Use the common encoder and single-layer linear decoder architecture, with the default initialization behavior of ChemoMAE v0.2.2.

| Setting | Adopted value |
| --- | --- |
| `seq_len` | 256, matching production preprocessing |
| `d_model` | 256 |
| `nhead` | 8, with 32 dimensions per head |
| `num_layers` | 8 |
| `dim_feedforward` | 1024 |
| `dropout` | 0.0 for all training conditions |
| `decoder_num_layers` | 1, a linear mapping |
| `latent_dim` / `latent_normalize` | 16 / True |
| `n_patches` | 16, with 16 consecutive channels per patch |

With this architecture, `n_mask` is 0 for A0 and 4, 8, and 12 for mask rates of 25%, 50%, and 75%, respectively.
The encoder receives 17 tokens including CLS under full visibility, and 9 tokens under the 50% mask rate used in the main comparison.

In the [ChemoMAE v0.2.2 model implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/models/chemo_mae.py),
`decoder_num_layers=1` means `Linear(16, 256)`, not one Transformer decoder layer.
Reconstruction is $`\hat{x}=Wz+b`$, recovering the full spectrum from a single 16-dimensional unit latent vector derived from CLS.

Parameter counts calculated from the implemented layers are 6,331,152 for the encoder, 4,352 for the decoder, and 6,335,504 in total; they have not been measured by executing the model.
The optimality of dropout=0.0 has not been tested.

The following initialization behavior was verified from the installed ChemoMAE v0.2.2 and PyTorch 2.13.0 sources and is adopted unchanged:

- ChemoMAE initializes the CLS token and learnable positional embeddings using `nn.init.trunc_normal_(..., std=0.02)`.
- The patch projection, `to_latent`, linear decoder, and similar layers use the default `nn.Linear` initialization.
  Weights and biases follow $`U(-1/\sqrt{d_{\mathrm{in}}},1/\sqrt{d_{\mathrm{in}}})`$ for input dimension $`d_{\mathrm{in}}`$.
- Combined attention QKV weights use PyTorch's Xavier uniform initialization; attention biases are 0, and LayerNorm uses weight=1 and bias=0.
- `nn.TransformerEncoder` creates 8 layers by deep-copying a single initialized encoder layer.
  Corresponding parameters start with identical values across layers but remain separate parameters, without weight sharing.
  ChemoMAE does not reinitialize individual layers after copying, and the calling code must not add such reinitialization.

Set the purpose-specific seed before constructing the model. Within the same fold and replicate, use the same construction seed across all training conditions.
Do not overwrite this initialization with that of the original MAE image model.

<a id="augmentation-clustering"></a>

#### 4.1.3 Reference Implementation, Augmentation, and Clustering

The reference version is ChemoMAE v0.2.2 pinned by the project, at commit
[`4ec7f6acecb82035c85001f5aee508910d40adac`](https://github.com/Mantis-Ryuji/ChemoMAE/commit/4ec7f6acecb82035c85001f5aee508910d40adac).
Use this version as the reference for the model, augmentation, extraction, optimizer, Trainer, Cosine-KMeans, and normalization helpers.

Random masking is also a denoising corruption. Apply training augmentation only to the input as additional corruption that preserves SNV constraints,
while retaining the observed SNV before the added perturbation as the target.
Learn across all bands through reconstruction loss on randomly hidden bands.
See [the positioning of ChemoMAE](../chemomae_positioning.md) for the rationale.

Use the [SpectraAugmenter implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/training/augmenter.py),
with noise angles $`U(0,5^\circ)`$ and shift and other operation settings fixed at their defaults.
The following settings are common to the main comparison, supplementary mask-rate experiment, and global training:

| Item | Adopted setting |
| --- | --- |
| Shift | $`\delta\sim U(-2,2)`$, a fractional shift in channel-index units; `shift_delta_range=(-2.0, 2.0)` (default) |
| Shift interpolation and endpoints | Linear interpolation; clamp reference indices to the endpoints and extend endpoint values outside the range. This is not a circular shift |
| Noise | Generate a zero-mean direction orthogonal to the input from Gaussian random numbers, then rotate toward that direction by the specified angle |
| Noise angle | $`\theta\sim U(0,5^\circ)`$; `noise_angle_deg_range=(0.0, 5.0)`. This differs from adding independent Gaussian noise to each channel |
| Application probability | During training, 0.5 for each enabled operation and 0 for disabled operations; sample application independently for each pixel |
| Operation order | `shuffle_order_per_batch=True`; randomize the order of the 2 operations for each batch (default) |
| Recentering and norm | `recenter_after_each_op=True` and `renorm_to_input_norm=True`; restore each pixel's mean to 0 and its norm to the pre-operation value after each operation (defaults) |
| Numerical stabilization | `eps=1e-8` (default) |

The adopted strengths were chosen subjectively and fixed through `sanity_check` inspection of SNV spectra;
they do not identify an instrument error distribution or a physicochemical generative process.
Do not optimize strengths using CV metrics or perform a strength ablation.
Use a 2×2 comparison of noise and shift presence/absence.
LFR uses the same strength distributions and operation settings, with the application probability of each evaluated operation set to 1.

Because the input axis is an equally spaced wavelength grid, express shift units as channel indices on the 256-point grid.
Use the saved wavelength spacing for conversion to nm; do not describe the shift as equally spaced on a wavenumber axis.
“Gaussian” in noise refers to generation of the direction, not to the distribution of rotation angles or the final additive residual.

Adopt the default algorithm and stopping settings from the
[Cosine-KMeans implementation](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/clustering/cosine_kmeans.py).
The values verified from the implementation are:

| Setting | Adopted value or behavior |
| --- | --- |
| Initialization | One k-means++-type initialization using cosine dissimilarity per fit; no additional restarts or best-run selection |
| Maximum iterations | `max_iter=500` |
| Tolerance | `tol=1e-4` |
| Stopping rule | Stop when the change in mean cosine dissimilarity from the previous iteration is relatively $`<10^{-4}`$ or absolutely $`<10^{-7}`$; the relative-change denominator is the absolute previous value $`+10^{-12}`$ |
| Empty clusters | Replace the center with a training pixel far from its current nearest center |
| Normalization and precision | Row-wise L2 normalization of inputs and updated centers; internal FP32, with `eps=1e-6` for normalization |
| Device | `device="cuda"` (default) |

Set `n_components` to each K in Section 6 and `random_state` to the purpose-specific seed in Section 3.3.
These settings implement the established K-dependence analysis and 3 replicates; do not overwrite all runs with the defaults K=8 and seed=42.
The library has no `n_init` argument. Distinguish the 3 replicates from the single initialization within each fit.

<a id="evaluation-precision"></a>

#### 4.1.4 Numerical Settings for Representation Extraction and Evaluation

Use FP32 for PCA fitting and transformation, full-visibility ChemoMAE extraction, evaluation perturbation generation,
L2 normalization, Cosine-KMeans, and continuous-valued metric calculations.
Keep inputs, model parameters, and extracted representations in FP32, and explicitly use `ExtractorConfig(amp=False)` for feature extraction.
Do not substitute FP16/BF16 computation followed by a final cast to FP32.
Also disable TF32 during evaluation. Retain FP16 AMP for training as specified in Section 4.2.

Explicitly set `dtype=torch.float32` for `silhouette_samples_cosine_gpu` and use the library defaults for numerical protection.
ChemoMAE latent normalization and silhouette use `eps=1e-12`, the Cosine-KMeans normalization helper uses `eps=1e-6`,
and SpectraAugmenter uses `eps=1e-8`. Use the same KMeans helper for explicit normalization after B0/PCA.
Each epsilon protects division; it is not a threshold that makes zero-norm vectors valid cosine inputs.
Follow the existing conventions in Section 2.4 of the evaluation document for non-finite values, zero norms, and related cases.
Save unit-norm errors as diagnostics, without introducing new pixel-exclusion thresholds based on the results.

Keep discrete labels, pixel counts, agreement counts, and contingency tables as integers.
Do not replace integer counting with floating-point operations because FP32 is specified.
Record seeds, actual computation settings, and library versions under Section 11.2.
Do not claim that small nondeterminism from parallel GPU reductions has been eliminated.

<a id="training-recipe"></a>

### 4.2 Training Settings

Adopt the pretraining recipe corresponding to Table 8 and the ablations in Table 1 of the
[MAE paper](https://arxiv.org/pdf/2111.06377).
Use commit [`efb2a8062c206524e35e47d04501ed4f544c0ae8`](https://github.com/facebookresearch/mae/commit/efb2a8062c206524e35e47d04501ed4f544c0ae8)
of the official PyTorch implementation as the reference. Use batch size 1024 without gradient accumulation.

| Setting | Adopted value |
| --- | --- |
| Optimizer | AdamW |
| Betas | $`(\beta_1,\beta_2)=(0.9,0.95)`$ |
| Epsilon / AMSGrad | $`10^{-8}`$ / disabled |
| Weight decay | 0.05; 0 for biases and normalization-layer parameters |
| Batch size | 1024 spectra / GPU |
| GPU count | Assumes a single GPU (`world_size=1`) |
| Gradient accumulation | `accum_iter=1`; update at every batch without accumulating gradients |
| Effective batch size | 1024 |
| Base learning rate | $`1.5\times10^{-4}`$, the reference value for an effective batch of 256 |
| Peak learning rate | $`6.0\times10^{-4}`$, from the linear scaling below |
| Training epochs | 800 |
| Warmup | Linear increase from 0 to the peak learning rate over the first 40 epochs |
| Schedule after warmup | Half-cycle cosine decay over the remaining 760 epochs, without restarts |
| Minimum learning rate | 0 |
| Learning-rate updates | Before each batch, including progress within the epoch |
| Gradient clipping | Not used |
| Drop path | 0 |
| Training precision | CUDA AMP with FP16 autocast and dynamic GradScaler; FP32 model parameters |
| Training DataLoader | Shuffle each epoch, `drop_last=True` |
| EMA | Not used (`use_ema=False`) |
| Checkpoint selection | Raw weights at completion of 800 epochs (`last_model.pt`); no EMA weights |

Apply these settings to A0, A1, M00, M10, M01, M11, and M11 with mask rates of 25% and 75%.
For A0, A1, M00, and M11 used in global visualization, also use the same condition-specific masking, loss region, and augmentation settings as CV,
and the same 800-epoch recipe.
Prioritize the [explicit settings in the official PRETRAIN.md](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/PRETRAIN.md).
The argument defaults in `main_pretrain.py` include 400 epochs and base lr $`10^{-3}`$;
do not copy these into the study's training settings.

For effective batch size $`B_{\mathrm{eff}}`$, set the learning rate as

```math
B_{\mathrm{eff}}=1024\times1\times1=1024,\qquad
\eta_{\max}=1.5\times10^{-4}\frac{B_{\mathrm{eff}}}{256}=6.0\times10^{-4}
```

This does not reproduce the original paper's effective batch size of 4096 through gradient accumulation.
If moving to multiple GPUs, respecify GPU count, effective batch size, and learning rate together.
The implementation must not silently change batch size or accumulation to address insufficient memory.

Let epoch progress be $`u=e+j/S_f`$, where $`e`$ is the zero-based epoch and $`j`$ the zero-based batch index.
Following the [official scheduler](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py), use

```math
\eta(u)=
\begin{cases}
\eta_{\max}\,u/40, & 0\le u<40 \\
\dfrac{\eta_{\max}}{2}\left[1+\cos\left(\pi\dfrac{u-40}{800-40}\right)\right], & 40\le u\le800
\end{cases}
```

Do not replace this with a scheduler that changes in epoch-wise steps.

For $`M_f`$ training specimens in fold $`f`$, the sampled-pixel count, batches per epoch, and scheduled update count are

```math
N_f=M_fq,\qquad S_f=\left\lfloor\frac{N_f}{1024}\right\rfloor,\qquad U_f=800S_f
```

Verify $`S_f\ge1`$ before execution. Use the same $`S_f`$ and $`U_f`$ for all conditions and replicates within a fold.
Update counts differ between folds with different training-specimen counts and in global training, so save the actual $`N_f`$, $`S_f`$, and $`U_f`$.
If AMP skips updates, also record the skipped-update count and actual update count.

With `drop_last=True`, omit the partial batch after shuffling from that epoch's neural-network training.
Within the same fold, replicate, and epoch, share the order and partial-batch omission across all training conditions.
Equal specimen counts are guaranteed for the sampled set, not necessarily for the pixels contributing in every epoch after the partial batch is dropped.
Fit PCA and KMeans on the entire sampled set without dropping the remainder. Do not drop a partial batch during test evaluation either.

Optimizer construction and weight-decay grouping follow
[main_pretrain.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py);
AMP and per-batch scheduler calls follow
[engine_pretrain.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py)
and [util/misc.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py).
Epsilon and AMSGrad explicitly use the [PyTorch AdamW defaults](https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html).

Only the training recipe in this section is adopted from MAE.
Use this study's specifications through Section 4.1 for the model, initialization, SNV, masking, augmentation, and loss.
Do not introduce ImageNet image augmentation, RGB normalization, or within-patch target normalization (`norm_pix_loss`).

When using the ChemoMAE v0.2.2
[Trainer](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/training/trainer.py),
explicitly set `amp_dtype="fp16"`, `grad_clip=None`, and `use_ema=False`.
The defaults of BF16, clipping at 1.0, and enabled EMA do not match this section.
Start independent runs with `resume_from=None`; when resuming, explicitly identify a checkpoint from the same run.
Do not automatically resume from another condition or replicate.

The [optimizer and scheduler helpers](https://github.com/Mantis-Ryuji/ChemoMAE/blob/4ec7f6acecb82035c85001f5aee508910d40adac/src/chemomae/training/optim.py)
also differ from this specification. `build_optimizer` exempts the CLS token and learnable positional embeddings from weight decay in addition to biases and LayerNorm.
The existing `build_scheduler` uses `step + 1` during warmup, and Trainer advances the scheduler after each batch update,
so it does not exactly match the learning-rate sequence starting at 0 in this section.
This repository implements parameter grouping and learning-rate updates before each batch in the
[calling-side optimizer construction](../../src/wood_degradation_map/experiments/neural.py)
and [ExperimentTrainer](../../src/wood_degradation_map/experiments/training.py).
Distinguish this specification from using the library's default helpers unchanged.

<a id="planned-comparisons"></a>

### 4.3 Planned Comparisons

| Comparison | Question examined |
| --- | --- |
| M11 vs B0 | Direct comparison of the proposed Aug-MAE with the input-space baseline |
| M11 vs B1 | Direct comparison of the proposed Aug-MAE with the linear dimensionality-reduction baseline |
| M11 vs M00 | Effect of combining 2 augmentations with standard MAE |
| M00 vs B0 | Comparison of standard MAE with the input-space baseline |
| M00 vs B1 | Comparison of standard MAE with PCA |
| M00 vs A0 | Effect of moving from unmasked, full-region reconstruction to masked reconstruction |
| M10 vs M00 | Effect of Gaussian noise alone |
| M01 vs M00 | Effect of shift alone |
| M11 vs M10 | Additional effect of shift |
| M11 vs M01 | Additional effect of Gaussian noise |
| A1 vs A0 | Effect of adding TGN + shift to an unmasked, full-region reconstruction AE (additional ablation) |
| M11 vs A1 | Comparison of denoising AE and Aug-MAE with matched TGN + shift (additional ablation) |

Treat M11 vs B0, M11 vs B1, and M11 vs M00 as the primary planned comparisons, and the remaining comparisons as ablations explaining individual components.
M00 vs B1 matches output dimension at 16 and norm at 1, but differs in learning objective and transformation, so do not interpret it as the effect of nonlinearity alone.
M00 vs A0 changes both masking and the loss region, so do not interpret it as the effect of mask rate alone.
A1 vs A0 compares added TGN + shift with 0% masking and full-region loss held fixed.
M11 vs A1 matches augmentation but differs in both masking and loss region.
Do not interpret it as the effect of mask rate alone or as a general advantage specific to combining augmentation with masked modeling.
Follow Section 4 for the status of these 2 comparisons. Report them alongside the 3 primary comparisons and the 2×2 factorial design below.

Treat M00, M10, M01, and M11 as a 2×2 factorial design for the presence/absence of Gaussian noise and shift.
The current report presents the following interaction contrast in CSV tables for corrected LLA-3/5/9 and LFR(TGN+FS), LFR(TGN), and LFR(FS),
using the same specimen, K, and replicate:

```math
(M11-M10)-(M01-M00)
```

LLA and LFR have opposite directions of improvement.
Show the contrast on the original metric scale and explicitly state the meaning of its sign.
The interaction describes these metrics under these strength settings; do not assert a general synergistic effect of augmentation.
The current reporting metrics and correspondence between the original snapshot and reporting CSVs are documented in
[Evaluation Metrics, Section 8.2](evaluation_metrics.md#reporting).

## 5. Supplementary Experiments

This section defines the fixed conditions and comparison scope of the mask-rate sweep.
See [Results and Figures](../results.md#mask-rate-handoff) for the dedicated OOF snapshot, figures, and tables.

<a id="mask-rate-sweep"></a>

### 5.1 Supplementary Mask-Rate Experiment

Compare mask rates of 25%, 50%, and 75% for the proposed M11 condition.

| ID | Mask rate | Gaussian noise | Shift |
| --- | ---: | ---: | ---: |
| M11-25 | 25% | Yes | Yes |
| M11 | 50% | Yes | Yes |
| M11-75 | 75% | Yes | Yes |

Use `M11`, the same execution ID as in the main comparison, for 50%; do not duplicate training under a different ID.
Use `M11(-50)` as its display name in mask-rate comparison figures.
Add M11-25 and M11-75, each with 5 folds × 3 replicates, for 30 additional runs.
Treat this experiment as a sensitivity analysis of the proposed condition's dependence on mask rate, not as selection of an “optimal mask rate.”
Report the results as a supplementary experiment, without replacing the main M11 condition after inspecting them.
Use the same folds, common K set, training budget, seed list for 3 replicates, and evaluation metrics as the main comparison.

Do not sweep mask rates for the Gaussian-noise-only or shift-only conditions.
Aggregate the 3 conditions into `mask_rate_oof_v1` separately from the main conditions, and generate figures and tables with the dedicated pipeline.
See the [runbook](../experiment_runbook.md#mask-rate-sweep) for execution steps.
Treat the appendix map comparison of M11(-50) and M11-75 over all 49 specimens as a
[supplementary global-fit comparison](visualization_and_interpretation.md#global-mask-rate), separate from CV.

<a id="cluster-counts"></a>

## 6. Common Cluster Counts and Dependence on K

### 6.1 Prespecification Independent of Results

Treat cluster count as the granularity for inspecting a continuous latent space, or a resolution parameter.
An elbow-based choice using all specimens would introduce the held-out distribution into the design;
even a choice based only on training folds would complicate paired comparisons if K differed. Therefore, use common K values.

Before inspecting results, fix the common K set used across all folds, conditions, and replicates at the following 7 values:

```math
\mathcal{K}=\{2,4,6,8,10,12,14\}
```

Do not calibrate K using an elbow from raw SNV across all specimens.
Do not change this set based on test-specimen spectral distributions, CV metrics, label maps, or degradation interpretations.
Do not select K separately from each condition's latent space, which would mix representation differences with differences in cluster count.

This setting examines coarse to fine partitions from 2 to 14 in increments of 2.
It is not assumed to represent the true number of tissue or surface states or a range capable of separating every anomaly.

### 6.2 Representative Display and Reporting All K Values

Fix the representative display value for main tables and global visualizations at the intermediate resolution of 8, the center of the common set.

```math
K_0=8
```

To provide a phenomenological guide for this intermediate resolution, coarsen the main variations potentially observed within one specimen
into a minimal 2 levels each: growth-ring tissue represented by earlywood/latewood, relatively sound/degraded condition,
and potentially environment-related surface condition. This gives the following 8 conceptual combinations:

```math
\underbrace{2}_{\text{growth-ring tissue}}
\times
\underbrace{2}_{\text{relative degradation state}}
\times
\underbrace{2}_{\text{environment-related surface state}}
=8
```

Because wood SNV spectra share major absorption structures, do not assume separate discrete states for each tree species in advance,
and do not add species to the number of combination levels.
Species differences may modify how each factor appears and contribute to cluster formation.
The $`2\times2\times2`$ count is a supplementary heuristic.
The actual factors are continuous and interact, so neither prior assignment to clusters nor a one-to-one correspondence is assumed.
The primary rationale remains the display convention of using the center of the common K set.
Do not interpret this as the true cluster count, optimal K, or 8 chemical states.

- Report primary metrics, diagnostics, and planned comparisons for $`K_0`$ in tables.
- Include curves across all of $`\mathcal{K}`$ and differences for planned comparisons; do not base conclusions only on the representative K.
- If the direction of improvement changes with K, state where it changes.
- Do not use the average or maximum over K as an overall method score.

Treat dependence on K as a supplementary analysis using the same representations obtained in the main experiment.
Perform clustering and evaluation at each K without additional representation learning.
A consistent trend within the chosen set is not a guarantee beyond it.

## 7. Processing Within Each Fold

Prepare the fixed training/test specimens and pixel sets for each fold. For each condition and replicate:

1. Fit the required representation transformation or train ChemoMAE using only training specimens.
2. Extract representations from training/test wood-region pixels using the dimensions, full-visibility extraction, and L2 normalization in Section 4.1.1.
3. For each K in the common set, fit Cosine-KMeans using only training representations.
4. Assign labels to clean test representations and perturbed representations of the same pixels using the fixed training centroids.
5. Calculate the primary metrics and supplementary diagnostics for the test specimens.
6. Save specimen-, K-, and replicate-level values, fold results, and results across all out-of-fold specimens.

After all replicates are complete, calculate ARI between their label partitions for the same fold, condition, and K.
Do not assume that cluster numbers have shared meanings across K values, replicates, or conditions.

Use the final checkpoint at completion of 800 epochs, as specified in Section 4.2.
Do not choose a stopping point based on test reconstruction loss, test metrics, or visualizations.
Introducing early stopping or another checkpoint-selection rule requires a design change that includes specimen-level validation
within the outer training set and an explicit selection rule.

## 8. Condition-Selection Policy

- Do not select a single “best condition” from the CV results.
- Use corrected LLA and LFR(TGN+FS) as the principal displays of primary evaluation.
  The metric priority order is LLA, LFR(TGN+FS), ARI, Cosine-Silhouette, and Cluster Occupancy.
- Report ARI as reproducibility across replicates, Cosine-Silhouette as geometry within each representation space, and occupancy as a diagnostic of degeneracy.
  Also retain uncorrected LLA and noise/shift LFR in the saved evaluations. Do not use any metric as a substitute for degradation accuracy.
- Do not create an overall score by arbitrarily combining metrics.
- For each planned comparison, report the direction of each metric and its trend with K.
- Do not select the main condition after inspecting interpretation or visualization results.

Follow the [visualization design](visualization_and_interpretation.md) for the targets and fitting conditions used in global interpretation,
rather than selecting them by CV ranking.

## 9. Searches Not Performed

- Encoder/decoder width or depth sweeps
- Gaussian-noise strength sweeps using CV metrics
- Shift-strength sweeps using CV metrics
- Mask-rate sweeps for the Gaussian-noise-only condition
- Mask-rate sweeps for the shift-only condition
- Condition-specific K selection
- Searches for the best condition based on CV results

The 3 replicates assess stochastic variation; they are not seed optimization or selection of favorable runs.

## 10. Execution Order

1. Generate and inspect the fixed 200 Hz production-preprocessed dataset and diagnostic figures.
2. Perform the checks and recording in Section 11.2. Create the execution configuration including the fixed $`q`$, common $`\mathcal{K}`$, and representative display $`K_0`$.
3. Create the common specimen-level 5-fold split and training-pixel manifests.
4. Evaluate all 8 main comparison conditions, including A1, over 3 replicates.
5. Create and check the main 8-condition OOF snapshot, then aggregate paired condition differences, dependence on K, 2×2 interactions, and stability across replicates. Generate and verify the main-condition reporting figures and tables.
6. Fit or train B0, B1, A0, A1, M00, and M11 on all specimens, then fit Cosine-KMeans once for each using the same representations, sampled pixels, and prespecified $`K_0`$.
7. Directly align labels to the common reference of M00 Cosine-KMeans using cosine similarity of observed-SNV representative spectra and Hungarian matching.
   Construct those representative spectra by averaging within-specimen cluster means with equal weights across specimens.
   Save maps, spectra, and PCA (PC1/PC2) for all 6 conditions, together with the SNV similarity matrix and inspection IoU/contingency/overlap.
8. Follow the separate [Bacteria-ID protocol](bacteria_id_experiment_plan.md) and [HPO protocol](bacteria_id_hpo_protocol.md) for the supplementary Raman experiment.
9. Evaluate the supplementary M11 mask-rate experiment over the same 3 replicates and create a separate OOF snapshot and mask-rate-dependence figures and tables.
10. Add one global training run for M11-75, and prepare 3-seed diagnostics using fixed representations at K8 and a 2×7 comparison figure from the first fits of M11(-50) and M11-75 for the appendix.

Follow the [reproducibility evaluation](spatial_map_reproducibility_plan.md) for global-fit map reproducibility,
and the [NIR analysis](visualization_and_interpretation.md#nir-interpretation) for chemical interpretation.
Keep the mask-rate CV and global-fit comparisons distinct from the main conditions, following the
[evaluation reporting conventions](evaluation_metrics.md#reporting) for principal metrics and display layouts.

## 11. Execution Records

### 11.1 Configuration References

Refer to Sections 3–5 for the main-condition and supplementary mask-rate settings.
Check them against the records of the saved runs.

<a id="execution-records"></a>

### 11.2 Data Checks and Execution Records

When executing the finalized design, check and record the following:

| Item | Information to verify, fix, and save before execution |
| --- | --- |
| Specimen set and splits | Correspondence between the production manifest and metadata, included specimen count, and the status of higher-level sampling relationships; resolve newly identified dependencies before splitting |
| Randomness | Purpose-specific seed lists for splits, pixel sampling, 3 training replicates, KMeans, evaluation perturbations, and other uses; fix numerical values in advance during implementation and do not reselect them after inspecting results |
| Manifests | Fold assignments, sampled coordinates, mapping between replicate IDs and seeds, and the pixel set for global training |
| Execution environment | GPU model, actual library versions, training AMP, FP32 extraction/evaluation, disabled evaluation TF32, actual seed and determinism settings, actual PCA solver, library-default epsilons, and unit-norm errors |
| Training process | Consistency with the recipe in Section 4.2, checkpoints per run, scheduled and actual update counts, AMP skips, and runtime |

Use schema 2 for CV training, clustering, and evaluation run records.
Save current loading requirements in `contract`, and the actual execution configuration, runtime, and source hashes in `execution`.
Validate loading requirements by exact matching, distinguishing them from execution provenance.
When resuming an interrupted run, require the entire run record to match.

Follow the [visualization design](visualization_and_interpretation.md#representative-samples) for displayed examples and the
[evaluation conventions](evaluation_metrics.md#occupancy) for occupancy.
Use the [runbook](../experiment_runbook.md) for procedures and artifact checks.
At production startup, generate new manifests in a dedicated directory and verify agreement with preflight for the configuration, seeds, splits, and sampled pixels.

Follow Section 4.2 for update counts; do not change the 800 epochs to equalize update counts across folds.
Because computation and runtime vary with mask rate, do not describe the comparison as having equal computational cost; report runtimes alongside results.
Distinguish design changes made in advance from adjustments made after inspecting results, and do not mix runs from different conditions.
