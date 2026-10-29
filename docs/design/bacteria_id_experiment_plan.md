# Raman Supplementary Experiments with Bacteria-ID

> Classification training for Table 5 follows the [5-fold CV protocol](bacteria_id_hpo_protocol.md).
> This document defines pretraining, Table 4, and the cited values.

The reference version is the **published SMAE PDF**. Settings established from public materials are distinguished
from supplementary choices made in this study. See [interpretation of the Raman supplementary experiments](../bacteria_id_interpretation_plan.md)
for the role of the results and limitations of the comparisons.

## 1. Purpose and Scope of Evaluation

Raman spectra from Bacteria-ID are used as a supplementary experiment to evaluate self-supervised representation
learning with ChemoMAE on publicly available spectra beyond historic wood NIR.
**ChemoMAE(M00) and ChemoMAE(M11) are the only two methods newly evaluated.**

The main comparison tables add both conditions to the following two SMAE tables.

- **Table 4 equivalent**: Clustering ACC, NMI, and AMI for Bacteria-4 and Bacteria-6.
- **Table 5 equivalent**: Accuracy for 30-class Bacteria-ID classification. Both conditions are added to the `w/ pretraining` column.

**Table 4 uses 80% training / 20% test data with no validation set.** For Table 5, classification settings selected
by 5-fold CV on M11 are applied to both M00 and M11. Configuration selection and full-subset training are described in Section 7.

Values reported by SMAE are quoted for other methods; SMAE, ResNet, RamanCluster, and others are not rerun.
The scope excludes ChemoMAE training from scratch, additional baselines, sweeps of latent dimensionality or augmentation strength,
additional datasets, Table 3, and additional visualizations.
M00 also uses masked-reconstruction pretraining;
it is not a `w/o pretraining` condition.

### Reference Version and Comparison Conditions

The reference paper is Ren, Zhou, and Li, *A self-supervised learning method for Raman spectroscopy based on masked autoencoders*,
**Expert Systems With Applications 292 (2025), 128576**,
[DOI: 10.1016/j.eswa.2025.128576](https://doi.org/10.1016/j.eswa.2025.128576).
Tables 4 and 5 are on p.8 of the published version; replicate settings are described in §3.3.

Additional references are S-5 and Fig. S3 of the SMAE supplement, and §3.1, §3.3, and supplementary Fig. S2 of
Sun, Wang, and Jiang, *RamanCluster: A deep clustering-based framework for unsupervised Raman spectral identification of pathogenic bacteria*,
**Talanta 275 (2024), 126076**, [DOI: 10.1016/j.talanta.2024.126076](https://doi.org/10.1016/j.talanta.2024.126076).

SMAE is the reference for data definitions, split proportions, metrics, replicates and reporting format,
classification head, and downstream training procedures that can be established from public materials.
For unspecified settings and discrepancies between sources, the **supplementary choices in Section 8** apply.
These choices are distinguished from SMAE's actual settings.
The published SMAE text, supplementary materials, and official code are referenced; for the discrepancy concerning Bacteria-6,
the composition in the SMAE supplementary figure takes precedence.

The inspected version of the official code is
[`pengjuRen99/SpectraMAENet@7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3`](https://github.com/pengjuRen99/SpectraMAENet/tree/7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3)
(commit dated 2025-06-23). This commit has not been confirmed as the version that generated the published results.

Intentional methodological differences are the specified ChemoMAE architecture, SNV, the WoodDegradationMap pretraining recipe,
TGN/FS, and Cosine-KMeans for Table 4. Known evaluation conditions are aligned, and these methodological differences
and the supplementary choices in Section 8 are stated in table footnotes.
The comparison is not described as a rerun of SMAE and other methods under the same environment and computational budget.

## 2. Model and Preprocessing

The settings are based on **the adopted WoodDegradationMap model and pretraining recipe**.
They are not replaced by ChemoMAE library argument defaults or SMAE pretraining settings.
The reference version is ChemoMAE v0.2.2, commit
[`4ec7f6acecb82035c85001f5aee508910d40adac`](https://github.com/Mantis-Ryuji/ChemoMAE/commit/4ec7f6acecb82035c85001f5aee508910d40adac).

| Item | Bacteria-ID setting |
| --- | --- |
| Input | 1000 channels per spectrum, `seq_len=1000` |
| Preprocessing | Per-spectrum SNV using the sample standard deviation (`ddof=1`) |
| Patches | `n_patches=20`; each patch contains 50 consecutive channels |
| Encoder width | `d_model=256` |
| Attention / layers | `nhead=8` / `num_layers=8` |
| FFN / dropout | `dim_feedforward=1024` / `dropout=0.0` |
| Latent representation | 128 dimensions from CLS through `to_latent`, `latent_dim=128`, `latent_normalize=True` |
| Decoder | `decoder_num_layers=1`, `Linear(128, 1000)` |
| Pretraining mask | 50%; 10 of the 20 patches are hidden |
| Pretraining loss | MSE on masked channels only, with no within-patch target normalization |
| Initialization | The same ChemoMAE v0.2.2 initialization as WoodDegradationMap |
| Normal feature extraction / classification | All 20 patches visible; no random mask |

Model settings other than input length, patch configuration, and latent dimensionality follow the
[historic wood architecture](experiment_protocol.md#model-architecture).
The 128-dimensional latent representation is a fixed condition of this supplementary experiment;
it does not imply an identical bottleneck to SMAE or other methods, or an optimal dimensionality.
Encoder width 256 is distinguished from the single latent vector's dimensionality of 128.
The 128-dimensional unit latent representation is used for pretraining reconstruction and Table 4 clustering.
The Table 5 classification head is attached to the CLS output before projection (Section 7).

The distributed spectra are retained as the original inputs, and SNV is applied.
The wood-specific 256-point interpolation, spatial masks, and pixel sampling are not reused.
The wavenumber axis, channel order, and shape are checked; no additional smoothing or 0–1 clipping is introduced.
Negative values after SNV are valid. If nonfinite values, zero standard deviations, or similar issues are found,
the affected spectra and counts are recorded and their handling is determined, rather than silently excluding them
and changing the evaluation set relative to SMAE.

## 3. TGN and Fractional Shift

**Strengths also use the same values as WoodDegradationMap.** They are not reduced for Raman or changed between training stages.
TGN denotes Tangent Gaussian Noise, and FS denotes Fractional Shift.

| Item | Adopted setting |
| --- | --- |
| TGN angle | Uniform over 0–5°, `noise_angle_deg_range=(0.0, 5.0)` |
| FS displacement | Uniform over −2–2 channels, `shift_delta_range=(-2.0, 2.0)` |
| Application probability | Independently 0.5 per spectrum for each enabled operation |
| Operation order | `shuffle_order_per_batch=True` |
| Recentering / norm | `recenter_after_each_op=True`, `renorm_to_input_norm=True` |
| FS interpolation / endpoints | Linear interpolation with endpoint-value extension; no circular shift |
| Numerical stabilization | `eps=1e-8` |

The processing order is **distributed input → SNV → TGN/FS**. Zero mean and constant norm are restored after each operation.
The procedure does not apply SMAE augmentation to the original input before SNV.
FS is measured in channel indices of the 1000-point input; it does not represent the same physical wavelength or wavenumber
displacement as in the wood experiment. See the [augmentation specification](experiment_protocol.md#augmentation-clustering)
for definitions and implementation details.

| Stage | ChemoMAE(M00) | ChemoMAE(M11) |
| --- | --- | --- |
| Pretraining | No additional augmentation; 50% mask | TGN and FS each applied with probability 0.5; 50% mask |
| Fine-tuning training | Neither TGN nor FS under the HPO-selected setting; all patches visible | Neither TGN nor FS under the HPO-selected setting; all patches visible |
| Table 4 test and feature extraction for clustering; Table 5 validation and test | No augmentation; all patches visible | No augmentation; all patches visible |

M00/M11 denote pretraining conditions. Fine-tuning augmentation is selected through [HPO](bacteria_id_hpo_protocol.md)
and applied to both conditions, so their difference is compared in terms of the presence or absence of additional pretraining augmentation.
The pretraining target for both conditions is the **observed SNV spectrum before additional perturbation**;
target intensities and channel positions are not shifted.
Here, clean does not mean a noise-free ground truth.

## 4. Pretraining Settings

The adopted settings follow the [WoodDegradationMap training recipe](experiment_protocol.md#training-recipe)
and [experiments/config.py](../../src/wood_degradation_map/experiments/config.py).

| Item | Adopted setting |
| --- | --- |
| Epochs / batch size | 800 / 1024 |
| GPU / gradient accumulation | Single GPU / `accum_iter=1` |
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8, no AMSGrad |
| Weight decay | 0.05; 0 for biases and normalization-layer parameters |
| Base / peak learning rate | 1.5e-4 / 6e-4 (effective batch size 1024) |
| Schedule | 40 epochs of linear warmup, then 760 epochs of cosine decay; minimum learning rate 0 |
| Learning-rate update | Before each batch, accounting for progress within the epoch |
| Precision | FP32 parameters, FP16 autocast, GradScaler |
| Gradient clipping / EMA | Neither is used |
| Training DataLoader | Shuffled, `drop_last=True` |
| Termination / checkpoint | No early stopping; raw weights after completing 800 epochs (`last_model.pt`) |

Purpose-specific seeds are recorded so that input sets, batch order, masks, and other factors can be paired between M00 and M11.
**Table 5 uses five distinct seeds per condition, following the published version. Table 4 also uses five seeds as a supplementary choice.**
Run seeds are `[0, 1, 2, 3, 4]`. Each seed independently repeats the process from pretraining.
For Table 5, the pretraining checkpoint for the same seed is passed to one fine-tuning run;
this is not five fine-tuning replicates from a single checkpoint.
Bacteria-4/6 in Table 4 and Table 5 use separate training corpora. Purpose-specific seeds and run counts are defined in Section 8.
The wood experiment's five folds × three replicates are not automatically carried over.
Actual training counts, batches per epoch, planned and actual update counts, and AMP skip counts are saved.
The 800 epochs are not described as the same computational budget as SMAE.

## 5. Data and Splits

Data sources are the [original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960993/) and the
[official Bacteria-ID repository](https://github.com/csho33/bacteria-ID).
The 30 classes are bacterial and yeast **isolate classes**, not 30 distinct species.

### Distributed Subsets and Splits for Table 5

| Distributed subset | Count | Role in Table 5 |
| --- | ---: | --- |
| reference | 60,000 (30 classes × 2,000) | Pretraining without labels in the loss |
| finetune | 3,000 (30 classes × 100) | Training and validation in 5-fold CV, followed by full-subset training with the selected configuration |
| test | 3,000 (30 classes × 100) | Final Accuracy evaluation |

For Table 5, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` is applied to the finetune subset,
and M11 seed 0 is used for HPO over all five splits. Each fold contains 2,400 training and 600 validation spectra;
after configuration selection, M00/M11 are trained on all 3,000 finetune spectra.
The published version's “five different seeds” is not reinterpreted as evaluation over five folds;
the 5-fold HPO is specific to this study.
Input-file row order and split indices are saved. The actual splits used in the published SMAE experiment have not been established from public materials.

Test data are not used for pretraining, HPO, configuration selection, or final training.
SNV uses only within-spectrum statistics. Relationships between the same biological units or measurement groups
across distributed subsets are recorded to the extent established by the sources;
this evaluation is not equated with specimen-level CV for wood or evaluation on unseen patients.

<a id="table4-split"></a>

### Counts, Split Proportions, and Class Composition for Table 4

SMAE §3.2 and supplementary S-5 state that Bacteria-4/6/8/10 follow the RamanCluster splitting procedure.
RamanCluster §3.1 (pp.5–6) defines 2,000 spectra for each of 10 species, datasets with 4, 6, 8, and 10 classes,
and **training / validation / test = 60% / 20% / 20%**.
This study uses **80% training / 20% test**. After obtaining results from the saved 60/20/20 split,
the training and validation sets were combined, and 80/20 was adopted for the main results.
The original test row indices and array order were preserved.
The saved indices below are authoritative and must not be replaced with a new random split.
The following table gives the final counts used in this study; it does not claim that literature methods used the same training proportion.

| Dataset | Total before splitting | Training | Test |
| --- | ---: | ---: | ---: |
| Bacteria-4 | 8,000 | 6,400 | 1,600 |
| Bacteria-6 | 12,000 | 9,600 | 2,400 |

This 80/20 split is specific to Table 4 and is distinct from the roles of the distributed reference/finetune/test subsets for Table 5.
The target isolates are extracted from the **distributed reference subset**, consistent with 2,000 spectra per class.
The files are `X_reference.npy` and `y_reference.npy`, listed in the
[official distribution guide](https://github.com/csho33/bacteria-ID/blob/3c00a712a6dbad9aefa19ac878a1e6db20590ca9/data/data.md).
The choice of source files is a supplementary choice made in this study.

The final Table 4 split is fixed from the saved indices as follows.

1. Read training, validation, and test indices from `data/processed/bacteria_id_v1/splits.npz`; verify disjointness, complete coverage of the target spectra, and per-class counts.
2. Use the union of training and validation as the 80% training set, saved in original row-index order.
3. Copy the original test row-index array, including its order.
4. Verify that the final training/test sets are disjoint and cover all target spectra, that test exactly matches the input test set, and that each class has 1,600 training and 400 test spectra.

Bacteria-4 and Bacteria-6 are split separately; each split is shared across M00/M11 and all five seeds.
Pretraining and CKmeans fitting use **all training spectra**, while test spectra are assigned to fixed cluster centers and scored.
Table 4 has no validation, HPO, or early stopping; it uses final raw weights trained from initialization for 800 epochs.
Checkpoints trained on the full reference subset for Table 5 are not reused for Table 4.

**Class composition differs between SMAE's statement that it “follows RamanCluster” and its supplementary figure.**
Inspection of both supplementary figures gives the following.

| Dataset and source | Class names and composition shown in the figure |
| --- | --- |
| RamanCluster supplementary Fig. S2(a): Bacteria-4 | C. albicans, C. glabrata, K. aerogenes, E. coli |
| SMAE supplementary Fig. S3(a): Bacteria-4 | C. albicans, C. glabrata, K. aerogenes, E. coli 1 |
| RamanCluster supplementary Fig. S2(b): Bacteria-6 | The four species above plus E. faecium and E. faecalis |
| SMAE supplementary Fig. S3(b): Bacteria-6 | The four isolate classes above plus E. coli 2 and E. faecium; E. faecalis is not shown |

The first six species listed in RamanCluster §3.1 are also consistent with Fig. S2(b).
In contrast, the legend and colors shared across all panels of SMAE Fig. S3 indicate the composition above,
with two of the six classes belonging to E. coli.
Comparison with `STRAINS` in the [official Bacteria-ID config.py](https://github.com/csho33/bacteria-ID/blob/3c00a712a6dbad9aefa19ac878a1e6db20590ca9/config.py)
maps the SMAE figure to IDs `[0, 1, 2, 3]` for Bacteria-4 and `[0, 1, 2, 3, 4, 5]` for Bacteria-6.
For comparison with SMAE, this study **adopts the IDs inferred from that figure (a supplementary choice made in this study)**.
This does not mean that the IDs used to produce SMAE Table 4 have been verified.
The sources alone do not identify whether RamanCluster's E. coli is ID 3 or 4, or whether its E. faecalis is ID 6 or 7.
The caption of SMAE Fig. S3 also describes a schematic of the self-attention mechanism, inconsistent with the clustering figure itself.

Whether this is a figure error or a difference in actual data composition remains unresolved;
Bacteria-6 is not asserted to be the same set in both papers.
The class composition in this experiment is fixed to the IDs above; no additional experiment using RamanCluster's composition is performed.
Table footnotes briefly state that composition follows the SMAE supplementary figure and note the discrepancy between the comparison sources.

## 6. Table 4: Unsupervised Clustering

### Evaluation Procedure

1. For the training set of each subset in Section 5, initialize and pretrain M00 and M11 for 800 epochs in each run with seeds 0–4.
   Labels are used to create subsets, stratify, verify splits, and score results, not in the pretraining loss or cluster-center updates.
2. Extract 128-dimensional unit latent representations from the pretrained checkpoint with no augmentation and all patches visible.
   The encoder after classification fine-tuning is not reused for this evaluation.
3. Fit **ChemoMAE's Cosine-KMeans as CKmeans** on the latent representations of all training spectra: K=4 for Bacteria-4 and K=6 for Bacteria-6.
   Latent representations are input directly, with no additional dimensionality reduction such as t-SNE or UMAP.
   Test representations are assigned to the trained center with the highest cosine similarity; centers are not updated or refitted on test data.
   This definition of fitting and scoring sets is a supplementary choice made in this study.
4. Compute ACC, NMI, and AMI over all test spectra as percentages. ACC uses the optimal one-to-one correspondence between cluster numbers and ground-truth labels.
   Label correspondence is used for scoring; cluster membership is not relearned. **Do not replace AMI with ARI.**
5. Report the arithmetic mean and sample SD (`ddof=1`) over five seeds for each subset and condition in the ChemoMAE rows.
   All three metrics for individual seeds are also recorded; the best seed is not selected.
   Reference rows quoted from SMAE retain point estimates only, as in the original table; unreported variability is not estimated and added.
   The “±” in Table 4 denotes sample SD, unlike the 95% CI in Table 5.

Cosine-KMeans uses k-means++-style initialization, one initialization per fit, `max_iter=500`, and `tol=1e-4`,
following the [existing specification](experiment_protocol.md#augmentation-clustering). No additional restarts are used to select the best value.
ACC correspondence uses Hungarian matching, as in
[`cluster_acc` in the public notebook](https://github.com/pengjuRen99/SpectraMAENet/blob/7939f2e6d3e6e975b687e4f07e56b9ce51cae5e3/Unsupervised_Mapping/t_SNE.ipynb).
Correspondence is determined over the entire test split; batch-specific ACC values are not averaged.
Feature extraction and scoring use `drop_last=False`, retaining incomplete final batches, including those in the training set used for fitting.

### NMI and AMI Definitions

The definitions explicitly stated in RamanCluster §3.3, Eqs. (15) and (16), the comparison source for SMAE, are adopted.
Let ground-truth labels be $`Y`$, cluster assignments $`\hat{Y}`$, mutual information $`I`$, and entropy $`H`$.

```math
\mathrm{NMI}=\frac{I(Y;\hat{Y})}{[H(Y)+H(\hat{Y})]/2}
```

```math
\mathrm{AMI}=\frac{I(Y;\hat{Y})-\mathbb{E}[I(Y;\hat{Y})]}
{\max\{H(Y),H(\hat{Y})\}-\mathbb{E}[I(Y;\hat{Y})]}
```

The implementation explicitly sets `average_method="arithmetic"` for
[NMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
and `average_method="max"` for
[AMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html),
then converts the scores to percentages. AMI is not left to the library default.
Negative AMI values are not clipped to zero. Values before output conversion and the library version are saved.
These definitions are based on RamanCluster's equations; equivalence to SMAE's unpublished scoring code has not been established.

### Main Table and Comparison Limitations

Values for the six reference methods are reported in the [published SMAE paper](https://doi.org/10.1016/j.eswa.2025.128576),
**p.8, Table 4**. Values are percentages, displayed to two decimal places in the saved main table, as in the published version.
Of the 30 values for the five reference methods excluding SMAE itself, 29 match
[RamanCluster Table 1](https://doi.org/10.1016/j.talanta.2024.126076).
This agreement suggests that previously reported values may have been retained, but the text and supplement do not establish
whether SMAE reran those five methods. They are therefore not described as “baselines recomputed by SMAE under identical conditions.”
SimCLR's Bacteria-4 NMI is 54.40 in SMAE Table 4 and 54.5 in RamanCluster Table 1.
Following the designated reference version, SMAE's 54.40 is retained; it is not silently corrected to the original paper's value.

The table combining measured and cited values is saved to `outputs/experiments/bacteria_id_table4_80_20_v1/table4.md`.

Footnotes state that the reference methods are quoted from SMAE Table 4, whereas ChemoMAE reports the five-seed mean ± sample SD
and uses SNV, a 128-dimensional unit latent representation, and Cosine-KMeans.
Isolate IDs based on the SMAE supplementary figure, the final 80/20 split, CKmeans fitting on training data,
and scoring on test data are listed as supplementary choices.
Because source materials disagree on class composition and split indices are unpublished,
the dataset is not described as exactly identical to the original experiment.
The comparison is not presented as rerunning all methods with the same preprocessing, clustering, and computational budget.
Reference rows lack variability estimates, and split/class-composition details remain unresolved;
point-estimate rankings alone do not establish statistical superiority or state-of-the-art performance under identical conditions.
The reasons for omitted variability in prior reports and the authors' intentions are not inferred.

## 7. Table 5: Classification after Pretraining

The finalized classification training settings are defined by the
[5-fold CV and full-subset training protocol](bacteria_id_hpo_protocol.md).
Using reference30 pretrained weights, configuration selection is based on the mean epoch-50 Accuracy across five CV folds
for M11 seed 0. The search and selection eligibility are defined in that protocol.
The selected configuration is applied to all five seeds of both M00 and M11, training for 50 epochs on all 3,000 finetune spectra.
The separate 3,000 test spectra are evaluated using only the selected epoch-50 weights.

Values for reference methods are quoted from the [published SMAE paper](https://doi.org/10.1016/j.eswa.2025.128576),
p.8, Table 5; those methods are not rerun in this study.

The table combining measured and cited values is saved to `outputs/experiments/bacteria_id_table5_hpo_cv_v1/table5.md`.

ChemoMAE aggregates test Accuracy as percentages across five seeds that are independent from pretraining onward.
With sample standard deviation $`s`$, the interval half-width uses a t-distribution with four degrees of freedom.

```math
\bar{x}\ \pm\ t_{0.975,4}\frac{s}{\sqrt{5}}
```

TCLP's 82.30 ± 0.5% is the value reported by the SMAE authors after modifying the encoder,
not the value from the original TCLP paper. The encoder update scope, checkpoint selection,
replicate starting point, and CI calculation for the published Table 5 cannot be determined.
The evaluation subset for Fig. 10 is also unspecified.
Comparison limitations are summarized in [interpretation of the results](../bacteria_id_interpretation_plan.md).

<a id="open-items"></a>

## 8. Study-Specific Choices and Randomness

For details absent from public materials, the following are adopted as conditions of this study.
Missing public information and discrepancies between SMAE and RamanCluster figures remain limitations of the literature comparison.

| Aspect | Supplementary choice made in this study |
| --- | --- |
| Table 4 isolates | SMAE supplementary Fig. S3 takes precedence: IDs `[0, 1, 2, 3]` for Bacteria-4 and `[0, 1, 2, 3, 4, 5]` for Bacteria-6 |
| Table 4 input and split | The 80/20 split from saved indices in Section 5 for target isolates in reference; fixed across M00/M11 and all runs |
| Table 4 fitting and scoring | Train 20 runs from initialization for 800 epochs, fit CKmeans on all training representations, and score all test spectra; no validation, HPO, or early stopping |
| Replicates and run seeds | Five seeds, `[0, 1, 2, 3, 4]`, for both tables; ChemoMAE rows in Table 4 report five-seed mean ± sample SD |
| Pretraining correspondence | Independently train for 800 epochs for every seed, corpus, and condition; Table 5 CV uses M11 seed 0, and full-subset training uses M00/M11 seeds 0–4 |
| Table 5 evaluation checkpoint | Exactly 50 epochs; CV uses epoch-50 validation, and full-subset training uses epoch-50 weights |
| Table 5 95% CI | t-based confidence interval with four degrees of freedom, derived from the sample SD of Accuracy across five seeds |

The five-replicate and 95%-confidence-interval reporting format for Table 5 comes from the published version;
seed values, replicate starting point, and the CI formula are supplementary choices.
Table 5's 5-fold CV is restricted to the finetune subset in Section 5 and is not replaced with the Table 4 split.

### Randomness Management

Run seeds are `s=0..4`, and corpus identifiers are `bacteria4`, `bacteria6`, and `reference30`.
Purpose-specific seeds are generated from SHA-256, following the existing [seed generation policy](experiment_protocol.md#seed-plan).
The input is ASCII JSON `["bacteria-id-seeds-v1", corpus, s, purpose]` without spaces;
the first four digest bytes are interpreted as a big-endian integer.
Purposes are distinguished as `model_init`, `train_order`, `mask`, `pretrain_augmentation`, `kmeans`,
`finetune_head`, `finetune_order`, and `finetune_augmentation`.
The M00/M11 condition name is excluded from seed generation, so the same corpus, run seed, and purpose share the same seed.
Random states for batch order, masks, and augmentation are separated so that unused augmentation does not alter other random streams.
For Table 5, reference30 pretraining and fine-tuning are paired by run seed.
The split seed 42 is outside this purpose-specific generation scheme; the actual seed list and split indices are saved before training begins.
Any seed collision is resolved and recorded before training, without selecting new seeds based on results.

### Training Corpora and Replicates

| Purpose | Training corpus | Conditions and replicates |
| --- | --- | --- |
| Table 4: Bacteria-4 | 6,400 training spectra from the four target classes in reference | Pretraining and CKmeans fitting for M00/M11 × five seeds |
| Table 4: Bacteria-6 | 9,600 training spectra from the six target classes in reference | Pretraining and CKmeans fitting for M00/M11 × five seeds |
| Table 5 pretraining | All 60,000 reference spectra | M00/M11 × five seeds |
| Table 5 HPO | 2,400 training spectra in each finetune fold | Uses the M11 seed-0 checkpoint; see the [classification protocol](bacteria_id_hpo_protocol.md) for search counts |
| Table 5 final training | All 3,000 finetune spectra | M00/M11 × five seeds from the corresponding pretrained checkpoints |

There are 30 pretraining runs in total. The full reference corpus for Table 5 is not mixed into the Table 4 training sets.
Per Table 4 pretraining run, Bacteria-4 has 4,800 attempted updates (six batches/epoch),
and Bacteria-6 has 7,200 (nine batches/epoch), using 800 epochs, batch size 1024, and `drop_last=True`.
Actual update and AMP skip counts are recorded per run, and all five seeds of each condition are aggregated.

## 9. Artifacts and Provenance

Processed inputs are saved under `data/processed/bacteria_id_v1/`, Table 4 main results under
`outputs/experiments/bacteria_id_table4_80_20_v1/`, Table 5 pretraining under
`outputs/experiments/bacteria_id_table5_reference30_pretrain_v1/`, and Table 5 search and final results under
`outputs/experiments/bacteria_id_table5_hpo_cv_v1/`.
See the [Table 4 runbook](../bacteria_id_runbook.md) and
[Table 5 runbook](../bacteria_id_hpo_runbook.md) for commands and checks.
Inputs and artifacts are excluded from Git; the Table 4 runbook records the source acquisition route and the status of comparison with distributed files.
