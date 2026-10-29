# Bacteria-ID Table 5: 5-Fold CV and Final Training

Table 4 and pretraining follow the [Bacteria-ID experiment specification](bacteria_id_experiment_plan.md).
Table 5 uses the existing reference30 pretrained checkpoints. Classification training settings are
selected using M11, then applied to both M00 and M11. The SMAE and other reference rows quote the
values reported in the published Table 5.

## Model and Search Space

The 30-class CLS classifier attaches LayerNorm→Linear(30) to the 256-dimensional CLS output before projection.
The last $`k`$ Transformer blocks and the classification head are updated; patch projection,
CLS/positional parameters, projection to $`z`$, and the decoder remain frozen. With $`k=0`$, only the head
is updated. Even $`k=8`$ does not unfreeze the entire encoder, including the stem.

Training uses AdamW (betas=(0.9, 0.95), eps=1e-8), CrossEntropyLoss, and FP32.
The head and updated blocks use separate constant learning rates, with the same weight decay
applied to all trainable parameters in both groups. No scheduler, early stopping, AMP, or Optuna pruning is used.
Each training run lasts exactly 50 epochs: only the head is updated in epochs 1–5, and the head plus the
last $`k`$ blocks are updated in epochs 6–50. With $`k=0`$, only the head is updated throughout all 50 epochs.

| Parameter | Candidates |
| --- | --- |
| $`k`$ | 0, 1, 2, 3, 4, 5, 6, 7, 8 |
| Head LR | $`(1,2,\ldots,10)\times10^{-4}`$ |
| Encoder block LR | $`(1,2,\ldots,10)\times10^{-5}`$; unused when $`k=0`$ |
| Batch size | 8, 16, 32, 64, 128 |
| TGN / FS | off/off, on/off, off/on, on/on |
| Weight decay | 0, $`10^{-5}`$, $`10^{-4}`$, $`10^{-3}`$, $`10^{-2}`$ |

TGN/FS strengths and their independent application probabilities of 0.5 use the WoodDegradationMap defaults.
Enabled augmentations are applied to fine-tuning training spectra after SNV.
No augmentation is applied to validation/test spectra. Excluding encoder LR when $`k=0`$,
the effective search space contains 41,000 configurations.

## Configuration Selection

StratifiedKFold(n_splits=5, shuffle=True, random_state=42) is applied to the 3,000 spectra
in the finetune subset (30 isolate classes × 100 spectra).
Each fold contains 2,400 training and 600 validation spectra; every spectrum is used for validation once.
Every trial independently fine-tunes M11 from pretraining seed 0 in each fold.
The objective maximizes the arithmetic mean of **epoch-50 validation Accuracy** across the folds.
The maximum over intermediate epochs is not used for scoring or checkpoint selection.

A single Optuna study searched up to 100 distinct configurations. The first 30 proposals were random,
followed by TPE proposals; the initial 30 included each $`k=0,\ldots,8`$ at least three times.
Repeated proposals are rejected without training; if duplicates recur,
unevaluated configurations are supplied at random. In total, 64 distinct configurations were completed,
and **the first 60 completed configurations ordered by trial number** were eligible for selection.
Selection was finalized before the 100-configuration limit; the remaining four configurations and
incomplete trials were excluded from selection and retained in the execution records.
Ties are resolved in favor of the smaller trial number. Test data are not used for search or selection.
Split indices, configurations, epoch-50 Accuracy, and execution settings are saved for each fold.

## Full-Subset Training and Test Evaluation

The **single selected configuration** is applied to both M00 and M11, each independently fine-tuned
from pretraining seeds 0–4. All 3,000 finetune spectra are used for training, with no validation split.
As in CV, training lasts exactly 50 epochs, with only the head updated for the first five epochs.
Only the **epoch-50 weights** are used for test evaluation. After all 10 runs are complete,
the separately distributed 3,000 test spectra are scored once per run.

Table 5 reports the mean Accuracy across five seeds ± the half-width of a t-based 95% confidence interval
for M00/M11 in the w/ pretraining column. Individual seed scores, fold-level search results, and the
selection eligibility range are saved. The shared configuration was optimized on M11 validation results;
it is not the result of a separate search for the best M00 configuration. This distinction is stated
when comparing M00 with M11 and when interpreting the literature values.
Pretraining, splits, HPO, classification, and checkpoints for Table 5 are independent of Table 4.
