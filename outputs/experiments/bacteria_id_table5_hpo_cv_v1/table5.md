# SMAE Table 5 + ChemoMAE (Bacteria-ID)

| Method | Supervised learning Accuracy | w/o pretraining Accuracy | w/ pretraining Accuracy |
| --- | ---: | ---: | ---: |
| ResNet | 83.40 ± 0.4 | 76.50 ± 1.2 | — |
| RamanNet | 85.40 ± 0.4 | 77.40 ± 1.2 | — |
| ConvMSANet | 84.80 ± 0.3 | 76.80 ± 1.4 | — |
| TCLP | — | — | 82.30 ± 0.5 |
| Jensen et al. (2024) | — | — | 81.60 ± 0.7 |
| SMAE | 85.40 ± 0.3 | 77.80 ± 1.4 | 83.90 ± 0.4 |
| ChemoMAE(M00) | — | — | 80.53 ± 1.6 |
| ChemoMAE(M11) | — | — | 80.11 ± 0.8 |

ChemoMAE uses the M11-selected shared setting: k=8, head lr=0.0003, encoder lr=8e-05, batch=8, TGN=off, FS=off, AdamW weight decay=0.0001. The first 60 of 64 unique completed settings were ranked by M11 seed-0 five-fold mean validation Accuracy at epoch 50. The planned 100-setting search was stopped early at user request; later settings and incomplete trials were excluded from selection. For each condition and five pretraining seeds, the selected setting was independently trained on all 3000 finetune spectra for 50 epochs; epochs 1–5 updated only the head and epochs 6–50 also updated the final k blocks. There was no LR scheduler or early stopping. ChemoMAE Accuracy is five-seed mean ± t(0.975,4) × sample SD / sqrt(5) in percent. Other rows are values reported in [published SMAE Table 5](https://doi.org/10.1016/j.eswa.2025.128576); their training settings and interval calculation are not asserted identical.
