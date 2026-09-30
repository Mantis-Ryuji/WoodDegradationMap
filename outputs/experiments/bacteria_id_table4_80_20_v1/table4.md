# Table 4 · Bacteria-ID clustering (80/20)

| Method | Bacteria-4 ACC | NMI | AMI | Bacteria-6 ACC | NMI | AMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SimCLR | 65.30 | 54.40 | 54.40 | 53.50 | 47.60 | 47.40 |
| SCCL | 71.00 | 61.70 | 61.60 | 52.00 | 46.40 | 46.20 |
| CC | 71.60 | 61.00 | 60.80 | 54.70 | 53.50 | 53.30 |
| TS-TCC | 75.00 | 73.50 | 73.40 | 70.50 | 65.70 | 64.80 |
| RamanCluster | 77.00 | 75.00 | 74.60 | 74.10 | 73.00 | 72.60 |
| SMAE | 83.80 | 76.20 | 76.10 | 81.30 | 76.80 | 76.70 |
| ChemoMAE(M00) | 81.34 ± 12.44 | 78.50 ± 1.34 | 76.25 ± 4.26 | 82.60 ± 8.93 | 76.89 ± 5.45 | 76.25 ± 6.29 |
| ChemoMAE(M11) | 78.40 ± 16.23 | 81.66 ± 5.86 | 78.42 ± 8.77 | 81.42 ± 11.24 | 80.59 ± 5.57 | 79.26 ± 6.52 |

All values are percentages. ChemoMAE: five independent seeds, mean ± sample SD (ddof=1); no seed selection. The six comparison rows are unchanged reported values from [Ren et al. (2025), Table 4](https://doi.org/10.1016/j.eswa.2025.128576). ChemoMAE uses SNV, 128-dimensional unit latents, 800 fixed pretraining epochs on the 80% train partition, and Cosine-KMeans fit on all train latents. The held-out 20% test partition is assigned to fixed centers. The literature methods' training fractions, exact split indices and evaluation procedures cannot be confirmed to match these conditions; this is not a strict same-condition comparison. Bacteria-4 uses IDs 0–3 and Bacteria-6 uses IDs 0–5 following SMAE supplementary Fig. S3.
