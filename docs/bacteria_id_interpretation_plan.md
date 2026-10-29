# Interpretation of the Raman Supplementary Experiments

Bacteria-ID is a supplementary experiment evaluating ChemoMAE representations in a spectroscopic modality different from historic wood NIR-HSI.
Unsupervised clustering is distinguished from 30-isolate classification after pretraining.
For the conditions, see the [experimental specification](design/bacteria_id_experiment_plan.md) and [classification protocol](design/bacteria_id_hpo_protocol.md);
for locations of numerical results, see the [artifact inventory](results.md).

## Unsupervised clustering

Table 4 evaluates whether 128-dimensional representations learned without using labels in the loss retain information
related to the isolate categories in Bacteria-4/6. Pretraining and Cosine-KMeans fitting use only the train data of each subset;
the fixed centers are then applied to test data. ACC, NMI, and AMI are reported as means and sample SDs over 5 seeds.

M11 has higher mean AMI and NMI than M00 on both subsets, whereas M00 has higher mean ACC.
These are distinct agreement measures; no single measure establishes overall superiority in discrimination.
Variation across replicates from training through clustering is also large, so the means should be read alongside individual seed results.

These results show that self-supervised learning can aggregate information useful for discrimination even in a Raman task
without spatial coordinates. However, the experiment does not identify whether this uses dependencies across multiple bands,
a limited set of peaks, or other factors. The inputs, latent dimensionality, and weights differ from the 16-dimensional historic wood representation,
so these results do not directly validate the chemical correctness of wood maps or the mechanism behind improved LLA.

## Classification after pretraining

Table 5 reports adaptation of models pretrained on 60,000 reference entries to 30-class classification using
100 labeled spectra per isolate, totaling 3,000 spectra.
One configuration selected by 5-fold CV within M11 finetune data is shared by M00/M11;
final test Accuracy and t-distribution 95% confidence intervals are reported for 5 seeds each.
Because the classification head is attached to the CLS representation before projection, this experiment does not feed the Table 4 128-dimensional latent representation directly to a classifier.
The two tables also use different training corpora and checkpoints.

Mean Accuracy exceeds 80% for both conditions, and M11 does not outperform M00.
This is an example of downstream adaptation with limited training labels, but there is no evaluation of the same model without
pretraining or with different label counts, so the causal benefit of pretraining and label efficiency remain unevaluated.
The shared classification configuration was selected with M11; this is not a comparison using a configuration optimized for M00.

Counts refer to spectra and do not imply 100 independent individuals or measurement groups per isolate.
The task classifies known isolates; generalization to unseen isolates or patients has not been evaluated.
Classification Accuracy should not be used to infer prediction accuracy for wood properties or the reliability of spatial maps.

## Comparison with published values

The existing-method values are reported values quoted from Tables 4 and 5 of the [published SMAE article](https://doi.org/10.1016/j.eswa.2025.128576);
those methods were not rerun in this study. The reference versions, class composition, preprocessing, splits, and metric definitions
are summarized in the [experimental specification](design/bacteria_id_experiment_plan.md).

- For Table 4, this study uses an 80/20% train/test split, whereas RamanCluster uses 60/20/20%.
  Bacteria-6 class composition also differs across sources, and the individual test indices in the literature cannot be reconciled.
  The quoted rows provide no per-seed values, so statistical superiority cannot be inferred from ChemoMAE's sample SD alone.
- For Table 5, preprocessing, pretraining, the scope of encoder updates, checkpoint selection, starting points of replicates,
  and confidence interval calculations do not fully agree. The TCLP values are reported by SMAE with the encoder replaced, rather than results from the original article.
- SMAE Fig. 10 shows classification accuracy under multiple design conditions; its best value, 83.9%, matches Table 5.
  Using the same evaluation set for configuration selection and final evaluation can introduce selection bias.
  This study selects settings by CV within finetune data and keeps test data separate.

Differences from published values are reference comparisons, not strict rankings or evidence of SOTA under identical preprocessing, training proportions, and search budgets.
Overlap of confidence intervals alone does not determine statistical significance.
