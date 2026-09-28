# Task 3 — measured frozen GNN–BERT fusion

**The frozen Task 3 experiment is complete on the available 3,964-clip MusicCaps cohort:** five heads,
three seeds, 15 trained and evaluated checkpoints, verified predictions, training/validation curves,
embedding plots, three case studies and failure analysis. The run is
`results/task3/available_run1/`, rebuilt and verified on 2026-09-10.
Missing Task 3 modules were recovered from the supplied source/test archives;
the real feature cache and experiment outputs were regenerated locally. The
measurements below describe this verified run and supersede the earlier draft.

This report covers frozen encoders. Joint encoder/fusion fine-tuning uses
`results/task3/joint_run2` is also complete; see the [joint results](task3_joint_results.md),
[current completion audit](task3_completion_audit.md) and [run guide](../docs/task3/README.md). MusicCaps dataset
substitution approval remains separate from experimental completion.

**Fusion did not improve over the text control on the selection metric.**
BERT-only wins mean validation Macro-F1 (0.4621). Gated fusion is the best fusion
mode on validation (0.4353), and its held-out Macro-F1 is 0.3705 versus 0.3850 for
BERT-only. This negative finding is part of the completed experiment.

## Frozen comparison

Values are mean ± sample standard deviation over head seeds 42, 43 and 44.
Every test run uses the same 606 clips. Standard deviations describe head
initialization/batch-order variation, not encoder variability, confidence
intervals or statistical significance.

| Head | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test mAP |
|---|---:|---:|---:|---:|
| BERT-only | 0.4621 ± 0.0040 | 0.3850 ± 0.0058 | 0.6564 ± 0.0105 | 0.3892 ± 0.0058 |
| GNN-only | 0.2967 ± 0.0050 | 0.2539 ± 0.0119 | 0.5632 ± 0.0352 | 0.2680 ± 0.0036 |
| Concatenation | 0.4228 ± 0.0070 | 0.3839 ± 0.0124 | 0.6560 ± 0.0034 | 0.3994 ± 0.0038 |
| Gated fusion | 0.4353 ± 0.0048 | 0.3705 ± 0.0013 | 0.6466 ± 0.0149 | 0.3995 ± 0.0110 |
| Cross-attention | 0.4232 ± 0.0058 | 0.3665 ± 0.0102 | 0.6453 ± 0.0061 | 0.3861 ± 0.0048 |

![Measured ablation comparison](../results/task3/available_run1/analysis/ablation.png)

Gated fusion improves test Macro-F1 by 0.1166 over the graph control but loses
0.0145 against BERT-only. Its mAP is 0.0103 higher than BERT-only, while its
thresholded F1 scores are lower. This describes a difference between ranking and
classification performance; it does not justify switching the validation-selected
winner. Concatenation nearly matches BERT-only test F1, but it also loses on
validation. No configuration was changed after viewing test results.

## Data and encoder reuse

The frozen cohort contains **2,775 train / 583 validation / 606 test** examples.
All heads share 30 independent `MusicCaps.audioset_positive_labels` targets,
vocabulary order, video-ID splits, captions and graphs. The vocabulary was fitted
on Task 1 training data. Missing audio was filtered without resplitting or
reselecting labels. These are available-audio subset results, not full-MusicCaps
or artist-disjoint results.

The text encoder is Task 1's two-epoch fine-tuned `distilbert-base-uncased`,
frozen during Task 3. Its 3,864 original training captions include the 2,775
paired training examples and 1,089 additional training-only examples without
usable paired audio. All paired held-out IDs/captions preserve their original
held-out partitions. The graph encoder is Task 2's seed-42 temporal GraphSAGE,
chosen from the architecture selected by mean validation Macro-F1; it trained on
the 2,775 paired training clips. Both encoders are identical across all 15 runs.
These controls are newly trained heads, not historical Task 1/2 scores.
The paired Task 3 inputs are identical across heads; historical encoder training
cohorts are not identical. Results therefore compare heads over reused encoders,
not encoders trained with equal numbers of examples.

Preparation verifies training CSV hashes and saved split metadata, IDs, captions,
label vectors/order, the graph cached-dataset fingerprint and training
normalization. It permits the relocated Task 2 path only after exact fingerprint
agreement. A disk-backed extraction buffer and mapped cache avoid duplicating
the 1.56 GB token tensor in RAM. The cache has tokens `[3964,128,768]`, masks
`[3964,128]`, graph embeddings `[3964,256]` and targets `[3964,30]`.
See [compact feature provenance](../results/task3/feature_summary.json).

## Heads and training

The BERT head projects the contextual CLS vector. The GNN head projects the
concatenated mean/max graph readout. Concatenation joins both width-64 projections;
gated fusion learns a per-dimension sigmoid mixture. Single-query cross-attention
uses the graph projection as query, masked contextual BERT tokens as keys/values,
and a graph residual. Every head ends with dropout 0.1 and a linear 30-label layer.

Settings fixed before testing: at most 20 epochs, batch size 16, hidden width 64,
AdamW learning rate 0.001 and default weight decay 0.01, BCEWithLogitsLoss,
gradient-norm clipping at 1.0, patience 5 and CPU with six PyTorch threads.
Every epoch selects a global threshold using validation Macro-F1 over
0.05–0.95 in steps of 0.05; lower thresholds break ties. Best validation epochs
and their thresholds are saved. The suite then selects the mode using mean
validation Macro-F1 across seeds and freezes checkpoint hashes before testing.

| Head | Trainable head parameters | Mean head-training seconds |
|---|---:|---:|
| BERT-only | 51,166 | 4.03 |
| GNN-only | 18,398 | 3.01 |
| Concatenation | 69,534 | 3.50 |
| Gated fusion | 75,870 | 3.32 |
| Cross-attention | 84,254 | 20.56 |

Counts exclude frozen encoders. Times include epoch validation/checkpoint saves
but exclude encoder extraction and final testing. The suite timer was 122.93
seconds including per-seed outputs. These are local elapsed measurements, not
isolated hardware benchmarks. The architectures have different parameter counts.

Macro-F1 averages all 30 label F1 scores; Micro-F1 pools decisions. mAP averages
per-label average precision over labels with test-positive support, rather than
trapezoidal precision–recall area. All 30 labels have test positives in this run.
Absent positive annotations are treated as zero operationally.

## Failures and three fixed cases

The selected gated model has weak test AP for Pop music (0.0316; 7 positives),
Funk (0.1514; 11) and Song (0.1692; 13). It has higher AP for Music (0.9394; 497),
Didgeridoo (0.9040; 9) and Wind instrument (0.8688; 30). Small support makes some
rankings unstable. See [per-label evidence](../results/task3/available_run1/analysis/per_label.csv).

Across the three paired seeds, gated fusion improves per-example F1 over BERT
in 438 seed/example pairs, worsens it in 452 and ties in 928. These 1,818 pairs
reuse the same 606 clips, so they are not independent observations. They show
mixed local changes rather than a consistent gain.

The first three validation IDs, fixed before looking at outcomes, are traced
through the seed-42 cross-attention head regardless of the winning architecture:

1. **Guitar tutorial, `-5xOcMJpTUk_70_80`:** 6 true positives, 3 false positives and 0 false negatives at threshold 0.15.
   False-positive labels: Effects unit, Electric guitar, Distortion. False-negative labels: none.
   The focal Guitar score is 0.9940 (target 1).
   Zeroing the 1–2 second normalized node changes this score by +0.1112 (original minus intervened).
   The largest lexical-vector effect is `technique` at +0.0010.

2. **Choir recording, `-8C-gydUbR8_30_40`:** 1 true positive, 0 false positives and 0 false negatives at threshold 0.15.
   False-positive labels: none. False-negative labels: none.
   The focal Singing score is 0.0277 (target 0).
   Zeroing the 7–8 second normalized node changes this score by -0.0199 (original minus intervened).
   The largest lexical-vector effect is `black` at +0.0173.

3. **Ambient synth clip, `-Bu7YaslRW0_30_40`:** 1 true positive, 1 false positive and 0 false negatives at threshold 0.15.
   False-positive labels: New-age music. False-negative labels: none.
   The focal New-age music score is 0.4821 (target 0).
   Zeroing the 8–9 second normalized node changes this score by +0.1272 (original minus intervened).
   The largest lexical-vector effect is `flute` at +0.0348.

[Full case traces and figures](../results/task3/available_run1/case_studies/cases.md)
retain all targets, probabilities and token/segment effects. Interventions zero
one cached contextual vector or one standardized node-feature vector. They retain
other token information and graph edges. Such out-of-distribution interventions
measure representation sensitivity, not causal importance of words or sounds.
t-SNE plots are also descriptive. The choir case has only Music annotated
within the selected vocabulary, while the ambient case has an unannotated
New-age prediction. Incomplete labels can contribute to apparent errors;
these cases were not independently relabeled.

The [selected gated-fusion embedding plot](../results/task3/available_run1/analysis/fusion_embeddings.png)
uses saved seed-42 validation representations; coordinates and sample IDs are
preserved alongside it. Per-seed overall-winner plots remain in each seed folder.

The weak graph control and stronger text control are consistent with audio adding
limited useful information under this setup. Coarse one-second segment features,
global graph pooling, fixed encoders and label imbalance are plausible constraints;
this experiment does not isolate which causes the negative fusion result. More
parameters and cross-attention did not yield a validation gain over BERT-only.

## Verification and reproducibility

- [Completion audit](task3_completion_audit.md) maps every Task 3 requirement
  to verified evidence and records the frozen-encoder scope qualifications.
- All **73 repository tests passed**, including 19 Task 3 tests for shapes,
  masks, gradients, feature extraction, provenance, restoration and suite audits.
- [Verification](../results/task3/available_run1/verification.json) recomputes all
  15 validation/test metric sets and validation thresholds, checks aligned
  IDs/targets, hashes and frozen selection, and verifies published CSV means/std.
- [Full comparison](../results/task3/available_run1/comparison.csv),
  [aggregates](../results/task3/available_run1/comparison_aggregate.csv),
  [failure analysis](../results/task3/available_run1/analysis/README.md),
  checkpoints, prediction archives and epoch curves are preserved locally.
- [Test verification](../results/task3/test_verification.json) records 73 passing tests and exit code 0.
- The [review notebook](../notebooks/task3_fusion.ipynb) reads measured results
  by default and runs saved-checkpoint inference on three real validation pairs;
  its synthetic training demonstration is opt-in.
- [Run guide](../docs/task3/README.md) gives exact preparation, training,
  evaluation, analysis and case-generation commands. Raw audio and large caches
  remain excluded from Git; preserve them locally for full reproduction.

The independent targets remove mechanical caption-derived target leakage, but
captions naturally mention labeled instruments and music types. Available-audio
selection bias, incomplete annotations, limited label support and reused encoders
remain limitations. No statistical significance, listener assessment or
cross-dataset generalization is claimed.
