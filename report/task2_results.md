# Task 2 — Measured MusicCaps audio results

**Task 2 training and final evaluation are complete on the available MusicCaps
subset.** The run is `results/task2/available_run1/`, using real audio and the
same 30 independent AudioSet labels as Task 1. This is a 3,964-clip subset,
not the complete 5,521-clip dataset.

## Held-out results

Values are mean ± sample standard deviation across seeds 42, 43 and 44, all
evaluated on the same 606 test clips. Standard deviations describe training-seed
variation; they are not confidence intervals or significance tests.

| Model | Test Macro-F1 | Test Micro-F1 | Test mAP | Parameters | Mean training seconds |
|---|---:|---:|---:|---:|---:|
| Pooled-feature MLP | 0.2610 ± 0.0097 | 0.5595 ± 0.0349 | 0.2747 ± 0.0086 | 83,614 | 116.46 |
| Audio CNN | 0.2215 ± 0.0104 | 0.3820 ± 0.0363 | 0.2320 ± 0.0067 | 25,470 | 2269.89 |
| Temporal GraphSAGE | 0.2648 ± 0.0114 | 0.5729 ± 0.0207 | 0.2762 ± 0.0006 | 120,862 | 177.15 |
| Temporal + similarity GraphSAGE | 0.2540 ± 0.0145 | 0.5651 ± 0.0157 | 0.2640 ± 0.0063 | 120,862 | 113.31 |
| Temporal + random GraphSAGE | 0.2475 ± 0.0140 | 0.5498 ± 0.0118 | 0.2560 ± 0.0117 | 120,862 | 97.93 |

Temporal GraphSAGE was selected **using validation only**: mean validation
Macro-F1 was 0.2817, compared with 0.2799 for pooled MLP. It achieves higher mean
test scores than CNN and the two graph-edge controls in this run. Its improvement
over pooled MLP is small (about 0.0037 Macro-F1), so the evidence does not establish
a substantial or statistically significant benefit from message passing.
Adding similarity edges did not improve over temporal adjacency; report this
negative ablation rather than assuming that denser graphs help.

## Dataset and shared splits

- 3,964 valid downloaded MusicCaps intervals: **2,775 train / 583 validation /
  606 test**. The frozen cohort excludes 10 candidate files failing audio checks.
- Targets are Task 1's saved 30 AudioSet label IDs in exactly the same order.
  Video-ID split assignments are inherited; missing audio does not cause re-splitting
  or label reselection. All models use identical retained samples and targets.
- Preparation decodes clips, checks finite samples and duration within 20 ms,
  and preserves rejected files. Details are in
  [the audio validation audit](../data/splits/task2_available.validation.json).
- Audio availability and acquisition order can bias this subset. The missing
  1,557 rows are not all necessarily permanently unavailable. No claim of
  artist-disjoint or full-dataset generalization is made.
- Task 1's reported scores use 829 test clips; evaluate its predictions on this
  same 606-clip test subset before direct cross-task numeric comparisons.

## Audio graphs and models

Audio is cropped to the annotated interval, converted to mono 22.05 kHz and
peak-normalized. Ten non-overlapping 1-second segments form ten nodes per clip.
Each node has 311 features: 128 log-mel means, 128 log-mel standard deviations,
12 chroma means, 20 MFCC means, 20 MFCC-delta means, mean RMS, spectral centroid
and zero-crossing rate. Feature normalization is fitted on training nodes only.
Peak normalization makes RMS a relative energy measure.

Temporal graphs use nine undirected adjacent-segment pairs. Similarity graphs
retain those edges and add symmetrized top-2 non-adjacent cosine neighbors on
normalized features. Random controls retain the temporal chain and add the same
number of random non-adjacent pairs as the corresponding similarity graph.
Similarity/random graphs have 19–26 undirected edges (mean 23.1413). Random
controls match edge count, not degree distribution; their graphs remain fixed
across training seeds. No random graph equals its similarity counterpart here.

The saved graph audit checked **11,892 graphs** for connectivity, finite values,
valid edge indices, symmetry, temporal structure and control edge counts, with
zero errors. Every edge stays within one audio clip.

GraphSAGE has two mean-aggregation layers, width 128, LayerNorm/ReLU/dropout,
concatenated global mean/max graph pooling and a multi-label head. Pooled MLP
uses mean/max raw node pooling without message passing. CNN uses three 2-D
convolution blocks with 16/32/64 channels on the same clip's log-mel image.
All heads use sigmoid outputs and binary cross-entropy with logits.

## Training and frozen evaluation

All five configurations ran for each of three seeds: **15 completed runs**.
Settings: up to 30 epochs, batch size 16, AdamW learning rate 0.001, weight decay
0.00001, dropout 0.2, early-stopping patience 5, no class-positive weighting.
Per-epoch validation Macro-F1 selects checkpoints, with a validation-only global
threshold search from 0.05 to 0.95 in 0.05 steps. Lower thresholds break ties.
The overall model selection uses mean validation Macro-F1 across seeds.

Final evaluation restored all 15 frozen checkpoints with their saved thresholds;
it did not retrain or use test scores to change selection. Macro-F1 averages all
30 label F1 scores. Micro-F1 pools decisions. mAP averages per-label average
precision over labels with positive support; it is not trapezoidal PR AUC.
Absent AudioSet positive annotations are treated as zero operationally.

The run used CPU PyTorch 2.13.0+cpu and six threads. Preprocessing took
563.92 seconds; the validation training suite took
8625.49 seconds (143.76 minutes).
Per-model training times above include epoch validation/checkpoint saves but
exclude preprocessing and final test evaluation. Per-run test inference timings
are in the comparison CSV. The architectures have different parameter counts.

## Deliverables and verification

- [Full model/seed comparison](../results/task2/available_run1/runs/comparison.csv)
  and [mean/std table](../results/task2/available_run1/runs/comparison_aggregate.csv).
- [Graph ablations](../results/task2/available_run1/runs/graph_ablation.csv).
- [20 real graph examples](../results/task2/available_run1/graph_examples/index.html).
- [Graph integrity audit](../data/processed/task2_available/graph_audit.json).
- Raw node/mel caches in `data/processed/task2_available/raw_features/`; normalized
  graph objects in `data/processed/task2_available/samples/`.
- Each of the 15 `runs/seed*/<model>/` directories contains `best_model.pt`,
  `training_curves.png`, `history.json`, validation/test prediction archives,
  sample IDs, settings and provenance. Test metrics are in `test_metrics.json`;
  the original training `metrics.json` remains validation-only.
- [Saved-output verification](../results/task2/available_run1/verification.json)
  recomputes all 15 models' validation/test metrics, checks sample IDs/targets/
  label order, verifies validation-derived thresholds, checkpoint hashes and
  aggregate means/standard deviations. All checks passed.

Cached-dataset fingerprint:
`1c37c2d2d5a476613291707596b271070abf555d592fa4f208b5949cd1f6f8fb`.
The checkpoints and exact cached dataset must be preserved together.

The earlier [implementation/synthetic audit](task2_initial_audit.md) is retained
as history; its missing-audio status is superseded by this measured run.
The notebook still defaults to its synthetic demonstration. Use this report and
the real run directory for final experimental results. Tasks 3/4 are separate.
