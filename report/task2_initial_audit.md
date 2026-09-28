# Task 2 — Completion audit and verified pipeline

Audit date: 8 September 2026.

**Task 2 is not yet complete as a real-data experiment.** The implementation
has been improved and exercised end to end. The remaining requirement is
training and evaluation on real MusicCaps audio using the inherited Task 1
labels/splits. No real Task 2 accuracy claim is made.

## Requirement-by-requirement status

| Requirement | Verified implementation/evidence | Real-data status |
|---|---|---|
| Load and preprocess audio | Mono resampling, explicit source interval, peak normalization and finite-waveform checks in `src/task2/features.py` | Awaiting local audio |
| Meaningful units and graph nodes | Fixed 1-second nodes by default; final partial segment padded | Awaiting real clip graphs |
| Node features | Log-mel mean/std, chroma, MFCC/delta, RMS, centroid, ZCR; 311 dimensions by default | Pipeline ready |
| Meaningful graph edges | Temporal chain; temporal + top-k cosine similarity | Pipeline ready |
| Explain construction | [Full method and defaults](../docs/task2/README.md#audio-and-graph-construction) | Documented |
| Connected graphs without invalid values | All saved graphs audited for connectivity, indices, symmetry, temporal edges and finite tensors | 90 synthetic graphs pass |
| GraphSAGE or GAT | GraphSAGE with mean-neighbor aggregation, LayerNorm/ReLU/dropout | Synthetic checkpoint only |
| One graph-level representation | Concatenated global mean/max readout | Verified |
| Same labels as other tasks | Real manifest inherits Task 1's 30 independent AudioSet IDs in their exact order | Source alignment verified |
| CNN comparison | Three-block log-mel CNN on the same cropped waveform | Synthetic checkpoint only |
| Non-message-passing baseline | Mean/max pooled node-feature MLP | Synthetic checkpoint only |
| Graph-structure controls | Temporal, temporal+similarity, temporal+random with matched added edge counts | Synthetic ablations saved |
| Identical Task 2 samples/splits | One processed manifest, saved split IDs and cached-tensor fingerprint for every run | Enforced by runner/checkpoints |
| Cached node features and saved graphs | Raw feature cache and per-clip dictionaries with all three edge tensors | 30 synthetic samples saved |
| At least 20 graph examples | 20 distinct clips, each shown under all three graph structures | Synthetic gallery only |
| GNN/CNN checkpoints, curves and test predictions | Five complete model output directories | Synthetic artifacts only |
| Macro-F1, Micro-F1 and mAP | Recomputable metrics with IDs, label order, targets and probabilities | No real-data metrics |
| Runtime and parameter counts | Per-model training/inference time, complete classifier parameters; preprocessing/suite runtime | Synthetic measurements only |

## Real-audio availability and label alignment

The [saved availability audit](../results/task2/audio_availability_audit.json)
checks the official metadata against Task 1's prepared label CSV and dataset
manifest. All **5,521 video IDs** agree with the independent AudioSet targets.
The label CSV checksum matches Task 1's saved checksum.

| Inherited split | Task 1 clips | Available local real audio |
|---|---:|---:|
| Train | 3,864 | 0 |
| Validation | 828 | 0 |
| Test | 829 | 0 |
| Total | 5,521 | 0 |

The local-audio manifest currently has zero records and explicitly reports
`ready_for_training=false`. Source annotation checks do not establish remote
audio availability. No audio was downloaded during this audit.

The previous Task 2 route joined captions to the old caption-derived proxy
labels and independently generated splits. It could not support comparisons
with the newer Task 1 run. Real manifests now join by `ytid`, preserve Task 1
split membership and label order, and retain the same available-audio subset
for all five Task 2 models. Missing audio does not trigger a new split or label
selection. Zero-positive examples are retained.

Task 1's published scores use the full 829-clip test population. If only a subset
has audio, text predictions must be evaluated on that same subset before any
cross-task numeric comparison. Tasks 3/4 were not retrained as part of this work.

## Delivered verification artifacts

The executed CLI run is
[`results/task2/synthetic_verification/`](../results/task2/synthetic_verification).
It uses 30 independently generated 2-second tone mixtures, three synthetic
targets, 8 kHz sampling, 0.25-second segments, 95 features per node, k=1, seed 42,
and three training epochs. Every model shares the 18/6/6 split. These settings
are intentionally small integration checks and differ from real-audio defaults.

- [Graph audit](../results/task2/synthetic_verification/processed/graph_audit.json):
  30 samples, **90 connected finite graphs**, no errors. Temporal graphs have
  seven undirected edges; similarity and random graphs each have 11–13.
  None of the 30 random graphs coincides with its similarity graph.
- [Graph gallery](../results/task2/synthetic_verification/graph_examples/index.html):
  **20 distinct clip visualizations**, 60 graph panels, with a machine-readable
  index. Example graph and learning-curve images were visually inspected.
- [Model comparison](../results/task2/synthetic_verification/runs/comparison.csv)
  and [graph ablations](../results/task2/synthetic_verification/runs/graph_ablation.csv):
  validation/test Macro-F1, Micro-F1, mAP, runtime and parameters. Values are
  **synthetic diagnostics, not music-context results**.
- [Frozen selection](../results/task2/synthetic_verification/runs/selection.json)
  and [shared split IDs](../results/task2/synthetic_verification/runs/split_ids.json).
  Validation ties selected temporal GraphSAGE by the declared ordering. This is
  not evidence that temporal graphs outperform other music representations.
- Five model directories under `runs/seed42/`: `mlp`, `cnn`,
  `gnn_temporal`, `gnn_similarity`, `gnn_random`. Each contains a checkpoint,
  history, learning curve, validation/test predictions, settings and provenance.
- [Verification record](../results/task2/synthetic_verification/verification.json):
  saved-prediction metric checks and artifact counts.

The synthetic MLP has 6,211 parameters, the CNN 23,715, and each GraphSAGE
classifier 8,515. GraphSAGE structures use the same parameter count. Architectures
are not parameter matched to CNN/MLP; parameter counts are reported to make that
limitation visible. Short CPU runtimes cannot establish real-data efficiency.

The shared cached-dataset SHA-256 is
`9038ed628ae7efd7d354d61d63dd742c70af32dfba828f559e0a4aad6bc03385`.

## Verification and experimental limits

The complete offline suite passed **55 tests**, including 23 Task 2 tests.
The Task 2 tests cover source-label alignment, inherited splits, crop handling,
cache invalidation, malformed/disconnected graphs, gallery export, all model
shapes and a complete five-model comparison. The suite verifies that saved test
metrics recompute, thresholds come from validation, test IDs/targets agree, and
changed cached features invalidate evaluation. Existing Task 1, Task 3 and
Task 4 tests also passed.

The updated notebook executed all **13 code cells** successfully while reviewing
the saved artifacts, without retraining or rewriting the cached tensors. See
[notebook verification](../results/task2/synthetic_verification/notebook_verification.json)
and [test verification](../results/task2/synthetic_verification/tests_verification.json).

Checkpoint/threshold selection uses validation only. The comparison runner
freezes all predeclared controls before test evaluation. Macro-F1 includes all
labels; mAP is mean per-label average precision over labels with positive
support, not trapezoidal PR AUC. Absence of an AudioSet positive annotation is
treated as zero operationally, not as proof that an event is absent.

The delivered synthetic run uses one training seed. The real-data command
defaults to three seeds and reports mean/standard deviation; no uncertainty or
generalization claim is supported by the current synthetic run. The random
control preserves the temporal chain and edge count but not the degree
distribution. Graph edges remain within clips. Artist-disjoint generalization
is not established by a video-ID split.

## Remaining work to finish Task 2

1. Make real MusicCaps clips available in `data/raw/musiccaps_audio/`, or
   provide their existing folder. Identify whether files are pretrimmed clips
   or full recordings so source offsets are applied correctly.
2. Rebuild the aligned manifest and inspect missing-clip and per-label coverage.
   Preprocess once to produce the common real-audio cohort and graph cache.
3. Run all five controls on that cache, using the declared seeds and validation
   selection. Export at least 20 real graph examples.
4. Freeze configurations, evaluate the fixed test set, and report real Macro-F1,
   Micro-F1, mAP, runtimes, parameter counts and graph ablations. Report any
   negative GNN-vs-CNN or GNN-vs-random result directly.

The [updated Task 2 guide](../docs/task2/README.md) contains executable commands.
The [updated notebook](../notebooks/task2_gnn_cnn.ipynb) reviews the synthetic
artifacts and provides the aligned real-data workflow.
