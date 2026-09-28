# Task 2 — Music graphs, GraphSAGE and audio baselines

Task 2 training and final test evaluation are **complete on 3,964 real MusicCaps
clips** in `results/task2/available_run1/`: 15 model/seed runs, 20 graph examples,
and 606 held-out test clips. Temporal GraphSAGE was selected on validation and
achieved mean test Macro-F1 0.2648, Micro-F1 0.5729 and mAP 0.2762 across three
seeds. See [the measured report](../../report/task2_results.md) for all controls,
uncertainty and verification. The synthetic run remains a separate pipeline check.

Open [the notebook](../../notebooks/task2_gnn_cnn.ipynb), or use the CLI below.
Both use the same modules in `src/task2/`.

## What changed in this audit

- Real manifests now join by MusicCaps `ytid`, use Task 1's independent 30-label
  AudioSet vocabulary in its saved order, and inherit its video-level splits.
  Missing audio is removed once, before all Task 2 comparisons.
- Audio input explicitly distinguishes pretrimmed clips from full recordings.
  Graph nodes and CNN inputs use the exact same requested interval.
- Invalid waveforms and features fail validation. Silence maps to a zero CNN
  image. Cached features include audio-content and crop fingerprints.
- Every saved graph is checked for finite features, valid indices, symmetry,
  duplicates, connectivity, temporal edges and matched random/similarity density.
- Twenty distinct clips can be exported as three-panel graph comparisons.
- A single runner trains pooled MLP, CNN, temporal GraphSAGE, similarity
  GraphSAGE and random-control GraphSAGE on identical samples and splits.
- Checkpoints bind to cached tensors, labels, normalization and splits by SHA-256.
  Evaluation rejects a changed dataset. Existing training directories are preserved.
- Validation predictions, histories, provenance, comparison and graph-ablation
  tables are saved. Test evaluation uses frozen checkpoints and thresholds.

## Audio and graph construction

Real-audio defaults: mono 22,050 Hz; peak normalization within the selected clip;
non-overlapping 1-second segments; zero padding for the last partial segment.
Silence is retained. Peak normalization makes RMS a relative energy descriptor,
not a measure of original recording loudness.

Each segment has **311 features**: 128 log-mel means, 128 log-mel standard
deviations, 12 chroma means, 20 MFCC means, 20 MFCC-delta means, and mean RMS,
spectral centroid and zero-crossing rate. Mel power uses an 80 dB dynamic range.
Feature-wise means and population standard deviations are fitted on training
nodes only; tiny variances have a numerical floor. Validation and test features
use the same saved normalization.

All graphs are within a single clip; there are no edges across clips or splits.

| Variant | Edges |
|---|---|
| `temporal` | Bidirectional edges between consecutive segments |
| `temporal_similarity` | Temporal chain plus each node's top-k non-adjacent cosine neighbors on normalized features, symmetrized |
| `random` | Temporal chain plus uniformly sampled non-adjacent pairs, with exactly the same added edge count as that clip's similarity graph |

Default k is 2. Similarity ties use segment index. Random edges use a stable
sample-ID-derived seed. Every graph remains connected through the temporal
chain, including the valid single-node case. The random control matches edge
count, **not node degrees**. With very short clips or dense graphs, controls can
coincide; `graph_audit.json` counts these cases. Similarity-only disconnected
graphs are not used: this comparison measures the value of adding similarity
edges to a connected temporal graph.

GraphSAGE uses mean-neighbor SAGEConv layers, LayerNorm, ReLU and dropout.
Concatenated global mean/max pooling produces one graph embedding, followed by
a multi-label linear head. The pooled MLP applies mean/max pooling directly to
node features, with no message passing. The CNN uses three convolution blocks
(16/32/64 channels), batch normalization, ReLU, max pooling and global average
pooling on the clip-level log-mel image. Models use sigmoid outputs and
BCEWithLogitsLoss. Parameter counts describe each complete classifier.

## Real-data workflow

To train using only the audio downloaded so far, first stop the downloader with
Ctrl+C. Then run these commands individually from the project root:

```powershell
python -m scripts.prepare_task2_available --output data/splits/task2_available.json
python -m src.task2.preprocess --manifest data/splits/task2_available.json --output-dir data/processed/task2_available
python -m src.task2.experiment --manifest data/processed/task2_available/manifest.json --output-dir results/task2/available_run1 --epochs 30 --seeds 42 43 44 --device auto
```

The preparation command fully decodes each candidate clip, excludes invalid or
incomplete audio without deleting files, and saves an exclusion audit beside the
frozen manifest. It preserves Task 1 labels and splits. Stop if a command fails.
If the frozen manifest already exists, reuse it and start at preprocessing;
new cohorts or training runs need fresh output paths. These commands perform
validation-based training; frozen test evaluation remains a separate step.

Run from the repository root with the project environment activated. Use fresh
output directories for training.

If you do not have the audio yet, follow
[the audio download setup](downloading_audio.md). The downloader checks FFmpeg,
ffprobe and a supported JavaScript runtime before making any media requests:

```powershell
.\scripts\download_musiccaps.ps1 -Limit 1
.\scripts\download_musiccaps.ps1
```

The first command verifies one clip; the second processes the CSV and skips
existing valid WAV clips. It resolves tool locations explicitly, so an older
terminal PATH need not prevent downloads after installation.

Place local clips in `data/raw/musiccaps_audio/`. Supported formats are WAV,
FLAC, MP3, M4A and OGG. Pretrimmed filenames may be `{ytid}.wav`,
`{ytid}_{start_s}.wav`, `{ytid}_{start_s}_{end_s}.wav`, or
`{ytid}-{start_s}.wav`. Exact interval names take precedence.

For files already cropped to the MusicCaps interval:

```powershell
python -m src.task2.manifest --metadata-csv data/raw/musiccaps_official.csv --labels-csv data/processed/musiccaps_audioset.csv --task1-manifest results/task1/audioset_cpu_20260907/dataset_manifest.json --audio-dir data/raw/musiccaps_audio --audio-mode pretrimmed --output data/splits/task2_manifest.json
python -m src.task2.preprocess --manifest data/splits/task2_manifest.json --output-dir data/processed/task2
python -m src.task2.experiment --manifest data/processed/task2/manifest.json --output-dir results/task2/audioset_run1 --epochs 30 --seeds 42 43 44 --device auto
```

Stop if a command fails. If using complete recordings named `{ytid}.wav`,
choose `--audio-mode full_source`; loading starts at `start_s` and lasts
`end_s-start_s`. Short recordings are rejected. Do not mark whole recordings
as pretrimmed clips.

The manifest checks the Task 1 label CSV checksum and official annotation
agreement. It reports missing clips, retained counts and label support by split.
It does not reselect labels or move held-out videos into training after missing
audio is removed. All three splits need available clips before training.

The runner exports the graph gallery, then trains all five models for each
declared seed. Model settings, sample membership, normalization and graph cache
are shared. Training seeds change initialization and batch order; random-control
edges stay fixed by the preprocessing seed.

## Selection and frozen test evaluation

Each epoch's global threshold is selected on validation Macro-F1 over
0.05–0.95 in 0.05 increments. Lower thresholds break ties. Checkpoints are
selected on that validation Macro-F1; early stopping defaults to patience 5.
This differs from Task 1's per-label calibration and should be disclosed in
cross-task comparisons. Learning curves show each epoch's validation-calibrated
F1, not F1 at a constant threshold.

The suite selects its overall model and GNN variant using mean validation
Macro-F1 across declared seeds, with declaration order breaking ties.
Training defaults to validation-only reporting in the experiment runner.
Once configuration selection is finished:

```powershell
python -m src.task2.experiment --evaluate-run results/task2/audioset_run1/runs --device auto
```

This evaluates **all predeclared frozen controls** on the same test IDs, to make
the required baseline/graph comparisons. It does not retrain, retune thresholds
or change the validation selection. A completed test report cannot be silently
overwritten by this command. Do not use the resulting test table to select
another model or seed.

The lower-level `src.task2.train` command remains available and requires
`--skip-test` during model selection. Standalone checkpoint evaluation is:

```powershell
python -m src.task2.evaluate --checkpoint results/task2/audioset_run1/runs/seed42/gnn_similarity/best_model.pt --manifest data/processed/task2/manifest.json --split test --output results/task2/audioset_run1/standalone_test.json
```

Old checkpoints without dataset fingerprints require retraining. Preserve the
processed cache with new checkpoints. Reprocessing changes that alter serialized
tensors may invalidate a checkpoint even if the new features are numerically
equivalent.

## Offline demonstration and verification

The delivered run is `results/task2/synthetic_verification/`: 30 generated
2-second tone mixtures, three synthetic targets, eight nodes per clip, 95 node
features, k=1, 18/6/6 train/validation/test samples, seed 42 and three epochs.
These are **not AudioSet labels or evidence of music-context accuracy**.

To reproduce into a new directory:

```powershell
python -m src.task2.experiment --demo --output-dir tmp/task2_demo_new --epochs 3 --hidden-dim 32 --batch-size 6 --device cpu --evaluate-test
python -m unittest discover -s tests -v
```

The notebook reviews the saved synthetic run when available and otherwise
builds it. Real mode uses the aligned manifest above; no audio is downloaded.

## Output files

```text
data/splits/task2_manifest.json       source audio, intervals, labels, inherited splits
data/processed/task2/
  raw_features/*.pt                  cached raw node features and CNN mel images
  samples/*.pt                       normalized x and three edge_index tensors
  normalization.json                 training-node statistics
  manifest.json                      processed paths and source provenance
  graph_audit.json                   all-sample connectivity/finiteness audit
results/task2/<experiment>/
  graph_examples/                    20 PNG comparisons + index.html/index.json
  runs/
    dataset_identity.json            exact cached-sample hashes
    split_ids.json                   one cohort for every control
    selection.json                   validation decision and checkpoint hashes
    experiment.json                  complete suite report
    comparison.csv                   every model/seed, validation and optional test
    graph_ablation.csv               the three GNN structures
    comparison_aggregate.csv         per-model means and across-seed standard deviations
    seed42/<model>/                   also seed43 and seed44 when requested
      best_model.pt
      metrics.json
      history.json
      training_curves.png
      validation_predictions.npz
      validation_sample_ids.json
      test_metrics.json              after frozen evaluation
      test_predictions.npz
      test_sample_ids.json
      run_config.json
      provenance.json
      dataset_identity.json
      split_ids.json
```

Saved graph objects are PyTorch dictionaries with node tensors and all three
edge-index tensors; the dataset converts them into PyG Data/Batch objects.
Checkpoints, audio and tensor caches are local artifacts; do not redistribute
raw dataset audio.

Macro-F1 averages all labels, including zero-support labels as zero.
Micro-F1 pools all decisions. mAP averages per-label **average precision** over
labels with evaluation positives; zero-support AP is null and excluded. mAP is
not trapezoidal PR AUC. Prediction archives contain IDs, label order, targets,
probabilities, decisions and the validation threshold.

Runtime reports distinguish training epochs (including validation/checkpoint
saving), test inference/data loading, preprocessing and total suite time.
Training times exclude audio preprocessing and gallery rendering. No fair
speed or accuracy advantage should be claimed from the tiny synthetic run.

Task 1's published full-cohort scores use 829 test clips. An audio subset requires
re-evaluation of text predictions on that same subset before cross-task numeric
comparisons. Tasks 3/4 have not been retrained by this Task 2 update.
