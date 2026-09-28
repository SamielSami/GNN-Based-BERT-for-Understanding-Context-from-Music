# Task 1 — Independent AudioSet text baseline

Task 1 now predicts the independently supplied `audioset_positive_labels` from
MusicCaps captions. The completed CPU run uses 5,521 rows, 30 training-selected
labels and disjoint video-ID partitions. Open
[`task1_bert_baseline.ipynb`](../../notebooks/task1_bert_baseline.ipynb) for the
executed walkthrough, metrics, label supports, inference, error cases and retraining.

## Measured held-out results

| Model | Test Macro-F1 | Test Micro-F1 | Test mAP |
|---|---:|---:|---:|
| DistilBERT, 2 epochs | 0.3840 | 0.5902 | 0.3686 |
| label_prior | 0.0528 | 0.3530 | 0.0591 |
| ontology_keyword | 0.1568 | 0.2257 | 0.1158 |
| tfidf_logistic | 0.3393 | 0.6530 | 0.3603 |


All models use the same 829 test IDs. DistilBERT improves Macro-F1 and mAP over
these controls; TF-IDF has higher Micro-F1. These independent-label scores are
not comparable to the old lexical-proxy experiment's near-perfect scores.
See [the result report](../../report/task1_results.md) for uncertainty and limitations.

## Data and selection protocol

1. Load official MusicCaps metadata and require nonempty `ytid`, `caption`, and
   explicit AudioSet labels. Model input is **caption only**. Aspect lists, IDs,
   label strings, author IDs and AudioSet split flags never enter the encoder.
2. Sort unique `ytid` values and assign a seeded random 70/15/15 partition
   **before parsing/counting/selecting labels**. All rows of one video stay together.
   Row order does not change ID membership. Cross-split exact caption duplicates
   fail validation. This is a custom random ID split, not the AudioSet benchmark split.
3. Fit label frequencies on the 3,864 training rows only. Keep the top 30 labels
   with at least 20 training positives, excluding any train-constant labels;
   frequency ties use label ID order. The `Music` root is not constant and remains
   selected. Apply this fixed vocabulary to 828 validation and 829 test rows.
4. Preserve all 5,521 rows, including 188/35/34 train/validation/test rows with no
   selected positive label. Serialize `ytid,text,<AudioSet IDs>` and a manifest
   containing ID and row splits, tag order, train supports and CSV SHA-256.
5. Fine-tune `distilbert-base-uncased`, CLS pooling, dropout 0.1, linear 768→30,
   sigmoid and unweighted BCEWithLogitsLoss. Use separate encoder/head learning
   rates (2e-5/1e-3), AdamW weight decay 1e-5, gradient clipping 1.0 and 10% linear
   warmup followed by decay. Select the checkpoint by minimum validation BCE.
6. Restore that checkpoint and maximize each label's validation F1 on the fixed
   grid 0.05, 0.10, …, 0.95. Ties favor the higher threshold. A label without
   validation positives falls back to the grid threshold maximizing validation
   Macro-F1. Save the full threshold vector in JSON and the inference checkpoint.
7. Fit TF-IDF/IDF and one-vs-rest logistic classifiers on training captions only;
   use unigrams/bigrams, min_df=2, max_features=20,000, C=1, liblinear and 1,000
   maximum iterations. Label prior uses training prevalence. Keyword matching
   uses fixed official AudioSet display-name aliases, without dataset-fitted rules.
   Calibrate each control on validation with the same threshold protocol.
8. Freeze all selections before test inference. Save full validation/test
   probabilities, targets, IDs and thresholds for every model. Test predictions
   are used only for reporting and descriptive error/uncertainty analysis.

The default grid is deliberately fixed, not optimized against held-out performance.
Two CPU epochs are a measured budget-limited run, not evidence of convergence.
The learning curves display F1 at 0.5; the final report uses calibrated thresholds.

## Reproduce

From the repository root, activate `.venv` and install `requirements.txt` if needed.
The defaults now prepare AudioSet labels. Use a **new** output path/directory;
preparation and training refuse to overwrite existing artifacts.

```powershell
python -m src.task1.data --hf-dataset --top-k 30 --min-train-support 20 --seed 42 --output data/processed/musiccaps_audioset_reproduction.csv
python -m src.task1.train --data-path data/processed/musiccaps_audioset_reproduction.csv --output-dir results/task1/audioset_reproduction --model-name distilbert-base-uncased --epochs 2 --batch-size 16 --max-length 128 --device cpu --cpu-threads 6 --patience 3 --seed 42
python -m src.task1.evaluate --run-dir results/task1/audioset_reproduction
python -m src.task1.predict --checkpoint results/task1/audioset_reproduction/best_model.pt --text "calm acoustic guitar with soft vocals" --top-k 5
python -m unittest discover -s tests -v
```

`--input-csv data/raw/musiccaps_official.csv` replaces `--hf-dataset` for the saved
official metadata. A changed metadata snapshot may yield a different CSV hash;
never reuse an old manifest with a changed CSV. CSV and raw data remain ignored
by Git, as do model checkpoints; they exist locally and are regenerable.

With CUDA available, use `--device cuda` and a predeclared larger epoch budget.
`--freeze-bert` is supported with encoder dropout disabled, but no new frozen
encoder result is claimed here. `--no-tune-thresholds` explicitly uses a fixed
threshold; the completed run uses validation calibration.

A prepared manifest is authoritative for partitions: `--split-path` must agree
with it exactly. Existing legacy text-only CSVs and aligned Task 3 CSVs remain
readable, but are marked `legacy_unverified`; they do not carry independent-label
provenance. The existing Tasks 2–4 artifacts have not been realigned or retrained.

## Notebook behavior

The notebook reviews `results/task1/audioset_cpu_20260907/`, uses the corresponding
AudioSet CSV, verifies label/threshold metadata, displays the measured curves,
shows per-label support and errors, and runs inference with saved thresholds.
Preparation, retraining and recomputation are opt-in. Retraining uses a fresh
timestamped directory, two epochs on CPU or ten on CUDA. Historical artifacts
are never substituted automatically if the new run is missing.

## Code and artifacts

| Location | Purpose |
|---|---|
| `src/task1/audioset.py` | ID-first split, train-only vocabulary, independent-label parsing, manifest integrity |
| `src/task1/data.py` | Preparation CLI; explicit `--label-source proxy` retains historical helper behavior |
| `src/task1/train.py` | Fine-tuning, validation selection/calibration, full prediction export |
| `src/task1/baselines.py` | Label-prior, fixed ontology keywords, train-fitted TF-IDF controls |
| `src/task1/predict.py` | Legacy/current checkpoint support and saved per-label threshold inference |
| `src/task1/evaluate.py` | Recompute every report, audit IDs/targets/thresholds, bootstrap and error analysis |
| `src/task1/audit.py` | Existing metadata/availability audit, preserved |
| `results/task1/audioset_cpu_20260907/` | Completed run and evidence |

The run directory includes `metrics.json`, `best_model.pt`, `run_config.json`,
`dataset_manifest.json`, `split_indices.json`, `thresholds.json`, `selection.json`,
`history.json`, `f1_curve.png`, `example_predictions.json`, `provenance.json`,
`verification.json`, `error_analysis.json`, all `*_predictions.npz` arrays, and
serialized classical models. `mean_auc_pr` is retained as a legacy field alias
for **mean average precision**, not trapezoidal area under the PR curve. Macro-F1
includes all labels; AP is null for labels without evaluation positives and those
labels are excluded from mAP. `ap_label_count` records the denominator.

`source_snapshot.zip` preserves the exact run-time source files whose hashes
appear in `provenance.json`; that file also records the pretrained model revision,
checkpoint hash and raw metadata hash. Later CLI-description edits do not alter
the archived training implementation. `training.log` preserves the process log.

`predicted_tags` returns all labels meeting their own thresholds. `top_k` limits
only the separate ranked `top_tags` display. Results retain canonical AudioSet
IDs and add official display names.

## Preserved historical work

- [Original notebook, preserved](../../notebooks/task1_lexical_proxy.ipynb)
- [Previous guide, preserved](legacy_proxy.md)
- `Task1_test/`: original BERT-base proxy checkpoint and outputs, unchanged.
- `results/task1/` and `results/task1/notebook_run/`: earlier outputs, unchanged.
- The prior source/tests/docs were backed up to
  `tmp/task1_before_academic_20260907_112845/` before edits.

The legacy proxy helper is retained for compatibility and explicitly labeled as
such. Its vocabulary selection does not implement the new academic protocol.

## Scope and attribution

MusicCaps publishes captions and AudioSet positive labels separately; the new
targets are read directly from the latter field rather than generated from text.
See the [official MusicCaps card and CC BY-SA 4.0 terms](https://huggingface.co/datasets/google/MusicCaps).
The checked-in ID-to-name mapping comes from the
[official AudioSet class CSV](https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv),
linked by the [AudioSet download page](https://research.google.com/audioset/download.html).
Only metadata was fetched for this task; audio was not required.

This remains caption-based audio-event classification with sparse annotations,
frequent-label imbalance and lexical shortcuts. It is not a validated general
mood/genre understanding benchmark or evidence of listening to audio. The split
protects video IDs, not artists, authors, near duplicates or possible pretrained
model exposure. Single-seed uncertainty and incomplete AudioSet negatives remain
limitations. Future multimodal comparisons must align all branches to this exact
ID manifest and training-fitted vocabulary.
