# Task 1 — Measured independent AudioSet results

## Completed run

Run: `results/task1/audioset_cpu_20260907/`, 7 September 2026. All results below
were recomputed from saved held-out predictions by `python -m src.task1.evaluate`.
The original proxy experiment remains preserved in the historical section below.

| Model | Test Macro-F1 | Test Micro-F1 | Test mAP |
|---|---:|---:|---:|
| DistilBERT, 2 epochs | 0.3840 | 0.5902 | 0.3686 |
| label_prior | 0.0528 | 0.3530 | 0.0591 |
| ontology_keyword | 0.1568 | 0.2257 | 0.1158 |
| tfidf_logistic | 0.3393 | 0.6530 | 0.3603 |


DistilBERT improves Macro-F1 and mAP over these controls, while TF-IDF achieves
higher Micro-F1. This is one controlled comparison, not a significance claim.
The labels and population differ from the old proxy experiment, so the old
near-perfect F1 values must not be compared as if they measured the same task.

## Dataset and leakage controls

- Source: [official MusicCaps metadata](https://huggingface.co/datasets/google/MusicCaps),
  5,521 captions with original `ytid` retained. Captions alone enter the text model.
- Targets: the separately supplied `audioset_positive_labels` field, interpreted
  as multi-hot AudioSet IDs. No regex/aspect extraction supplies these targets.
  Official label names come from [AudioSet](https://research.google.com/audioset/download.html).
- Sorted unique video IDs are split 70/15/15, seed 42, before any label counting:
  3,864 train / 828 validation / 829 test. All 5,521 IDs are unique in this snapshot.
- Vocabulary: top 30 nonconstant labels with at least 20 training positives,
  ranked using training counts only. Label-ID order resolves ties. The selected
  training supports range from 66 to 3108.
- Keep all examples, including 188 / 35 / 34 rows with zero selected positives.
  A missing label is treated as zero for the operational task, although absence
  of positive annotation is not proof that an audio event is absent.
- The CSV hash is `1d5c892bd705d49be07d831655ea423e29f713ea7f3183c509172382b2996ead`. The saved manifest binds label order,
  train supports, video membership and row indices to that exact dataset.
- All classical feature/model fitting uses training data. Checkpoint and threshold
  selection use validation only. No test-driven rerun or threshold change was made.

## Model, training and selection

DistilBERT (`distilbert-base-uncased`) uses CLS pooling, dropout 0.1 and a linear
768-to-30 output head with sigmoid/BCEWithLogitsLoss. All encoder layers were
fine-tuned for two epochs, maximum length 128, batch size 16, encoder/head learning
rates 2e-5 / 1e-3, AdamW weight decay 1e-5, gradient clipping 1.0, 10% warmup and
linear decay. Minimum validation BCE selects epoch 2, whose validation loss
was 0.090148. Patience was 3; the two-epoch budget ended before early stopping.

The run used an Intel Core i5-12400 (6 cores, 12 logical CPUs), six PyTorch threads,
CPU-only PyTorch 2.13.0+cpu, and Intel UHD 730 graphics without CUDA. Recorded
training, calibration and evaluation time was **1461.17 seconds
(24.35 minutes)**. Dependency versions and input/source hashes
are recorded in `provenance.json`.

After checkpoint restoration, per-label thresholds maximize validation F1 over
the predeclared grid 0.05 to 0.95 in 0.05 steps. Higher thresholds break ties.
Labels without validation positives use the validation-selected global threshold;
there were 0 such selected labels in this run. Thresholds are saved in the
checkpoint and used by inference. All baseline thresholds are calibrated on the
same validation set before test inference. TF-IDF uses fixed unigram/bigram,
20,000-feature and C=1 settings, not a broad hyperparameter search.

Learning-curve F1 values use 0.5 throughout; final test F1 uses the saved
validation-calibrated vector. At the fixed 0.5 threshold, the same BERT test
probabilities give Macro-F1 **0.1999** and Micro-F1
**0.6791**. Calibration improves macro balance at the cost of Micro-F1;
this comparison is descriptive and did not alter the selected threshold vector.

## Metric definitions, uncertainty and errors

Macro-F1 averages all 30 label F1 values, using zero for undefined precision/recall.
Micro-F1 pools label decisions. mAP is the arithmetic mean of per-label average
precision, **not trapezoidal PR AUC**. Labels without evaluation positives have
null AP and are excluded from mAP; all 30 labels have positives in this test set.

For the fixed DistilBERT model and threshold vector, 1,000 bootstrap resamples of
test IDs (seed 2026) give 95% percentile intervals:

- Macro-F1: **0.3499–0.4112**.
- Micro-F1: **0.5709–0.6095**.

These intervals do not include training-seed or hyperparameter-selection
uncertainty. Each test ID occurs once, making row and ID resampling equivalent.
The verification artifact also includes intervals for the three controls.

The frequent `Music` label is influential: excluding it without refitting gives
descriptive Macro-F1 **0.3656**, Micro-F1
**0.3718**, and mAP
**0.3491** over the remaining 29 labels.
Exact match across all labels is **23.76%**. The error artifact
includes five highest and five lowest sample-F1 cases, missed/extra labels and
ontology-name-overlap strata. These are post-evaluation descriptions, not
additional model-selection experiments. Ontology-name overlap is affected by
the frequent `Music` label and does not isolate a causal shortcut effect.

## Per-label evidence

| AudioSet label | Train positives | Test positives | Test F1 | Test AP |
|---|---:|---:|---:|---:|
| Music (`/m/04rlf`) | 3108 | 679 | 0.9182 | 0.9342 |
| Musical instrument (`/m/04szw`) | 462 | 108 | 0.5354 | 0.5058 |
| Guitar (`/m/0342h`) | 304 | 80 | 0.6667 | 0.7210 |
| Plucked string instrument (`/m/0fx80y`) | 297 | 70 | 0.6269 | 0.6695 |
| Singing (`/m/015lz1`) | 235 | 54 | 0.3019 | 0.2367 |
| Percussion (`/m/0l14md`) | 152 | 40 | 0.4286 | 0.4375 |
| Wind instrument, woodwind instrument (`/m/085jw`) | 146 | 41 | 0.7500 | 0.8083 |
| Brass instrument (`/m/01kcd`) | 143 | 18 | 0.5909 | 0.5245 |
| Speech (`/m/09x0r`) | 117 | 24 | 0.2439 | 0.1741 |
| Drum (`/m/026t6`) | 108 | 27 | 0.3182 | 0.3036 |
| Song (`/m/074ft`) | 100 | 17 | 0.0930 | 0.0654 |
| Effects unit (`/m/02rr_`) | 97 | 19 | 0.4545 | 0.5242 |
| Keyboard (musical) (`/m/05148p4`) | 97 | 18 | 0.4615 | 0.3743 |
| Electric guitar (`/m/02sgy`) | 87 | 25 | 0.2778 | 0.3496 |
| Acoustic guitar (`/m/042v_gx`) | 85 | 22 | 0.2222 | 0.3967 |
| Electronic music (`/m/02lkt`) | 84 | 27 | 0.2222 | 0.1966 |
| Rock music (`/m/06by7`) | 82 | 26 | 0.3636 | 0.3611 |
| Music of Africa (`/m/0164x2`) | 78 | 9 | 0.0792 | 0.0705 |
| Pop music (`/m/064t9`) | 77 | 9 | 0.0235 | 0.0269 |
| Distortion (`/m/0g12c5`) | 77 | 21 | 0.5385 | 0.4294 |
| Bowed string instrument (`/m/0l14_3`) | 76 | 14 | 0.1905 | 0.1713 |
| Blues (`/m/0155w`) | 75 | 13 | 0.3333 | 0.3767 |
| Country (`/m/01lyv`) | 70 | 14 | 0.3448 | 0.2550 |
| Hip hop music (`/m/0glt670`) | 69 | 10 | 0.2727 | 0.3287 |
| Didgeridoo (`/m/02bxd`) | 68 | 13 | 0.9600 | 0.9291 |
| Angry music (`/t/dd00036`) | 68 | 17 | 0.4878 | 0.3832 |
| New-age music (`/m/02v2lh`) | 67 | 11 | 0.1111 | 0.0756 |
| Funk (`/m/02x8m`) | 67 | 15 | 0.0678 | 0.0848 |
| Traditional music (`/m/02p0sh1`) | 66 | 11 | 0.1600 | 0.0689 |
| Independent music (`/m/05rwpb`) | 66 | 17 | 0.4762 | 0.2753 |

## Reproducibility and remaining limits

`metrics.json`, `selection.json`, `thresholds.json`, `dataset_manifest.json`,
the checkpoint, and all validation/test `*_predictions.npz` arrays form the run's
evidence. The verifier checks saved IDs and targets against the hashed CSV,
rederives thresholds from validation probabilities, and exactly reproduces every
aggregate and per-label report. See [the run guide](../docs/task1/README.md) for
commands and [the notebook](../notebooks/task1_bert_baseline.ipynb) for exploration.

Independent label provenance removes deterministic caption-to-target generation;
it does not remove caption lexical shortcuts, sparse/incomplete annotations,
artist overlap, near-duplicate content or possible pretrained exposure. This is
a custom random ID split, not official AudioSet evaluation or a stratified
multilabel split. Only one training seed and two full fine-tuning epochs were
run; no claim of convergence, seed robustness, or statistically significant
superiority is made. There is no new frozen-encoder result. Existing Tasks 2–4
must be aligned to this manifest before cross-modal comparisons.

The former notebook and documentation are preserved as
`notebooks/task1_lexical_proxy.ipynb` and `docs/task1/legacy_proxy.md`. Original
checkpoints and previous measured results remain untouched.

---

## Historical lexical-proxy experiment (preserved; superseded for academic evaluation)

### Original report

### Setup

- Dataset: MusicCaps, 5,299 non-empty captions retained after proxy labeling
- Labels: top 50 genre, mood, instrument, and production tags
- Model: `bert-base-uncased` CLS representation plus dropout and a 50-output linear head
- Objective: binary cross-entropy with logits
- Split: 70% train, 15% validation, 15% test; seed 42
- Training: 10 epochs; threshold 0.5

### Results

| Metric | Test value |
|---|---:|
| Macro-F1 | 0.9894 |
| Micro-F1 | 0.9969 |
| Macro precision | 0.9995 |
| Macro recall | 0.9811 |

The experiment artifacts include the Macro/Micro-F1 learning curve,
`metrics.json`, five qualitative predictions, and the trained checkpoint under
`Task1_test/files/results/`.

### Interpretation and limitation

These values are valid outputs of the implemented lexical proxy experiment,
but they are not evidence of general music-context understanding. The target
tags were generated by deterministic keyword matching on the same caption used
as model input. BERT can therefore recover explicit lexical rules (for example,
the word "guitar" directly creates the `guitar` label). The random split has no
exact duplicate captions, but that does not remove this target leakage.

The experiment should be described as a pipeline verification baseline. A
stronger semantic evaluation would obtain tags from an independent annotation
source, mask the label terms from captions, split by clip/artist where possible,
or use held-out paraphrases. The current preprocessing also does not model
negation, so phrases such as "no vocals" may still create a positive `vocals`
proxy label.

### Deliverable mapping

- BERT fine-tuning: `src/task1/train.py`
- MusicCaps caption-to-tag preprocessing: `src/task1/data.py`
- Macro/Micro-F1 curve: `Task1_test/files/results/f1_curve.png`
- Five predictions: `Task1_test/files/results/example_predictions.json`
- Reusable inference: `src/task1/predict.py`
- Offline checks: `tests/task1/`
