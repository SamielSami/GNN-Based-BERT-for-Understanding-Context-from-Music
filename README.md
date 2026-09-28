# GNN-BERT Music Context Understanding

Course project implementation for music-context prediction using audio structure graphs and text. The project supports multi-label genre/mood/tag classification, optional valence-arousal regression, and optional caption-audio retrieval.

> **Results policy:** all metric and plot directories start empty. Add only results produced by your own experiments.

## Pipeline

```text
                  MUSIC
                    |
          +---------+---------+
          |                   |
        AUDIO                TEXT
          |                   |
    Feature Extraction       BERT
          |                   |
     Graph Construction       |
          |                   |
         GNN                  |
          |                   |
          +---------+---------+
                    |
                  FUSION
                    |
                    v
          Music Context Prediction
          genre | mood | tags | emotion
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Place audio in `data/raw/`, then build a graph:

```bash
python -m src.task2.graphs --audio data/raw/example.wav --output data/processed/example.pt
```

Each processed sample is a PyTorch dictionary containing `graph`, `text`, `labels`, and optionally `emotion`. Create JSON split files that list sample paths, then run:

```bash
python -m src.train --config config.yaml --train-split data/splits/train.json --val-split data/splits/val.json
python -m src.evaluate --config config.yaml --checkpoint results/checkpoints/best.pt --split data/splits/test.json
```

## Milestones

Task 4 retrieval is measured on the same 3,964 paired clips with three projection seeds
over frozen encoders. Held-out Recall@1/5/10 is **2.59% / 10.62% / 17.77%**
for text-to-graph and **2.81% / 11.11% / 18.65%** for graph-to-text.
See the [measured report](report/task4_results.md),
[review notebook](notebooks/task4_retrieval.ipynb) and
[training/evaluation guide](docs/task4/README.md). [CLAP and zero-shot caption-tag
comparisons](report/task4_extensions.md) are now measured, with ten caption-query
examples. [Five-listener evaluation](report/task4_listening_guide.md) awaits real
responses; the [frozen-encoder scope deviation](report/task4_scope_deviation.md)
remains explicit.

Task 3's frozen experiment is complete on 3,964 paired MusicCaps clips: five heads,
three seeds, 15 verified validation/test runs, embedding plots and three
token/segment case studies. BERT-only wins validation selection; held-out mean
Macro-F1 is **0.3850** versus **0.3705** for validation-selected gated fusion.
See [the measured report](report/task3_results.md),
[executed review notebook](notebooks/task3_fusion.ipynb) and
[run guide](docs/task3/README.md). The five-model joint experiment is also
complete in `joint_run2`, including verified test evaluation, genre/mood t-SNE,
three cases and executed inference notebooks. Validation-selected gated fusion
achieved test Macro-F1 **0.4073**, compared with **0.3727** for joint BERT-only
and **0.2700** for joint GNN-only (one seed). See the
[joint results](report/task3_joint_results.md),
[joint review notebook](notebooks/task3_joint.ipynb) and
[raw audio/caption demo](notebooks/demo_context.ipynb).

Verify the completed run, or train a fresh run using the dedicated configuration:

```powershell
.venv/Scripts/python.exe -m src.task3.pipeline verify --config configs/task3.yaml
.venv/Scripts/python.exe -m src.task3.pipeline train --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline evaluate --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline verify --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline plot --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline analyze --output-dir results/task3/joint_run3
```

The configuration defaults to the completed `results/task3/joint_run2`; the
training commands above override that with `joint_run3`. Do not rerun `train` on an existing output folder;
use a fresh `--output-dir` consistently for another experiment. `joint_run1`
is incomplete. `config1` duplicates the legacy root `config.yaml`; neither is
the joint-training configuration. Content from `README1` is integrated in the
[Task 3 guide](docs/task3/README.md). Full assignment compliance still depends
on the scope and deliverables in the [submission audit](report/submission_readiness_audit.md).

- Task 1: text-only BERT baseline (`--model bert`)
- Task 2: GraphSAGE audio-graph baseline (`--model gnn`)
- Task 3: GNN-BERT fusion (`--model fusion`)
- Task 4: contrastive retrieval (`src/task4/`)

## Task 1: independent AudioSet text baseline

Task 1 now preserves MusicCaps `ytid`, splits by video ID before label selection,
fits its 30-label AudioSet vocabulary on training rows only, and tunes thresholds
on validation only. The executed
[`notebooks/task1_bert_baseline.ipynb`](notebooks/task1_bert_baseline.ipynb) reviews
the completed run, controls, per-label evidence, inference and optional retraining.

CPU fine-tuning of DistilBERT for two epochs on 3,864 training captions achieved
held-out **Macro-F1 0.3840, Micro-F1 0.5902 and mAP 0.3686** on 829 test IDs.
TF-IDF reached 0.3393 / 0.6530 / 0.3603 respectively. Saved predictions reproduce
the metrics; all models share the same prepared vocabulary and ID split.

```bash
python -m src.task1.data --hf-dataset --top-k 30 --output data/processed/musiccaps_audioset_new.csv
python -m src.task1.train --data-path data/processed/musiccaps_audioset_new.csv --output-dir results/task1/audioset_new --epochs 2 --device cpu --cpu-threads 6
python -m src.task1.evaluate --run-dir results/task1/audioset_new
```

Use fresh paths; preparation and training preserve existing artifacts. Independent
AudioSet labels replace caption-derived proxy targets for this run. The original
work remains in `Task1_test/`, earlier `results/task1/` outputs,
[`notebooks/task1_lexical_proxy.ipynb`](notebooks/task1_lexical_proxy.ipynb), and
[`docs/task1/legacy_proxy.md`](docs/task1/legacy_proxy.md).
See [the run guide](docs/task1/README.md) and [measured report](report/task1_results.md)
for methods, limitations and reproduction. Task 2's real-data manifest now
inherits this vocabulary and video split; the completed Task 2 and Task 3
comparisons use the available-audio cohort.

## Part 1: dataset viability audit

The planning roadmap's Part 1 audit is implemented separately from the model
task above. Reproduce the metadata audit and deterministic 200-ID, no-download
availability pilot with:

```bash
python -m src.task1.audit --hf-dataset --probe-count 200 --seed 42
```

The signed-off evidence is in `results/data_audit.json`, with the scope decision
and remaining instructor-approval gate in `docs/scope.md`. Availability probing
requires `yt-dlp` and is sensitive to time, region, network, and platform
throttling; technical errors are reported separately from unavailable clips.

## Task 2: graph and CNN audio baselines

Task 2 now has an aligned AudioSet manifest, audio-interval loading, cached
features, connected graph validation, a 20-example gallery, and one controlled
suite for GraphSAGE (temporal/similarity/random), pooled MLP and mel-CNN.
Checkpoints are bound to identical cached samples, labels and splits.

Real-data Task 2 is complete on the available 3,964-clip MusicCaps subset:
15 trained/evaluated model runs, 20 graph visualizations, curves and predictions.
Validation-selected temporal GraphSAGE achieved mean test Macro-F1 **0.2648**,
Micro-F1 **0.5729** and mAP **0.2762** across three seeds on 606 held-out clips.
See [the measured report](report/task2_results.md) for baseline comparisons and
limitations. The notebook's default synthetic demonstration remains separate.

Open [`notebooks/task2_gnn_cnn.ipynb`](notebooks/task2_gnn_cnn.ipynb) to review the
saved verification, or reproduce it in a fresh directory:

```powershell
python -m src.task2.experiment --demo --output-dir tmp/task2_demo_new --epochs 3 --hidden-dim 32 --batch-size 6 --device cpu --evaluate-test
```

See [the updated guide](docs/task2/README.md) for real-data commands and
[the completion audit](report/task2_results.md) for evidence and remaining work.

## Collaboration

Suggested branches: `member1/audio-graph`, `member2/bert`, and `member3/gnn-fusion`. Protect `main`; merge changes through pull requests after a reproducibility check.

## Data and ethics

Do not commit copyrighted dataset audio, credentials, or generated caches. Respect each dataset licence and document all preprocessing and split decisions to avoid artist leakage.
