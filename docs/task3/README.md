# Task 3: joint and frozen GNN-BERT fusion

Task 3 compares BERT-only, GNN-only, concatenation, gated fusion and graph-query
cross-attention on identical paired samples. See [the measured report](../../report/task3_results.md)
and [review notebook](../../notebooks/task3_fusion.ipynb) for the completed real-data run.

The [joint experiment](../../report/task3_joint_results.md) is also complete:
all five models trained and passed held-out test verification in `joint_run2`.
Review its status with `.venv/Scripts/python.exe -m src.task3.pipeline verify`.

## Joint training: current entry point

The joint runner forwards captions and normalized audio graphs through the live
encoders on every batch. It updates the final DistilBERT block, GraphSAGE and the
fusion/prediction head. Lower transformer blocks remain frozen. BERT-only and
GNN-only update their respective active branches; all five modes use the same
paired training/validation IDs and ordered 30-label vocabulary.

To train another experiment, run these commands from the project root in
PowerShell. The fresh `joint_run3` destination preserves the completed run:

```powershell
.venv/Scripts/python.exe -m src.task3.pipeline check --config configs/task3.yaml --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline smoke --config configs/task3.yaml
.venv/Scripts/python.exe -m src.task3.pipeline train --config configs/task3.yaml --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline evaluate --config configs/task3.yaml --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline verify --config configs/task3.yaml --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline plot --config configs/task3.yaml --output-dir results/task3/joint_run3
.venv/Scripts/python.exe -m src.task3.pipeline analyze --config configs/task3.yaml --output-dir results/task3/joint_run3
```

`configs/task3.yaml` declares all five modes, seed 42, three epochs, batch size 8,
width 64 and six CPU threads. It initializes the encoders from Task 1/2 and each
head from its existing seed-42 frozen checkpoint. This is continued fine-tuning,
not training from random initialization. Epochs, thresholds and the winning mode
are selected using validation Macro-F1. `evaluate` runs the predeclared test
comparisons only after every mode completes; it refuses repeated test evaluation.
Use `verify` to review an evaluated run without rerunning test inference.

The configuration's default destination is the completed `results/task3/joint_run2`.
The commands above override it with `joint_run3`. Training requires an
empty destination and does not resume partial runs. To repeat the experiment,
use `--output-dir results/task3/joint_run3` on **every** command, or change the
YAML's `output_dir`. Existing `joint_run1` is an interrupted run; its BERT
checkpoint alone does not establish completion of the five-mode experiment.
An existing `selection.json` plus a successful `verify` with `evaluated_test:
true` establishes completion of training and evaluation.

Each mode saves a complete encoder/head `best_model.pt`, `history.json`,
`training_curves.png`, `training_proof.json` and validation/test predictions and
metrics. The root contains a tokenizer, immutable selection/provenance records
and `comparison.csv`. Metrics include Macro-F1, Micro-F1, mAP and separately
defined trapezoidal mean PR-AUC. Gradient/update checks record active encoder
changes and verify that lower frozen transformer weights stay unchanged.

`analyze` produces an ablation figure, per-label and paired failure comparisons,
and three fixed validation cases under `analysis/`. Open
[`notebooks/task3_joint.ipynb`](../../notebooks/task3_joint.ipynb) to review these
artifacts and run live encoder inference on three real caption/graph pairs. It
restores the full selected fusion checkpoint and checks its predictions against
the saved validation results. Both `plot` and `analyze` require fresh destination
subfolders; for alternative destinations, use their module CLIs with
`--output-dir`. They read saved predictions and do not repeat test inference.

[`notebooks/demo_context.ipynb`](../../notebooks/demo_context.ipynb) starts one
step earlier: it loads the first validation clip's local audio, rebuilds segment
features and graph edges with the saved training normalization, tokenizes its
caption, and runs the selected joint fusion model. It checks graph and prediction
agreement with the saved validation artifacts. The local WAV file referenced by
the manifest is required; this demonstration does not download audio.

`config1` and root `config.yaml` contain identical legacy generic-runner settings;
they are not compatible with the dedicated Task 3 runner. Their hardcoded
`num_labels: 10` must not replace this experiment's manifest-derived vocabulary.
The Task 3 guidance from `README1` is merged into this guide; the root README
links here. Original import files are preserved.

## Genre and mood embedding views

`pipeline plot` creates validation t-SNE views for the validation-selected joint
fusion model. For the completed frozen experiment, the generated views are in
`results/task3/available_run1/semantic_plots/`. Reproduce them in a fresh folder:

```powershell
.venv/Scripts/python.exe -m src.task3.semantic_plots --frozen-run results/task3/available_run1 --output-dir results/task3/available_run1/semantic_plots_new
```

Colors use the explicit AudioSet mapping in `src/task3/semantic_label_map.json`.
There are 95 genre/style-annotated validation clips and only 12 with the available
Angry music mood annotation. Missing annotations are displayed explicitly;
overlapping genres also have individual panels. These views do not establish
broad mood discrimination. Coordinates, IDs, labels and mapping hashes are saved.
The assignment's MusicCaps dataset substitution still needs documented
instructor acceptance; completed training cannot supply that approval.

## Data and encoder protocol

The available MusicCaps cohort contains 3,964 clips: 2,775 train, 583 validation,
606 test. It inherits Task 1's video-ID partition and 30 independent AudioSet
targets. Targets are not extracted from caption words. Missing audio introduces
selection bias, and incomplete AudioSet annotations are operationally treated as
negative labels. Artist-disjoint generalization is not established.

The frozen text encoder is the Task 1 fine-tuned DistilBERT checkpoint. Its 3,864
training captions include the paired training cohort plus training-only clips
without usable audio; none of the paired held-out IDs belongs to its training
split. The frozen graph encoder is Task 2's seed-42 temporal GraphSAGE, whose
architecture was selected using mean validation Macro-F1. It trained on the
2,775 paired training clips. Both encoders are reused unchanged across all heads
and seeds. This is a comparison of heads conditional on these trained encoders;
seed variation does not include encoder retraining.

Preparation checks checkpoint/dataset hashes, exact labels and order, sample IDs,
captions, split assignments, graph cache fingerprint and training normalization.
A relocated graph manifest is accepted only when its cached-dataset fingerprint
matches the checkpoint. Keep training CSVs, split metadata and caches with the
checkpoints. Legacy checkpoints without immutable split hashes still rely on
preserving their original split metadata.

## Reproduce the frozen real-data run

Run from the project root in the project's Python environment. These commands
need the completed Task 1/2 local artifacts; use fresh output paths when rerunning.
No API key or audio download is required.

```powershell
python -m src.task3.prepare --processed-manifest data/processed/task2_available/manifest.json --source-manifest data/splits/task2_available.json --text-checkpoint results/task1/audioset_cpu_20260907/best_model.pt --graph-checkpoint results/task2/available_run1/runs/seed42/gnn_temporal/best_model.pt --output data/processed/task3_available/features.pt --batch-size 4 --cpu-threads 6 --device cpu
python -m src.task3.experiment --features data/processed/task3_available/features.pt --output-dir results/task3/available_run1 --seeds 42 43 44 --epochs 20 --batch-size 16 --hidden-dim 64 --patience 5 --learning-rate 0.001 --cpu-threads 6 --device cpu
python -m src.task3.experiment --evaluate-run results/task3/available_run1 --device cpu
python -m src.task3.experiment --verify-run results/task3/available_run1
python -m src.task3.analyze --run-dir results/task3/available_run1
python -m src.task3.cases --checkpoint results/task3/available_run1/seed42/cross_attention.pt --features data/processed/task3_available/features.pt --processed-manifest data/processed/task2_available/manifest.json --graph-checkpoint results/task2/available_run1/runs/seed42/gnn_temporal/best_model.pt --output-dir results/task3/available_run1/case_studies --device cpu
```

The suite freezes mode selection using mean validation Macro-F1 before evaluating
any test predictions. It evaluates all five predeclared controls with their saved
thresholds. Test scores never select an architecture, epoch, threshold or seed.
The selected fusion mode and the overall winner are recorded separately.

The standalone runner remains available for a single seed:

```powershell
python -m src.task3 --features data/processed/task3_available/features.pt --epochs 20 --output-dir results/task3/single_seed_new --seed 42 --device cpu
```

## Architecture and resources

```text
caption -> frozen DistilBERT -> contextual tokens [128,768] + padding mask
audio   -> frozen temporal GraphSAGE -> mean/max graph embedding [256]
                                      |
                learned projections to width 64
                                      |
      text CLS | graph | concatenate | sigmoid gate | cross-attention
                                      |
               dropout + linear multi-label head
                                      |
             BCE loss / validation threshold selection
```

Cross-attention uses one projected graph query and projected text tokens as keys
and values, masks padding, and adds the attended output to the graph projection.
The other text heads use CLS only. Encoders remain frozen; projection and head
parameters train with AdamW, gradient clipping and early stopping.

Preparation writes tokens through a disk-backed array; training reads the cache
with memory mapping and validates finite features in small chunks. The float32
token cache is about 1.6 GB. Allow additional disk space for the temporary array
during preparation. CPU training defaults to six threads. Head parameter counts
exclude encoders; reported head training time excludes encoder extraction.

## Evidence and outputs

- `comparison.csv` and `comparison_aggregate.csv`: all model/seed measurements,
  sample standard deviations, active head parameters and measured training time.
- `selection.json`: validation-only decision, checkpoint hashes and feature hash.
- Each `seed*/`: five checkpoints, epoch histories, validation curves, saved
  validation/test predictions and per-label metrics, selected-head t-SNE.
- `verification.json`: recomputed metrics, validation thresholds and provenance.
- `analysis/`: measured comparison plot, per-label AP/support and paired failure
  analysis from saved predictions.
- `case_studies/`: three predeclared validation examples for seed-42
  cross-attention, including token and graph-segment perturbation sensitivity.

Cases are advanced-head diagnostics even when another mode wins. Setting a
contextual token vector or normalized graph node to zero measures sensitivity to
that representation intervention. It does not establish raw-word, audio-event or
causal importance. Graph interventions preserve nodes and edges. t-SNE is
descriptive. Interpret false positives/negatives in light of incomplete labels.

## Offline checks

Synthetic fixtures require no audio or model download and are never research
results. The review notebook keeps them opt-in.

```powershell
python -m src.task3 --demo --epochs 2 --device cpu --output-dir tmp/task3_demo_new
python -m unittest discover -s tests -v
```
