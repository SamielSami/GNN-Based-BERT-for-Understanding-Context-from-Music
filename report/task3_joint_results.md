# Task 3: measured joint GNN-BERT fusion

**Training and held-out test evaluation are complete for all five models.**
The run is `results/task3/joint_run2`, seed 42, on 2,775 training, 583 validation
and 606 test MusicCaps clips with the same ordered 30-label AudioSet vocabulary.
The saved verifier reports `valid: true`, `runs_checked: 5` and
`evaluated_test: true`.

## Ablation comparison

| Model | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test mAP | Test mean PR-AUC |
|---|---:|---:|---:|---:|---:|
| BERT-only | 0.4602 | 0.3727 | 0.6739 | 0.3922 | 0.3751 |
| GNN-only | 0.2861 | 0.2700 | 0.5398 | 0.2706 | 0.2562 |
| Concatenation | 0.4326 | 0.3877 | 0.6810 | 0.4056 | 0.3884 |
| Gated fusion | 0.4496 | **0.4073** | 0.6563 | **0.4148** | **0.3986** |
| Cross-attention | 0.4104 | 0.3867 | 0.6503 | 0.3943 | 0.3777 |

Validation selects **BERT-only overall** and **gated fusion among fusion models**.
These selections were frozen before test inference. Gated fusion exceeds BERT-only
test Macro-F1 by **0.0347** and GNN-only by **0.1374**. It also exceeds both on
mAP and trapezoidal PR-AUC, but its Micro-F1 is lower than BERT-only. Thus fusion
improves the primary held-out metric in this run, not every metric. Test scores
do not replace the saved validation-based model selection.

This is one joint-training seed, with no claim of statistical significance.
The [earlier frozen experiment](task3_results.md) used three head seeds and is a
separate comparison; its scores must not be combined with these as repeated
joint-training runs. mAP uses average precision; mean PR-AUC uses trapezoidal
integration. Both average labels with positive support.

## Training and verification

All modes ran three epochs with batch size 8 and width-64 projections. Each
starts from the original Task 1/2 encoders and its seed-42 frozen head. The final
DistilBERT block, GraphSAGE and the active fusion/prediction head update through
live encoder forwards; the lower transformer blocks stay frozen. The unimodal
controls update their respective active branches.

Best epochs for BERT, GNN, concatenation, gated and cross-attention are
3, 2, 3, 1 and 1. Their validation-selected thresholds are 0.20, 0.10, 0.25,
0.15 and 0.20. Per-mode training proofs record nonzero active gradients and
parameter changes, plus matching hashes for the frozen lower text weights.
The complete saved suite passed checkpoint, vocabulary, sample-ID, split,
dataset-fingerprint, threshold and recomputed-metric verification.

Historical encoder pretraining cohorts differ: DistilBERT used 1,089 additional
training-only captions. All joint fine-tuning branches use the same paired cohort.
The model is DistilBERT plus temporal GraphSAGE; cross-attention uses a graph query,
text-token keys/values, a padding mask and a residual graph connection.

## Evidence and interpretation

- [Full comparison table](../results/task3/joint_run2/comparison.csv)
- [Saved verification](../results/task3/joint_run2/verification.json)
- [Validation selection](../results/task3/joint_run2/selection.json)
- [Failure analysis and per-label results](../results/task3/joint_run2/analysis/README.md)
- [Three fixed validation cases](../results/task3/joint_run2/analysis/cases.md)
- [Genre/mood t-SNE](../results/task3/joint_run2/semantic_plots/README.md)
- [Live encoder inference notebook](../notebooks/task3_joint.ipynb)
- [Raw audio/caption demonstration](../notebooks/demo_context.ipynb)
- [Training and evaluation instructions](../docs/task3/README.md)

Each model directory contains its full checkpoint, measured training/validation
curves, epoch history and prediction artifacts. Cases use the first three
validation IDs in processed-manifest order, independently of outcomes. The
analysis compares fusion and unimodal errors at their saved thresholds.
Attention weights are not presented as causal explanations.

Against BERT-only, gated fusion improves per-example F1 on **124** test clips,
worsens it on **159**, and ties on **323**. Against GNN-only the counts are
**286 / 85 / 235**. Its Macro-F1 improvement is therefore uneven across examples;
Macro-F1 averages labels, while these counts compare individual clips.

Genre/mood colors use explicit positive AudioSet annotations with overlap and
missing-annotation handling. There are 95 genre/style-annotated validation clips
and only 12 with the available Angry music mood annotation. Broad mood separation
is not established. Incomplete labels, missing-audio selection, coarse temporal
graphs and the absence of artist-disjoint splits limit generalization.

MusicCaps substitution approval remains undocumented. Completed experiments do
not resolve the assignment's dataset requirement or final submission packaging.
