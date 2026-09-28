# Task 3 completion audit

## Current update: 11 September 2026

The checklist below certifies the **frozen experiment**, not full assignment
compliance. Joint fine-tuning is implemented in `src/task3/joint.py`, with live
encoder forwards, active-branch gradient/update checks and fixed lower text
layers. The five-mode run `results/task3/joint_run2` has **completed training and
held-out test evaluation**. Verification passed for all five models with
`evaluated_test: true`. `joint_run1` remains an interrupted BERT run.
See the [joint results](task3_joint_results.md): validation-selected gated fusion
achieves test Macro-F1 **0.4073**, versus **0.3727** for BERT-only and **0.2700**
for GNN-only. This is a single-seed result; gated Micro-F1 is lower than BERT-only.

Genre/mood t-SNE is now generated and checked in
both `results/task3/available_run1/semantic_plots/` and
`results/task3/joint_run2/semantic_plots/`. The 583 validation clips include
95 with selected genre/style annotations and 12 with the available Angry music
mood annotation. Missing annotations and overlapping genres are explicit.
This limited mood coverage does not support claims about broad mood separation.

Joint-run curves, the five-model comparison, three fixed validation cases and
failure analysis are generated. Both `notebooks/task3_joint.ipynb` (five code
cells) and `notebooks/demo_context.ipynb` (three code cells) executed with zero
errors. The former reproduces three live-encoder validation predictions; the
latter reconstructs a graph from real audio and verifies its resulting fusion
prediction. Evidence: `results/task3/joint_run2/notebook_verification.json`.

The missing training-curve output was restored. All **86 repository tests**
passed, including **28 Task 3 tests**, after the joint-analysis tests were added
(`tmp/task3_full_regression.log`). `src/task3/pipeline.py` and `configs/task3.yaml` now exist; the run guide
contains matching commands. `README1` was integrated into that guide; `config1`
is identical to the legacy root configuration and is preserved.

MusicCaps substitution approval remains undocumented. Consult the
[submission audit](submission_readiness_audit.md) for the remaining project-wide
deliverables; successful training does not resolve instructor approval.

## Historical frozen-experiment audit

**Verdict: complete for the documented frozen DistilBERT–GraphSAGE experiment
on 3,964 available MusicCaps clips.** No required implementation or experiment
output is missing from `results/task3/available_run1/`.

This audit checked source code, actual cached features, checkpoint hashes,
saved predictions, report numbers and notebook outputs. The repository tests
were rerun: **73 passed**, including **19 Task 3 tests**. Both the original suite
and the later `my_run` suite passed metric/provenance verification without
retraining or new test inference.

| Requirement | Status | Evidence |
|---|---|---|
| Each caption and audio graph identify the same clip | Complete | All 3,964 cached IDs, captions, targets and splits agree with the source manifest; encoder and dataset hashes verified. |
| Shared IDs, labels, vocabulary and splits for both Task 3 branches | Complete | Every head uses the same 2,775 train / 583 validation / 606 test pairs and ordered 30-label vocabulary. See the encoder-training qualification below. |
| Text and graph representations | Complete | Frozen DistilBERT tokens `[3964,128,768]`; temporal GraphSAGE representations `[3964,256]`. |
| Compatible projections, concatenation and prediction head | Complete | Learned width-64 projections, concatenated representation, dropout and linear multi-label head trained with BCE loss. |
| BERT-only / GNN-only / concat / gated / cross-attention ablations | Complete | Five modes across seeds 42, 43 and 44; 15 nonempty checkpoints with matching saved hashes. |
| Graph-query-to-text-token attention and padding masks | Complete | Graph query, contextual token keys/values and padding mask implemented and tested. |
| Shapes, batching, masks and gradient tests | Complete | Tests check batch/single consistency, padded NaN invariance, zero padded gradients and gradient flow through active branches. |
| Validation-only model/threshold selection, then held-out test | Complete | Frozen suite selection, per-checkpoint validation thresholds, matching prediction IDs/targets and recomputed validation/test metrics for all 15 runs. |
| Macro-F1, Micro-F1 and mAP | Complete | Per-run and mean/sample-standard-deviation tables, plus per-label metrics. mAP uses average precision rather than trapezoidal PR area. |
| Training curves and embedding visualization | Complete | Training BCE and validation Macro-F1 curves for each seed; selected-fusion and per-seed t-SNE plots. |
| At least three qualitative cases and failure analysis | Complete | Three fixed validation cases, prediction errors, token/segment sensitivity figures, per-label AP/support and paired fusion/control comparisons. |
| No causal claims from attention weights | Complete | Cases use explicit representation interventions; limitations distinguish sensitivity from causal attribution. |
| Fusion inference notebook or demonstration | Complete | Notebook restores the selected gated head and predicts three real cached validation pairs with its saved threshold. |

## Result and scope

The validation-selected gated fusion has mean test Macro-F1 **0.3705**, compared
with **0.3850** for BERT-only and **0.2539** for GNN-only. Fusion therefore improves
over the graph control but not the text control on Macro-F1. The required
comparison is complete; obtaining a positive fusion result is not a completion
condition.

The text encoder is **DistilBERT**, permitted by the project's frozen-encoder
plan. Its earlier training used **3,864 captions**, including **1,089 additional
training-only captions** beyond the paired cohort. GraphSAGE and the Task 3
heads trained on **2,775 paired clips**. The Task 3 branch inputs and held-out
partitions are shared; the historical encoder training cohorts are not equal.
This is a disclosed comparison of heads over reused encoders, not a comparison
of encoders trained with equal numbers of examples. New raw audio/caption
inference requires the existing encoder/preprocessing stages; the delivered
notebook demonstrates inference from their cached paired representations.

The later `results/task3/my_run/` also contains all 15 trained/evaluated heads,
curves, analysis and basic JSON cases. Its dedicated `case_studies/` directory
has not been generated. The completed report and notebook use `available_run1`,
which contains the detailed case narratives and figures required for Task 3.

## Audit corrections and evidence

The only defects found in the completed deliverables were presentation issues:
literal question marks replacing `±` and interval dashes in the report, plus
singular/plural wording in two cases. These were corrected without changing
measurements. The notebook was refreshed to display the corrected report.

- [Measured Task 3 report](task3_results.md)
- [Executed inference notebook](../notebooks/task3_fusion.ipynb)
- [Full comparison](../results/task3/available_run1/comparison.csv)
- [Aggregate comparison](../results/task3/available_run1/comparison_aggregate.csv)
- [15-run verification](../results/task3/available_run1/verification.json)
- [Detailed cases](../results/task3/available_run1/case_studies/cases.md)
- [Failure analysis](../results/task3/available_run1/analysis/README.md)
- [Audit record](../results/task3/completion_audit.json)
- [Run instructions](../docs/task3/README.md)
