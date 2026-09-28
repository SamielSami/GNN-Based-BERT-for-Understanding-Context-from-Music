# Submission readiness audit against the assignment PDF

Audit date: 10 September 2026.

**11 September update:** The five-mode Task 3 `joint_run2` experiment completed
training and held-out test evaluation; all five models passed saved-result and
dataset verification (`evaluated_test: true`). Genre/mood t-SNE, curves, three
cases and failure analysis are generated. Both the joint review notebook and
the real audio/caption demo executed with zero errors. Mood coverage remains
limited to 12 Angry music annotations. `joint_run1` remains interrupted.
The missing frozen-head `training_curves.png` generation was fixed; all 86
repository tests passed, including 28 Task 3 tests (`tmp/task3_full_regression.log`).
See [joint results](task3_joint_results.md) and
[notebook verification](../results/task3/joint_run2/notebook_verification.json).
Dataset approval, the final report and submission packaging remain separate
requirements. The current blockers below exclude the resolved Task 3 items.

**Verdict: not ready for final submission.** The repository contains substantial,
measured work across Tasks 1-4, and the current tests and saved-result checks pass.
However, several explicit PDF deliverables are missing or differ from the
implemented scope. The existing task completion audits certify narrower
experiments; they do not establish compliance with the entire assignment.

Source of requirements: `CSE425_Project_GNN_BERT_Music_Context.pdf`, all nine
pages read and visually inspected. The stated deadline is **2 October 2026**.
This audit covers the current local workspace; it does not certify the contents
of the live GitHub repository or instructor approvals outside this workspace.

## Submission blockers

| Priority | Finding | Evidence | Required action |
|---|---|---|---|
| 1 | Final report PDF is missing. | PDF section 10, pp. 8-9 requires a 6-10 page NeurIPS/IEEE/ICML-style report. `report/README.md` still instructs the author to place it there; only task Markdown reports exist. | Write and render the final report with actual results, diagrams, methods, citations and limitations. |
| 1 | Dataset substitution lacks documented sign-off. | PDF p. 4 specifies GTZAN/FMA-small for Task 2 and FMA-medium/MagnaTagATune for Task 3. The measured runs use MusicCaps. `docs/scope.md` explicitly records instructor approval as pending. | Record any existing instructor approval, or obtain approval for the MusicCaps substitution; otherwise run the required dataset experiments. This audit cannot assume approval from completed training. |
| 2 | Required graph files are not included in tracked Git contents. | 3,964 processed `.pt` samples exist locally, but `.gitignore` excludes `data/processed/*` and all `*.pt`; `git ls-files '*.pt'` returns none. The 20-example gallery contains PNGs and an index, not a portable graph-sample bundle. | Package at least 20 loadable `.pt` or `.json` graphs with feature/edge schema and label mapping in the submission ZIP or explicitly tracked sample directory. |
| 2 | Final source/results package is not established. | The working tree contains modified and untracked source, tests, reports and result directories. No final project submission ZIP was found; the Task 1 source snapshot ZIP is not the complete project. | Include the current files in the selected repository commit or ZIP, provide model/data dependencies, and test the submitted copy. A Git remote being configured does not prove the current local work is published. |

## Requirement-by-requirement assessment

### Dataset and preprocessing: implemented with a scope qualification

- **Paired audio and text:** 3,964 available MusicCaps audio/caption pairs, with
  independent AudioSet label IDs. MusicCaps supplies both modalities; acceptance
  as a substitute for the task-specific datasets remains unresolved above.
- **Audio preprocessing:** 22.05 kHz mono, per-track peak normalization, 128-bin
  log-mel features plus chroma/MFCC features are documented and implemented.
- **Segmentation and graphs:** ten one-second segment nodes; temporal,
  temporal-plus-similarity and edge-count-matched random controls. The brief gives
  segment durations as examples. The top-k similarity and mean/max readout choices
  should be described as implementation choices rather than exact copies of the
  displayed threshold graph and mean-only readout.
- **Text:** DistilBERT tokenization with maximum length 128 is implemented.
- **Splits:** video-ID train/validation/test separation, training-only vocabulary
  and normalization, validation-only thresholds and model selection. Paired
  cohort: 2,775 train / 583 validation / 606 test. No artist-disjoint claim is
  supported; the reports disclose that limitation.

### Task 1: implemented and measured

- DistilBERT fine-tuning code, held-out Macro/Micro-F1, epoch curves and example
  predictions are present.
- Current independent-label run: `results/task1/audioset_cpu_20260907/`.
- The original MusicCaps caption-to-tag proxy run remains preserved and is
  explicitly identified as a lexical proxy, as allowed by Task 1's wording.
- The current 30-label independent-target experiment is a disclosed variation;
  do not present its results as top-50 MagnaTagATune results or mix its scores with
  the old near-perfect proxy scores.
- Task 1's full test set has 829 clips; its headline score must not be directly
  compared with Task 2's 606-clip test set as if the cohorts were identical.

### Task 2: implemented and measured on the substitute dataset

- Graph construction scripts, PyTorch Geometric GraphSAGE, mel-CNN and pooled
  MLP are present.
- Five configurations across three seeds give 15 saved checkpoints and
  validation/test prediction sets in `results/task2/available_run1/runs/`.
- Graph ablations, training curves and a 20-example real graph gallery exist.
- The specific GTZAN/FMA-small deliverable is not demonstrated by these runs.
- `notebooks/task2_gnn_cnn.ipynb` defaults to synthetic verification. The report
  and `available_run1` contain the real results; the final reviewer entry point
  should make this distinction clear.

### Task 3: frozen and joint ablations completed on the substitute dataset

- BERT-only, GNN-only, concatenation and cross-attention are measured; gated
  fusion is an additional comparison. All five heads have three seeds.
- Saved outputs, comparisons, curves and three graph/caption case studies exist
  under `results/task3/available_run1/`. The case figures show temporal paths and
  segment/token sensitivities; these are representation interventions, not causal
  alignment proofs.
- Joint training now updates GraphSAGE, the fusion head and the final transformer
  block. All five modes completed three epochs and passed held-out verification
  in `joint_run2`. Genre/mood t-SNE is also complete, with limited mood coverage.
- Both inference notebooks executed successfully, including the raw audio/caption
  demonstration. Their external input/checkpoint dependencies still need packaging.
- In the single-seed joint run, validation-selected gated fusion has test
  Macro-F1 0.4073 versus 0.3727 for BERT-only and 0.2700 for GNN-only. Its
  Micro-F1 remains below BERT-only; no statistical significance is claimed.
- The specific FMA-medium/MagnaTagATune results are not supplied.
- In the historical frozen experiment, BERT-only wins mean validation Macro-F1; gated fusion is the best fusion mode.
  Test Macro-F1 is 0.3850 for BERT-only and 0.3705 for gated fusion. Failure to beat
  BERT is an honest experimental outcome, not by itself a missing deliverable.
- Text encoder training used 1,089 additional training-only captions beyond the
  paired cohort. The report discloses this; characterize these as controlled head
  comparisons over reused encoders, not equally trained encoder comparisons.

### Task 4: useful retrieval experiment, not the full listed advanced task

Task 4 is marked optional/bonus in the rubric on p. 7. If claiming completion
of Task 4, its listed requirements still need to be addressed.

| Requirement | Status and evidence |
|---|---|
| Shared embedding space and InfoNCE training | Implemented with trained projection heads over frozen DistilBERT/GraphSAGE features. |
| Encoder updates in Algorithm 4, p. 7 | Not demonstrated: encoders remain frozen. Document acceptance of this variation or run the specified training. |
| Bidirectional Recall@1/5/10 table | Present for 606 held-out pairs, three seeds, with untrained/random controls. |
| Ten caption queries with top-three matched clips, p. 5 | Incomplete in the saved test examples: five caption queries and five reverse-direction queries per seed. There are only seven distinct caption queries across the three test seeds. Top-ten matches contain the top three, but the caption-query count is still short. |
| Zero-shot tag prediction versus Task 3 supervised model, p. 5 | No implementation/result deliverable found. This is distinct from an optional CLAP comparison. |
| At least five listeners rating matches from 1 to 5, p. 6 | Not performed; report explicitly disclaims listening evaluation. The PDF does not mark this subrequirement optional separately from Task 4 as a whole. |

Do not label Task 4 fully complete against the PDF without resolving these
items. Human ratings must come from actual participants; they cannot be inferred
from retrieval ranks or generated by the model.

### Baselines, metrics and optional extensions

- **At least two baselines:** satisfied by multiple implemented controls,
  including label-prior, text-only, CNN and pooled MLP comparisons. Use matched
  populations when making cross-model claims.
- **Macro-F1 and Micro-F1:** present with saved predictions and threshold policy.
- **AUC-PR terminology:** the joint Task 3 report now includes both mAP and
  separately calculated trapezoidal mean PR-AUC. The historical reports provide mean average precision and explicitly
  say this is not trapezoidal PR area. The brief asks for mean AUC-PR. Specify the
  chosen definition in the final report; for literal coverage, also compute and
  report trapezoidal PR-AUC from the same frozen predictions. Do not simply
  relabel AP as a different numerical metric.
- **Emotion regression:** optional when DEAM targets are available; its absence
  is not a core blocker for the stated non-DEAM scope.
- **Attention visualization for Task 1, graph coherence score and PCA+MLP:**
  explicitly optional; no completion claim is needed for omitted extensions.
- **Performance values in the PDF:** illustrative, not minimum required scores.

## Historical verification performed on 10 September

1. Read and rendered all nine assignment pages, inspected relevant source,
   task reports/audits, graph/case artifacts and notebook cell contents.
2. Ran `python -m unittest discover -s tests -v`: **77 tests passed**, 22.017 s.
   Log: `tmp/submission_audit_tests.log`.
3. Re-ran `python -m src.task1.evaluate --run-dir results/task1/audioset_cpu_20260907`:
   passed; saved IDs, targets, calibration and metrics reproduced for DistilBERT
   and its controls. Log: `tmp/submission_task1_verify.log`.
4. Recomputed validation/test aggregate and per-label metrics for all **15 Task 2
   runs**, rederived validation thresholds, checked saved classifications and
   identical IDs/targets/label order. Passed. Record:
   `tmp/submission_task2_verify.json`. This check did not repeat audio extraction
   or the full saved-graph integrity audit.
5. Re-ran `python -m src.task3.experiment --verify-run results/task3/available_run1`:
   passed. Checks include the feature fingerprint, frozen selections, prediction
   metrics/thresholds, identities and comparison tables. Log:
   `tmp/submission_task3_verify.log`.
6. Re-ran `python -m src.task4.experiment verify --output-dir results/task4/available_run1`:
   passed for validation/test retrieval, baselines, normalized embeddings,
   identities, checkpoint/artifact hashes and tables. Log:
   `tmp/submission_task4_verify.log`.

These are software and saved-evidence checks, not full retraining or a clean
installation test of a final submission archive. No experimental results were
fabricated or retrained during this audit. Existing verifier commands refreshed
their verification artifacts; model/source files were not edited.

## Recommended completion order

1. Resolve/document instructor acceptance of the dataset substitution and any Task 4 scope variations.
2. Package the executed real end-to-end demo with its dependencies plus 20 graph files.
3. Incorporate the completed genre/mood t-SNE and separately defined PR-AUC metrics into the final report.
4. Complete the remaining Task 4 deliverables if claiming full advanced-task credit.
5. Assemble the 6-10 page report with measured evidence and explicit limitations.
6. Correct the scope of completion claims and stale quick-start instructions;
   package/commit the final files and execute the demo from that delivered copy.

The project does not need to be restarted. The existing measured experiments
are useful evidence, but successful tests and task-specific completion notes do
not replace the missing submission artifacts or instructor-approved scope.
