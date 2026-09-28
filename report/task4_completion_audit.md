# Task 4 completion audit

**Revised verdict (2026-09-11): the frozen-encoder retrieval experiment is
measured, but full original-proposal completion remains pending five real
listener responses and resolution of the encoder-training scope deviation.**
CLAP, caption-tag comparison and ten distinct caption queries are now added.
See [extension results](task4_extensions.md) and [scope deviation](task4_scope_deviation.md).

| Requirement | Status | Evidence |
|---|---|---|
| Attempt after stable Tasks 1–3 | Complete | Task 3 completion audit, unchanged audited feature hash and 77 passing repository tests. |
| Freeze trained text and graph encoders | Complete | Audited trained DistilBERT/temporal GraphSAGE cache; detached features; encoders absent from the Task 4 optimizer. |
| Small projection heads and shared normalized space | Complete | Two linear heads; 768/256 inputs → 64 outputs; 65,664 parameters; unit embeddings verified. |
| Symmetric contrastive objective | Complete | Symmetric InfoNCE at temperature 0.07; paired sample IDs are positives and other batch items are negatives. |
| Both retrieval directions | Complete | All 583 validation and 606 test pairs used as queries and galleries. |
| Validation selection separate from test | Complete | Three seeds trained first; earliest best validation epoch and hashes frozen before the separate test command. |
| Recall@1/5/10 | Complete | Both directions, every seed, trained and untrained baselines; CSV means and sample standard deviations. |
| Median rank and MRR | Complete | Both directions in saved metrics and report. |
| Untrained projection baseline | Complete | Exact initial weights saved in each checkpoint and evaluated on identical galleries. |
| Random-ranking expectation | Complete | Analytical K/N Recall and mean reciprocal rank under uniform random ranking. |
| Multiple random seeds | Complete | Projection seeds 42, 43 and 44; frozen encoders shared. |
| Contrastive checkpoints | Complete | `seed42/best_model.pt`, `seed43/best_model.pt`, `seed44/best_model.pt`. |
| Training/validation curves | Complete | Per-seed `learning_curves.png` and combined analysis plot. |
| Saved embeddings and ranked examples | Complete | Trained/untrained NPZ files, all paired ranks and top-ten query examples per split/seed. |
| Failure cases and duplicate discussion | Complete | Fixed and outcome-selected diagnostics, caption/video audit and false-negative limitations. |
| Pretrained CLAP retrieval | Measured | Same 606 test pairs, pinned pretrained checkpoint, no local training. |
| Zero-shot caption tags vs Task 3 | Measured | Fixed CLAP text prompt-similarity baseline; aligned targets and saved supervised predictions. |
| Ten caption queries → top-three clips | Complete | Ten distinct fixed test queries for graph seed42 and CLAP. |
| Human listening | Pending five real responses | Blinded L01–L05 forms, audio and strict response-summary script prepared. |
| Original proposal encoder updates | Scope deviation | Projection-only training; no documented acceptance of the frozen-backbone adjustment. |

The held-out means are **2.59% / 10.62% / 17.77%** text-to-graph and
**2.81% / 11.11% / 18.65%** graph-to-text at Recall@1/5/10. Random expectations
are 0.165% / 0.825% / 1.650%. Improvement over baselines is measured, while the
low absolute exact-match recall remains a limitation.

Verification recomputes all 12 split/seed/baseline metric sets and checks
normalized embeddings, ordered sample IDs, checkpoint and artifact hashes,
first-best validation epochs, per-query ranks and aggregate tables. There are
no exact/normalized duplicate captions or repeated video IDs in the cohort.
Semantic false negatives remain possible.

The encoder/data qualifications from Task 3 carry over: DistilBERT's historical
training cohort includes 1,089 additional training-only captions, available
audio defines the paired subset, and the evaluation is not artist-disjoint.
Only projection initialization and batch ordering change across seeds.

- [Measured report](task4_results.md)
- [Executed notebook](../notebooks/task4_retrieval.ipynb)
- [Frozen selection](../results/task4/available_run1/selection.json)
- [Independent verification](../results/task4/available_run1/verification.json)
- [Full comparison](../results/task4/available_run1/comparison.csv)
- [Pairing audit](../results/task4/available_run1/pairing_audit.json)
- [Case evidence](../results/task4/available_run1/analysis/cases.md)
- [Run guide](../docs/task4/README.md)
