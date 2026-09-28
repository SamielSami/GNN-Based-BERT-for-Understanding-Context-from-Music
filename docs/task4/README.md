# Task 4: contrastive graph-text retrieval

Task 4 is implemented and measured on the same 3,964 paired MusicCaps clips as
Task 3. See [measured results](../../report/task4_results.md),
[completion audit](../../report/task4_completion_audit.md) and the
[executed review notebook](../../notebooks/task4_retrieval.ipynb).

New: [zero-shot comparisons and ten queries](../../report/task4_extensions.md),
[five-listener instructions](../../report/task4_listening_guide.md) and
[frozen-encoder scope deviation](../../report/task4_scope_deviation.md).
The full original proposal remains pending real listener ratings and resolution
of its encoder-update requirement; frozen projection training alone is narrower.

## Frozen encoders and objective

Reuse `data/processed/task3_available/features.pt`, produced and audited in
Task 3. Task 1's trained DistilBERT and Task 2's temporal GraphSAGE are frozen;
Task 4 never loads them into its optimizer. Their cached outputs are detached.
The CLS vector (768 dimensions) and graph readout (256 dimensions) each pass
through a linear projection into 64 dimensions, followed by L2 normalization.
Only these 65,664 projection parameters train. AudioSet labels do not enter the loss.

Each sample ID defines one positive caption/graph pair. Other pairs in the
training batch are negatives. Symmetric InfoNCE averages caption-to-graph and
graph-to-caption cross-entropy with temperature 0.07. A final singleton batch
is merged into the previous batch so every update has negatives.

## Reproduce the real-data suite

Run from the repository root in the configured Python environment, using a new
output directory. Prepare features using the [Task 3 guide](../task3/README.md)
if the local cache is unavailable. Large features and checkpoints are excluded
from Git and must be retained locally or transferred separately.

```powershell
python -m src.task4.experiment run --features data/processed/task3_available/features.pt --output-dir results/task4/my_run --seeds 42 43 44 --epochs 30 --batch-size 32 --projection-dim 64 --temperature 0.07 --learning-rate 0.001 --patience 5 --cpu-threads 6 --device cpu
python -m src.task4.experiment evaluate --output-dir results/task4/my_run --device cpu
python -m src.task4.analyze --output-dir results/task4/my_run
python -m src.task4.experiment verify --output-dir results/task4/my_run
```

The first command uses training pairs for gradients and validation pairs for
checkpoint selection/early stopping. Mean bidirectional validation Recall@1
selects the checkpoint; the earliest epoch wins ties. Every seed finishes before
`selection.json` freezes all checkpoints, initial projection weights, validation
artifacts and the suite declaration using SHA-256 hashes. The second command
checks those hashes before any test inference. It evaluates all declared seeds
and baselines once; it does not choose a seed using test results.

`verify` recomputes metrics from saved embeddings, checks their unit norms and
ordered split IDs, validates per-query ranks, frozen checkpoint/configuration
hashes and the first-best validation epoch, and checks both comparison CSVs.
It performs no new inference or training. Use `--features <relocated-cache>`
with evaluate/verify/analyze after moving an identical cache to another machine.

## Metrics and artifacts

The complete chosen split is both query set and gallery: 583 validation pairs or
606 test pairs. Report Recall@1/5/10, median rank and mean reciprocal rank for
both text-to-graph and graph-to-text. Recall is a fraction in JSON/CSV, a
percentage where explicitly marked in plots/reports. Effective K is clipped at
gallery size. Tied negatives rank before the paired positive; example rankings
use the same rule. Scoring uses bounded query blocks and float64 normalization.
Feature loading memory-maps the token cache and copies only compact CLS vectors.

The untrained baseline uses each seed's exact initial projection weights.
Random-ranking expected Recall@K is `min(K, N) / N` and expected MRR is
`sum(1/r for r in 1..N) / N`. This is an analytical expectation, not a sampled
random run. Mean/sample-standard-deviation summaries measure projection-seed
variation; encoders and data stay fixed across seeds.

- Suite: `suite_config.json`, `pairing_audit.json`, `selection.json`,
  `test_evaluation.json`, `comparison.csv`, `comparison_aggregate.csv`,
  `verification.json`.
- Each seed: `best_model.pt` (trained and initial heads), `config.json`,
  `initial_validation.json`, `history.json`, `timing.json`, `learning_curves.png`.
- Each seed/split: `metrics.json`, `embeddings.npz`, `untrained_embeddings.npz`,
  `ranks.json` and `examples.json` with IDs, captions, ranks and cosine scores.
- Analysis: comparison/learning plots, `cases.md` and `rank_summary.json`.

Recompute a single metric set independently:

```powershell
python -m src.task4.metrics --embeddings results/task4/available_run1/seed42/test/embeddings.npz
python -m src.task4.metrics --embeddings results/task4/available_run1/seed42/test/untrained_embeddings.npz
```

## Single-run API and offline check

The original single-run commands remain supported:

```powershell
python -m src.task4 train --demo --epochs 5 --device cpu --output-dir tmp/task4_demo_new
python -m src.task4 evaluate --checkpoint tmp/task4_demo_new/best_model.pt --features tmp/task4_demo_new.demo.pt --output-dir tmp/task4_demo_new/test --device cpu
python -m unittest discover -s tests/task4 -v
```

The random-feature demonstration tests execution only. Its six-pair held-out
gallery makes Recall@10 trivial; it is not evidence of music retrieval quality.
The review notebook reads real saved results by default and keeps synthetic
retraining opt-in. New raw queries require the same encoder/preprocessing stages;
the implemented retrieval interface operates on cached embeddings.

## Interpretation limits

The audit detects exact and normalized duplicate captions and repeated video
IDs; the completed cohort has none. Semantically similar music can still create
false negatives: an unpaired but suitable track is penalized by one-positive
InfoNCE and exact-pair retrieval metrics. Do not interpret a failed exact match
as proof of poor listening quality. Future work could audit semantic duplicates,
group truly equivalent pairs, or use multiple positives without consulting test
results to tune the current model.

The reused text encoder trained on 3,864 original training captions, including
1,089 without paired audio; the graph encoder trained on 2,775 paired clips.
Held-out IDs preserve the original partitions. Missing-audio selection bias,
coarse graph features, fixed classification-trained representations and lack of
artist-disjoint evaluation remain limitations. CLAP retrieval and a caption-tag
prompt-similarity comparison have now been measured; see the extension report.
Human listening is pending five real participants, and no subjective quality
claim is made. Forms and local audio are in `results/task4/listening_study/`.
