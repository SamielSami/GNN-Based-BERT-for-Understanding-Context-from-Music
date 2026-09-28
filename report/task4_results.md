# Task 4 — measured contrastive graph–text retrieval

**Frozen-encoder exact-pair retrieval is measured on the available 3,964-clip
MusicCaps cohort; full proposal completion remains pending human ratings and
resolution of the encoder-training scope deviation.** See the
[extension results](task4_extensions.md), [listening guide](task4_listening_guide.md)
and [scope note](task4_scope_deviation.md). Three projection seeds were trained over the audited frozen
Task 3 features, selected using validation only, and evaluated on 606 held-out
pairs. Checkpoints, embeddings, ranks, baselines, curves and verification are in
`results/task4/available_run1/`. The experiment was run on 2026-09-10.

Contrastive training improves both retrieval directions over untrained
projections and random ranking. Absolute performance remains modest:
**text-to-graph Recall@10 is 17.77%; graph-to-text Recall@10 is 18.65%.**
Most paired items are outside the first ten results. These are exact-ID matching
measurements, not listener-rated relevance.

## Held-out comparison

Values are mean ± sample standard deviation over seeds 42, 43 and 44.
Recall columns are percentages; MRR is a fraction. Median rank is computed
within each seed before averaging. All rows use the same 606-query/gallery pairs.

| Direction | Model | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | Median rank | MRR |
|---|---|---:|---:|---:|---:|---:|
| Text → graph | Trained | 2.59 ± 0.10 | 10.62 ± 0.42 | 17.77 ± 0.91 | 52.17 ± 4.25 | 0.0800 ± 0.0015 |
| Text → graph | Untrained projections | 0.11 ± 0.19 | 0.44 ± 0.19 | 0.83 ± 0.17 | 314.83 ± 4.93 | 0.0091 ± 0.0018 |
| Graph → text | Trained | 2.81 ± 0.44 | 11.11 ± 0.42 | 18.65 ± 0.17 | 50.67 ± 1.44 | 0.0829 ± 0.0015 |
| Graph → text | Untrained projections | 0.17 ± 0.17 | 0.83 ± 0.33 | 1.71 ± 1.10 | 321.83 ± 12.79 | 0.0112 ± 0.0031 |
| Either direction | Random expectation | 0.165 | 0.825 | 1.650 | — | 0.011526 |

![Held-out comparison](../results/task4/available_run1/analysis/retrieval_comparison.png)

Random Recall@K is `min(K, N) / N`; expected MRR is the mean of reciprocals
`1, 1/2, ..., 1/N`. The random row is analytical, with no seed standard deviation.
It does not report an expected sample median. Untrained projections are each
seed's exact initial weights, evaluated on the identical gallery.

The text-to-graph Recall@10 gain over untrained projections is **16.94 percentage
points**; graph-to-text gains **16.94 points**. The standard deviations measure
projection initialization and batch-order variation with fixed encoders and
fixed samples. They are not confidence intervals or independent dataset trials.
No statistical significance or generalization beyond this cohort is claimed.

## Validation selection and training

The inherited split contains **2,775 train / 583 validation / 606 test** pairs.
Matching sample IDs define positives. Every other training-batch item is a
negative; AudioSet target labels do not enter this contrastive objective.

The text representation is the 768-dimensional CLS vector of Task 1's trained
DistilBERT. The graph representation is Task 2's 256-dimensional temporal
GraphSAGE readout. Both encoders were frozen during feature extraction and remain
outside the Task 4 optimizer. Detached cached features enter two linear
projections, each producing an L2-normalized 64-dimensional embedding.
There are **65,664 trainable parameters** across the two projection heads.

Settings were declared before final evaluation: symmetric InfoNCE, temperature
0.07, AdamW learning rate 0.001, weight decay 0.01, batch size 32, gradient-norm
clipping at 1.0, at most 30 epochs and patience 5. The final 23-item training
batch remains valid; singleton batches in other cohorts are merged into the
preceding batch. CPU execution uses six PyTorch threads.

Checkpoint selection maximizes mean bidirectional validation Recall@1, with
the earliest epoch breaking ties. All three seeds finished and their artifact
hashes were frozen in `selection.json` before the separate test command ran.
Every declared seed was evaluated; no test-based seed or hyperparameter
selection occurred.

| Seed | Best epoch | Stopped epoch | Validation mean R@1 (%) | Test text → graph R@1 (%) | Test graph → text R@1 (%) |
|---|---:|---:|---:|---:|---:|
| 42 | 6 | 11 | 3.602 | 2.640 | 2.970 |
| 43 | 9 | 14 | 3.516 | 2.475 | 2.310 |
| 44 | 12 | 17 | 3.859 | 2.640 | 3.135 |

Validation means for trained text → graph are **3.89% / 11.95% / 20.98%**
at R@1/5/10; graph → text gives **3.43% / 12.86% / 21.15%**.
Full per-seed validation and test values, including baselines, are preserved in
[comparison.csv](../results/task4/available_run1/comparison.csv) and
[comparison_aggregate.csv](../results/task4/available_run1/comparison_aggregate.csv).

![Training and validation curves](../results/task4/available_run1/analysis/learning_curves.png)

Training loss continues to fall while validation Recall@1 fluctuates. This
supports early stopping and does not justify selecting a later checkpoint using
test results. Projection training plus epoch validation/checkpoint saves took
1.78, 2.39 and 2.69 seconds for seeds 42, 43 and 44. These timers exclude feature
loading/validation, fingerprinting, initial-baseline scoring, plotting and final
evaluation, so they are not end-to-end runtimes or hardware benchmarks.

## Ranked examples and failure cases

The first three test IDs in cache order are fixed diagnostic queries, shown for
the first declared seed, 42. Both retrieval directions use all 606 candidates.

| Query ID | Caption description | Text → graph paired rank | Graph → text paired rank |
|---|---|---:|---:|
| `-4NLarMj4xU_30_40` | Female pop vocal, synths and piano | 200 | 175 |
| `-6pcgdLfb_A_110_120` | Male R&B/soul vocal, string sample and electronic beat | 48 | 68 |
| `-7wUQP6G5EQ_30_40` | Renaissance-style flute and wooden percussion | 322 | 402 |

The first caption retrieves a graph annotated as electrical-current experimental
music, a substantial mismatch in the provided descriptions. The second retrieves
a female-vocal pop track, retaining a broad vocal/music association but missing
the paired clip and several annotated attributes. For the third graph query, the
top caption describes a movie instrumental with a flute: there is an annotated
instrument overlap, while genre and arrangement differ. These interpretations
compare dataset captions; the returned audio was not independently listened to.

Outcome-selected examples are explicitly marked separately. A harmonica/acoustic
guitar clip (`4VbXYuCz4M0_30_40`) achieves text-to-graph rank 1, and a female-vocal
piano ballad (`1h2sb2xeCt8_30_40`) achieves graph-to-text rank 1. At the other
extreme, a video-game cover ranks 587 in text-to-graph retrieval, and a fast
drum/synth piece ranks 595 in graph-to-text retrieval. These best/worst examples
are diagnostics, not representative quality estimates.

[Full case evidence](../results/task4/available_run1/analysis/cases.md) includes
query captions and the first three returned captions, IDs and cosine scores.
Each seed's `validation/examples.json` and `test/examples.json` retain ten ranked
candidates per selected query. `ranks.json` retains the paired rank for every
query in each direction. Tied negatives are ranked ahead of the positive in both
metrics and examples; K is capped at gallery size.

## False negatives, duplicates and scope

The [pairing audit](../results/task4/available_run1/pairing_audit.json) checked all
3,964 unique IDs and captions. There are **zero exact duplicate-caption groups,
zero normalized duplicate-caption groups and zero repeated video IDs**.
Normalization uses Unicode NFKC, case folding, word tokens and whitespace
collapse. No video ID crosses train/validation/test.

This does not rule out semantic duplicates. Different captions may describe
very similar music, and different clips can satisfy the same query. One-positive
InfoNCE pushes such unpaired items apart; exact-ID Recall@K counts them as wrong
even when a listener might find them relevant. Broad genre/instrument overlap
in examples is not sufficient to relabel them as positives. A future training-only
semantic audit, verified multiple-positive annotations or a predefined listening
protocol could address this limitation. None was used to adjust these test scores.

The text encoder previously trained on 3,864 original training captions,
including 1,089 without available paired audio. The graph encoder trained on
2,775 paired clips. Both preserve the original held-out partitions. This is a
comparison of projections over reused encoders, not equally sized historical
encoder-training cohorts. The feature SHA-256 matches Task 3's audited cache:
`77c667b7312a6579ef17f0a29da5170781829a1a7849154b024794fa9d6a7513`.

Other limitations include missing-audio selection bias, coarse one-second graph
features, classification-trained frozen representations, no artist-disjoint
split and a modest gallery size. The implemented interface retrieves cached
items; new raw queries require the same preprocessing and encoder stages.
The **CLAP and caption-tag zero-shot comparisons are now measured** in the
[extension report](task4_extensions.md). **Human listening remains pending**,
so no subjective relevance or listening-quality claim is made.

## Deliverables and verification

- Three nonempty contrastive checkpoints contain selected and initial heads.
- Saved normalized embeddings cover validation and test for trained/untrained
  heads at every seed, with IDs for independent metric recomputation.
- Training-loss and validation-retrieval curves, aggregate comparisons, ranks,
  examples and failure diagnostics are saved with the experiment.
- [Verification](../results/task4/available_run1/verification.json) confirms
  checkpoint/configuration hashes, first-best validation selection, split IDs,
  unit embeddings, recomputed metrics/ranks and comparison tables.
- All **77 repository tests passed**, including **8 Task 4 tests** for ranks,
  gradients, detached inputs, label independence, checkpoint restoration,
  baseline comparison, split separation and artifact tampering.
- [Executed notebook](../notebooks/task4_retrieval.ipynb),
  [completion audit](task4_completion_audit.md) and
  [reproduction guide](../docs/task4/README.md) provide review and rerun paths.

The exact-pair, frozen-encoder experiment is delivered. See the extension report
for the new comparisons and ten queries; five human responses and resolution of
the original proposal's encoder-update requirement remain outstanding.
