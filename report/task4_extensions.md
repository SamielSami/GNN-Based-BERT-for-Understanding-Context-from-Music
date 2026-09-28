# Task 4 extension results and outstanding human work

Updated 2026-09-11. This supplements the original frozen-head retrieval report
and corrects its earlier broad completion claim. Zero-shot comparisons and ten
caption queries are now measured/generated. **Human evaluation remains pending
five real listeners; encoder fine-tuning remains a documented scope deviation.**

## Pretrained CLAP retrieval comparison

The fixed `laion/clap-htsat-unfused` checkpoint at revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` was evaluated without local training,
prompt search or checkpoint selection. All 606 test IDs and captions match the
original graph retrieval gallery in the same order. The same local audio
intervals are loaded as mono, resampled to 48 kHz and peak-normalized. This is a
pretrained **audio–text comparator**, not an alternative graph architecture.
See the [official model card](https://huggingface.co/laion/clap-htsat-unfused)
and [Transformers CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap).

| Direction | System | R@1 (%) | R@5 (%) | R@10 (%) | Median rank | MRR |
|---|---|---:|---:|---:|---:|---:|
| Caption → graph | Trained graph projections, three-seed mean | 2.59 | 10.62 | 17.77 | 52.17 | 0.0800 |
| Caption → audio | CLAP zero-shot | 13.70 | 32.67 | 45.38 | 13 | 0.2380 |
| Graph → caption | Trained graph projections, three-seed mean | 2.81 | 11.11 | 18.65 | 50.67 | 0.0829 |
| Audio → caption | CLAP zero-shot | 11.39 | 31.52 | 40.92 | 16 | 0.2178 |

CLAP substantially outperforms the current frozen graph representation on these
exact-pair metrics. This comparison changes both backbones and external training
data; it cannot isolate the effect of graph structure. CLAP has one fixed
checkpoint, so there is no invented three-seed standard deviation for it.
Zero-shot means no project-specific parameter updates. External pretraining
overlap with these clips has **not** been ruled out.

The reused retrieval metric code calls the second modality `graph` internally.
The CLAP artifact explicitly maps that key to **CLAP audio**, so it must not be
interpreted as a GNN embedding. Metrics were independently recomputed from the
saved 512-dimensional embeddings. Input audio hashes, ordered IDs, settings,
model revision and artifact hashes are retained in
[`results/task4/clap_zeroshot/`](../results/task4/clap_zeroshot/verification.json).

## Zero-shot caption tags versus supervised Task 3

The proposal separately asks for zero-shot tag prediction **from captions**.
We evaluate a declared CLAP **text-to-text prompt-similarity baseline** on the
same 606 captions and 30 AudioSet targets, without local training or validation
calibration. For each label name, embed two fixed prompts:

- `This music contains {label}.`
- `This music does not contain {label}.`

The score is the sigmoid of the positive-minus-negative cosine-similarity margin.
A fixed threshold of 0.5 predicts positive when the margin is nonnegative. These
are bounded similarity scores, not calibrated probabilities. Prompt wording and
the threshold were fixed before scoring; no test-driven prompt search occurred.

| Method | Macro-F1 | Micro-F1 | mAP |
|---|---:|---:|---:|
| Zero-shot caption prompt similarity | 0.1090 | 0.1530 | 0.1025 |
| Task 3 supervised gated fusion, three-seed mean | 0.3705 | 0.6466 | 0.3995 |

The simple zero-shot tag baseline performs poorly. CLAP was contrastively
trained for audio/text alignment, not textual entailment, and its representation
may not distinguish the negated prompt reliably. This is **not** a supervised
NLI classifier, nor CLAP audio-tag classification. Its weakness is not evidence
that all zero-shot tagging methods fail. The supervised comparator also has
audio and project training labels, so this is a task-level reference comparison,
not an equal-information architecture ablation.

Task 3 predictions were reused without new inference or threshold tuning.
Their IDs, targets and recomputed metrics match the original artifacts. Saved
zero-shot prompts, embeddings, scores, targets and per-label metrics are in
[`zeroshot_tags/`](../results/task4/zeroshot_tags/verification.json), with the
[per-seed comparison](../results/task4/zeroshot_tags/comparison.csv).

## Ten queries and five-listener study

[Ten distinct caption queries](../results/task4/listening_study/ten_caption_queries.md)
now show the top-three clips for graph seed 42 and CLAP, including IDs,
similarity scores and paired ranks. The queries are the first ten unchanged
test IDs, selected by order rather than retrieval quality.

The blinded package has **60 caption/clip judgments per listener** and five
assigned forms, L01–L05. Every model contributes three clips per query. Method
names, ranks, source IDs and candidate captions are hidden; order is shuffled
separately for each listener. Ratings use the proposal's 1–5 scale. Analysis
averages within query and listener before reporting the mean/sample standard
deviation across five listeners.

**No human ratings have been collected.** The user confirmed there are none.
The generated forms and unit-test responses do not count as listener evidence.
Follow the [step-by-step listening guide](task4_listening_guide.md) and return
the five downloaded JSON files. The summary script rejects incomplete responses;
the coordinator must also confirm that five different humans participated.

## Frozen-encoder scope

The later plan and implementation request specify frozen encoders, while the
original PDF's Algorithm 4 updates encoders with the contrastive loss. The
current measured result is **projection-only contrastive training over previously
supervised backbones**. It does not satisfy a strict joint-fine-tuning requirement.
[The scope note](task4_scope_deviation.md) records this difference and its effect
on claims. Acceptance of the narrower scope is not documented; otherwise a
separate encoder fine-tuning experiment remains required.

## Reproduce and verify

Use fresh output directories for a new model run; preserve the original test
results and never tune against them.

```powershell
python -m src.task4.zeroshot --output-dir results/task4/clap_new
python -m src.task4.zeroshot_tags --clap-dir results/task4/clap_new --output-dir results/task4/tags_new
python -m src.task4.listening prepare --clap-embeddings results/task4/clap_new/embeddings.npz --output-dir results/task4/listening_new
python -m unittest discover -s tests/task4 -v
```

The CLAP download and CPU inference actually ran. Eleven Task 4 tests pass,
including fixed-query pairing, blinded form construction, complete five-response
validation and rejection of mismatched galleries. Human results remain the only
part that requires participants, alongside the separate scope-acceptance issue.
