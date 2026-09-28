# Part 1 - Scope approval and MusicCaps viability gate

Date: 29 August 2026  
Status: **conditional go for MusicCaps; instructor approval still required**

## Decision

Use MusicCaps as the provisional paired dataset for Tasks 1-3. The deterministic
200-ID pilot (seed 42) found 187 clips available and 13 private/unavailable:
**93.5% usable**, above the plan's 80% gate. The probe used yt-dlp metadata
extraction with `skip_download=True`; it did not download or save media.

The official metadata passed its structural checks: 5,521 unique 10-second clip
windows, no missing required fields, no duplicate video IDs, no duplicate clip
windows, and no exact duplicate captions. The detailed machine-readable evidence
is in `results/data_audit.json`; the 13 failures are recorded in
`results/musiccaps_availability_failures.jsonl`.

A full 5,521-ID probe was attempted after the pilot passed, but YouTube began
returning "confirm you are not a bot" responses after the first few hundred
requests. Those responses are platform throttling, not evidence that clips are
missing, so the run was stopped and its partial output was not used. The clean,
pre-throttle 200-ID pilot remains the gate result.

The route cannot be fully signed off until the instructor approves MusicCaps for
Tasks 1-3. Send this exact question:

> May we use MusicCaps as the paired dataset for Tasks 1-3 so every graph,
> caption, and label refers to the same 10-second clip, while keeping Task 4
> optional?

## Requirement checklist

| Proposal requirement | Part 1 interpretation | Status |
| --- | --- | --- |
| At least one audio dataset and one text/tag source | MusicCaps supplies aligned YouTube clip references, captions, aspects, and AudioSet labels | Conditional pass; audio rights remain separate |
| Task 1 BERT multi-label baseline | Caption-only input; top tag vocabulary; BCEWithLogitsLoss; Macro/Micro-F1 curves; five predictions | In scope |
| Task 2 GNN music graph and CNN comparison | Segment graphs from the same 10-second clips; temporal and similarity edges | In scope |
| Task 3 GNN-BERT fusion and ablations | BERT-only, GNN-only, concat, and advanced fusion on identical clip IDs | In scope |
| Task 4 contrastive retrieval | Frozen-encoder stretch only after the core gate | Optional |
| Correct splits and no leakage | Disjoint clip IDs; train-only vocabulary/statistics; validation-only thresholds; fixed test set | Required in Part 2 |
| At least two fair baselines | Label prior/keyword/TF-IDF, CNN, BERT-only, and GNN-only | In scope |
| Final artifacts | Repository/ZIP, 20 graph samples, plots/tables, 6-10 page report, demo notebook | In scope |
| No invented experimental results | Every reported value must trace to a saved run or audit artifact | Required |

## Measured metadata audit

| Check | Result | Gate implication |
| --- | ---: | --- |
| Rows / unique clip windows | 5,521 / 5,521 | Pass |
| Missing required values | 0 | Pass |
| Duplicate video IDs / clip windows / captions | 0 / 0 / 0 | Pass |
| Duration | exactly 10 s for every row | Pass |
| Caption authors | 10 | Record author-stratified errors; do not treat authors as labels |
| Balanced subset rows | 1,000 | Available for a smaller controlled experiment |
| AudioSet-eval rows | 2,858 | Preserve the flag; do not silently call it a MusicCaps test split |
| Unique canonical aspect phrases | 13,082 | Synonym consolidation is mandatory |
| Unique AudioSet label IDs | 346 | Map IDs to names before label selection |
| Rows with an exact aspect phrase in the caption | 5,387 (97.57%) | High lexical leakage risk |
| 200-ID availability pilot | 187/200 (93.5%) | Passes 80% gate |
| Decoded 22.05 kHz mono PCM estimate | 2.435 GB | Allow 10-20 GB with caches/checkpoints |

The label-frequency plot is
`results/plots/musiccaps_label_frequency.png`. Its top-30 counts are a global
**audit only**, not the final vocabulary. Part 2 must create the split first and
fit the label vocabulary, synonym map, normalization statistics, and thresholds
without using validation or test labels.

## Leakage decision

MusicCaps aspects and captions were written to describe the same clips, and the
audit found at least one exact aspect phrase in 97.57% of captions. Therefore:

1. BERT input is the `caption` field only.
2. `aspect_list`, AudioSet label fields, and any derived target columns are never
   concatenated into the input text.
3. The current keyword-derived Task 1 experiment remains a clearly labelled
   pipeline/proxy demonstration; its near-perfect F1 is not evidence of music
   understanding.
4. The final Task 1 report must include label-prior, keyword-overlap, and TF-IDF
   controls plus overlap-stratified results.
5. Prefer a documented canonical top-30 target vocabulary selected from the
   training split. Exclude recording-quality artifacts such as "low quality"
   unless the instructor explicitly wants production-quality tags.

## Scope freeze

- **Core:** MusicCaps metadata/audio viability; deterministic preprocessing;
  BERT-only, GNN-only, compact CNN, prior/keyword/TF-IDF baselines; concat
  fusion; Macro-F1, Micro-F1, and mAP; graph ablations; 20 graph samples; demo;
  final report.
- **Advanced comparison:** gated fusion or one-way graph-to-text
  cross-attention only after concat is stable.
- **Stretch:** contrastive retrieval only after the core table, tests, demo, and
  report draft are complete.
- **Out of scope for the core:** DEAM multitask regression, training a large
  audio-text encoder from scratch, and cross-dataset fusion without verified
  track-level identities.

## Legal and operational note

The MusicCaps metadata card identifies the dataset as CC BY-SA 4.0 and exposes
YouTube IDs/timestamps rather than redistributed audio. Referenced audio retains
separate rights and platform constraints. Do not commit or publish raw audio;
store only permitted local copies and document attribution. See the
[official MusicCaps dataset card](https://huggingface.co/datasets/google/MusicCaps/blob/main/README.md).

## Exit status and next action

| Gate | Status |
| --- | --- |
| Metadata integrity | Pass |
| 200-ID audio availability threshold | Pass (93.5% >= 80%) |
| License/attribution note | Written |
| Input/target ambiguity | Resolved, subject to final train-only taxonomy |
| FMA fallback | Not activated because the primary gate passed |
| Instructor approval | **Pending human action** |

After instructor approval, Part 2 may freeze split manifests and the train-only
label taxonomy. If approval is denied, or a permitted acquisition run later
finds less than 80% usable audio, switch immediately to FMA-small.

