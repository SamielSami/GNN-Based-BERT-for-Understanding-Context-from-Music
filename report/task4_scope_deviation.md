# Task 4 frozen-encoder scope and proposal compliance

## What was implemented

The current contrastive experiment reuses Task 1's trained DistilBERT and
Task 2's trained temporal GraphSAGE. Their cached outputs are detached. Only
two linear projection heads, with 65,664 parameters in total, receive InfoNCE
gradients. The encoders were trained previously for supervised classification;
they are **not fine-tuned jointly for retrieval**.

This matches the later project plan's frozen-encoder stretch scope and the
explicit Task 4 implementation request. It supports the claim that contrastive
projection learning improves alignment between these fixed representations.
It does not establish the benefit of contrastively updating the GNN or BERT,
nor is it a from-scratch dual-encoder experiment.

## Difference from the original proposal

`CSE425_Project_GNN_BERT_Music_Context.pdf`, page 7, Algorithm 4 explicitly says
“Update encoders to minimize” the contrastive loss. Frozen-backbone projection
training is a **scope deviation from that algorithm**. The original proposal's
Task 3 algorithm likewise includes encoder backpropagation, whereas the historical
`available_run1` Task 3 comparison reused here uses frozen encoders. This statement
does not assess any later Task 3 joint-training implementation or experiment.

The revised plan (`CSE425_GNN_BERT_Music_Project_Plan.md`, full boundary and
Part 6) intentionally narrows Task 4 to frozen encoders plus projection heads.
No instructor acceptance of this adjustment is documented here. Do not describe
the implementation as fully satisfying the original joint-training algorithm.
If that algorithm is mandatory, a separately declared fine-tuning experiment
is still required; the existing test results must not be used to tune it.

## Other original requirements

- Page 5 requires **ten caption queries with top-three matched clips**. The
  new `listening_study/ten_caption_queries.md` supplies ten distinct queries,
  rather than counting both retrieval directions as separate caption queries.
- Page 5 requires **zero-shot tag prediction from captions versus Task 3**.
  This is distinct from a pretrained CLAP retrieval comparison. The new
  caption/tag prompt-similarity experiment reports its fixed rule, predictions
  and supervised comparison, with its limitations explicitly labeled.
- Page 6 requires **at least five listeners rating caption/clip matches 1–5**.
  The prepared study cannot satisfy this until five people supply real ratings.

## Comparison qualifications

The frozen text encoder trained on 3,864 original training captions, including
1,089 without paired audio; GraphSAGE trained on 2,775 paired clips. The paired
held-out IDs preserve their original partitions. The three retrieval seeds vary
only the projection initialization and batch order, not backbone training.

CLAP uses different pretrained text/audio encoders and external training data;
its audio representation is not a graph. “Zero-shot” means no local parameter
updates. It does not certify that the external pretraining corpus contains no
overlapping audio. The caption-to-tag baseline uses CLAP text prompt similarity,
not a supervised NLI model; sensitivity to negation and prompt wording limits it.

**Status:** frozen-scope retrieval is measured; full original-proposal completion
remains conditional on real listener results and acceptance of the documented
encoder-training scope deviation (or completion of the required fine-tuning).
