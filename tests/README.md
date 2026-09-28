# Tests

Run the offline test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

`tests/task1/test_pipeline.py` checks proxy-label extraction, the preserved 5,299 x 50
MusicCaps artifact, deterministic non-overlapping splits, metric calculations,
and consistency between the legacy checkpoint metadata and saved deliverables.

`tests/task1/test_audit.py` checks MusicCaps field parsing, label normalization,
duplicate/overlap diagnostics, and availability-gate handling when platform
errors make a probe inconclusive.

`tests/task1/test_academic.py` checks ID grouping and order-invariant splits,
held-out label perturbations, caption-independent targets, training-only TF-IDF
vocabulary, prepared-manifest hashes and split integrity, validation calibration,
and checkpoint inference with per-label thresholds. `test_training_safety.py`
covers configuration, frozen encoders and AP edge cases. `test_notebook.py`
checks the independent-label notebook workflow and legacy/current example support.

For the completed real-data run, recompute all metrics and verify IDs, targets,
and validation-derived thresholds without retraining:

```bash
python -m src.task1.evaluate --run-dir results/task1/audioset_cpu_20260907
```

`tests/task2/` checks segment features, finite values, all model shapes, and
complete synthetic audio -> graphs -> training -> prediction workflows.
It also covers Task 1 AudioSet label/ID alignment, inherited video splits,
source annotation/hash checks, audio interval handling, stale-cache rejection,
disconnected/malformed graphs, and export of 20 distinct graph comparisons.
The five-model suite test recomputes each saved test metric, verifies that
thresholds came from validation, checks identical IDs/targets across models,
rejects changed caches, and protects existing experiment directories.

`tests/task3/` covers fusion shapes/masks/gradients, frozen checkpoint restoration,
aligned real-cache extraction, relocated provenance, independent targets, output
preservation, token/segment sensitivity and the five-mode three-seed suite.
Its suite audit recomputes thresholds/metrics and rejects altered checkpoints,
tables or experiment declarations. Recheck the measured suite without retraining:

```bash
python -m src.task3.experiment --verify-run results/task3/available_run1
```

Joint-training tests additionally run all five modes through tiny offline BERT
and GraphSAGE training/evaluation, verify active and frozen parameter behavior,
exercise config/preflight/smoke validation, and check genre/mood overlap handling.
The configured pipeline's real pretrained smoke check is:

```bash
python -m src.task3.pipeline smoke --report results/task3/pipeline_smoke.json
```

`tests/task4/` covers retrieval metrics, gradients, frozen features, validation-only
selection and artifact verification. Extension tests check ten-query gallery
alignment, blinded listener form generation, five complete responses, rating
validation and the fixed caption/tag score boundary. Synthetic listener ratings
exist only in temporary test fixtures and are never project evidence.

```bash
python -m unittest discover -s tests/task4 -v
```
