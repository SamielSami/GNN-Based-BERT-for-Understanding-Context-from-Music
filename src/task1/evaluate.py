"""Recompute held-out metrics and audit saved IDs, targets and calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audioset import load_prepared_manifest
from .baselines import OntologyKeywords
from .train import classification_report, select_thresholds


def bootstrap_f1(targets, predictions, *, repeats=1000, seed=2026):
    """Percentile intervals conditional on this fitted model and fixed thresholds."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        rows = rng.integers(0, len(targets), len(targets))
        truth, pred = targets[rows].astype(bool), predictions[rows].astype(bool)
        tp = (truth & pred).sum(axis=0)
        denom = truth.sum(axis=0) + pred.sum(axis=0)
        macro = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0).mean()
        micro = 2 * tp.sum() / denom.sum() if denom.sum() else 0.0
        values.append([macro, micro])
    intervals = np.quantile(values, [0.025, 0.975], axis=0)
    return {"method": "test-row percentile bootstrap; fixed fitted model and thresholds",
            "repeats": repeats, "seed": seed,
            "macro_f1_95_ci": intervals[:, 0].tolist(), "micro_f1_95_ci": intervals[:, 1].tolist()}


def verify_run(run_dir, *, repeats=1000):
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    data_path = Path(metrics["dataset"]["path"])
    manifest = load_prepared_manifest(data_path)
    if manifest is None:
        raise ValueError("Independent-label manifest required")
    archived = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if archived != manifest or metrics["dataset"]["sha256"] != manifest["csv_sha256"]:
        raise ValueError("Run dataset provenance mismatch")
    frame = pd.read_csv(data_path, dtype={"ytid": str})
    tags = manifest["tags"]
    reports = {"distilbert": metrics, **metrics.get("baselines", {})}
    verified = {}
    for name, report in reports.items():
        prefix = "" if name == "distilbert" else name + "_"
        for split in ("validation", "test"):
            saved = np.load(run_dir / f"{prefix}{split}_predictions.npz", allow_pickle=False)
            indices = manifest["split_indices"][split]
            if saved["tags"].tolist() != tags or saved["row_indices"].tolist() != indices:
                raise ValueError("Prediction row/tag alignment mismatch")
            if saved["ytid"].tolist() != frame.iloc[indices].ytid.tolist():
                raise ValueError("Prediction ytid alignment mismatch")
            np.testing.assert_array_equal(saved["targets"], frame.iloc[indices][tags].to_numpy())
            if split == "validation" and report["calibration"]["fit_split"] == "validation":
                calibration = select_thresholds(saved["targets"], saved["probabilities"])
                if calibration != report["calibration"]:
                    raise ValueError("Thresholds are not reproducible from validation predictions")
            np.testing.assert_array_equal(saved["thresholds"], report["calibration"]["thresholds"])
            recomputed = classification_report(saved["targets"], saved["probabilities"], tags, saved["thresholds"])
            for key, value in recomputed.items():
                if value != report[split][key]:
                    raise ValueError(f"Saved/recomputed metrics disagree: {name}/{split}/{key}")
        # Unique IDs in MusicCaps make a row bootstrap also a video-ID bootstrap.
        if len(set(saved["ytid"])) == len(saved["ytid"]):
            verified[name] = bootstrap_f1(saved["targets"], saved["probabilities"] >= saved["thresholds"], repeats=repeats)
        else:
            verified[name] = {"bootstrap_skipped": "Repeated IDs require a group bootstrap"}

    saved = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    targets, probabilities, thresholds = saved["targets"], saved["probabilities"], saved["thresholds"]
    predicted = probabilities >= thresholds
    denominator = targets.sum(axis=1) + predicted.sum(axis=1)
    row_f1 = np.divide(2 * (targets * predicted).sum(axis=1), denominator,
                       out=np.ones(len(targets)), where=denominator > 0)
    test_frame = frame.iloc[manifest["split_indices"]["test"]]
    names = json.loads(Path(__file__).with_name("audioset_names.json").read_text(encoding="utf-8"))
    def example(i):
        return {"ytid": str(saved["ytid"][i]), "text": test_frame.iloc[i].text,
                "sample_f1": float(row_f1[i]),
                "true_tags": [names.get(tags[j], tags[j]) for j in np.flatnonzero(targets[i])],
                "predicted_tags": [names.get(tags[j], tags[j]) for j in np.flatnonzero(predicted[i])],
                "missed": [names.get(tags[j], tags[j]) for j in np.flatnonzero(targets[i] & ~predicted[i])],
                "extra": [names.get(tags[j], tags[j]) for j in np.flatnonzero(~targets[i].astype(bool) & predicted[i])]}
    overlap = (OntologyKeywords(tags).predict_proba(test_frame.text.tolist()) * targets).any(axis=1)
    analysis = {
        "selection": "Descriptive test analysis only; no model or threshold changes",
        "five_highest_sample_f1": [example(i) for i in np.argsort(-row_f1, kind="stable")[:5]],
        "five_lowest_sample_f1": [example(i) for i in np.argsort(row_f1, kind="stable")[:5]],
        "exact_match_fraction": float(np.all(targets == predicted, axis=1).mean()),
        "keyword_overlap_strata": {str(flag): classification_report(targets[overlap == flag], probabilities[overlap == flag], tags, thresholds)
                                   for flag in (True, False) if (overlap == flag).any()},
    }
    non_root = [j for j, tag in enumerate(tags) if tag != "/m/04rlf"]
    analysis["excluding_music_root_descriptive"] = classification_report(
        targets[:, non_root], probabilities[:, non_root], [tags[j] for j in non_root], thresholds[non_root])
    result = {"verified": True, "models": verified, "dataset_sha256": manifest["csv_sha256"],
              "test_ids": len(targets), "labels": len(tags)}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "error_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir, repeats=args.bootstrap_repeats), indent=2))


if __name__ == "__main__":
    main()
