"""Audit saved predictions and describe the completed frozen Task 3 comparison.

No model inference, threshold tuning on test data, or test-driven selection is
performed. The suite must already have a successful completed verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.task2.train import classification_metrics, select_global_threshold
from .data import feature_fingerprint
from .experiment import FUSION_MODES, _assert_equal
from .models import MODES


METRICS = {"macro_f1": "macro_f1", "micro_f1": "micro_f1", "map": "mean_average_precision"}
MODE_NAMES = {"bert": "BERT only", "gnn": "GNN only", "concat": "Concatenation",
              "gated": "Gated fusion", "cross_attention": "Cross-attention"}
LIMITATIONS = [
    "All comparisons are descriptive; no statistical significance is claimed.",
    "Seeds vary fusion-head initialization and training order only; encoders and features are fixed. "
    "Sample standard deviation across head seeds does not measure dataset or encoder uncertainty.",
    "The selected fusion and every decision threshold were frozen using validation data before test evaluation. "
    "Test results are not used to select, retrain, or tune models.",
    "Per-example F1 uses each run's saved validation threshold and zero_division=0. "
    "Seed-example counts repeat the same test examples across head seeds and are not independent observations.",
    "Average precision is undefined for labels with no test positives; these labels are excluded from AP rankings "
    "and mAP, but retained in per_label.csv. Low-support label rankings can be unstable.",
]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Suite artifact escapes run directory: {relative}")
    return path


def _check_hash(path, expected, description):
    if feature_fingerprint(path) != expected:
        raise ValueError(f"{description} changed since the frozen suite was verified")


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None}


def _sample_f1(targets, predictions):
    numerator = 2 * np.logical_and(targets, predictions).sum(axis=1)
    denominator = targets.sum(axis=1) + predictions.sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros(len(targets), dtype=float), where=denominator != 0)


def _load_audited_predictions(root):
    """Recheck persisted evidence without mutating the suite or restoring models."""
    report, selection = _read(root / "experiment.json"), _read(root / "selection.json")
    verification_path = root / "verification.json"
    if not verification_path.exists():
        raise ValueError("Run python -m src.task3.experiment --verify-run RUN_DIR before analysis")
    verification = _read(verification_path)
    if not report.get("evaluated_test") or not verification.get("valid") or not verification.get("evaluated_test"):
        raise ValueError("Analysis requires a successfully verified suite with completed test evaluation")
    _check_hash(root / "selection.json", report["selection_sha256"], "Validation selection")
    if (verification["selection_sha256"] != report["selection_sha256"] or
            verification["feature_sha256"] != report["feature_sha256"] or
            selection["feature_sha256"] != report["feature_sha256"]):
        raise ValueError("Verification, selection, and experiment fingerprints differ")
    if "suite_config_sha256" in selection:
        _check_hash(root / "suite_config.json", selection["suite_config_sha256"], "Suite declaration")
    declaration = _read(root / "suite_config.json")
    seeds = selection["seeds"]
    expected = [(seed, mode) for seed in seeds for mode in MODES]
    if (not seeds or len(seeds) != len(set(seeds)) or declaration["seeds"] != seeds or
            declaration["modes"] != list(MODES) or declaration["feature_sha256"] != report["feature_sha256"] or
            selection.get("selected_on") != "validation" or verification["runs_checked"] != len(expected)):
        raise ValueError("Suite declaration or completed verification differs from the frozen selection")
    for records in (report["rows"], selection["runs"]):
        if [(row["seed"], row["mode"]) for row in records] != expected:
            raise ValueError("Suite does not contain every declared mode and seed exactly once")
    if report.get("selection") is not None:
        _assert_equal(report["selection"], selection, "embedded frozen selection")
    anchors, loaded, validation_means = {}, {}, {mode: [] for mode in MODES}
    label_names = None
    for row, run in zip(report["rows"], selection["runs"]):
        _check_hash(_artifact(root, run["checkpoint"]), run["checkpoint_sha256"], "Checkpoint")
        threshold = float(row["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid saved validation threshold")
        for split in ("validation", "test"):
            source = run if split == "validation" else row
            prediction_path = _artifact(root, source[f"{split}_predictions"])
            _check_hash(prediction_path, source[f"{split}_predictions_sha256"], f"{split} predictions")
            saved = _read(_artifact(root, source[f"{split}_metrics"]))
            saved_metrics = saved if split == "validation" else saved["metrics"]
            names = list(saved_metrics["per_label"])
            if label_names is None:
                label_names = names
            if names != label_names:
                raise ValueError("Prediction metric label order differs between runs")
            with np.load(prediction_path, allow_pickle=False) as archive:
                targets = archive["targets"].copy()
                probabilities = archive["probabilities"].copy()
                sample_ids = archive["sample_ids"].copy()
            if sample_ids.ndim != 1 or len(sample_ids) != len(targets) or len(set(sample_ids.tolist())) != len(sample_ids):
                raise ValueError("Prediction IDs must uniquely identify every example")
            if split in anchors:
                if not np.array_equal(sample_ids, anchors[split][0]) or not np.array_equal(targets, anchors[split][1]):
                    raise ValueError(f"{split} IDs/order or targets differ across compared modes/seeds")
            else:
                anchors[split] = sample_ids, targets
            metrics = classification_metrics(targets, probabilities, label_names, threshold)
            _assert_equal(metrics, saved_metrics, f"{split} saved metrics")
            for short, key in METRICS.items():
                _assert_equal(row[f"{split}_{short}"], metrics[key], f"{split} comparison metrics")
            if split == "validation":
                _assert_equal(threshold, select_global_threshold(targets, probabilities), "validation-only threshold")
                validation_means[run["mode"]].append(metrics["macro_f1"])
            else:
                loaded[(run["seed"], run["mode"])] = dict(
                    probabilities=probabilities, targets=targets, sample_ids=sample_ids,
                    predictions=probabilities >= threshold, metrics=metrics, threshold=threshold)
    means = {mode: float(np.mean(values)) for mode, values in validation_means.items()}
    _assert_equal(means, selection["validation_means"], "validation selection means")
    if (selection["selected_mode"] != max(MODES, key=means.get) or
            selection["selected_fusion"] != max(FUSION_MODES, key=means.get)):
        raise ValueError("Selected fusion does not follow frozen mean validation Macro-F1")
    if set(anchors["validation"][0].tolist()) & set(anchors["test"][0].tolist()):
        raise ValueError("Validation and test prediction IDs overlap")
    return report, selection, loaded, label_names


def _plot_ablation(summary, selected_fusion, seeds, synthetic, destination):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)
    x, width = np.arange(len(MODES)), 0.34
    for axis, (metric, title) in zip(axes, (("macro_f1", "Macro-F1"), ("micro_f1", "Micro-F1"), ("map", "mAP"))):
        for split, shift, color in (("validation", -width / 2, "#597b9c"), ("test", width / 2, "#cf7952")):
            values = [summary[mode][f"{split}_{metric}"]["mean"] for mode in MODES]
            deviations = [summary[mode][f"{split}_{metric}"]["std"] or 0 for mode in MODES]
            axis.bar(x + shift, values, width, yerr=deviations, capsize=3,
                     label=split.capitalize(), color=color, error_kw={"elinewidth": 1})
        axis.set(title=title, ylim=(0, 1.06), xticks=x)
        axis.set_xticklabels([MODE_NAMES[mode].replace(" ", "\n") + (" *" if mode == selected_fusion else "")
                              for mode in MODES], fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    axes[-1].legend(loc="upper right", fontsize=9)
    qualifier = "SYNTHETIC SMOKE DATA — " if synthetic else ""
    fig.suptitle(f"{qualifier}Frozen-feature ablation | {len(seeds)} head seeds", fontsize=14)
    fig.text(0.5, 0.035, "Bars: mean; whiskers: sample SD across head seeds only. "
             "* Fusion selected by mean validation Macro-F1.\n"
             "Encoders and examples fixed; whiskers are not confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _markdown(report):
    lines = ["# Task 3 saved-prediction analysis", "",
             f"Frozen selected fusion: **{report['selected_fusion']}**. "
             f"Head seeds: {', '.join(map(str, report['seeds']))}. "
             f"Test examples per run: {report['test_samples']}.", ""]
    if report["synthetic"]:
        lines.extend(["**Synthetic smoke data: these numbers are software checks, not project findings.**", ""])
    lines.extend(["## Frozen five-mode comparison", "", "Mean ± sample SD across head seeds.", "",
                  "| Mode | Validation Macro-F1 | Test Macro-F1 | Test Micro-F1 | Test mAP |",
                  "|---|---:|---:|---:|---:|"])
    for mode in MODES:
        cells = []
        for metric in ("validation_macro_f1", "test_macro_f1", "test_micro_f1", "test_map"):
            item = report["metrics_by_mode"][mode][metric]
            cells.append(f"{item['mean']:.4f} ± {item['std']:.4f}" if item["std"] is not None else f"{item['mean']:.4f} (SD undefined)")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.extend(["", "![Frozen-feature comparison](ablation.png)", "", "## Paired test comparisons", "",
                  "Deltas are selected fusion minus the named control, computed within each declared seed. "
                  "Improved/worse/tie counts compare per-example F1 at each model's saved validation threshold.", "",
                  "| Control | Δ Macro-F1 mean ± SD | Δ Micro-F1 mean ± SD | Δ mAP mean ± SD | Improved / worse / tied seed-example pairs |",
                  "|---|---:|---:|---:|---:|"])
    for mode, item in report["paired_comparisons"].items():
        deltas = []
        for metric in METRICS:
            value = item["metric_deltas"][metric]
            deltas.append(f"{value['mean']:+.4f} ± {value['std']:.4f}" if value["std"] is not None else f"{value['mean']:+.4f} (SD undefined)")
        counts = item["seed_example_counts"]
        lines.append(f"| {mode} | " + " | ".join(deltas) +
                     f" | {counts['improved']} / {counts['worse']} / {counts['tie']} |")
    for heading, key in (("Highest selected-fusion AP", "strongest_ap_labels"), ("Lowest selected-fusion AP", "weakest_ap_labels")):
        lines.extend(["", f"## {heading}", "", "| Label | Test positives | AP mean ± SD |", "|---|---:|---:|"])
        for row in report[key]:
            name = row["name"].replace("|", "\\|")
            deviation = row["average_precision_std"]
            ap = f"{row['average_precision_mean']:.4f} ± {deviation:.4f}" if deviation is not None else f"{row['average_precision_mean']:.4f} (SD undefined)"
            lines.append(f"| {name} (`{row['label_id']}`) | {row['support']} | {ap} |")
    lines.extend(["", "## Interpretation limits", ""] + [f"- {item}" for item in report["limitations"]])
    lines.extend(["", "Exact per-seed paired values and counts are in `failure_analysis.json`; "
                  "all labels and modes are in `per_label.csv`. No test outcomes alter the frozen selection.", ""])
    return "\n".join(lines)


def analyze_run(run_dir, output_dir=None, top_labels=10):
    """Write reproducible descriptive outputs from a completed audited suite."""
    if top_labels < 1:
        raise ValueError("top_labels must be positive")
    root = Path(run_dir).resolve()
    experiment, selection, loaded, label_names = _load_audited_predictions(root)
    destination = Path(output_dir).resolve() if output_dir else root / "analysis"
    seeds, selected = selection["seeds"], selection["selected_fusion"]
    metrics_by_mode = {
        mode: {f"{split}_{short}": _summary([row[f"{split}_{short}"] for row in experiment["rows"] if row["mode"] == mode])
               for split in ("validation", "test") for short in METRICS} for mode in MODES}
    display_names = _read(Path(__file__).resolve().parents[1] / "task1" / "audioset_names.json")
    per_label = []
    for mode in MODES:
        for identifier in label_names:
            metrics = [loaded[(seed, mode)]["metrics"]["per_label"][identifier] for seed in seeds]
            row = {"mode": mode, "label_id": identifier, "name": display_names.get(identifier, identifier),
                   "support": metrics[0]["support"], "test_samples": len(loaded[(seeds[0], mode)]["targets"])}
            for metric in ("precision", "recall", "f1", "average_precision"):
                values = [value[metric] for value in metrics]
                summary = _summary(values) if all(value is not None for value in values) else {"mean": None, "std": None}
                row.update({f"{metric}_{name}": value for name, value in summary.items()})
            per_label.append(row)
    comparisons = {}
    for control in ("bert", "gnn", "concat"):
        per_seed = []
        for seed in seeds:
            fusion, baseline = loaded[(seed, selected)], loaded[(seed, control)]
            differences = _sample_f1(fusion["targets"], fusion["predictions"]) - _sample_f1(baseline["targets"], baseline["predictions"])
            tie = np.isclose(differences, 0, atol=1e-12, rtol=0)
            per_seed.append({"seed": seed, "fusion_threshold": fusion["threshold"], "control_threshold": baseline["threshold"],
                             "metric_deltas": {short: float(fusion["metrics"][key] - baseline["metrics"][key]) for short, key in METRICS.items()},
                             "per_example_f1_delta_mean": float(differences.mean()),
                             "improved": int(((differences > 0) & ~tie).sum()),
                             "worse": int(((differences < 0) & ~tie).sum()), "tie": int(tie.sum())})
        comparisons[control] = {"per_seed": per_seed,
                               "metric_deltas": {metric: _summary([row["metric_deltas"][metric] for row in per_seed]) for metric in METRICS},
                               "seed_example_counts": {key: sum(row[key] for row in per_seed) for key in ("improved", "worse", "tie")},
                               "same_mode_comparison": control == selected}
    ranked = [row for row in per_label if row["mode"] == selected and row["average_precision_mean"] is not None]
    report = dict(selected_fusion=selected, selected_mode=selection["selected_mode"], seeds=seeds,
                  synthetic=bool(experiment.get("synthetic", False)),
                  feature_sha256=experiment["feature_sha256"], selection_sha256=experiment["selection_sha256"],
                  experiment_sha256=feature_fingerprint(root / "experiment.json"),
                  prediction_audit="All declared predictions, checkpoints, metric values, validation thresholds, "
                  "selection, and paired sample IDs/targets verified; no inference performed.",
                  test_samples=len(loaded[(seeds[0], selected)]["targets"]), label_count=len(label_names),
                  metrics_by_mode=metrics_by_mode, paired_comparisons=comparisons,
                  strongest_ap_labels=sorted(ranked, key=lambda row: (-row["average_precision_mean"], row["label_id"]))[:top_labels],
                  weakest_ap_labels=sorted(ranked, key=lambda row: (row["average_precision_mean"], row["label_id"]))[:top_labels],
                  labels_without_test_positives=[row["label_id"] for row in per_label if row["mode"] == selected and not row["support"]],
                  limitations=LIMITATIONS)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "failure_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(per_label).to_csv(destination / "per_label.csv", index=False)
    _plot_ablation(metrics_by_mode, selected, seeds, report["synthetic"], destination / "ablation.png")
    # Visualize the selected fusion representation even when a unimodal head wins.
    run = next(run for run in selection["runs"] if run["mode"] == selected and run["seed"] == seeds[0])
    with np.load(_artifact(root, run["validation_predictions"]), allow_pickle=False) as archive:
        embeddings, targets, sample_ids = archive["embeddings"], archive["targets"], archive["sample_ids"]
        if len(sample_ids) >= 3:
            points = TSNE(n_components=2, perplexity=min(30, len(sample_ids) - 1), random_state=seeds[0],
                          init="random", learning_rate="auto").fit_transform(embeddings)
            fig, axis = plt.subplots(figsize=(7, 5))
            scatter = axis.scatter(points[:, 0], points[:, 1], c=targets.sum(1), cmap="viridis", s=15)
            fig.colorbar(scatter, ax=axis, label="Number of true labels")
            axis.set_title(f"{MODE_NAMES[selected]} | seed {seeds[0]} validation\nDescriptive t-SNE of saved fusion embeddings")
            fig.tight_layout()
            fig.savefig(destination / "fusion_embeddings.png", dpi=150)
            plt.close(fig)
            np.savez_compressed(destination / "fusion_embedding_projection.npz", coordinates=points,
                                sample_ids=sample_ids, label_counts=targets.sum(1))
    (destination / "README.md").write_text(_markdown(report) +
        "\n## Selected fusion embeddings\n\n"
        f"Seed {seeds[0]} validation representations of {selected}; descriptive t-SNE, not a quality metric.\n\n"
        "![Selected fusion embeddings](fusion_embeddings.png)\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/task3/available_run1"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN_DIR/analysis; never rewrites suite predictions")
    parser.add_argument("--top-labels", type=int, default=10)
    args = parser.parse_args()
    report = analyze_run(args.run_dir, args.output_dir, args.top_labels)
    print(json.dumps({key: report[key] for key in ("selected_fusion", "seeds", "test_samples", "synthetic")}, indent=2))


if __name__ == "__main__":
    main()
