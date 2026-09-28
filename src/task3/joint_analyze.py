"""Create comparison, failure analysis and fixed validation cases from a joint run.

Reads verified saved predictions; never trains, retunes, or runs test inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cases import label_outcomes
from .joint import read_json, verify_joint, write_json
from .prepare import file_hash


def sample_f1(targets, predictions):
    denominator = targets.sum(axis=1) + predictions.sum(axis=1)
    return np.divide(2 * (targets & predictions).sum(axis=1), denominator,
                     out=np.zeros(len(targets), dtype=float), where=denominator != 0)


def markdown_table(frame):
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return "undefined" if np.isnan(value) else f"{value:.4f}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "| " + " | ".join("---" for _ in frame.columns) + " |"] +
                     ["| " + " | ".join(format_value(v) for v in row) + " |"
                      for row in frame.itertuples(index=False, name=None)])


def analyze_joint(run_dir, output_dir=None, case_count=3):
    root = Path(run_dir).resolve()
    verification = verify_joint(root)
    if not verification["evaluated_test"]:
        raise ValueError("Complete the frozen joint-suite test evaluation before analysis")
    output = Path(output_dir).resolve() if output_dir else root / "analysis"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty joint-analysis output directory")
    config, selection = read_json(root / "run_config.json"), read_json(root / "selection.json")
    table = pd.read_csv(root / "comparison.csv")
    modes, fusion = config["modes"], selection["selected_fusion"]
    if not fusion or not {"bert", "gnn"}.issubset(modes):
        raise ValueError("Failure comparison requires both unimodal controls and a fusion model")
    source = read_json(config["source_manifest"])
    names = read_json(Path(__file__).resolve().parents[1] / "task1/audioset_names.json")
    source_rows = {r["sample_id"]: r for r in source["records"]}
    loaded = {}
    for mode in modes:
        loaded[mode] = {}
        for split in ("validation", "test"):
            with np.load(root / mode / f"{split}_predictions.npz", allow_pickle=False) as archive:
                loaded[mode][split] = {k: archive[k].copy() for k in ("sample_ids", "targets", "probabilities")}
    thresholds = dict(zip(table["mode"], table["threshold"]))
    val_ids = loaded[fusion]["validation"]["sample_ids"]
    if case_count < 3 or len(val_ids) < case_count:
        raise ValueError("At least three validation cases are required")
    output.mkdir(parents=True, exist_ok=True)

    per_label = []
    for mode in modes:
        metrics = read_json(root / mode / "test_metrics.json")
        prediction = loaded[mode]["test"]
        target, predicted = prediction["targets"].astype(bool), prediction["probabilities"] >= thresholds[mode]
        for j, label in enumerate(config["label_names"]):
            per_label.append(dict(mode=mode, label_id=label, name=names.get(label, label),
                                  **metrics["per_label"][label],
                                  false_positives=int((predicted[:, j] & ~target[:, j]).sum()),
                                  false_negatives=int((~predicted[:, j] & target[:, j]).sum())))
    label_table = pd.DataFrame(per_label)
    label_table.to_csv(output / "per_label_test.csv", index=False)
    example_scores = {}
    for mode in modes:
        prediction = loaded[mode]["test"]
        example_scores[mode] = sample_f1(prediction["targets"].astype(bool), prediction["probabilities"] >= thresholds[mode])
    comparisons = {}
    indexed = table.set_index("mode")
    for control in ("bert", "gnn"):
        delta = example_scores[fusion] - example_scores[control]
        comparisons[control] = dict(fusion_better=int((delta > 1e-12).sum()),
                                    fusion_worse=int((delta < -1e-12).sum()),
                                    tied=int((np.abs(delta) <= 1e-12).sum()),
                                    test_macro_f1_delta=float(indexed.loc[fusion, "test_macro_f1"] - indexed.loc[control, "test_macro_f1"]))
    write_json(output / "failure_summary.json", dict(selected_fusion=fusion, controls=comparisons,
               definition="Per-example F1 at each model's validation-selected threshold; zero denominator returns zero"))

    cases, paragraphs = [], ["# Three fixed validation cases", "",
        "The first three validation IDs in processed-manifest order are fixed independently of predictions. "
        "Thresholds and the fusion mode were selected on validation. These are in-sample validation diagnostics, "
        "not independent estimates of generalization. False positives mean missing positive annotations, "
        "which may reflect incomplete labels. No attention weights are used as causal explanations."]
    for number, (index, identifier) in enumerate(zip(range(case_count), val_ids[:case_count]), 1):
        record = source_rows[str(identifier)]
        case = dict(sample_id=str(identifier), ytid=record["ytid"], caption=record["text"],
                    selection="First validation IDs in processed-manifest order", models={})
        paragraphs.extend(["", f"## Case {number}: {identifier}", "", record["text"], ""])
        true_names = [names.get(label, label) for label, value in zip(config["label_names"], record["labels"]) if value]
        paragraphs.append("Annotated labels: " + ", ".join(true_names) + ".")
        for mode in ("bert", "gnn", fusion):
            prediction = loaded[mode]["validation"]
            outcomes = label_outcomes(prediction["probabilities"][index], record["labels"],
                                      config["label_names"], names, thresholds[mode])
            case["models"][mode] = dict(threshold=float(thresholds[mode]), labels=outcomes)
            categories = {status: [r["name"] for r in outcomes if r["status"] == status]
                          for status in ("true_positive", "false_positive", "false_negative")}
            paragraphs.extend(["", f"**{mode}** (threshold {thresholds[mode]:.2f}): " + "; ".join(
                f"{status.replace('_', ' ')}: {', '.join(values) or 'none'}" for status, values in categories.items()) + "."])
        outcomes = case["models"][fusion]["labels"]
        paragraphs.extend(["", "Selected-fusion top probabilities:", "",
            markdown_table(pd.DataFrame(sorted(outcomes, key=lambda r: -r["probability"])[:8])[
                ["name", "probability", "target", "predicted", "status"]])])
        cases.append(case)
    write_json(output / "cases.json", cases)
    (output / "cases.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, columns, title in ((axes[0], ["validation_macro_f1", "test_macro_f1"], "Macro-F1"),
                                 (axes[1], ["test_micro_f1", "test_mean_average_precision", "test_mean_pr_auc_trapezoidal"], "Held-out metrics")):
        indexed[columns].plot.bar(ax=axis, rot=15)
        axis.set(title=title, ylim=(0, 1), xlabel="Model")
        axis.legend(fontsize=7)
    fig.suptitle(f"Joint experiment, seed {config['seed']}; validation-selected fusion: {fusion}")
    fig.savefig(output / "ablation.png", dpi=150)
    plt.close(fig)

    columns = ["mode", "best_epoch", "threshold", "validation_macro_f1", "test_macro_f1", "test_micro_f1",
               "test_mean_average_precision", "test_mean_pr_auc_trapezoidal"]
    lines = ["# Task 3 joint-fusion results", "",
        f"Completed five-mode joint fine-tuning experiment, seed {config['seed']}. "
        f"Validation selects **{selection['selected_mode']}** overall and **{fusion}** among fusion models.", "",
        "## Protocol", "",
        f"Configured maximum {config['epochs']} epochs; patience {config['patience']}; batch size {config['batch_size']}; "
        f"projection width {config['hidden_dim']}. Initial heads: {config['head_initialization']}. "
        f"The final {config['unfreeze_layers']} transformer block(s), GraphSAGE and active head train using live "
        "encoder forwards. Lower text layers remain frozen. Checkpoints record active encoder/head updates and unchanged frozen weights.", "",
        "All modes share paired IDs, label order and splits. Validation alone selects epochs, thresholds and mode. "
        "All predeclared modes are then evaluated on held-out test clips. This is one joint-training seed; "
        "the earlier three-seed frozen experiment is separate. Historical encoder pretraining cohorts differ, as documented in the run guide.", "",
        "## Ablation matrix", "", markdown_table(table[columns]), "",
        "mAP is mean per-label average precision. Trapezoidal mean PR-AUC is a separate metric. "
        "Both average only labels with positive evaluation support.", "", "![Comparison](ablation.png)", "",
        "## Failure analysis", ""]
    for control, values in comparisons.items():
        lines.append(f"Selected {fusion} versus {control}: test Macro-F1 difference {values['test_macro_f1_delta']:+.4f}; "
                     f"per-example F1 improves for {values['fusion_better']} clips, worsens for {values['fusion_worse']}, "
                     f"and ties for {values['tied']}. These descriptive counts use each model's saved threshold and are not significance tests.")
        lines.append("")
    worst = label_table[label_table["mode"] == fusion].sort_values(["f1", "support", "label_id"]).head(8)
    lines.extend(["Lowest selected-fusion per-label test F1 (ties ordered by support and label ID):", "",
        markdown_table(worst[["name", "support", "f1", "average_precision", "false_positives", "false_negatives"]]), "",
        "Low-support labels have unstable estimates. Incomplete AudioSet positives can make sensible predictions count as false positives. "
        "Captions often directly describe instruments, while ten audio graph segments compress temporal detail; these are plausible "
        "limitations, not established causal explanations of an individual error. Missing-audio selection and the absence of artist-disjoint "
        "splits limit generalization. A negative fusion result is reported without selecting a replacement using test scores.", "",
        "[Three fixed validation cases](cases.md) include captions, annotations, predicted labels, errors and probabilities. "
        "They describe observed model behavior; attention weights do not establish causal importance.", "",
        "## Remaining scope qualifications", "",
        "MusicCaps is a substitute for the assignment's named datasets; instructor acceptance remains undocumented. "
        "Genre/mood t-SNE uses explicit positive AudioSet annotations, retains genre overlap and displays missing annotations. "
        "Only Angry music is available as a selected mood label, so broad mood coverage cannot be claimed.", "",
        "The run's per-mode training_curves.png files contain measured BCE and validation Macro-F1 curves. "
        "The semantic_plots directory is generated separately using the saved validation-selected fusion embeddings."])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    evidence = dict(run_dir=str(root), selected_fusion=fusion, case_ids=val_ids[:case_count].tolist(),
                    selection_sha256=file_hash(root / "selection.json"), comparison_sha256=file_hash(root / "comparison.csv"),
                    case_selection="First validation IDs in processed-manifest order", test_inference=False,
                    source_prediction_hashes={mode: {split: file_hash(root / mode / f"{split}_predictions.npz")
                                                     for split in ("validation", "test")} for mode in modes})
    write_json(output / "analysis_metadata.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(analyze_joint(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
