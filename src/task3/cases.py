"""Trace three validation predictions through frozen graph and token interventions.

These are representation-sensitivity checks, not causal explanations of words or
sounds. In particular, contextual BERT vectors already contain other words.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch

from src.task2.dataset import ProcessedTask2Dataset, load_manifest
from src.task2.models import build_task2_model
from src.task2.train import resolve_device
from .data import feature_fingerprint, load_features
from .models import FusionClassifier


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}
INTERPRETATION = (
    "Delta = original probability minus probability after intervention; positive "
    "means the intervention lowers the score. Each intervention is independent. "
    "Graph interventions replace one normalized node-feature vector with zero "
    "(the training feature mean), retaining every edge, and rerun the frozen GNN. "
    "Text interventions zero one cached contextual BERT vector, preserving the "
    "attention mask and other vectors. They do not delete or mask the raw word, "
    "and contextual information remains in other vectors. These out-of-distribution "
    "representation checks are neither causal word/audio attributions nor proof "
    "that a particular instrument occurs in a segment. Sensitivities need not add up."
)


def select_case_indices(splits, count=3):
    """Take the first validation rows in frozen-cache order, never rank by outcome."""
    indices = [i for i, split in enumerate(splits) if split == "validation"]
    if len(indices) < count:
        raise ValueError(f"At least {count} validation examples are required")
    return indices[:count]


@torch.inference_mode()
def representation_sensitivity(model, graph_encoder, tokens, mask, graph_embedding,
                               graph, token_strings, segment_seconds=1.0, excluded_label_indices=()):
    """Compute signed leave-one-vector-zero effects without changing inputs."""
    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive and finite")
    if tokens.shape[0] != 1 or len(token_strings) != tokens.shape[1]:
        raise ValueError("Exactly one caption with aligned token strings is required")
    model.eval()
    graph_encoder.eval()
    baseline = model(tokens, mask, graph_embedding).sigmoid()[0]
    candidates = [i for i in range(len(baseline)) if i not in excluded_label_indices]
    if not candidates:
        candidates = list(range(len(baseline)))
    label_index = max(candidates, key=lambda i: float(baseline[i].item()))
    recomputed = graph_encoder(graph)
    if not torch.allclose(recomputed, graph_embedding, atol=1e-4, rtol=1e-4):
        raise ValueError("Recomputed graph embedding differs from frozen feature cache")

    def outcome(probabilities):
        values = probabilities[0]
        return dict(probability=float(values[label_index].item()),
                    probability_delta=float((baseline[label_index] - values[label_index]).item()),
                    probabilities=values.cpu().tolist(),
                    probability_deltas=(baseline - values).cpu().tolist())

    token_evidence = []
    for position in mask[0].bool().nonzero(as_tuple=False).flatten().tolist():
        modified = tokens.clone()
        modified[:, position] = 0
        text = token_strings[position]
        token_evidence.append(dict(position=position, token=text,
                                   special=text in SPECIAL_TOKENS,
                                   **outcome(model(modified, mask, graph_embedding).sigmoid())))
    node_evidence = []
    for node in range(graph.num_nodes):
        modified = graph.clone()
        modified.x[node] = 0
        embedding = graph_encoder(modified)
        node_evidence.append(dict(node_index=node,
                                  start_seconds=node * segment_seconds,
                                  end_seconds=(node + 1) * segment_seconds,
                                  **outcome(model(tokens, mask, embedding).sigmoid())))
    return dict(probabilities=baseline.cpu().tolist(), explained_label_index=label_index,
                tokens=token_evidence, segments=node_evidence,
                graph_embedding_max_abs_error=float((recomputed - graph_embedding).abs().max().item()))


def label_outcomes(probabilities, targets, label_names, display_names, threshold):
    rows = []
    for identifier, probability, target in zip(label_names, probabilities, targets):
        predicted, target = probability >= threshold, bool(target)
        status = ("true_positive" if target else "false_positive") if predicted else (
            "false_negative" if target else "true_negative")
        rows.append(dict(label_id=identifier, name=display_names.get(identifier, identifier),
                         probability=float(probability), target=int(target),
                         predicted=int(predicted), status=status))
    return rows


def _top_effects(rows, limit=10):
    return sorted(rows, key=lambda row: -abs(row["probability_delta"]))[:limit]


def _plot_case(case, path):
    """Show graph topology, segment sensitivity, token sensitivity and errors."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"{case['sample_id']} | {case['mode']} | explained label: {case['explained_label']['name']}", fontsize=13)
    segments = case["segments"]
    positions = [row["node_index"] for row in segments]
    deltas = [row["probability_delta"] for row in segments]
    extent = max(max(abs(value) for value in deltas), 1e-6)
    axis = axes[0, 0]
    edges = {tuple(sorted(pair)) for pair in case["graph"]["edges"] if pair[0] != pair[1]}
    for first, second in sorted(edges):
        axis.add_patch(FancyArrowPatch((first, 0), (second, 0), arrowstyle="-",
                                      connectionstyle="arc3,rad=-0.35", color="#a3aab5", alpha=0.55))
    points = axis.scatter(positions, [0] * len(positions), c=deltas, cmap="RdBu_r",
                          vmin=-extent, vmax=extent, s=190, zorder=3)
    for position in positions:
        axis.text(position, 0, str(position), ha="center", va="center", fontsize=8)
    max_span = max((second - first for first, second in edges), default=1)
    axis.set(xlim=(-0.7, len(positions) - 0.3), ylim=(-0.35, max(0.35, max_span * 0.22)),
             xlabel="Segment/node index (clip-relative)", title=f"Fixed graph edges: {case['graph']['variant']}")
    axis.set_yticks([])
    fig.colorbar(points, ax=axis, label="Original - intervened probability", shrink=0.75)

    axis = axes[1, 0]
    axis.bar(positions, deltas, color=["#bd4438" if value >= 0 else "#3476a8" for value in deltas])
    axis.axhline(0, color="#45505e", linewidth=0.8)
    axis.set_xticks(positions, [f"{row['start_seconds']:g}-{row['end_seconds']:g}" for row in segments], rotation=45)
    axis.set(xlabel="Clip-relative segment interval (seconds)", ylabel="Original - intervened probability",
             title="Zero one normalized node; rerun frozen GNN")

    axis = axes[0, 1]
    strongest = list(reversed(_top_effects([row for row in case["tokens"] if not row["special"]], 12)))
    axis.barh(range(len(strongest)), [row["probability_delta"] for row in strongest],
              color=["#bd4438" if row["probability_delta"] >= 0 else "#3476a8" for row in strongest])
    axis.set_yticks(range(len(strongest)), [f"{row['position']}: {row['token']}" for row in strongest])
    axis.axvline(0, color="#45505e", linewidth=0.8)
    axis.set(xlabel="Original - intervened probability", title="Largest contextual-wordpiece sensitivities\n(special tokens excluded)")

    axis = axes[1, 1]
    rows = sorted(case["label_outcomes"], key=lambda row: (-int(row["target"] or row["predicted"]), -row["probability"]))[:12]
    rows.reverse()
    colors = {"true_positive": "#397d4e", "false_positive": "#c95745",
              "false_negative": "#d59a24", "true_negative": "#8591a0"}
    axis.barh(range(len(rows)), [row["probability"] for row in rows], color=[colors[row["status"]] for row in rows])
    axis.set_yticks(range(len(rows)), [f"{row['name']} ({row['status'].replace('_', ' ')})" for row in rows], fontsize=8)
    axis.axvline(case["threshold"], color="black", linestyle="--", label=f"threshold {case['threshold']:.2f}")
    axis.set(xlim=(0, 1), xlabel="Probability", title="Labels and errors (up to 12; all labels in JSON)")
    axis.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _markdown(report):
    lines = ["# Task 3 validation case studies", "",
             f"Checkpoint mode: **{report['mode']}**. {report['selection']}", "",
             report["focal_label_selection"], "", report["interpretation"], "", report["text_limitation"], ""]
    for number, case in enumerate(report["cases"], 1):
        explained = case["explained_label"]
        lines.extend([f"## Case {number}: {case['sample_id']}", "", case["text"], "",
                      f"Explained label: **{explained['name']}** (`{explained['label_id']}`); "
                      f"probability **{explained['probability']:.4f}**, target **{explained['target']}**, "
                      f"status **{explained['status']}**. Threshold: {case['threshold']:.4f}.", ""])
        for status in ("true_positive", "false_positive", "false_negative"):
            names = [f"{row['name']} ({row['probability']:.3f})" for row in case["label_outcomes"] if row["status"] == status]
            lines.append(f"- {status.replace('_', ' ').capitalize()}: {', '.join(names) or 'none'}.")
        lines.extend(["", "Representation trace: cached normalized segment features → frozen GraphSAGE "
                      "mean/max readout; cached contextual BERT vectors → frozen fusion head → sigmoid score.", "",
                      "| Segment (clip seconds) | Original − intervened probability |", "|---|---:|"])
        for row in _top_effects(case["segments"], 3):
            lines.append(f"| {row['start_seconds']:g}–{row['end_seconds']:g} | {row['probability_delta']:+.6f} |")
        lexical = [row for row in case["tokens"] if not row["special"]]
        nonzero = [row for row in lexical if abs(row["probability_delta"]) > 1e-8]
        if nonzero:
            lines.extend(["", "| Contextual wordpiece position / token | Original − intervened probability |", "|---|---:|"])
            for row in _top_effects(nonzero, 5):
                token = row["token"].replace("|", "\\|")
                lines.append(f"| {row['position']} / `{token}` | {row['probability_delta']:+.6f} |")
        else:
            lines.extend(["", "No nonzero lexical-token sensitivity under this intervention. "
                          "This does not establish that the caption words are irrelevant."])
        lines.extend(["", f"![Case {number} evidence]({case['figure']})", ""])
    return "\n".join(lines)


def generate_cases(checkpoint, features, processed_manifest, graph_checkpoint, output_dir, device="cpu"):
    """Write three fixed validation cases with checkpoint/cache identity checks."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_sha256") != feature_fingerprint(features):
        raise ValueError("Feature file differs from the checkpoint training data")
    data = load_features(features, mmap=True)
    provenance = data.get("provenance", {})
    for key, path in (("graph_checkpoint_sha256", graph_checkpoint),
                      ("processed_manifest_sha256", processed_manifest)):
        if provenance.get(key) != feature_fingerprint(path):
            raise ValueError(f"Case input differs from frozen feature provenance: {key}")
    if data["label_names"] != saved["label_names"]:
        raise ValueError("Feature label names/order differ from checkpoint")
    indices = select_case_indices(data["splits"])
    if "token_strings" not in data:
        raise ValueError("Case studies require aligned token_strings; rebuild the feature cache")
    # Keep only three rows before graph inference; full BERT cache can be large.
    samples = [dict(sample_id=data["sample_ids"][i], text=data["texts"][i],
                    tokens=data["tokens"][i:i + 1].float().clone(),
                    mask=data["mask"][i:i + 1].bool().clone(),
                    graph=data["graph"][i:i + 1].float().clone(),
                    labels=data["labels"][i].tolist(), token_strings=data["token_strings"][i]) for i in indices]
    del data
    graph_saved = torch.load(graph_checkpoint, map_location="cpu", weights_only=True)
    gc = graph_saved["config"]
    if gc["model"] != "gnn" or graph_saved["label_names"] != saved["label_names"]:
        raise ValueError("Expected the matching trained GraphSAGE checkpoint")
    manifest = load_manifest(processed_manifest)
    if manifest["label_names"] != saved["label_names"]:
        raise ValueError("Processed manifest label names/order differ from checkpoint")
    segment_seconds = float(manifest["feature_config"]["segment_seconds"])
    records = {record["sample_id"]: record for record in manifest["records"]}
    dataset = ProcessedTask2Dataset(processed_manifest, "validation", gc["graph_variant"])
    graph_indices = {record["sample_id"]: i for i, record in enumerate(dataset.records)}
    device = resolve_device(device)
    model = FusionClassifier(**saved["dimensions"], mode=saved["mode"]).to(device)
    model.load_state_dict(saved["state_dict"])
    graph_model = build_task2_model("gnn", input_dim=graph_saved["input_dim"],
                                   num_labels=len(saved["label_names"]), hidden_dim=gc["hidden_dim"],
                                   layers=gc["layers"], dropout=gc["dropout"]).to(device)
    graph_model.load_state_dict(graph_saved["state_dict"])
    names_path = Path(__file__).resolve().parents[1] / "task1" / "audioset_names.json"
    display_names = json.loads(names_path.read_text(encoding="utf-8"))
    threshold = float(saved["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Saved threshold must be finite and between zero and one")
    cases = []
    for number, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        if sample_id not in graph_indices:
            raise ValueError(f"Case sample is not in the graph validation split: {sample_id}")
        graph_sample = dataset[graph_indices[sample_id]]
        if graph_sample["labels"].tolist() != sample["labels"]:
            raise ValueError(f"Graph and frozen feature targets differ: {sample_id}")
        graph = graph_sample["graph"].to(device)
        evidence = representation_sensitivity(model, graph_model.encoder,
                                             sample["tokens"].to(device), sample["mask"].to(device),
                                             sample["graph"].to(device), graph, sample["token_strings"], segment_seconds,
                                             excluded_label_indices=[i for i, name in enumerate(saved["label_names"])
                                                                     if name == "/m/04rlf"])
        outcomes = label_outcomes(evidence.pop("probabilities"), sample["labels"], saved["label_names"],
                                  display_names, threshold)
        explained = outcomes[evidence.pop("explained_label_index")]
        record = records[sample_id]
        cases.append(dict(sample_id=sample_id, split="validation", text=sample["text"], mode=saved["mode"],
                          threshold=threshold, explained_label=explained, label_outcomes=outcomes,
                          figure=f"case_{number}.png", graph=dict(variant=gc["graph_variant"],
                              node_count=graph.num_nodes, edges=graph.edge_index.t().cpu().tolist(),
                              segment_seconds=segment_seconds, time_origin="start of loaded audio clip",
                              source_start_seconds=record.get("start_s"), source_end_seconds=record.get("end_s")),
                          **evidence))
    mode = saved["mode"]
    if mode == "cross_attention":
        limitation = "Cross-attention reads all unmasked contextual vectors; wordpiece sensitivity still is not raw-word importance."
    elif mode == "gnn":
        limitation = "The GNN-only head ignores text, so all token interventions have zero effect."
    else:
        limitation = ("This head reads only BERT's contextual [CLS] vector. Non-CLS vector interventions "
                      "therefore have zero effect by construction. [CLS] sensitivity measures a whole-caption "
                      "representation change; individual influential words cannot be identified by this method.")
    if mode == "bert":
        limitation += " The BERT-only head ignores graph embeddings, so graph interventions also have zero effect."
    report = dict(mode=mode, split="validation", selection="First three validation rows in frozen-cache order; no outcome filtering. "
                  "The supplied head is an advanced-fusion diagnostic and is not necessarily the validation winner.",
                  focal_label_selection="Fixed focal-label rule: highest-probability label excluding the generic "
                  "AudioSet Music label (/m/04rlf); fallback to the highest-probability label if Music is the only label. "
                  "All label probabilities and intervention deltas are retained in JSON.",
                  checkpoint_sha256=feature_fingerprint(checkpoint), feature_sha256=saved["feature_sha256"],
                  graph_checkpoint_sha256=feature_fingerprint(graph_checkpoint),
                  processed_manifest_sha256=feature_fingerprint(processed_manifest), provenance=provenance,
                  interpretation=INTERPRETATION, text_limitation=limitation, cases=cases)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty case-study output directory")
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        _plot_case(case, output / case["figure"])
    (output / "cases.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "cases.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("checkpoint", "features", "processed-manifest", "graph-checkpoint", "output-dir"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    report = generate_cases(**vars(parser.parse_args()))
    print(json.dumps(dict(mode=report["mode"], sample_ids=[case["sample_id"] for case in report["cases"]]), indent=2))


if __name__ == "__main__":
    main()
