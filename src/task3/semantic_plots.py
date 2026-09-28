"""Genre/style and mood t-SNE from saved validation embeddings and true labels."""
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

from .prepare import file_hash


def label_categories(targets, label_names, mapping, group):
    columns = [(label_names.index(key), name) for key, name in mapping[group].items() if key in label_names]
    active = [[name for index, name in columns if row[index] == 1] for row in targets]
    categories = [values[0] if len(values) == 1 else f"Multiple selected {group} labels" if values
                  else f"No selected {group} annotation" for values in active]
    return active, categories, columns


def plot_semantic_embeddings(predictions, label_names, output_dir, title, seed=42):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty semantic-plot output directory")
    mapping_path = Path(__file__).with_name("semantic_label_map.json")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    with np.load(predictions, allow_pickle=False) as archive:
        embeddings, targets, ids = archive["embeddings"], archive["targets"], archive["sample_ids"]
    if (embeddings.ndim != 2 or len(embeddings) < 3 or targets.shape != (len(ids), len(label_names))
            or len(embeddings) != len(ids) or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(embeddings).all() or not np.isin(targets, [0, 1]).all()):
        raise ValueError("Require finite aligned embeddings, unique IDs and binary targets")
    points = TSNE(n_components=2, perplexity=min(30, len(ids) - 1), init="random", learning_rate="auto", random_state=seed).fit_transform(embeddings)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_predictions=str(Path(predictions).resolve()), predictions_sha256=file_hash(predictions),
                    seed=seed, samples=len(ids), label_names=label_names, mapping=mapping,
                    mapping_sha256=file_hash(mapping_path), title=title, labels="Ground-truth positive annotations, not predictions")
    table = {"sample_id": ids, "tsne_x": points[:, 0], "tsne_y": points[:, 1]}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for axis, group in zip(axes, ("genre", "mood")):
        active, categories, columns = label_categories(targets, label_names, mapping, group)
        categories = np.asarray(categories)
        table[f"{group}_labels"] = [" | ".join(values) for values in active]
        table[f"{group}_category"] = categories
        unknown = f"No selected {group} annotation"
        overlap = f"Multiple selected {group} labels"
        ordered = [unknown] + [name for _, name in columns] + [overlap]
        colors = plt.get_cmap("tab20")
        for j, name in enumerate(ordered):
            mask = categories == name
            if not mask.any():
                continue
            color = "#b7bbc2" if name == unknown else "#222222" if name == overlap else colors((j - 1) % 20)
            axis.scatter(points[mask, 0], points[mask, 1], s=12 if name == unknown else 25,
                         alpha=.55 if name == unknown else .85, color=color,
                         label=f"{name} (n={int(mask.sum())})", rasterized=True)
        axis.set(title="Genre/style annotations" if group == "genre" else "Mood annotations (limited coverage)",
                 xlabel="t-SNE 1", ylabel="t-SNE 2")
        axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.13), ncol=2 if group == "genre" else 1)
        metadata[group] = dict(annotated_samples=sum(bool(v) for v in active),
                              unannotated_samples=sum(not v for v in active),
                              overlapping_samples=sum(len(v) > 1 for v in active),
                              per_label_positive_support={name: int(targets[:, j].sum()) for j, name in columns})
    fig.suptitle(f"{title}\nSame validation coordinates in both views; colors use annotations")
    fig.savefig(output / "genre_mood_tsne.png", dpi=160)
    plt.close(fig)
    # One-vs-rest annotation panels expose every positive, including overlap.
    _, _, columns = label_categories(targets, label_names, mapping, "genre")
    if columns:
        fig, axes = plt.subplots(int(np.ceil(len(columns) / 4)), 4, figsize=(14, 3 * int(np.ceil(len(columns) / 4))), squeeze=False, layout="constrained")
        for axis, (j, name) in zip(axes.flat, columns):
            positive = targets[:, j] == 1
            axis.scatter(points[:, 0], points[:, 1], s=6, color="#d5d7db", alpha=.6)
            axis.scatter(points[positive, 0], points[positive, 1], s=19, color="#176a9b")
            axis.set_title(f"{name} (n={int(positive.sum())})", fontsize=10)
            axis.set_xticks([]); axis.set_yticks([])
        for axis in list(axes.flat)[len(columns):]:
            axis.set_visible(False)
        fig.suptitle("Genre/style positives; overlapping annotations retained in each panel")
        fig.savefig(output / "genre_label_panels.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(table).to_csv(output / "sample_annotations.csv", index=False)
    np.savez_compressed(output / "embedding_coordinates.npz", sample_ids=ids, coordinates=points, targets=targets)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        f"# Genre and mood embedding views\n\n{title}. {len(ids)} validation examples.\n\n"
        "![Genre and mood](genre_mood_tsne.png)\n\n"
        + mapping["multilabel_policy"] + "\n\n"
        + "\n".join(f"- {s}" for s in mapping["limitations"]) + "\n\n"
        "`metadata.json` records the complete mapping, source hash and positive support. "
        "`sample_annotations.csv` and `embedding_coordinates.npz` preserve labels, IDs and coordinates.\n",
        encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--joint-run", type=Path)
    source.add_argument("--frozen-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.joint_run:
        from .joint import verify_joint
        verify_joint(args.joint_run)
        root = args.joint_run
        selection = json.loads((root / "selection.json").read_text())
        config = json.loads((root / "run_config.json").read_text())
        mode = selection["selected_fusion"] or selection["selected_mode"]
        predictions = root / mode / "validation_predictions.npz"
        labels, seed = config["label_names"], config["seed"]
        title = f"Joint {mode}, seed {seed}"
    else:
        from .experiment import verify_experiment
        verify_experiment(args.frozen_run)
        root = args.frozen_run
        selection = json.loads((root / "selection.json").read_text())
        mode, seed = selection["selected_fusion"], selection["seeds"][0]
        predictions = root / f"seed{seed}" / f"{mode}_validation_predictions.npz"
        metric = json.loads((root / f"seed{seed}" / f"{mode}_validation_metrics.json").read_text())
        labels = list(metric["per_label"])
        title = f"Frozen {mode}, seed {seed}"
    result = plot_semantic_embeddings(predictions, labels, args.output_dir or root / "semantic_plots", title, seed)
    print(json.dumps({k: result[k] for k in ("samples", "genre", "mood")}, indent=2))


if __name__ == "__main__":
    main()
