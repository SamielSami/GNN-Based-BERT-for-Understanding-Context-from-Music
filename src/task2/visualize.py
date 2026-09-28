"""Export reproducible graph comparisons for at least 20 distinct clips when available."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

from .dataset import load_manifest, resolve_record_path
from .graphs import GRAPH_VARIANTS, validate_graph_variants


def visualize_graph_examples(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Save one three-panel PNG per distinct sample plus JSON/HTML gallery indices.

    Node positions and colors follow segment order and are shared across panels.
    This layout makes extra edges comparable without implying spatial proximity.
    The selection cycles through splits without inspecting labels or predictions.
    """
    if count < 1:
        raise ValueError("count must be positive")
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    groups = []
    for split in ("train", "validation", "test"):
        records = sorted(
            [record for record in manifest["records"] if record["split"] == split],
            key=lambda record: record["sample_id"],
        )
        rng.shuffle(records)
        groups.append(records)
    selected = []
    while any(groups) and len(selected) < count:
        for records in groups:
            if records and len(selected) < count:
                selected.append(records.pop())
    entries = []
    titles = ("Temporal chain", "Temporal + cosine similarity", "Temporal + random control")
    colors = {"temporal": "#6b7280", "temporal_similarity": "#2563eb", "random": "#059669"}
    for position, record in enumerate(selected, start=1):
        sample = torch.load(
            resolve_record_path(manifest_path, record["processed_path"]),
            map_location="cpu", weights_only=False,
        )
        summaries = validate_graph_variants(sample["x"], sample["edge_indices"])
        nodes = len(sample["x"])
        angle = np.pi / 2 - np.arange(nodes) * 2 * np.pi / nodes
        coordinates = np.column_stack([np.cos(angle), np.sin(angle)])
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
        for axis, variant, title in zip(axes, GRAPH_VARIANTS, titles):
            pairs = [tuple(pair) for pair in sample["edge_indices"][variant].t().tolist() if pair[0] < pair[1]]
            temporal = [pair for pair in pairs if abs(pair[0] - pair[1]) == 1]
            extra = [pair for pair in pairs if abs(pair[0] - pair[1]) > 1]
            for edge_group, color, width, alpha in (
                (temporal, colors["temporal"], 1.6, 0.75),
                (extra, colors[variant], 1.1, 0.6),
            ):
                if edge_group:
                    axis.add_collection(LineCollection(
                        [[coordinates[source], coordinates[target]] for source, target in edge_group],
                        colors=color, linewidths=width, alpha=alpha,
                    ))
            axis.scatter(
                coordinates[:, 0], coordinates[:, 1], c=np.arange(nodes), cmap="viridis",
                s=max(24, min(220, 1600 / nodes)), edgecolors="white", linewidths=0.8, zorder=3,
                vmin=0, vmax=max(1, nodes - 1),
            )
            for index, point in enumerate(coordinates):
                if nodes <= 30 or index % max(1, nodes // 20) == 0:
                    axis.text(*(point * 1.14), str(index), ha="center", va="center", fontsize=8)
            axis.set_title(f"{title}\n{nodes} nodes | {len(pairs)} undirected edges", fontsize=11)
            axis.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal")
            axis.axis("off")
        figure.suptitle(
            f"{record['sample_id']}  |  {record['split']}\n"
            "Nodes = audio segments in clockwise time order; gray = temporal edges; colored = added edges",
            fontsize=11,
        )
        digest = hashlib.sha1(record["sample_id"].encode("utf-8")).hexdigest()[:12]
        filename = f"graph_{position:02d}_{digest}.png"
        figure.savefig(output_dir / filename, dpi=150, facecolor="white")
        plt.close(figure)
        entries.append({
            "sample_id": record["sample_id"], "split": record["split"],
            "image": filename, "graphs": summaries,
        })
    index = {
        "manifest": str(manifest_path), "seed": seed, "requested_examples": count,
        "available_samples": len(manifest["records"]), "examples_written": len(entries),
        "at_least_20_examples": len(entries) >= 20,
        "graph_panels_written": 3 * len(entries),
        "note": "Each example is one distinct clip shown under all three graph constructions. These are data visualizations, not model results.",
        "examples": entries,
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    cards = "\n".join(
        f'<figure><img loading="lazy" src="{entry["image"]}" alt="Graph variants for {html.escape(entry["sample_id"], quote=True)}">'
        f'<figcaption>{html.escape(entry["sample_id"])} ({entry["split"]})</figcaption></figure>'
        for entry in entries
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Task 2 graph examples</title>'
        '<style>body{font:16px system-ui;margin:2rem auto;max-width:1300px;padding:0 1rem;color:#172033;}'
        'figure{margin:2rem 0;border:1px solid #ddd;padding:1rem;border-radius:10px;}'
        'img{width:100%;height:auto;}figcaption{font-weight:600;}</style>'
        f'<h1>Task 2 graph examples</h1><p>{len(entries)} distinct clips; three structures per clip.</p>'
        '<p>Clockwise node order represents time. Gray edges join adjacent segments. Blue edges join '
        'similar segments; green edges are a random control matched to the number of blue edges. '
        'All structures keep the temporal chain and are checked for connectivity and finite features.</p>'
        + cards + '</html>'
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task2/graph_examples"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    index = visualize_graph_examples(args.manifest, args.output_dir, count=args.count, seed=args.seed)
    print(json.dumps({key: value for key, value in index.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
