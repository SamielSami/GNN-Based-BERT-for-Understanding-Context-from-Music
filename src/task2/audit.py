"""Audit every saved Task 2 graph, feature tensor, label vector, and split identity."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import load_manifest, resolve_record_path
from .graphs import GRAPH_VARIANTS, validate_graph_variants


def audit_saved_graphs(
    manifest_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Inspect all cached samples and save an auditable per-sample report.

    Errors are collected so a single run reports all damaged samples; callers
    must check ``valid`` before training. Missing split coverage is reported
    separately because small diagnostic fixtures need not have all three splits.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    errors: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    widths: set[int] = set()
    mel_heights: set[int] = set()
    split_counts = Counter(record["split"] for record in manifest["records"])
    for record in manifest["records"]:
        try:
            sample = torch.load(
                resolve_record_path(manifest_path, record["processed_path"]),
                map_location="cpu", weights_only=False,
            )
            if sample["sample_id"] != record["sample_id"] or sample["split"] != record["split"]:
                raise ValueError("Cached sample ID or split disagrees with the manifest")
            if record["split"] not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown split: {record['split']}")
            labels = torch.as_tensor(sample["labels"])
            if labels.shape != (len(manifest["label_names"]),):
                raise ValueError("Cached label vector width differs from label_names")
            if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
                raise ValueError("Cached labels must be finite and binary")
            if "labels" in record and labels.tolist() != record["labels"]:
                raise ValueError("Cached label values disagree with the manifest")
            mel = torch.as_tensor(sample["mel"])
            if mel.ndim != 3 or mel.shape[0] != 1 or min(mel.shape) < 1:
                raise ValueError("Cached mel must have shape [1, n_mels, frames]")
            if not torch.isfinite(mel).all() or torch.any(mel < 0) or torch.any(mel > 1):
                raise ValueError("Cached mel must be finite and within [0, 1]")
            graphs = validate_graph_variants(sample["x"], sample["edge_indices"])
            widths.add(graphs["temporal"]["feature_dim"])
            mel_heights.add(mel.shape[1])
            similarity = set(map(tuple, sample["edge_indices"]["temporal_similarity"].t().tolist()))
            random_edges = set(map(tuple, sample["edge_indices"]["random"].t().tolist()))
            samples.append({
                "sample_id": record["sample_id"], "split": record["split"],
                "graphs": graphs,
                "random_equals_similarity": similarity == random_edges,
                "mel_frames": mel.shape[-1],
            })
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError) as error:
            errors.append({"sample_id": record["sample_id"], "error": str(error)})
    if not manifest["records"]:
        errors.append({"sample_id": "__dataset__", "error": "No cached samples to audit"})
    if len(widths) > 1 or len(mel_heights) > 1:
        errors.append({"sample_id": "__dataset__", "error": "Inconsistent feature widths or mel heights"})
    variants = {}
    for variant in GRAPH_VARIANTS:
        counts = [sample["graphs"][variant]["undirected_edges"] for sample in samples]
        variants[variant] = {
            "graphs_checked": len(counts),
            "min_undirected_edges": min(counts) if counts else None,
            "max_undirected_edges": max(counts) if counts else None,
            "mean_undirected_edges": float(np.mean(counts)) if counts else None,
        }
    report = {
        "manifest": str(manifest_path),
        "valid": not errors,
        "sample_count": len(manifest["records"]),
        "valid_samples": len(samples),
        "graphs_checked": len(samples) * len(GRAPH_VARIANTS),
        "all_graphs_connected_and_finite": not errors,
        "split_counts": dict(split_counts),
        "all_splits_present": all(split_counts[split] > 0 for split in ("train", "validation", "test")),
        "at_least_20_samples": len(samples) >= 20,
        "random_similarity_identical_samples": sum(sample["random_equals_similarity"] for sample in samples),
        "control_note": "Random graphs retain temporal edges and match similarity edge counts. Small or dense graphs can have identical controls.",
        "variants": variants,
        "errors": errors,
        "samples": samples,
    }
    destination = Path(output_path) if output_path else manifest_path.parent / "graph_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_saved_graphs(args.manifest, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
