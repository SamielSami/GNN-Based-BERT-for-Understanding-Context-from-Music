"""Build deterministic temporal, similarity, and random-control segment graphs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .features import AudioFeatureConfig, extract_track_features


GRAPH_VARIANTS = ("temporal", "temporal_similarity", "random")


def _validate_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("features must have shape [num_segments, feature_dim]")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    return features


def _edge_tensor(edges: set[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def temporal_edge_set(num_nodes: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for index in range(num_nodes - 1):
        edges.add((index, index + 1))
        edges.add((index + 1, index))
    return edges


def similarity_edge_set(features: np.ndarray, top_k: int = 2) -> set[tuple[int, int]]:
    """Return symmetric top-k cosine edges between non-adjacent segments."""
    features = _validate_features(features)
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0 or len(features) < 3:
        return set()
    features = features.astype(np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    undirected: set[tuple[int, int]] = set()
    for source in range(len(features)):
        candidates = [target for target in range(len(features)) if abs(source - target) > 1]
        candidates.sort(key=lambda target: (-float(similarity[source, target]), target))
        for target in candidates[:top_k]:
            undirected.add((min(source, target), max(source, target)))
    return {(a, b) for edge in undirected for a, b in (edge, edge[::-1])}


def random_edge_set(num_nodes: int, directed_edge_count: int, seed: int) -> set[tuple[int, int]]:
    """Create a symmetric random control with the requested directed edge count."""
    if directed_edge_count < 0 or directed_edge_count % 2:
        raise ValueError("directed_edge_count must be a non-negative even number")
    candidates = [(a, b) for a in range(num_nodes) for b in range(a + 2, num_nodes)]
    needed = min(directed_edge_count // 2, len(candidates))
    chosen = random.Random(seed).sample(candidates, needed) if needed else []
    return {(a, b) for edge in chosen for a, b in (edge, edge[::-1])}


def build_edge_variants(
    features: np.ndarray, *, top_k: int = 2, seed: int = 42
) -> dict[str, torch.Tensor]:
    features = _validate_features(features)
    temporal = temporal_edge_set(len(features))
    similarity = similarity_edge_set(features, top_k)
    random_control = random_edge_set(len(features), len(similarity), seed)
    return {
        "temporal": _edge_tensor(temporal),
        "temporal_similarity": _edge_tensor(temporal | similarity),
        "random": _edge_tensor(temporal | random_control),
    }


def build_segment_graph(
    features: np.ndarray,
    *,
    variant: str = "temporal_similarity",
    top_k: int = 2,
    seed: int = 42,
) -> Data:
    features = _validate_features(features)
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"variant must be one of {GRAPH_VARIANTS}")
    edge_index = build_edge_variants(features, top_k=top_k, seed=seed)[variant]
    validate_graph(features, edge_index)
    return Data(x=torch.from_numpy(features), edge_index=edge_index)


def validate_graph(features: np.ndarray | torch.Tensor, edge_index: torch.Tensor) -> dict:
    """Reject malformed/disconnected graphs; one node with no edges is connected."""
    x = _validate_features(torch.as_tensor(features).detach().cpu().numpy())
    edges = torch.as_tensor(edge_index).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edges.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("edge_index must contain integer node indices")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= len(x)):
        raise ValueError("edge_index contains out-of-range node indices")
    pairs = [tuple(pair) for pair in edges.t().tolist()]
    edge_set = set(pairs)
    if len(edge_set) != len(pairs):
        raise ValueError("edge_index contains duplicate edges")
    if any(source == target for source, target in edge_set):
        raise ValueError("edge_index contains unexpected self loops")
    if any((target, source) not in edge_set for source, target in edge_set):
        raise ValueError("edge_index must be symmetric")
    adjacency: list[list[int]] = [[] for _ in range(len(x))]
    for source, target in edge_set:
        adjacency[source].append(target)
    visited = {0}
    pending = [0]
    while pending:
        for target in adjacency[pending.pop()]:
            if target not in visited:
                visited.add(target)
                pending.append(target)
    if len(visited) != len(x):
        raise ValueError(f"Graph is disconnected: {len(visited)}/{len(x)} nodes reachable")
    return {
        "nodes": len(x),
        "feature_dim": x.shape[1],
        "directed_edges": len(edge_set),
        "undirected_edges": len(edge_set) // 2,
        "connected": True,
        "finite_features": True,
        "symmetric": True,
    }


def validate_graph_variants(features: np.ndarray | torch.Tensor, edge_indices: dict) -> dict:
    """Check all three graphs and the edge-count-matched random control."""
    if set(edge_indices) != set(GRAPH_VARIANTS):
        raise ValueError(f"Expected exactly these graph variants: {GRAPH_VARIANTS}")
    summaries = {name: validate_graph(features, edge_indices[name]) for name in GRAPH_VARIANTS}
    temporal = temporal_edge_set(summaries["temporal"]["nodes"])
    for name in GRAPH_VARIANTS:
        edges = set(map(tuple, torch.as_tensor(edge_indices[name]).t().tolist()))
        if not temporal.issubset(edges):
            raise ValueError(f"{name} graph is missing temporal backbone edges")
        if name == "temporal" and edges != temporal:
            raise ValueError("Temporal graph contains non-temporal edges")
    if summaries["random"]["directed_edges"] != summaries["temporal_similarity"]["directed_edges"]:
        raise ValueError("Random and similarity graphs must have identical edge counts")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--variant", choices=GRAPH_VARIANTS, default="temporal_similarity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = AudioFeatureConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
    )
    features = extract_track_features(args.audio, config)
    graph = build_segment_graph(
        features, variant=args.variant, top_k=args.top_k, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "variant": args.variant,
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
                "feature_dim": graph.num_node_features,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
