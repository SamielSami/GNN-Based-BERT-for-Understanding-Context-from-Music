"""Validate aligned cached encoder outputs; generate an offline smoke fixture."""
from pathlib import Path
import hashlib
import torch


def feature_fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_features(path, *, mmap=False):
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    ids, splits, names = data["sample_ids"], data["splits"], data["label_names"]
    n = len(ids)
    if not n or len(set(ids)) != n or len(splits) != n:
        raise ValueError("Sample IDs must be unique and aligned with splits")
    if not names or len(set(names)) != len(names):
        raise ValueError("Label names must be non-empty and unique")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected non-empty train, validation, and test splits")
    tokens, mask, graph, labels = (data[k] for k in ("tokens", "mask", "graph", "labels"))
    if tokens.ndim != 3 or min(tokens.shape) < 1 or len(tokens) != n or mask.shape != tokens.shape[:2]:
        raise ValueError("Token/mask shapes do not align with sample IDs")
    if graph.ndim != 2 or min(graph.shape) < 1 or len(graph) != n or labels.shape != (n, len(names)):
        raise ValueError("Graph/label shapes do not align with sample IDs")
    if not all(torch.isfinite(chunk).all() for x in (tokens, graph, labels) for chunk in x.split(16)):
        raise ValueError("Features and targets must be finite")
    if not ((labels == 0) | (labels == 1)).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Labels and masks must be binary")
    if not mask.bool()[:, 0].all():
        raise ValueError("First/CLS token must be valid")
    if len(data["texts"]) != n:
        raise ValueError("Captions must align with sample IDs")
    return data


def make_demo(path, seed=42):
    """Random paired features: no pretrained BERT or real audio is used."""
    generator = torch.Generator().manual_seed(seed)
    n = 30
    tokens = torch.randn(n, 6, 12, generator=generator)
    graph = torch.randn(n, 8, generator=generator)
    labels = torch.stack([tokens[:, 0, 0] > 0, graph[:, 0] > 0, tokens[:, 0, 1] + graph[:, 1] > 0], 1).float()
    data = dict(tokens=tokens, mask=torch.ones(n, 6, dtype=torch.bool), graph=graph,
                labels=labels, sample_ids=[f"synthetic_{i}" for i in range(n)],
                splits=["train"] * 18 + ["validation"] * 6 + ["test"] * 6,
                texts=[f"Synthetic fixture {i}" for i in range(n)],
                label_names=["text_signal", "graph_signal", "joint_signal"],
                provenance={"synthetic": True, "description": "Random features; pipeline diagnostics only"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path
