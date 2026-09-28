"""Evaluate a saved fusion checkpoint without retraining or retuning its threshold."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task2.train import classification_metrics, resolve_device
from .data import load_features, feature_fingerprint
from .models import FusionClassifier


def predict_features(model, data, indices, batch_size, device):
    probabilities, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            tokens = data["tokens"] if model.mode == "cross_attention" else data["tokens"][:, :1]
            mask = data["mask"] if model.mode == "cross_attention" else data["mask"][:, :1]
            logits, embedding = model(tokens[chunk].float().to(device), mask[chunk].bool().to(device),
                                      data["graph"][chunk].float().to(device), return_embedding=True)
            probabilities.append(logits.sigmoid().cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings)


def evaluate_checkpoint(checkpoint, features, output_dir, split="test", batch_size=16, device="auto"):
    if split not in {"train", "validation", "test"} or batch_size < 1:
        raise ValueError("Choose a valid split and positive batch_size")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not saved.get("feature_sha256"):
        raise ValueError("Checkpoint lacks feature provenance; train a new Task 3 run")
    if saved["feature_sha256"] != feature_fingerprint(features):
        raise ValueError("Feature file differs from the checkpoint training data")
    data = load_features(features, mmap=True)
    if data["label_names"] != saved["label_names"]:
        raise ValueError("Label names/order differ from checkpoint")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty evaluation output directory")
    device = resolve_device(device)
    model = FusionClassifier(**saved["dimensions"], mode=saved["mode"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    indices = [i for i, value in enumerate(data["splits"]) if value == split]
    probabilities, embeddings = predict_features(model, data, indices, batch_size, device)
    targets = data["labels"][indices].numpy()
    report = dict(mode=saved["mode"], split=split, best_epoch=saved["best_epoch"],
                  feature_sha256=saved["feature_sha256"], provenance=data.get("provenance", {}),
                  metrics=classification_metrics(targets, probabilities, saved["label_names"], saved["threshold"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", probabilities=probabilities, targets=targets, embeddings=embeddings,
                        sample_ids=np.array([data["sample_ids"][i] for i in indices]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "features", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    print(json.dumps(evaluate_checkpoint(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
