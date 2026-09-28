"""Evaluate a saved Task 2 checkpoint on a processed manifest split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ProcessedTask2Dataset, collate_task2, dataset_identity
from .models import build_task2_model
from .train import classification_metrics, resolve_device, run_loader


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest: str | Path,
    *,
    split: str = "test",
    device_name: str = "auto",
    output: str | Path | None = None,
) -> dict:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset = ProcessedTask2Dataset(manifest, split, config["graph_variant"])
    if dataset.label_names != checkpoint["label_names"]:
        raise ValueError("Manifest label names/order do not match the checkpoint")
    if not checkpoint.get("dataset_fingerprint"):
        raise ValueError("Checkpoint lacks dataset provenance; retrain with the current Task 2 pipeline")
    if dataset_identity(manifest)["sha256"] != checkpoint["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint does not match the checkpoint (samples, splits or features changed)")
    torch.set_num_threads(config.get("cpu_threads", 6))
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_task2,
    )
    model = build_task2_model(
        config["model"],
        input_dim=checkpoint["input_dim"],
        num_labels=checkpoint["num_labels"],
        hidden_dim=config["hidden_dim"],
        layers=config["layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    start_time = time.perf_counter()
    result = run_loader(model, config["model"], loader, device, nn.BCEWithLogitsLoss())
    elapsed = time.perf_counter() - start_time
    metrics = classification_metrics(
        result["targets"],
        result["probabilities"],
        checkpoint["label_names"],
        checkpoint["threshold"],
    )
    metrics.update(split=split, inference_seconds=elapsed, dataset_fingerprint=checkpoint["dataset_fingerprint"],
                   synthetic=checkpoint.get("synthetic", False), parameter_count=checkpoint["parameter_count"])
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(
            output.with_suffix(".predictions.npz"),
            probabilities=result["probabilities"],
            targets=result["targets"],
            sample_ids=np.asarray(result["sample_ids"]),
            label_names=np.asarray(checkpoint["label_names"]),
            threshold=np.asarray(checkpoint["threshold"]),
            predictions=(result["probabilities"] >= checkpoint["threshold"]).astype(np.int8),
        )
        output.with_suffix(".sample_ids.json").write_text(
            json.dumps(result["sample_ids"], indent=2), encoding="utf-8"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/task2/evaluation.json"))
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        split=args.split,
        device_name=args.device,
        output=args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
