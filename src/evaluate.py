"""Evaluate a saved classifier checkpoint and save genuine test metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch, yaml
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from .train import MusicDataset, build_model, collate, forward

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config.yaml"); parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", required=True); parser.add_argument("--output", default="results/metrics/test_metrics.json")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text()); device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); choice = checkpoint["model"]
    model = build_model(cfg, choice).to(device); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    loader = DataLoader(MusicDataset(args.split), batch_size=cfg["training"]["batch_size"], collate_fn=collate); targets, scores = [], []
    with torch.no_grad():
        for batch in loader:
            scores.append(torch.sigmoid(forward(model, choice, batch, cfg["data"]["max_text_length"], device)).cpu().numpy()); targets.append(batch["labels"].numpy())
    y_true, y_score = np.concatenate(targets), np.concatenate(scores); y_pred = (y_score >= 0.5).astype(int)
    metrics = {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)), "mean_auc_pr": float(average_precision_score(y_true, y_score, average="macro")), "threshold": 0.5, "samples": int(len(y_true))}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(metrics, indent=2), encoding="utf-8"); print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
