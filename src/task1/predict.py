"""Run Task 1 inference from either a new metadata-rich or legacy checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .train import BertTagClassifier


def load_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path | None = None,
) -> tuple[dict, dict]:
    """Return ``(state_dict, metadata)`` for current and legacy checkpoints."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and "state_dict" in saved:
        metadata = {
            "model_name": saved["config"]["model_name"],
            "max_length": saved["config"]["max_length"],
            "threshold": saved["config"]["threshold"],
            "thresholds": saved.get("thresholds", [saved["config"]["threshold"]] * len(saved["tags"])),
            "label_source": saved.get("label_source", "legacy_unverified"),
            "tags": saved["tags"],
        }
        return saved["state_dict"], metadata

    metadata_path = metadata_path or checkpoint_path.with_name("model_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Legacy checkpoint requires metadata; expected {metadata_path} or pass --metadata"
        )
    return saved, json.loads(metadata_path.read_text(encoding="utf-8"))


def predict(
    texts: list[str],
    checkpoint_path: Path,
    *,
    metadata_path: Path | None = None,
    device_name: str = "auto",
    top_k: int = 5,
) -> list[dict]:
    state_dict, metadata = load_checkpoint(checkpoint_path, metadata_path)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    tags = metadata["tags"]
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
    model = BertTagClassifier(metadata["model_name"], len(tags)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(metadata.get("max_length", 128)),
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        probabilities = torch.sigmoid(model(**encoded)).cpu()
    thresholds = torch.tensor(metadata.get("thresholds", [metadata.get("threshold", 0.5)] * len(tags)))
    names_path = Path(__file__).with_name("audioset_names.json")
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    results = []
    for text, row in zip(texts, probabilities):
        order = torch.argsort(row, descending=True)[:top_k].tolist()
        results.append(
            {
                "text": text,
                "predicted_tags": [
                    {"tag": tags[index], "display_name": names.get(tags[index], tags[index]), "probability": round(float(row[index]), 4)}
                    for index in torch.argsort(row, descending=True).tolist()
                    if row[index] >= thresholds[index]
                ],
                "top_tags": [
                    {"tag": tags[index], "display_name": names.get(tags[index], tags[index]), "probability": round(float(row[index]), 4)}
                    for index in order
                ],
                "thresholds": thresholds.tolist(),
                "label_source": metadata.get("label_source", "legacy_unverified"),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--text", action="append", required=True, help="Caption; repeat for a batch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    results = predict(
        args.text,
        args.checkpoint,
        metadata_path=args.metadata,
        device_name=args.device,
        top_k=args.top_k,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
