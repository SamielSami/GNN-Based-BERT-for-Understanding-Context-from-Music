"""Export Task 1 text training inputs using the exact Task 2 sample splits."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.task2.dataset import load_manifest


def align_text(manifest_path, output_dir):
    manifest = load_manifest(manifest_path)
    rows, splits = [], {s: [] for s in ("train", "validation", "test")}
    names = manifest["label_names"]
    for i, record in enumerate(manifest["records"]):
        if record["split"] not in splits or not record.get("text", "").strip():
            raise ValueError("Each record requires a caption and valid split")
        if len(record["labels"]) != len(names) or any(v not in (0, 1) for v in record["labels"]):
            raise ValueError("Targets must be binary and match the label vocabulary")
        rows.append({"text": record["text"].strip(), **dict(zip(names, record["labels"]))})
        splits[record["split"]].append(i)
    frame = pd.DataFrame(rows)
    if not all(splits.values()) or frame["text"].duplicated().any():
        raise ValueError("Require three non-empty splits and unique captions")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "tags.csv", index=False)
    (output / "split_indices.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(align_text(args.manifest, args.output_dir))
