"""ID-first MusicCaps preparation with independent, training-fitted AudioSet targets."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "validation", "test")
LABEL_SOURCE = "MusicCaps.audioset_positive_labels"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_labels(value) -> list[str]:
    """Read official comma-separated IDs; never infer a target from caption text."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            value = ast.literal_eval(value)
        else:
            value = value.split(",") if value else []
    if not isinstance(value, (list, tuple)):
        raise ValueError("AudioSet labels must be an explicit list or comma-separated IDs")
    labels = sorted({str(label).strip() for label in value})
    if any(not re.fullmatch(r"/(?:m|t)/[A-Za-z0-9_]+", label) for label in labels):
        raise ValueError("Invalid AudioSet label ID")
    return labels


def split_by_id(ids, *, seed=42, val_size=0.15, test_size=0.15):
    """Assign sorted unique video IDs, independent of row order and all labels."""
    unique = sorted(set(ids))
    if len(unique) < 7:
        raise ValueError("At least seven unique ytid values are required")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    train_ids, held = train_test_split(unique, test_size=val_size + test_size, random_state=seed)
    val_ids, test_ids = train_test_split(held, test_size=test_size / (val_size + test_size), random_state=seed)
    return dict(zip(SPLIT_NAMES, [sorted(train_ids), sorted(val_ids), sorted(test_ids)]))


def build_audioset_frame(raw, *, top_k=30, min_train_support=20, seed=42,
                         val_size=0.15, test_size=0.15, caption_col="caption"):
    required = {"ytid", caption_col, "audioset_positive_labels"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing official fields: {sorted(required - set(raw.columns))}")
    if top_k <= 0 or min_train_support <= 0:
        raise ValueError("top_k and min_train_support must be positive")
    work = raw.copy().reset_index(drop=True)
    for name in ("ytid", caption_col):
        if work[name].isna().any() or work[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty {name}; resolve explicitly before splitting")
        work[name] = work[name].astype(str).str.strip()
    # This assignment precedes parsing, counting, filtering, or selecting labels.
    split_ids = split_by_id(work.ytid, seed=seed, val_size=val_size, test_size=test_size)
    membership = {value: name for name, values in split_ids.items() for value in values}
    row_splits = work.ytid.map(membership)
    if pd.DataFrame({"caption": work[caption_col], "split": row_splits}).groupby("caption")["split"].nunique().gt(1).any():
        raise ValueError("Duplicate captions cross ID splits; resolve duplicate groups before preparation")
    hits = work.audioset_positive_labels.map(parse_labels)
    train_hits = hits[row_splits == "train"]
    counts = Counter(label for labels in train_hits for label in labels)
    # Exclude train-constant targets (e.g. a universal Music root) using train only.
    eligible = [label for label, n in counts.items() if min_train_support <= n < len(train_hits)]
    tags = sorted(eligible, key=lambda label: (-counts[label], label))[:top_k]
    if not tags:
        raise ValueError("No nonconstant labels meet training support; choose a lower support cutoff")
    result = pd.DataFrame({"ytid": work.ytid, "text": work[caption_col]})
    for tag in tags:
        result[tag] = hits.map(lambda labels: int(tag in labels))
    # Keep every row, including held-out examples with no in-vocabulary positives.
    splits = {name: row_splits.index[row_splits == name].tolist() for name in SPLIT_NAMES}
    manifest = {
        "schema_version": 1, "label_source": LABEL_SOURCE,
        "source_url": "https://huggingface.co/datasets/google/MusicCaps",
        "input_fields": [caption_col], "id_field": "ytid",
        "vocabulary_fit_split": "train", "split_before_vocabulary": True,
        "seed": seed, "val_size": val_size, "test_size": test_size,
        "top_k": top_k, "min_train_support": min_train_support,
        "tags": tags, "train_support": {tag: counts[tag] for tag in tags},
        "excluded_train_constant_labels": sorted(label for label, n in counts.items() if n == len(train_hits)),
        "split_ids": split_ids, "split_indices": splits,
        "rows": len(work), "unique_ids": work.ytid.nunique(),
        "zero_positive_rows": {name: int(result.loc[indices, tags].sum(axis=1).eq(0).sum()) for name, indices in splits.items()},
    }
    return result, manifest


def save_audioset_dataset(raw, output: str | Path, **kwargs):
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Use a new dataset path; artifacts already exist at {output}")
    frame, manifest = build_audioset_frame(raw, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    manifest["csv_sha256"] = sha256_file(output)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def load_prepared_manifest(path):
    """Verify prepared CSV identity and ID partition before any training."""
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("label_source") != LABEL_SOURCE or manifest.get("vocabulary_fit_split") != "train" or not manifest.get("split_before_vocabulary"):
        raise ValueError("Invalid independent-label preparation provenance")
    if manifest.get("csv_sha256") != sha256_file(path):
        raise ValueError("Prepared CSV hash mismatch; re-prepare to a new path")
    frame = pd.read_csv(path, dtype={"ytid": str})
    if frame.columns.tolist() != ["ytid", "text", *manifest["tags"]]:
        raise ValueError("Prepared schema/tag order mismatch")
    splits = manifest["split_indices"]
    flat = [i for values in splits.values() for i in values]
    if set(splits) != set(SPLIT_NAMES) or not all(splits.values()) or any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(frame))):
        raise ValueError("Prepared row splits must be nonempty, disjoint and exhaustive")
    seen = set()
    for name, indices in splits.items():
        ids = set(frame.iloc[indices].ytid)
        if seen & ids or ids != set(manifest["split_ids"][name]):
            raise ValueError("Prepared ytid split mismatch or overlap")
        seen.update(ids)
    actual = frame.iloc[splits["train"]][manifest["tags"]].sum().to_dict()
    if actual != manifest["train_support"]:
        raise ValueError("Training label support mismatch")
    return manifest
