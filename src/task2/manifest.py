"""Build a Task 2 audio manifest with Task 1's independent labels and ID splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..task1.audioset import LABEL_SOURCE, parse_labels

SPLIT_NAMES = ("train", "validation", "test")
AUDIO_MODES = ("pretrimmed", "full_source")


def assign_splits(
    sample_count: int,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> list[str]:
    """Return deterministic split names without touching labels or test metrics."""
    if sample_count < 7:
        raise ValueError("At least seven audio samples are required")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than one")
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(sample_count * test_size))
    validation_count = max(1, round(sample_count * val_size))
    if test_count + validation_count >= sample_count:
        raise ValueError("Split sizes leave no training samples")
    names = ["train"] * sample_count
    for index in indices[:test_count]:
        names[index] = "test"
    for index in indices[test_count : test_count + validation_count]:
        names[index] = "validation"
    return names


def _audio_candidates(audio_dir: Path, video_id: str, start_s: float, end_s: float) -> list[Path]:
    start_s, end_s = f"{start_s:g}", f"{end_s:g}"
    stems = (
        video_id,
        f"{video_id}_{start_s}",
        f"{video_id}_{start_s}_{end_s}",
        f"{video_id}-{start_s}",
    )
    extensions = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
    return [audio_dir / f"{stem}{extension}" for stem in stems for extension in extensions]


def find_audio(
    audio_dir: Path, video_id: str, start_s: float, end_s: float, *, audio_mode: str | None = None
) -> Path | None:
    candidates = _audio_candidates(audio_dir, video_id, start_s, end_s)
    if audio_mode == "full_source":
        # Only unsuffixed sources can unambiguously satisfy source-time offsets.
        candidates = [path for path in candidates if path.stem == video_id]
    elif audio_mode == "pretrimmed":
        # Prefer an explicitly interval-named clip if both source and clip exist.
        candidates.sort(key=lambda path: (path.stem == video_id, path.stem != f"{video_id}_{start_s:g}_{end_s:g}"))
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        None,
    )


def _validate_task1_labels(
    label_frame: pd.DataFrame, task1_manifest: dict[str, Any]
) -> tuple[list[str], dict[str, str], pd.DataFrame]:
    """Validate the complete source cohort before intersecting with available audio."""
    if (task1_manifest.get("label_source") != LABEL_SOURCE
            or task1_manifest.get("vocabulary_fit_split") != "train"
            or task1_manifest.get("split_before_vocabulary") is not True):
        raise ValueError("Task 1 manifest must use independent AudioSet labels fitted on train")
    tags = task1_manifest.get("tags")
    if (not isinstance(tags, list) or not tags or len(tags) != len(set(tags))
            or any(not isinstance(tag, str) for tag in tags)):
        raise ValueError("Task 1 manifest must contain unique, nonempty tags")
    if label_frame.columns.tolist() != ["ytid", "text", *tags]:
        raise ValueError("Task 1 label CSV schema/tag order must equal ['ytid', 'text', *tags]")
    frame = label_frame.copy()
    if frame.ytid.isna().any() or frame.ytid.astype(str).str.strip().eq("").any():
        raise ValueError("Task 1 label CSV contains empty video IDs")
    frame["ytid"] = frame.ytid.astype(str)
    if frame.ytid.duplicated().any():
        raise ValueError("Task 1 label CSV must contain unique video IDs")
    values = frame[tags].apply(pd.to_numeric, errors="raise")
    if not np.isin(values.to_numpy(), [0, 1]).all():
        raise ValueError("Task 1 labels must be finite binary values")
    frame[tags] = values.astype(int)
    split_ids = task1_manifest.get("split_ids")
    if not isinstance(split_ids, dict) or set(split_ids) != set(SPLIT_NAMES):
        raise ValueError("Task 1 manifest must define train, validation, and test split_ids")
    membership: dict[str, str] = {}
    for split in SPLIT_NAMES:
        ids = split_ids[split]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) or not value.strip() for value in ids)):
            raise ValueError(f"Task 1 {split} split_ids must be a nonempty list of video IDs")
        for video_id in ids:
            if video_id in membership:
                raise ValueError("Task 1 video split_ids contain duplicate IDs or split overlap")
            membership[video_id] = split
    if set(frame.ytid) != set(membership):
        raise ValueError("Task 1 label CSV video IDs must exactly match its split_ids")
    if task1_manifest.get("rows", len(frame)) != len(frame):
        raise ValueError("Task 1 manifest row count disagrees with label CSV")
    if "train_support" in task1_manifest:
        train_rows = frame.ytid.map(membership).eq("train")
        if frame.loc[train_rows, tags].sum().to_dict() != task1_manifest["train_support"]:
            raise ValueError("Task 1 training label support disagrees with label CSV")
    return tags, membership, frame


def build_aligned_musiccaps_manifest(
    metadata: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    task1_manifest: dict[str, Any],
    audio_dir: Path,
    audio_mode: str = "pretrimmed",
    include_missing: bool = False,
) -> dict[str, Any]:
    """Intersect local audio with Task 1 video IDs without refitting labels or splits.

    ``pretrimmed`` means each local file already starts at MusicCaps ``start_s``;
    ``full_source`` means it starts at source time zero. Both read exactly the
    metadata interval duration. The mode is an explicit caller assertion, never
    inferred from a filename or the presence of missing audio.
    """
    if audio_mode not in AUDIO_MODES:
        raise ValueError(f"audio_mode must be one of {AUDIO_MODES}")
    required = {"ytid", "start_s", "end_s", "caption"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata is missing columns: {sorted(required - set(metadata.columns))}")
    tags, membership, labels = _validate_task1_labels(label_frame, task1_manifest)
    metadata = metadata.copy()
    if metadata.ytid.isna().any() or metadata.ytid.astype(str).str.strip().eq("").any():
        raise ValueError("Metadata contains empty video IDs")
    metadata["ytid"] = metadata.ytid.astype(str)
    if metadata.ytid.duplicated().any():
        raise ValueError("Metadata must contain one MusicCaps interval per video ID")
    if metadata.caption.isna().any():
        raise ValueError("Metadata contains missing captions")
    # Sorting keeps cohort order independent of input CSV order or directory listing.
    joined = metadata.merge(labels[["ytid", *tags]], on="ytid", how="inner", validate="one_to_one")
    joined = joined.sort_values("ytid")
    records: list[dict[str, Any]] = []
    missing_audio: list[str] = []
    audio_found_by_split = {split: 0 for split in SPLIT_NAMES}
    missing_by_split = {split: 0 for split in SPLIT_NAMES}
    for row in joined.to_dict("records"):
        video_id = row["ytid"]
        if any(character in video_id for character in ("/", "\\")) or video_id in (".", ".."):
            raise ValueError("Video IDs cannot contain path separators")
        start_s, end_s = float(row["start_s"]), float(row["end_s"])
        if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < 0 or end_s <= start_s:
            raise ValueError(f"Invalid MusicCaps interval for {video_id}")
        if "audioset_positive_labels" in row:
            official = set(parse_labels(row["audioset_positive_labels"]))
            if [int(row[tag]) for tag in tags] != [int(tag in official) for tag in tags]:
                raise ValueError(f"Task 1 labels disagree with official AudioSet labels for {video_id}")
        sample_id = f"{video_id}_{start_s:g}_{end_s:g}"
        split = membership[video_id]
        audio_path = find_audio(audio_dir, video_id, start_s, end_s, audio_mode=audio_mode)
        if audio_path is None:
            missing_audio.append(sample_id)
            missing_by_split[split] += 1
            if not include_missing:
                continue
            audio_path = _audio_candidates(audio_dir, video_id, start_s, end_s)[0].resolve()
        else:
            audio_found_by_split[split] += 1
        records.append({
            "sample_id": sample_id, "ytid": video_id, "audio_path": str(audio_path),
            "start_s": start_s, "end_s": end_s, "audio_mode": audio_mode,
            "audio_offset_seconds": start_s if audio_mode == "full_source" else 0.0,
            "audio_duration_seconds": end_s - start_s,
            "labels": [int(row[tag]) for tag in tags], "text": str(row["caption"]),
            "split": split,
        })
    split_counts = {split: sum(record["split"] == split for record in records) for split in SPLIT_NAMES}
    label_support = {
        split: {tag: sum(record["labels"][i] for record in records if record["split"] == split)
                for i, tag in enumerate(tags)} for split in SPLIT_NAMES
    }
    return {
        "schema_version": 2, "dataset": "MusicCaps local audio / Task 1 AudioSet intersection",
        "label_source": LABEL_SOURCE, "label_names": tags,
        "seed": task1_manifest.get("seed"), "audio_mode": audio_mode,
        "split_strategy": "inherited_task1_video_ids_available_audio_intersection",
        "task1_alignment": {
            "label_order_preserved": True, "video_splits_preserved": True,
            "vocabulary_fit_split": "train", "source_csv_sha256": task1_manifest.get("csv_sha256"),
            "source_split_counts": {split: len(task1_manifest["split_ids"][split]) for split in SPLIT_NAMES},
            "comparison_scope": "Use this same available-audio cohort for all Task 2 models; Task 1 full-cohort scores use different samples.",
        },
        "records": records,
        "audit": {
            "metadata_rows": len(metadata), "label_rows": len(labels), "id_matches": len(joined),
            "metadata_ids_outside_task1": sorted(set(metadata.ytid) - set(membership)),
            "task1_ids_without_metadata": sorted(set(membership) - set(metadata.ytid)),
            "audio_found": sum(audio_found_by_split.values()), "audio_missing": len(missing_audio),
            "missing_sample_ids": missing_audio, "audio_found_by_split": audio_found_by_split,
            "audio_missing_by_split": missing_by_split, "split_counts": split_counts,
            "label_support_by_split": label_support,
            "ready_for_training": all(split_counts.values()) and not (include_missing and missing_audio),
        },
    }


def build_musiccaps_manifest(
    metadata: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    audio_dir: Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    include_missing: bool = False,
) -> dict[str, Any]:
    """Legacy caption-label join, retained for old smoke fixtures only.

    Use ``build_aligned_musiccaps_manifest`` for actual Task 2 experiments.
    This function cannot preserve Task 1's current independent targets or splits.
    """
    required_metadata = {"ytid", "start_s", "end_s", "caption"}
    missing = sorted(required_metadata - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    if "text" not in label_frame.columns:
        raise ValueError("Label CSV must contain a 'text' caption column")
    if "ytid" in label_frame.columns:
        raise ValueError("Use build_aligned_musiccaps_manifest for ID-based Task 1 labels")
    label_names = [column for column in label_frame.columns if column != "text"]
    if not label_names:
        raise ValueError("Label CSV must contain at least one binary label column")
    numeric = label_frame[label_names].apply(pd.to_numeric, errors="raise")
    if not set(np.unique(numeric.to_numpy())).issubset({0, 1}):
        raise ValueError("All label columns must contain only 0 and 1")
    labels = label_frame.copy()
    labels[label_names] = numeric.astype(int)
    if labels["text"].duplicated().any() or metadata["caption"].duplicated().any():
        raise ValueError("Caption join keys must be unique")
    joined = metadata.merge(labels, left_on="caption", right_on="text", how="inner", validate="one_to_one")
    records = []
    missing_audio = []
    for row in joined.to_dict("records"):
        video_id = str(row["ytid"])
        start_s, end_s = int(row["start_s"]), int(row["end_s"])
        audio_path = find_audio(audio_dir, video_id, start_s, end_s)
        sample_id = f"{video_id}_{start_s}_{end_s}"
        if audio_path is None:
            missing_audio.append(sample_id)
            if not include_missing:
                continue
            audio_path = _audio_candidates(audio_dir, video_id, start_s, end_s)[0].resolve()
        records.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "labels": [int(row[label]) for label in label_names],
                "text": str(row["caption"]),
            }
        )
    if records:
        for record, split in zip(
            records,
            assign_splits(len(records), val_size=val_size, test_size=test_size, seed=seed),
        ):
            record["split"] = split
    return {
        "schema_version": 1,
        "dataset": "MusicCaps local audio",
        "label_source": "legacy_caption_derived_labels",
        "split_strategy": "legacy_independent_random_rows_not_task1_aligned",
        "seed": seed,
        "label_names": label_names,
        "records": records,
        "audit": {
            "metadata_rows": int(len(metadata)),
            "label_rows": int(len(label_frame)),
            "caption_matches": int(len(joined)),
            "audio_found": len(records) - (len(missing_audio) if include_missing else 0),
            "audio_missing": len(missing_audio),
            "missing_sample_ids": missing_audio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task1-manifest", type=Path, help="Task 1 independent AudioSet dataset_manifest.json")
    source.add_argument("--legacy-caption-labels", action="store_true", help="Explicitly opt into old caption-derived labels and independently generated splits")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-mode", choices=AUDIO_MODES, help="Required for Task 1 alignment: pretrimmed clips or full_source recordings")
    parser.add_argument("--output", type=Path, default=Path("data/splits/task2_manifest.json"))
    parser.add_argument("--missing-log", type=Path, default=Path("results/task2/missing_audio.json"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()
    if args.task1_manifest:
        if args.audio_mode is None:
            parser.error("--audio-mode is required with --task1-manifest; declare whether audio is already trimmed")
        task1_manifest = json.loads(args.task1_manifest.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(args.labels_csv.read_bytes()).hexdigest()
        if actual_hash != task1_manifest.get("csv_sha256"):
            parser.error("Task 1 label CSV hash does not match --task1-manifest")
        manifest = build_aligned_musiccaps_manifest(
            pd.read_csv(args.metadata_csv, dtype={"ytid": str}),
            pd.read_csv(args.labels_csv, dtype={"ytid": str}),
            task1_manifest=task1_manifest, audio_dir=args.audio_dir,
            audio_mode=args.audio_mode, include_missing=args.include_missing,
        )
        manifest["task1_alignment"].update({
            "source_manifest": str(args.task1_manifest.resolve()),
            "source_manifest_sha256": hashlib.sha256(args.task1_manifest.read_bytes()).hexdigest(),
            "source_csv": str(args.labels_csv.resolve()), "source_csv_hash_verified": True,
        })
    else:
        manifest = build_musiccaps_manifest(
            pd.read_csv(args.metadata_csv), pd.read_csv(args.labels_csv),
            audio_dir=args.audio_dir, val_size=args.val_size, test_size=args.test_size,
            seed=args.seed, include_missing=args.include_missing,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.missing_log.parent.mkdir(parents=True, exist_ok=True)
    args.missing_log.write_text(
        json.dumps(manifest["audit"]["missing_sample_ids"], indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest["audit"].items() if key != "missing_sample_ids"}, indent=2))
    print(f"Saved {len(manifest['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
