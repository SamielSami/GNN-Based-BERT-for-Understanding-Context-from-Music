"""Audit MusicCaps metadata and optionally probe clip availability without downloading audio.

This module implements the data-viability checks from Part 1 of the project
plan.  It deliberately keeps availability probing separate from audio
acquisition: ``yt-dlp`` is invoked with ``skip_download=True`` and no media is
saved.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "ytid",
    "start_s",
    "end_s",
    "audioset_positive_labels",
    "aspect_list",
    "caption",
    "author_id",
    "is_balanced_subset",
    "is_audioset_eval",
}


def parse_list(value: Any, *, separator: str = ",") -> list[str]:
    """Parse a list-valued MusicCaps field into stripped strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text[:1] in "[({" and text[-1:] in "])}":
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(separator) if item.strip()]


def canonicalize_label(label: str) -> str:
    """Return a conservative canonical form for frequency/overlap auditing."""
    text = unicodedata.normalize("NFKC", str(label)).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    """Check a normalized phrase as a whole phrase in normalized text."""
    phrase = canonicalize_label(phrase)
    text = canonicalize_label(text)
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _count_values(rows: Iterable[list[str]], *, canonicalize: bool = True) -> Counter[str]:
    def cleaned(item: str) -> str:
        return canonicalize_label(item) if canonicalize else str(item).strip()

    return Counter(cleaned(item) for row in rows for item in row if cleaned(item))


def _top_counts(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def validate_frame(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"MusicCaps metadata is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("MusicCaps metadata is empty")


def build_metadata_audit(frame: pd.DataFrame, *, top_k: int = 30) -> dict[str, Any]:
    """Compute deterministic metadata, label, and lexical-overlap diagnostics."""
    validate_frame(frame)
    work = frame.copy()
    work["ytid"] = work["ytid"].fillna("").astype(str).str.strip()
    work["caption"] = work["caption"].fillna("").astype(str).str.strip()
    aspect_rows = work["aspect_list"].map(parse_list)
    audioset_rows = work["audioset_positive_labels"].map(parse_list)
    aspect_counts = _count_values(aspect_rows)
    # AudioSet machine identifiers (for example ``/m/04rlf``) are opaque IDs;
    # normalizing punctuation would corrupt them.
    audioset_counts = _count_values(audioset_rows, canonicalize=False)

    any_overlap = []
    overlap_counts = []
    for caption, aspects in zip(work["caption"], aspect_rows):
        overlaps = {canonicalize_label(item) for item in aspects if phrase_in_text(item, caption)}
        any_overlap.append(bool(overlaps))
        overlap_counts.append(len(overlaps))

    duplicate_clip_ids = int(work["ytid"].duplicated(keep=False).sum())
    sample_keys = work["ytid"] + ":" + work["start_s"].astype(str) + ":" + work["end_s"].astype(str)
    duplicate_sample_keys = int(sample_keys.duplicated(keep=False).sum())
    duration = pd.to_numeric(work["end_s"], errors="coerce") - pd.to_numeric(
        work["start_s"], errors="coerce"
    )
    missing_by_column = {column: int(work[column].isna().sum()) for column in REQUIRED_COLUMNS}
    missing_by_column["ytid"] += int((work["ytid"] == "").sum())
    missing_by_column["caption"] += int((work["caption"] == "").sum())

    return {
        "metadata": {
            "rows": int(len(work)),
            "unique_video_ids": int(work["ytid"].nunique()),
            "duplicate_video_id_rows": duplicate_clip_ids,
            "unique_clip_windows": int(sample_keys.nunique()),
            "duplicate_clip_window_rows": duplicate_sample_keys,
            "exact_duplicate_captions": int(work["caption"].duplicated(keep=False).sum()),
            "missing_by_column": dict(sorted(missing_by_column.items())),
            "duration_seconds": {
                "minimum": float(duration.min()),
                "median": float(duration.median()),
                "maximum": float(duration.max()),
                "invalid_or_nonpositive": int((duration <= 0).sum() + duration.isna().sum()),
            },
            "authors": int(work["author_id"].nunique()),
            "balanced_subset_rows": int(work["is_balanced_subset"].fillna(False).astype(bool).sum()),
            "audioset_eval_rows": int(work["is_audioset_eval"].fillna(False).astype(bool).sum()),
        },
        "label_analysis": {
            "aspect_unique_canonical_labels": len(aspect_counts),
            "audioset_unique_label_ids": len(audioset_counts),
            "top_aspects_global_audit_only": _top_counts(aspect_counts, top_k),
            "top_audioset_label_ids_global_audit_only": _top_counts(audioset_counts, top_k),
            "aspect_count_per_row": {
                "minimum": int(aspect_rows.map(len).min()),
                "median": float(aspect_rows.map(len).median()),
                "maximum": int(aspect_rows.map(len).max()),
            },
            "warning": (
                "These global counts are for viability analysis only. The final label vocabulary "
                "and canonical synonym map must be fitted on the training split only."
            ),
        },
        "lexical_overlap": {
            "rows_with_any_exact_aspect_phrase_in_caption": int(sum(any_overlap)),
            "row_rate": float(sum(any_overlap) / len(work)),
            "mean_overlapping_aspects_per_row": float(sum(overlap_counts) / len(work)),
            "interpretation": (
                "Aspect phrases and captions describe the same clip and often overlap lexically. "
                "Aspect labels must never be concatenated into the BERT input; keyword and TF-IDF "
                "controls plus overlap-stratified reporting are required."
            ),
        },
    }


def _probe_one(record: dict[str, Any], timeout: int) -> dict[str, Any]:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install yt-dlp to use --probe-count") from exc

    video_id = str(record["ytid"])
    url = f"https://www.youtube.com/watch?v={video_id}"
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": timeout,
        "retries": 0,
        "extractor_retries": 0,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as client:
            info = client.extract_info(url, download=False)
        status = "available" if info and info.get("id") == video_id else "technical_error"
        reason = None if status == "available" else "Extractor returned no matching video metadata"
    except Exception as exc:  # yt-dlp intentionally exposes extractor-specific exception types
        reason = re.sub(r"\s+", " ", str(exc)).strip()[:500]
        unavailable_markers = (
            "video unavailable",
            "private video",
            "has been removed",
            "not available",
            "copyright",
            "account associated with this video has been terminated",
        )
        status = "unavailable" if any(marker in reason.lower() for marker in unavailable_markers) else "technical_error"
    return {
        "ytid": video_id,
        "start_s": int(record["start_s"]),
        "end_s": int(record["end_s"]),
        "status": status,
        "reason": reason,
    }


def probe_availability(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
    workers: int,
    timeout: int,
) -> list[dict[str, Any]]:
    """Probe a deterministic sample with yt-dlp metadata extraction only."""
    if count < 1:
        return []
    if count > len(frame):
        raise ValueError(f"probe count {count} exceeds dataset size {len(frame)}")
    indices = random.Random(seed).sample(range(len(frame)), count)
    records = frame.iloc[indices][["ytid", "start_s", "end_s"]].to_dict("records")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_probe_one, record, timeout): record for record in records}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 20 == 0 or completed == count:
                print(f"Availability probes completed: {completed}/{count}", flush=True)
    return sorted(results, key=lambda item: (item["ytid"], item["start_s"]))


def summarize_availability(results: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    if not results:
        return {
            "status": "not_run",
            "probe_count": 0,
            "threshold": threshold,
            "gate": "pending",
            "note": "Run with --probe-count 200 to execute the plan's deterministic pilot.",
        }
    counts = Counter(item["status"] for item in results)
    available_rate = counts["available"] / len(results)
    technical_rate = counts["technical_error"] / len(results)
    # A high technical-error rate cannot establish real clip availability.
    conclusive = technical_rate <= 0.10
    gate = "pass" if conclusive and available_rate >= threshold else "fail" if conclusive else "inconclusive"
    return {
        "status": "completed",
        "method": "yt-dlp metadata extraction with skip_download=True; no media saved",
        "probe_count": len(results),
        "seed": None,
        "counts": dict(sorted(counts.items())),
        "available_rate": available_rate,
        "technical_error_rate": technical_rate,
        "threshold": threshold,
        "gate": gate,
        "interpretation": (
            "Technical errors above 10% make the pilot inconclusive rather than proving clips unavailable. "
            "Availability is time-, region-, network-, and platform-dependent."
        ),
    }


def save_label_plot(audit: dict[str, Any], destination: Path) -> None:
    values = audit["label_analysis"]["top_aspects_global_audit_only"]
    labels = [row["label"] for row in reversed(values)]
    counts = [row["count"] for row in reversed(values)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    height = max(6.0, len(labels) * 0.28)
    fig, axis = plt.subplots(figsize=(10, height))
    axis.barh(labels, counts, color="#3568a8")
    axis.set_xlabel("MusicCaps rows containing aspect")
    axis.set_title("MusicCaps candidate aspect frequency (global audit only)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.input_csv:
        return pd.read_csv(args.input_csv), str(args.input_csv)
    if args.input_arrow:
        from datasets import Dataset

        return Dataset.from_file(str(args.input_arrow)).to_pandas(), str(args.input_arrow)
    from datasets import load_dataset

    dataset = load_dataset("google/MusicCaps", split="train", cache_dir=str(args.cache_dir))
    return dataset.to_pandas(), "huggingface:google/MusicCaps@train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", action="store_true")
    source.add_argument("--input-csv", type=Path)
    source.add_argument("--input-arrow", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache/huggingface"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.json"))
    parser.add_argument(
        "--failure-log", type=Path, default=Path("results/musiccaps_availability_failures.jsonl")
    )
    parser.add_argument(
        "--label-plot", type=Path, default=Path("results/plots/musiccaps_label_frequency.png")
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--probe-count", type=int, default=0)
    parser.add_argument("--availability-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    if not 0 < args.availability_threshold <= 1:
        parser.error("--availability-threshold must be in (0, 1]")
    if args.top_k < 1 or args.workers < 1 or args.timeout < 1:
        parser.error("--top-k, --workers, and --timeout must be positive")

    frame, source_name = load_source(args)
    audit = {
        "audit_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        **build_metadata_audit(frame, top_k=args.top_k),
    }
    probes = probe_availability(
        frame,
        count=args.probe_count,
        seed=args.seed,
        workers=args.workers,
        timeout=args.timeout,
    )
    availability = summarize_availability(probes, threshold=args.availability_threshold)
    if probes:
        availability["seed"] = args.seed
    audit["availability"] = availability
    audit["storage"] = {
        "decoded_pcm_22050hz_mono_16bit_gb": round(len(frame) * 10 * 22050 * 2 / 1e9, 3),
        "recommended_workspace_allowance_gb": "10-20",
    }
    audit["licensing"] = {
        "metadata": "MusicCaps metadata/annotations: CC BY-SA 4.0 per the dataset card",
        "audio": "Referenced YouTube audio has separate rights and platform constraints; do not redistribute it",
    }
    audit["decision"] = {
        "metadata_gate": "pass",
        "audio_availability_gate": availability["gate"],
        "instructor_approval": "pending_human_action",
        "route": (
            "musiccaps_pending_instructor_approval"
            if availability["gate"] == "pass"
            else "fma_fallback_recommended"
            if availability["gate"] == "fail"
            else "pending_conclusive_availability_probe"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    failures = [item for item in probes if item["status"] != "available"]
    args.failure_log.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures), encoding="utf-8"
    )
    save_label_plot(audit, args.label_plot)
    print(json.dumps(audit["decision"], indent=2))
    print(f"Saved audit to {args.output}")
    print(f"Saved {len(failures)} non-available probe records to {args.failure_log}")
    print(f"Saved label-frequency plot to {args.label_plot}")


if __name__ == "__main__":
    main()
