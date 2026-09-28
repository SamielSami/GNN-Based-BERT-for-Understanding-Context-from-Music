"""Cache Task 2 node features, graph variants, and log-mel tensors from a manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import load_manifest, resolve_record_path
from .features import (
    AudioFeatureConfig,
    extract_log_mel_from_waveform,
    extract_track_features_from_waveform,
    load_audio,
)
from .graphs import GRAPH_VARIANTS, build_edge_variants, validate_graph_variants


FEATURE_CACHE_VERSION = 2


def safe_sample_filename(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}-{digest}.pt"


def source_fingerprint(record: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Identify source content, including edits that keep the same sample ID."""
    path = resolve_record_path(manifest_path, record["audio_path"]).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "audio_path": str(path), "sha256": digest.hexdigest(),
        "audio_offset_seconds": record.get("audio_offset_seconds", 0.0),
        "audio_duration_seconds": record.get("audio_duration_seconds"),
    }


def extract_raw_sample(
    record: dict[str, Any], manifest_path: Path, config: AudioFeatureConfig
) -> dict[str, Any]:
    audio_path = resolve_record_path(manifest_path, record["audio_path"])
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found for {record['sample_id']}: {audio_path}")
    waveform = load_audio(
        audio_path, config.sample_rate,
        offset_seconds=record.get("audio_offset_seconds", 0.0),
        duration_seconds=record.get("audio_duration_seconds"),
    )
    x = extract_track_features_from_waveform(waveform, config)
    mel = extract_log_mel_from_waveform(waveform, config)[None, ...]
    return {
        "sample_id": record["sample_id"],
        "audio_path": str(audio_path),
        "split": record["split"],
        "labels": torch.tensor(record["labels"], dtype=torch.float32),
        "x_raw": torch.from_numpy(x),
        "mel": torch.from_numpy(mel),
        "feature_config": config.to_dict(),
        "audio_samples": len(waveform),
        "audio_offset_seconds": record.get("audio_offset_seconds", 0.0),
        "audio_duration_seconds": len(waveform) / config.sample_rate,
    }


def fit_train_normalizer(raw_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    mean: np.ndarray | None = None
    squared_deviations: np.ndarray | None = None
    count = 0
    for path in raw_paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if sample["split"] != "train":
            continue
        x = torch.as_tensor(sample["x_raw"], dtype=torch.float64).numpy()
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError(f"Invalid training-node features in {path}")
        batch_mean = x.mean(axis=0)
        batch_deviations = ((x - batch_mean) ** 2).sum(axis=0)
        if mean is None:
            mean, squared_deviations = batch_mean, batch_deviations
        else:
            delta = batch_mean - mean
            squared_deviations += batch_deviations + delta ** 2 * count * len(x) / (count + len(x))
            mean += delta * len(x) / (count + len(x))
        count += len(x)
    if count == 0 or mean is None or squared_deviations is None:
        raise ValueError("Cannot fit normalization: manifest has no training nodes")
    variance = np.maximum(squared_deviations / count, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count


def preprocess_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: AudioFeatureConfig,
    top_k: int = 2,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Run resumable two-pass preprocessing and return the processed manifest path."""
    started = time.perf_counter()
    config.validate()
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if not manifest["records"]:
        raise ValueError("Input manifest contains no records")
    output_dir = Path(output_dir).resolve()
    raw_dir = output_dir / "raw_features"
    sample_dir = output_dir / "samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    for position, record in enumerate(manifest["records"], start=1):
        raw_path = raw_dir / safe_sample_filename(record["sample_id"])
        fingerprint = source_fingerprint(record, manifest_path)
        reuse = False
        if raw_path.exists() and not overwrite:
            cached = torch.load(raw_path, map_location="cpu", weights_only=False)
            reuse = (
                cached.get("feature_config") == config.to_dict()
                and cached.get("source_fingerprint") == fingerprint
                and cached.get("feature_cache_version") == FEATURE_CACHE_VERSION
            )
        if not reuse:
            cached = extract_raw_sample(record, manifest_path, config)
        # Labels and splits belong to the current manifest, never to the cache.
        cached["labels"] = torch.tensor(record["labels"], dtype=torch.float32)
        cached["split"] = record["split"]
        cached["source_fingerprint"] = fingerprint
        cached["feature_cache_version"] = FEATURE_CACHE_VERSION
        torch.save(cached, raw_path)
        raw_paths.append(raw_path)
        if position % 25 == 0 or position == len(manifest["records"]):
            print(f"Raw audio features: {position}/{len(manifest['records'])}", flush=True)

    mean, std, train_node_count = fit_train_normalizer(raw_paths)
    normalization = {
        "fitted_on": "training nodes only",
        "train_node_count": train_node_count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )

    processed_records = []
    for position, (record, raw_path) in enumerate(zip(manifest["records"], raw_paths), start=1):
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        x = (torch.as_tensor(raw["x_raw"]).numpy() - mean) / std
        sample_seed = seed + int(hashlib.sha1(record["sample_id"].encode()).hexdigest()[:8], 16)
        edge_indices = build_edge_variants(x, top_k=top_k, seed=sample_seed)
        validate_graph_variants(x, edge_indices)
        processed_path = sample_dir / safe_sample_filename(record["sample_id"])
        torch.save(
            {
                "sample_id": record["sample_id"],
                "audio_path": raw["audio_path"],
                "split": record["split"],
                "labels": raw["labels"],
                "x": torch.from_numpy(x.astype(np.float32)),
                "mel": raw["mel"].float(),
                "edge_indices": edge_indices,
                "feature_config": config.to_dict(),
                "top_k": top_k,
                "seed": sample_seed,
                "source_fingerprint": raw["source_fingerprint"],
                "feature_cache_version": FEATURE_CACHE_VERSION,
                "audio_samples": raw.get("audio_samples"),
                "audio_offset_seconds": record.get("audio_offset_seconds", 0.0),
                "audio_duration_seconds": raw.get("audio_duration_seconds"),
            },
            processed_path,
        )
        processed_records.append(
            {
                **record,
                "sample_id": record["sample_id"],
                "split": record["split"],
                "audio_path": raw["audio_path"],
                "processed_path": str(processed_path.relative_to(output_dir)),
                "labels": record["labels"],
            }
        )
        if position % 25 == 0 or position == len(raw_paths):
            print(f"Normalized graph samples: {position}/{len(raw_paths)}", flush=True)

    processed_manifest = {
        **{key: value for key, value in manifest.items() if key != "records"},
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "label_names": manifest["label_names"],
        "feature_config": config.to_dict(),
        "normalization_path": "normalization.json",
        "graph_variants": list(GRAPH_VARIANTS),
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "graph_audit_path": "graph_audit.json",
        "preprocessing_seconds": time.perf_counter() - started,
        "top_k": top_k,
        "seed": seed,
        "records": processed_records,
    }
    destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(processed_manifest, indent=2), encoding="utf-8")
    # Import locally to keep graph/dataset utilities independent of preprocessing.
    from .audit import audit_saved_graphs

    audit = audit_saved_graphs(destination)
    if not audit["valid"]:
        raise ValueError(f"Saved graph audit failed; inspect {output_dir / 'graph_audit.json'}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/task2"))
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-fft", type=int, default=2_048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = AudioFeatureConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    destination = preprocess_manifest(
        args.manifest,
        args.output_dir,
        config=config,
        top_k=args.top_k,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Saved processed Task 2 manifest to {destination}")


if __name__ == "__main__":
    main()
