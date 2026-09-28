"""Reproducible five-model Task 2 comparison, with separately frozen test evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import soundfile as sf

from .dataset import dataset_identity, file_sha256, load_manifest
from .evaluate import evaluate_checkpoint
from .train import Task2TrainConfig, train_task2


EXPERIMENTS = (
    ("mlp", "mlp", "temporal"),
    ("cnn", "cnn", "temporal"),
    ("gnn_temporal", "gnn", "temporal"),
    ("gnn_similarity", "gnn", "temporal_similarity"),
    ("gnn_random", "gnn", "random"),
)


def make_demo_manifest(output_dir: str | Path, sample_count: int = 30, seed: int = 42) -> Path:
    """Create clearly marked synthetic audio for offline integration checks only."""
    if sample_count < 20:
        raise ValueError("Use at least 20 samples for the graph-gallery demonstration")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh demo-data directory: {output_dir}")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    sr = 8000
    t = np.arange(2 * sr, dtype=np.float32) / sr
    # Assign splits before generating independent waveforms; no repeated source clips.
    permutation = rng.permutation(sample_count)
    held_out = max(3, round(sample_count * .2))
    test = set(permutation[:held_out].tolist())
    validation = set(permutation[held_out:2 * held_out].tolist())
    records = []
    for i in range(sample_count):
        labels = [int(i % 2 == 0), int(i % 3 == 0), int(i % 5 < 2)]
        phase = rng.uniform(0, 2 * np.pi)
        tone = np.sin(2 * np.pi * (220 + 220 * labels[0] + rng.uniform(-15, 15)) * t + phase)
        envelope = .55 + .35 * np.sin(2 * np.pi * (2 + labels[1] * 4) * t)
        waveform = (.4 * tone * envelope + .1 * labels[2] * np.sin(2 * np.pi * 1100 * t)
                    + rng.normal(0, .01, len(t))).astype(np.float32)
        audio_path = audio_dir / f"synthetic_{i:03d}.wav"
        sf.write(audio_path, waveform, sr)
        records.append({"sample_id": f"synthetic_{i:03d}", "audio_path": str(audio_path),
                        "labels": labels, "split": "test" if i in test else "validation" if i in validation else "train"})
    manifest = {"schema_version": 1, "dataset": "Synthetic tones; pipeline verification only", "synthetic": True,
                "label_source": "synthetic waveform recipe; not MusicCaps or AudioSet", "seed": seed,
                "label_names": ["synthetic_high_tone", "synthetic_fast_pulse", "synthetic_harmonic"], "records": records}
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_tables(root: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "comparison.csv", index=False)
    frame[frame["model"] == "gnn"].to_csv(root / "graph_ablation.csv", index=False)
    measurements = [name for name in frame.columns if name.startswith(("validation_", "test_"))
                    and name.endswith(("macro_f1", "micro_f1", "map"))]
    measurements += ["training_seconds", "parameter_count"]
    aggregate = frame.groupby("experiment", sort=False)[measurements].agg(["mean", "std"])
    # A single seed has no estimate of across-seed variation (leave std blank).
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.to_csv(root / "comparison_aggregate.csv")


def run_experiment(config: Task2TrainConfig, seeds=(42,), evaluate_test: bool = False) -> dict:
    """Train all predeclared controls on one frozen manifest; select only on validation."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct training seeds")
    root = Path(config.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a fresh experiment output directory: {root}")
    manifest_path = Path(config.manifest).resolve()
    manifest = load_manifest(manifest_path)
    from .audit import audit_saved_graphs
    audit = audit_saved_graphs(manifest_path)
    if not audit["valid"]:
        raise ValueError("Graph audit failed; inspect graph_audit.json before training")
    identity = dataset_identity(manifest_path)
    for split in ("train", "validation", "test"):
        if not any(record["split"] == split for record in manifest["records"]):
            raise ValueError(f"The comparison requires a non-empty {split} partition")
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    split_ids = {split: [r["sample_id"] for r in manifest["records"] if r["split"] == split]
                 for split in ("train", "validation", "test")}
    (root / "split_ids.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    rows, runs = [], []
    started = time.perf_counter()
    for seed in seeds:
        for name, model, variant in EXPERIMENTS:
            relative = Path(f"seed{seed}") / name
            summary = train_task2(replace(config, manifest=str(manifest_path), output_dir=str(root / relative),
                                           model=model, graph_variant=variant, seed=seed, evaluate_test=False))
            if summary["dataset_fingerprint"] != identity["sha256"]:
                raise ValueError("The cached dataset changed during the comparison")
            # Compare row order and target values, not merely dataset lengths.
            prediction_path = root / relative / "validation_predictions.npz"
            with np.load(prediction_path) as prediction:
                if prediction["sample_ids"].tolist() != split_ids["validation"]:
                    raise ValueError("Validation samples/order differ across runs")
                if runs:
                    with np.load(root / runs[0]["directory"] / "validation_predictions.npz") as reference:
                        if not np.array_equal(prediction["targets"], reference["targets"]):
                            raise ValueError("Validation targets differ across runs")
            checkpoint = relative / "best_model.pt"
            runs.append({"experiment": name, "seed": seed, "directory": relative.as_posix(),
                         "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": file_sha256(root / checkpoint)})
            metrics = summary["validation"]
            rows.append({"experiment": name, "model": model, "graph_variant": variant, "seed": seed,
                         "synthetic": summary["synthetic"], "dataset_fingerprint": identity["sha256"],
                         "validation_macro_f1": metrics["macro_f1"], "validation_micro_f1": metrics["micro_f1"],
                         "validation_map": metrics["mean_average_precision"], "threshold": metrics["threshold"],
                         "parameter_count": summary["parameter_count"], "training_seconds": summary["training_seconds"],
                         "best_epoch": summary["best_epoch"], "device": summary["device"]})
            _write_tables(root, rows)
    means = {name: float(np.mean([r["validation_macro_f1"] for r in rows if r["experiment"] == name]))
             for name, _, _ in EXPERIMENTS}
    selection = {"criterion": "Mean validation Macro-F1 over declared seeds; declaration order breaks ties",
                 "selected_experiment": max(means, key=means.get),
                 "selected_gnn": max((name for name in means if name.startswith("gnn_")), key=means.get),
                 "validation_means": means, "seeds": list(seeds), "runs": runs,
                 "graph_control": "Random edges are fixed by preprocessing seed; training seeds vary initialization/order",
                 "test_policy": "Evaluate all predeclared frozen controls; never choose configurations using test metrics"}
    report = {"manifest": str(manifest_path), "dataset_fingerprint": identity["sha256"],
              "synthetic": manifest.get("synthetic", False), "rows": rows, "selection": selection,
              "evaluated_test": False, "suite_seconds": time.perf_counter() - started}
    (root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (root / "experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return evaluate_experiment(root, device=config.device) if evaluate_test else report


def evaluate_experiment(output_dir: str | Path, manifest: str | Path | None = None, device: str = "auto") -> dict:
    """Evaluate the frozen suite without retraining, reselecting, or recalibrating."""
    root = Path(output_dir).resolve()
    report = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if report["evaluated_test"]:
        raise FileExistsError("This frozen suite already has a test report; inspect its saved predictions")
    manifest_path = Path(manifest or report["manifest"])
    if dataset_identity(manifest_path)["sha256"] != report["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint changed since validation selection")
    for run in report["selection"]["runs"]:
        if file_sha256(root / run["checkpoint"]) != run["checkpoint_sha256"]:
            raise ValueError("A checkpoint changed since validation selection")
    reference_ids = reference_targets = None
    for row, run in zip(report["rows"], report["selection"]["runs"]):
        destination = root / run["directory"]
        metrics = evaluate_checkpoint(root / run["checkpoint"], manifest_path, device_name=device,
                                      output=destination / "test_metrics.json")
        shutil.move(destination / "test_metrics.predictions.npz", destination / "test_predictions.npz")
        shutil.move(destination / "test_metrics.sample_ids.json", destination / "test_sample_ids.json")
        with np.load(destination / "test_predictions.npz") as predictions:
            if reference_ids is None:
                reference_ids, reference_targets = predictions["sample_ids"].copy(), predictions["targets"].copy()
            elif not np.array_equal(reference_ids, predictions["sample_ids"]) or not np.array_equal(reference_targets, predictions["targets"]):
                raise ValueError("Test IDs/targets differ across the predeclared controls")
        row.update(test_macro_f1=metrics["macro_f1"], test_micro_f1=metrics["micro_f1"],
                   test_map=metrics["mean_average_precision"], test_seconds=metrics["inference_seconds"])
    report["evaluated_test"] = True
    _write_tables(root, report["rows"])
    (root / "experiment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task2/experiment"))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--evaluate-run", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    if args.evaluate_run:
        if args.demo:
            parser.error("--demo cannot be combined with --evaluate-run")
        result = evaluate_experiment(args.evaluate_run, args.manifest, args.device)
    else:
        if args.demo and args.manifest:
            parser.error("Choose either --demo or --manifest")
        root = args.output_dir.resolve()
        if args.demo:
            from .features import AudioFeatureConfig
            from .preprocess import preprocess_manifest
            source = make_demo_manifest(root / "demo_data")
            manifest = preprocess_manifest(source, root / "processed", config=AudioFeatureConfig(
                sample_rate=8000, segment_seconds=.25, n_mels=32, n_mfcc=8, n_fft=256, hop_length=64), top_k=1)
        elif args.manifest:
            manifest = args.manifest
        else:
            parser.error("Provide --manifest or --demo")
        from .visualize import visualize_graph_examples
        visualize_graph_examples(manifest, root / "graph_examples", count=20)
        config = Task2TrainConfig(manifest=str(manifest), output_dir=str(root / "runs"), epochs=args.epochs,
                                 hidden_dim=args.hidden_dim, layers=args.layers, batch_size=args.batch_size,
                                 patience=args.patience, learning_rate=args.learning_rate, device=args.device,
                                 cpu_threads=args.cpu_threads, evaluate_test=False)
        result = run_experiment(config, seeds=args.seeds or ([42] if args.demo else [42, 43, 44]),
                                evaluate_test=args.evaluate_test)
    print(json.dumps({key: result[key] for key in ("synthetic", "dataset_fingerprint", "evaluated_test")}, indent=2))


if __name__ == "__main__":
    main()
