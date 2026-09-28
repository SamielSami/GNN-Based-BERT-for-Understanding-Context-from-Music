"""Run and audit the predeclared five-mode, three-seed frozen-feature comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.task2.train import classification_metrics, resolve_device, select_global_threshold
from .data import feature_fingerprint, load_features, make_demo
from .evaluate import predict_features
from .models import FusionClassifier, MODES
from .train import run_experiments


FUSION_MODES = ("concat", "gated", "cross_attention")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _comparison_tables(rows):
    frame = pd.DataFrame(rows)
    measurements = [name for name in frame if name.startswith(("validation_", "test_"))
                    and name.endswith(("macro_f1", "micro_f1", "map"))]
    measurements += ["training_seconds", "parameter_count"]
    aggregate = frame.groupby("mode", sort=False)[measurements].agg(["mean", "std"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    return {"comparison.csv": frame, "comparison_aggregate.csv": aggregate.reset_index()}


def _write_tables(root, rows):
    for filename, frame in _comparison_tables(rows).items():
        frame.to_csv(root / filename, index=False)


def _verify_tables(root, rows):
    for filename, expected in _comparison_tables(rows).items():
        try:
            actual = pd.read_csv(root / filename)
            # CSV loses dtype details; preserve columns/order and compare values.
            # Sample std is NaN for one seed, which must stay NaN in the CSV.
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                          check_exact=False, rtol=1e-10, atol=1e-12)
        except (AssertionError, ValueError, OSError) as error:
            raise ValueError(f"Saved {filename} differs from verified experiment rows") from error


def _prediction_metrics(path, data, split, threshold):
    indices = [i for i, value in enumerate(data["splits"]) if value == split]
    expected_ids = np.asarray([data["sample_ids"][i] for i in indices])
    expected_targets = data["labels"][indices].numpy()
    with np.load(path, allow_pickle=False) as prediction:
        if not np.array_equal(prediction["sample_ids"], expected_ids):
            raise ValueError(f"{split} prediction IDs/order differ from feature cache")
        if not np.array_equal(prediction["targets"], expected_targets):
            raise ValueError(f"{split} prediction targets differ from feature cache")
        metrics = classification_metrics(expected_targets, prediction["probabilities"],
                                         data["label_names"], threshold)
        if split == "validation":
            selected = select_global_threshold(expected_targets, prediction["probabilities"])
            if not np.isclose(selected, threshold, rtol=0, atol=1e-12):
                raise ValueError("Saved threshold differs from validation-only threshold search")
    return metrics


def _assert_equal(actual, expected, description):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{description}: metric keys differ")
        for key in actual:
            _assert_equal(actual[key], expected[key], f"{description}.{key}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{description}: values differ")
    elif actual != expected:
        raise ValueError(f"{description}: values differ")


def _audit(root, report, selection, data):
    if report["selection_sha256"] != feature_fingerprint(root / "selection.json"):
        raise ValueError("Frozen validation selection changed")
    if report["feature_sha256"] != selection["feature_sha256"]:
        raise ValueError("Selection feature fingerprint differs")
    _assert_equal(report["selection"], selection, "Embedded validation selection")
    if not selection.get("suite_config_sha256"):
        raise ValueError("Frozen selection lacks the suite declaration fingerprint; train a new suite")
    if feature_fingerprint(root / "suite_config.json") != selection["suite_config_sha256"]:
        raise ValueError("Predeclared suite configuration changed")
    declaration = _read_json(root / "suite_config.json")
    for key in ("feature_sha256", "seeds"):
        _assert_equal(declaration[key], selection[key], f"Suite declaration {key}")
    _assert_equal(declaration["modes"], list(MODES), "Suite declaration modes")
    _assert_equal(declaration["features"], report["features"], "Suite declaration features")
    _assert_equal(declaration["seed_scope"], report["seed_scope"], "Suite declaration seed scope")
    seeds = selection["seeds"]
    expected = [(seed, mode) for seed in seeds for mode in MODES]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid declared seeds")
    if [(run["seed"], run["mode"]) for run in selection["runs"]] != expected:
        raise ValueError("Frozen suite differs from the predeclared five-mode matrix")
    if [(row["seed"], row["mode"]) for row in report["rows"]] != expected:
        raise ValueError("Comparison rows differ from the frozen suite")
    checked, values = [], {mode: [] for mode in MODES}
    for run, row in zip(selection["runs"], report["rows"]):
        checkpoint_path = root / run["checkpoint"]
        if feature_fingerprint(checkpoint_path) != run["checkpoint_sha256"]:
            raise ValueError("A checkpoint changed since validation selection")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (saved["feature_sha256"] != report["feature_sha256"] or
                saved["label_names"] != data["label_names"] or saved["mode"] != run["mode"] or
                saved["config"]["seed"] != run["seed"]):
            raise ValueError("Checkpoint feature/label/mode/seed identity differs")
        for key, value in declaration["training_options"].items():
            # 'auto' records the resolved device in each checkpoint.
            if key == "device" and value == "auto":
                continue
            _assert_equal(saved["config"][key], value, f"Declared training option {key}")
        _assert_equal(row["threshold"], saved["threshold"], "checkpoint threshold")
        validation_path = root / run["validation_predictions"]
        if feature_fingerprint(validation_path) != run["validation_predictions_sha256"]:
            raise ValueError("Validation predictions changed since selection")
        validation = _prediction_metrics(validation_path, data, "validation", saved["threshold"])
        _assert_equal(validation, _read_json(root / run["validation_metrics"]), "validation metrics")
        for name, key in (("macro_f1", "macro_f1"), ("micro_f1", "micro_f1"), ("map", "mean_average_precision")):
            _assert_equal(row[f"validation_{name}"], validation[key], "comparison validation metric")
        values[run["mode"]].append(validation["macro_f1"])
        record = dict(seed=run["seed"], mode=run["mode"], validation_recomputed=True,
                      validation_threshold_verified=True, test_recomputed=False)
        if report["evaluated_test"]:
            test_path = root / row["test_predictions"]
            if feature_fingerprint(test_path) != row["test_predictions_sha256"]:
                raise ValueError("Test predictions changed since evaluation")
            test = _prediction_metrics(test_path, data, "test", saved["threshold"])
            test_saved = _read_json(root / row["test_metrics"])
            _assert_equal(test, test_saved["metrics"], "test metrics")
            for name, key in (("macro_f1", "macro_f1"), ("micro_f1", "micro_f1"), ("map", "mean_average_precision")):
                _assert_equal(row[f"test_{name}"], test[key], "comparison test metric")
            record["test_recomputed"] = True
        checked.append(record)
    means = {mode: float(np.mean(values[mode])) for mode in MODES}
    _assert_equal(means, selection["validation_means"], "validation selection means")
    if (selection["selected_mode"] != max(MODES, key=means.get) or
            selection["selected_fusion"] != max(FUSION_MODES, key=means.get)):
        raise ValueError("Selection does not follow mean validation Macro-F1")
    _verify_tables(root, report["rows"])
    return dict(valid=True, feature_sha256=report["feature_sha256"],
                selection_sha256=report["selection_sha256"], runs_checked=len(checked),
                evaluated_test=report["evaluated_test"], validation_only_selection=True,
                suite_declaration_verified=True, comparison_tables_verified=True,
                checks=checked)


def run_experiment(features, output_dir, seeds=(42, 43, 44), evaluate_test=False, **training_options):
    """Train every control, then freeze the mean-validation selection before testing."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct training seeds")
    root, features = Path(output_dir).resolve(), Path(features).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh experiment output directory")
    if "seed" in training_options or "evaluate_test" in training_options:
        raise ValueError("Seeds and test evaluation are controlled by the suite")
    data = load_features(features, mmap=True)
    fingerprint = feature_fingerprint(features)
    synthetic = data.get("provenance", {}).get("synthetic", False)
    root.mkdir(parents=True, exist_ok=True)
    declaration = dict(features=str(features), feature_sha256=fingerprint, seeds=list(seeds),
                       modes=list(MODES), training_options=training_options,
                       seed_scope="Fusion-head initialization and training order; encoders/features fixed",
                       test_policy="Evaluate all predeclared frozen controls after validation selection")
    _write_json(root / "suite_config.json", declaration)
    declaration_sha256 = feature_fingerprint(root / "suite_config.json")
    rows, runs = [], []
    started = time.perf_counter()
    for seed in seeds:
        directory = Path(f"seed{seed}")
        run_experiments(features, root / directory, seed=seed, evaluate_test=False, **training_options)
        comparison = pd.read_csv(root / directory / "comparison.csv")
        for mode in MODES:
            entry = comparison.loc[comparison["mode"] == mode].iloc[0]
            checkpoint = directory / f"{mode}.pt"
            saved = torch.load(root / checkpoint, map_location="cpu", weights_only=True)
            if saved["feature_sha256"] != fingerprint:
                raise ValueError("Feature cache changed during suite training")
            prediction = directory / f"{mode}_validation_predictions.npz"
            metric_path = directory / f"{mode}_validation_metrics.json"
            metrics = _prediction_metrics(root / prediction, data, "validation", saved["threshold"])
            _assert_equal(metrics, _read_json(root / metric_path), "saved validation metrics")
            rows.append(dict(mode=mode, seed=seed, synthetic=synthetic,
                             validation_macro_f1=metrics["macro_f1"], validation_micro_f1=metrics["micro_f1"],
                             validation_map=metrics["mean_average_precision"], threshold=saved["threshold"],
                             best_epoch=saved["best_epoch"], epochs_run=int(entry["epochs_run"]),
                             training_seconds=float(entry["seconds"]), parameter_count=int(entry["parameters"])))
            runs.append(dict(mode=mode, seed=seed, checkpoint=checkpoint.as_posix(),
                             checkpoint_sha256=feature_fingerprint(root / checkpoint),
                             validation_predictions=prediction.as_posix(),
                             validation_predictions_sha256=feature_fingerprint(root / prediction),
                             validation_metrics=metric_path.as_posix()))
        _write_tables(root, rows)
    if feature_fingerprint(features) != fingerprint:
        raise ValueError("Feature cache changed during suite training")
    means = {mode: float(np.mean([row["validation_macro_f1"] for row in rows if row["mode"] == mode]))
             for mode in MODES}
    selection = dict(criterion="Mean validation Macro-F1; declaration order breaks ties", selected_on="validation",
                     selected_mode=max(MODES, key=means.get), selected_fusion=max(FUSION_MODES, key=means.get),
                     validation_means=means, seeds=list(seeds), runs=runs, feature_sha256=fingerprint,
                     suite_config_sha256=declaration_sha256)
    _write_json(root / "selection.json", selection)
    report = dict(features=str(features), feature_sha256=fingerprint, synthetic=synthetic, rows=rows,
                  selection_sha256=feature_fingerprint(root / "selection.json"), selection=selection,
                  evaluated_test=False, suite_seconds=time.perf_counter() - started,
                  seed_scope=declaration["seed_scope"])
    _write_json(root / "experiment.json", report)
    _write_json(root / "verification.json", _audit(root, report, selection, data))
    if evaluate_test:
        return evaluate_experiment(root, device=training_options.get("device", "auto"),
                                   batch_size=training_options.get("batch_size", 16),
                                   cpu_threads=training_options.get("cpu_threads", 6))
    return report


def evaluate_experiment(output_dir, features=None, device="auto", batch_size=16, cpu_threads=6):
    """Restore every declared checkpoint, using saved thresholds and one mapped cache."""
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("batch_size and cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    root = Path(output_dir).resolve()
    report, selection = _read_json(root / "experiment.json"), _read_json(root / "selection.json")
    if report["evaluated_test"]:
        raise FileExistsError("This frozen suite already has a test report; inspect saved predictions")
    features = Path(features or report["features"])
    if feature_fingerprint(features) != report["feature_sha256"]:
        raise ValueError("Feature file differs from the frozen suite")
    data = load_features(features, mmap=True)
    _audit(root, report, selection, data)
    device = resolve_device(device)
    indices = [i for i, value in enumerate(data["splits"]) if value == "test"]
    targets = data["labels"][indices].numpy()
    sample_ids = np.asarray([data["sample_ids"][i] for i in indices])
    for row, run in zip(report["rows"], selection["runs"]):
        checkpoint = root / run["checkpoint"]
        destination = checkpoint.parent
        if (destination / f"{run['mode']}_test_predictions.npz").exists() or (destination / f"{run['mode']}_test_metrics.json").exists():
            raise FileExistsError("Partial test artifacts already exist; preserve and inspect the run")
    # No test forward pass occurs until the full declaration and all checkpoints pass their audit.
    for row, run in zip(report["rows"], selection["runs"]):
        saved = torch.load(root / run["checkpoint"], map_location="cpu", weights_only=True)
        model = FusionClassifier(**saved["dimensions"], mode=saved["mode"]).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        started = time.perf_counter()
        probabilities, embeddings = predict_features(model, data, indices, batch_size, device)
        elapsed = time.perf_counter() - started
        metrics = classification_metrics(targets, probabilities, data["label_names"], saved["threshold"])
        relative = Path(run["checkpoint"]).parent
        prediction_path = relative / f"{run['mode']}_test_predictions.npz"
        metrics_path = relative / f"{run['mode']}_test_metrics.json"
        np.savez_compressed(root / prediction_path, probabilities=probabilities, targets=targets,
                            sample_ids=sample_ids, embeddings=embeddings)
        _write_json(root / metrics_path, dict(mode=run["mode"], seed=run["seed"], split="test", metrics=metrics,
                                             checkpoint_sha256=run["checkpoint_sha256"],
                                             feature_sha256=report["feature_sha256"], inference_seconds=elapsed))
        row.update(test_macro_f1=metrics["macro_f1"], test_micro_f1=metrics["micro_f1"],
                   test_map=metrics["mean_average_precision"], test_seconds=elapsed,
                   test_predictions=prediction_path.as_posix(), test_metrics=metrics_path.as_posix(),
                   test_predictions_sha256=feature_fingerprint(root / prediction_path))
        print(f"test seed={run['seed']} mode={run['mode']} macro_f1={metrics['macro_f1']:.4f}", flush=True)
    report["evaluated_test"] = True
    _write_tables(root, report["rows"])
    _write_json(root / "experiment.json", report)
    _write_json(root / "verification.json", _audit(root, report, selection, data))
    return report


def verify_experiment(output_dir, features=None):
    """Recompute metrics and threshold selection from persisted predictions, without inference."""
    root = Path(output_dir).resolve()
    report, selection = _read_json(root / "experiment.json"), _read_json(root / "selection.json")
    features = Path(features or report["features"])
    if feature_fingerprint(features) != report["feature_sha256"]:
        raise ValueError("Feature file differs from the frozen suite")
    result = _audit(root, report, selection, load_features(features, mmap=True))
    _write_json(root / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/task3/experiment"))
    parser.add_argument("--demo", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--evaluate-run", type=Path)
    action.add_argument("--verify-run", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    if args.evaluate_run or args.verify_run:
        if args.demo or args.evaluate_test:
            parser.error("Evaluation/verification cannot be combined with --demo or --evaluate-test")
        if args.verify_run:
            result = verify_experiment(args.verify_run, args.features)
        else:
            result = evaluate_experiment(args.evaluate_run, args.features, args.device, args.batch_size, args.cpu_threads)
    else:
        if args.demo == bool(args.features):
            parser.error("Choose exactly one of --demo or --features")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error("Use a fresh output directory")
        demo_path = args.output_dir.with_suffix(".demo.pt")
        if args.demo and demo_path.exists():
            parser.error("Synthetic feature cache already exists; choose a fresh output directory")
        features = make_demo(demo_path, args.seeds[0]) if args.demo else args.features
        result = run_experiment(features, args.output_dir, seeds=args.seeds, evaluate_test=args.evaluate_test,
                                epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
                                patience=args.patience, learning_rate=args.learning_rate, device=args.device,
                                cpu_threads=args.cpu_threads)
    print(json.dumps({key: result[key] for key in ("feature_sha256", "evaluated_test")}, indent=2))


if __name__ == "__main__":
    main()
