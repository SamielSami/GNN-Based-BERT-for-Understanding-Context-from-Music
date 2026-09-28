"""Predeclare, train, freeze, evaluate and verify a multi-seed retrieval suite."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

from src.task3.data import feature_fingerprint, load_features, make_demo
from .metrics import normalized_pairs, paired_ranks, recompute
from .train import empty_output, evaluate, train

DIRECTIONS = ("text_to_graph", "graph_to_text")
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "median_rank", "mean_reciprocal_rank")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def pairing_audit(data):
    """Audit exact/normalized caption duplicates and MusicCaps video grouping."""
    exact, normalized, videos = defaultdict(list), defaultdict(list), defaultdict(list)
    for sample, caption, split in zip(data["sample_ids"], data["texts"], data["splits"]):
        row = dict(sample_id=sample, split=split)
        exact[caption].append(row)
        key = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", caption).casefold()))
        normalized[key].append(row)
        # MusicCaps IDs are <video ID>_<start>_<end>; don't infer synthetic groups.
        parts = sample.rsplit("_", 2)
        video = parts[0] if len(parts) == 3 and all(p.replace(".", "", 1).isdigit() for p in parts[1:]) else sample
        videos[video].append(row)

    def duplicates(groups):
        return [dict(value=key, examples=rows, crosses_splits=len({r["split"] for r in rows}) > 1)
                for key, rows in groups.items() if len(rows) > 1]

    video_duplicates = duplicates(videos)
    if any(group["crosses_splits"] for group in video_duplicates):
        raise ValueError("Video IDs overlap across splits")
    return dict(samples=len(data["sample_ids"]), split_counts=dict(Counter(data["splits"])),
                unique_sample_ids=len(set(data["sample_ids"])),
                exact_caption_duplicates=duplicates(exact), normalized_caption_duplicates=duplicates(normalized),
                repeated_videos=video_duplicates, video_split_overlap=False,
                normalization="Unicode NFKC, casefold, word tokens, whitespace collapse",
                limitation="Distinct captions can still describe perceptually equivalent music; no listening judgments made.")


def tables(root, selection, include_test):
    rows = []
    for run in selection["runs"]:
        for split in (("validation", "test") if include_test else ("validation",)):
            report = read(root / f"seed{run['seed']}" / split / "metrics.json")
            for baseline, key in (("trained", "metrics"), ("untrained", "untrained_projection_metrics")):
                for direction in DIRECTIONS:
                    rows.append(dict(seed=run["seed"], split=split, baseline=baseline, direction=direction,
                                     gallery_size=report[key]["gallery_size"], **report[key][direction]))
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby(["split", "baseline", "direction"], sort=False)[list(METRICS)].agg(["mean", "std"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    return frame, aggregate.reset_index()


def save_tables(root, selection, include_test):
    for name, frame in zip(("comparison.csv", "comparison_aggregate.csv"), tables(root, selection, include_test)):
        frame.to_csv(root / name, index=False)


def check_report(root, run, split, data):
    directory = root / f"seed{run['seed']}" / split
    report = read(directory / "metrics.json")
    if report["split"] != split or report["feature_sha256"] != run["feature_sha256"] or report["best_epoch"] != run["best_epoch"]:
        raise ValueError("Evaluation identity differs from frozen selection")
    ids = np.asarray([sid for sid, s in zip(data["sample_ids"], data["splits"]) if s == split])
    for filename, key in (("embeddings.npz", "metrics"), ("untrained_embeddings.npz", "untrained_projection_metrics")):
        with np.load(directory / filename, allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], ids):
                raise ValueError("Embedding IDs/order differ from the declared split")
            for modality in ("text", "graph"):
                if not np.allclose(np.linalg.norm(saved[modality], axis=1), 1, atol=1e-6):
                    raise ValueError("Saved embeddings are not normalized")
        if recompute(directory / filename) != report[key]:
            raise ValueError("Saved retrieval metrics differ from embeddings")
    gain = report["metrics"]["mean_recall_at_1"] - report["untrained_projection_metrics"]["mean_recall_at_1"]
    if gain != report["mean_recall_at_1_gain"]:
        raise ValueError("Saved baseline gain differs")
    with np.load(directory / "embeddings.npz", allow_pickle=False) as saved:
        t, g = normalized_pairs(saved["text"], saved["graph"])
    ranks = read(directory / "ranks.json")
    if ranks != dict(sample_ids=ids.tolist(), text_to_graph=paired_ranks(t, g).tolist(), graph_to_text=paired_ranks(g, t).tolist()):
        raise ValueError("Saved per-query ranks differ")
    for example in read(directory / "examples.json"):
        query = ids.tolist().index(example["query_id"])
        if example["paired_rank"] != ranks[example["direction"]][query]:
            raise ValueError("Example rank differs from aggregate scoring")
    return report


def validate_selection(root, features, data):
    declaration, selection = read(root / "suite_config.json"), read(root / "selection.json")
    if feature_fingerprint(root / "suite_config.json") != selection["suite_config_sha256"]:
        raise ValueError("Predeclared suite configuration changed")
    fingerprint = feature_fingerprint(features)
    if fingerprint != declaration["feature_sha256"] or fingerprint != selection["feature_sha256"]:
        raise ValueError("Feature cache differs from frozen suite")
    if selection["seeds"] != declaration["seeds"] or [r["seed"] for r in selection["runs"]] != declaration["seeds"]:
        raise ValueError("Frozen runs differ from declared seeds")
    for run in selection["runs"]:
        for path, digest in run["artifacts"].items():
            if feature_fingerprint(root / path) != digest:
                raise ValueError(f"Frozen artifact changed: {path}")
        checkpoint = torch.load(root / run["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint["feature_sha256"] != fingerprint or checkpoint["config"]["seed"] != run["seed"]:
            raise ValueError("Checkpoint identity differs")
        for key, value in declaration["training_options"].items():
            if key == "device" and value == "auto":
                continue
            if checkpoint["config"][key] != value:
                raise ValueError(f"Checkpoint training option differs: {key}")
        history = read(root / f"seed{run['seed']}/history.json")
        best = max(history, key=lambda row: row["validation"]["mean_recall_at_1"])
        if checkpoint["best_epoch"] != best["epoch"] or run["best_epoch"] != best["epoch"]:
            raise ValueError("Checkpoint was not selected by first maximum validation Recall@1")
        validation = check_report(root, run, "validation", data)
        if best["validation"] != validation["metrics"] or run["validation_mean_recall_at_1"] != validation["metrics"]["mean_recall_at_1"]:
            raise ValueError("Validation selection score differs")
    if read(root / "pairing_audit.json") != pairing_audit(data):
        raise ValueError("Pairing audit differs")
    return selection


def verify(output_dir, features=None):
    root = Path(output_dir).resolve()
    declaration = read(root / "suite_config.json")
    features = Path(features or declaration["features"])
    torch.set_num_threads(declaration["training_options"]["cpu_threads"])
    data = load_features(features, mmap=True)
    selection = validate_selection(root, features, data)
    tested = (root / "test_evaluation.json").exists()
    if tested:
        final = read(root / "test_evaluation.json")
        if final["selection_sha256"] != feature_fingerprint(root / "selection.json"):
            raise ValueError("Validation selection changed after final evaluation")
        for path, digest in final["artifacts"].items():
            if feature_fingerprint(root / path) != digest:
                raise ValueError(f"Test artifact changed: {path}")
        for run in selection["runs"]:
            check_report(root, run, "test", data)
    for name, expected in zip(("comparison.csv", "comparison_aggregate.csv"), tables(root, selection, tested)):
        try:
            pd.testing.assert_frame_equal(pd.read_csv(root / name), expected, check_dtype=False, rtol=1e-10, atol=1e-12)
        except AssertionError as error:
            raise ValueError(f"Comparison table changed: {name}") from error
    result = dict(valid=True, seeds=selection["seeds"], feature_sha256=selection["feature_sha256"],
                  evaluated_test=tested, validation_only_selection=True, unit_embeddings_verified=True,
                  metrics_recomputed=True, baseline_metrics_recomputed=True, split_ids_verified=True,
                  checkpoint_hashes_verified=True, comparison_tables_verified=True)
    write(root / "verification.json", result)
    return result


def evaluate_suite(output_dir, features=None, device="cpu"):
    root = Path(output_dir).resolve()
    declaration = read(root / "suite_config.json")
    features = Path(features or declaration["features"])
    torch.set_num_threads(declaration["training_options"]["cpu_threads"])
    data = load_features(features, mmap=True)
    selection = validate_selection(root, features, data)
    # All checkpoints must be frozen before any final inference starts.
    if any((root / f"seed{r['seed']}/test").exists() for r in selection["runs"]):
        raise ValueError("Test outputs already exist; verify them without repeating final inference")
    del data
    artifacts = {}
    for run in selection["runs"]:
        directory = root / f"seed{run['seed']}/test"
        evaluate(root / run["checkpoint"], features, directory, device=device)
        for path in directory.iterdir():
            artifacts[path.relative_to(root).as_posix()] = feature_fingerprint(path)
    write(root / "test_evaluation.json", dict(selection_sha256=feature_fingerprint(root / "selection.json"), artifacts=artifacts))
    save_tables(root, selection, True)
    return verify(root, features)


def run_experiment(features, output_dir, seeds=(42, 43, 44), evaluate_test=False, **options):
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    settings = dict(epochs=30, batch_size=32, projection_dim=64, temperature=0.07,
                    learning_rate=0.001, patience=5, device="cpu", cpu_threads=6)
    settings.update(options)
    torch.set_num_threads(settings["cpu_threads"])
    features = Path(features).resolve()
    data = load_features(features, mmap=True)
    audit = pairing_audit(data)
    fingerprint = feature_fingerprint(features)
    root = empty_output(Path(output_dir).resolve())
    declaration = dict(features=str(features), feature_sha256=fingerprint, seeds=seeds, training_options=settings,
                       synthetic=data.get("provenance", {}).get("synthetic", False), provenance=data.get("provenance", {}),
                       seed_scope="Projection initialization and batch order only; both encoders fixed",
                       selection_metric="Mean bidirectional validation Recall@1; earliest epoch breaks ties",
                       test_policy="All declared seeds and their exact initial projections; no test-based selection")
    write(root / "suite_config.json", declaration)
    write(root / "pairing_audit.json", audit)
    del data
    runs = []
    for seed in seeds:
        directory = root / f"seed{seed}"
        report = train(features, directory, seed=seed, **settings)
        paths = [p for p in directory.rglob("*") if p.is_file()]
        runs.append(dict(seed=seed, checkpoint=f"seed{seed}/best_model.pt", best_epoch=report["best_epoch"],
                         feature_sha256=fingerprint, validation_mean_recall_at_1=report["metrics"]["mean_recall_at_1"],
                         artifacts={p.relative_to(root).as_posix(): feature_fingerprint(p) for p in paths}))
    selection = dict(seeds=seeds, feature_sha256=fingerprint, suite_config_sha256=feature_fingerprint(root / "suite_config.json"), runs=runs)
    write(root / "selection.json", selection)
    save_tables(root, selection, False)
    return evaluate_suite(root, features, settings["device"]) if evaluate_test else verify(root, features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "evaluate", "verify"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    for name, default in (("epochs", 30), ("batch-size", 32), ("projection-dim", 64), ("patience", 5), ("cpu-threads", 6)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    args = vars(parser.parse_args())
    command, root, features = args.pop("command"), args.pop("output_dir"), args.pop("features")
    demo = args.pop("demo")
    if command == "run":
        if demo:
            features = root.with_suffix(".demo.pt")
            if features.exists():
                parser.error("Demo cache already exists; use a fresh output path")
            make_demo(features)
        if features is None:
            parser.error("run requires --features or --demo")
        result = run_experiment(features, root, **args)
    elif command == "evaluate":
        result = evaluate_suite(root, features, args["device"])
    else:
        result = verify(root, features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
