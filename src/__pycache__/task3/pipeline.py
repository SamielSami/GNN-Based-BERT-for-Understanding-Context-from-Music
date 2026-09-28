"""Configured entry point for Task 3 joint training and saved-run evaluation."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from .joint import aligned_manifests, evaluate_joint, train_joint, verify_joint

ROOT = Path(__file__).resolve().parents[2]
PATH_FIELDS = ("processed_manifest", "source_manifest", "text_checkpoint", "graph_checkpoint", "head_run", "output_dir")


def load_config(path, output_dir=None):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    options = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(options, dict):
        raise ValueError("Task 3 configuration must be a YAML mapping")
    signature = inspect.signature(train_joint)
    unknown = set(options) - set(signature.parameters)
    missing = [name for name, value in signature.parameters.items()
               if value.default is inspect.Parameter.empty and name not in options]
    if unknown or missing:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}; missing required keys: {missing}")
    if output_dir is not None:
        options["output_dir"] = output_dir
    for key in PATH_FIELDS:
        if options.get(key) is not None:
            value = Path(options[key])
            options[key] = value.resolve() if value.is_absolute() else (ROOT / value).resolve()
    return options


def check_inputs(options):
    for key in ("processed_manifest", "source_manifest", "text_checkpoint", "graph_checkpoint"):
        if not options[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {options[key]}")
    processed, source = aligned_manifests(options["processed_manifest"], options["source_manifest"])
    text = torch.load(options["text_checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    graph = torch.load(options["graph_checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    if text["tags"] != source["label_names"] or graph["label_names"] != source["label_names"]:
        raise ValueError("Encoder vocabularies differ from paired data")
    if options.get("head_run"):
        for mode in options.get("modes", ("bert", "gnn", "concat", "gated", "cross_attention")):
            if not (options["head_run"] / f"{mode}.pt").is_file():
                raise FileNotFoundError(f"Missing initial frozen head: {mode}")
    output = options["output_dir"]
    if (output / "experiment.json").is_file():
        status = "test_evaluated" if json.loads((output / "experiment.json").read_text())["evaluated_test"] else "trained_validation_selected"
    elif output.exists() and any(output.iterdir()):
        status = "partial_run_requires_inspection"
    else:
        status = "not_started"
    return dict(inputs_present=True, paired_manifest_alignment=True, label_count=len(source["label_names"]),
                samples=len(processed["records"]),
                split_counts={s: sum(r["split"] == s for r in source["records"]) for s in ("train", "validation", "test")},
                output_dir=str(output), run_status=status,
                note="Full encoder/dataset fingerprint checks also run before training; check does not certify completed training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "train", "evaluate", "verify", "plot", "analyze"))
    parser.add_argument("--config", type=Path, default=Path("configs/task3.yaml"))
    parser.add_argument("--output-dir", type=Path, help="Fresh training folder, or the existing run for evaluation/verification")
    args = parser.parse_args()
    options = load_config(args.config, args.output_dir)
    if args.action == "check":
        result = check_inputs(options)
    elif args.action == "smoke":
        code = subprocess.run([sys.executable, "-m", "unittest", "tests.task3.test_joint",
                               "tests.task3.test_fusion", "tests.task3.test_semantic_plots", "-v"], cwd=ROOT).returncode
        raise SystemExit(code)
    elif args.action == "train":
        result = train_joint(**options)
    elif args.action == "evaluate":
        result = evaluate_joint(options["output_dir"], device=options.get("device", "auto"), cpu_threads=options.get("cpu_threads", 6))
    elif args.action == "verify":
        result = verify_joint(options["output_dir"])
    elif args.action == "analyze":
        from .joint_analyze import analyze_joint
        result = analyze_joint(options["output_dir"])
    else:
        code = subprocess.run([sys.executable, "-m", "src.task3.semantic_plots", "--joint-run", str(options["output_dir"])], cwd=ROOT).returncode
        raise SystemExit(code)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
