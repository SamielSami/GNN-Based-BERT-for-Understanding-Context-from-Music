"""Jointly fine-tune paired caption/graph models, then evaluate frozen selections.

Unlike task3.train, this runner recomputes transformer and GNN representations
inside every batch. Normalized input graphs and integer token IDs may be cached;
encoder representations are never cached during optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.task2.dataset import ProcessedTask2Dataset, dataset_identity, load_manifest
from src.task2.models import GraphSAGEEncoder
from src.task2.train import classification_metrics, resolve_device, select_global_threshold, set_seed
from .models import FusionClassifier, MODES
from .prepare import SPLITS, _load_text_encoder, _verify_graph_alignment, _verify_text_alignment, file_hash


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aligned_manifests(processed_path, source_path):
    processed, source = load_manifest(processed_path), load_manifest(source_path)
    if processed["label_names"] != source["label_names"] or processed.get("label_source") != source.get("label_source"):
        raise ValueError("Paired manifests disagree on vocabulary or label provenance")
    rows = {r["sample_id"]: r for r in source["records"]}
    if set(rows) != {r["sample_id"] for r in processed["records"]}:
        raise ValueError("Paired sample IDs differ")
    if {r["split"] for r in source["records"]} != set(SPLITS):
        raise ValueError("Three nonempty splits are required")
    videos = {}
    for record in processed["records"]:
        row = rows[record["sample_id"]]
        if not row.get("text", "").strip() or not row.get("ytid"):
            raise ValueError("Each paired clip needs a caption and video ID")
        for key in ("split", "ytid", "labels"):
            if record.get(key) != row.get(key):
                raise ValueError(f"Paired {key} differs for {row['sample_id']}")
        previous = videos.setdefault(row["ytid"], row["split"])
        if previous != row["split"]:
            raise ValueError("A video occurs in multiple partitions")
    return processed, source


class PairedDataset(Dataset):
    def __init__(self, processed_path, source, split, variant, tokenizer, max_length):
        base = ProcessedTask2Dataset(processed_path, split, variant)
        rows = {r["sample_id"]: r for r in source["records"]}
        self.records = [rows[r["sample_id"]] for r in base.records]
        self.graphs, targets = [], []
        for index, row in enumerate(self.records):
            sample = base[index]
            if sample["labels"].tolist() != row["labels"]:
                raise ValueError("Cached graph targets differ from paired source")
            self.graphs.append(sample["graph"])
            targets.append(sample["labels"])
        self.targets = torch.stack(targets)
        self.tokens = tokenizer([r["text"] for r in self.records], truncation=True,
                                padding="max_length", max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {"graph": self.graphs[index], "labels": self.targets[index],
                "tokens": {k: v[index] for k, v in self.tokens.items()},
                "sample_id": self.records[index]["sample_id"]}


def collate_pairs(samples):
    return {"graph": Batch.from_data_list([s["graph"] for s in samples]),
            "labels": torch.stack([s["labels"] for s in samples]),
            "tokens": {k: torch.stack([s["tokens"][k] for s in samples]) for k in samples[0]["tokens"]},
            "sample_ids": [s["sample_id"] for s in samples]}


def transformer_blocks(encoder):
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return encoder.transformer.layer  # DistilBERT
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer  # BERT
    raise ValueError("Partial fine-tuning currently supports BERT and DistilBERT")


class JointClassifier(nn.Module):
    def __init__(self, text_encoder, graph_encoder, dimensions, mode, unfreeze_layers=1):
        super().__init__()
        self.mode, self.dimensions = mode, dimensions
        self.text_encoder = text_encoder if mode != "gnn" else None
        self.graph_encoder = graph_encoder if mode != "bert" else None
        self.head = FusionClassifier(**dimensions, mode=mode)
        self.unfrozen_blocks = []
        if self.text_encoder is not None:
            blocks = transformer_blocks(self.text_encoder)
            if not 1 <= unfreeze_layers <= len(blocks):
                raise ValueError("unfreeze_layers must select at least one existing transformer block")
            self.text_encoder.requires_grad_(False)
            self.unfrozen_blocks = list(blocks[-unfreeze_layers:])
            for block in self.unfrozen_blocks:
                block.requires_grad_(True)
        if self.graph_encoder is not None:
            self.graph_encoder.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen lower layers also retain deterministic dropout behavior.
        if self.text_encoder is not None:
            self.text_encoder.eval()
            for block in self.unfrozen_blocks:
                block.train(mode)
        return self

    def forward(self, graph, tokens, return_embedding=False):
        mask = tokens["attention_mask"].bool()
        batch = len(mask)
        if self.text_encoder is not None:
            text = self.text_encoder(**tokens).last_hidden_state
        else:
            text = torch.zeros(batch, 1, self.dimensions["text_dim"], device=mask.device)
            mask = mask[:, :1]
        if self.graph_encoder is not None:
            audio = self.graph_encoder(graph)
        else:
            audio = torch.zeros(batch, self.dimensions["graph_dim"], device=mask.device)
        return self.head(text, mask, audio, return_embedding=return_embedding)


def parameter_group(name):
    return "text" if name.startswith("text_encoder.") else "graph" if name.startswith("graph_encoder.") else "head"


def frozen_text_hash(model):
    if model.text_encoder is None:
        return None
    digest = hashlib.sha256()
    for name, parameter in model.text_encoder.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_anchors(model):
    anchors = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if parameter.requires_grad and group not in anchors:
            anchors[group] = (name, parameter.detach().cpu().clone())
    return anchors


def changed_parameters(model, anchors):
    params = dict(model.named_parameters())
    return {group: {"parameter": name,
                    "max_abs_change": float((params[name].detach().cpu() - before).abs().max())}
            for group, (name, before) in anchors.items()}


def metric_report(targets, probabilities, label_names, threshold):
    result = classification_metrics(targets, probabilities, label_names, threshold)
    areas = []
    for j, name in enumerate(label_names):
        value = None
        if targets[:, j].sum():
            precision, recall, _ = precision_recall_curve(targets[:, j], probabilities[:, j])
            value = float(auc(recall, precision))
            areas.append(value)
        result["per_label"][name]["pr_auc_trapezoidal"] = value
    result["mean_pr_auc_trapezoidal"] = float(np.mean(areas)) if areas else None
    result["pr_auc_definition"] = "Mean trapezoidal precision-recall AUC over labels with positive support"
    return result


@torch.no_grad()
def predict(model, dataset, batch_size, device):
    model.eval()
    probabilities, embeddings, targets, ids = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_pairs):
        logits, embedding = model(batch["graph"].to(device),
                                  {k: v.to(device) for k, v in batch["tokens"].items()}, True)
        probabilities.append(logits.sigmoid().cpu().numpy())
        embeddings.append(embedding.cpu().numpy())
        targets.append(batch["labels"].numpy())
        ids.extend(batch["sample_ids"])
    return dict(probabilities=np.concatenate(probabilities), embeddings=np.concatenate(embeddings),
                targets=np.concatenate(targets), sample_ids=np.asarray(ids))


def restore_model(saved, device="cpu"):
    metadata = saved["architecture"]
    config = dict(metadata["text_config"])
    model_type = config.pop("model_type")
    text = AutoModel.from_config(AutoConfig.for_model(model_type, **config)) if saved["mode"] != "gnn" else None
    graph = GraphSAGEEncoder(**metadata["graph_config"]) if saved["mode"] != "bert" else None
    model = JointClassifier(text, graph, metadata["dimensions"], saved["mode"], metadata["unfreeze_layers"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def _tables(root, rows):
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)


def train_joint(processed_manifest, source_manifest, text_checkpoint, graph_checkpoint,
                output_dir, head_run=None, epochs=3, batch_size=8, hidden_dim=64,
                unfreeze_layers=1, text_lr=1e-5, graph_lr=1e-4, head_lr=1e-4,
                patience=2, seed=42, device="auto", cpu_threads=6, modes=MODES):
    if min(epochs, batch_size, hidden_dim, patience, unfreeze_layers, cpu_threads) < 1:
        raise ValueError("Training counts must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (text_lr, graph_lr, head_lr)):
        raise ValueError("Learning rates must be positive and finite")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("Declare distinct supported modes")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh joint-training output directory")
    torch.set_num_threads(cpu_threads)
    device = resolve_device(device)
    processed_manifest, source_manifest = Path(processed_manifest).resolve(), Path(source_manifest).resolve()
    processed, source = aligned_manifests(processed_manifest, source_manifest)
    text_saved = torch.load(text_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    graph_saved = torch.load(graph_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    labels = source["label_names"]
    if labels != text_saved["tags"] or labels != graph_saved["label_names"]:
        raise ValueError("Encoder checkpoint vocabulary/order differs")
    provenance = _verify_text_alignment(text_saved, text_checkpoint, source)
    fingerprint = _verify_graph_alignment(graph_saved, processed_manifest, processed)
    provenance.update(graph_dataset_fingerprint=fingerprint, text_checkpoint_sha256=file_hash(text_checkpoint),
                      graph_checkpoint_sha256=file_hash(graph_checkpoint), source_manifest_sha256=file_hash(source_manifest),
                      processed_manifest_sha256=file_hash(processed_manifest), synthetic=bool(processed.get("synthetic", False)),
                      training="Joint GNN, fusion/head and final transformer blocks; live encoder forward passes")
    gc, tc = graph_saved["config"], text_saved["config"]
    tokenizer = AutoTokenizer.from_pretrained(tc["model_name"])
    text_config = AutoConfig.from_pretrained(tc["model_name"]).to_dict()
    graph_config = dict(input_dim=graph_saved["input_dim"], hidden_dim=gc["hidden_dim"],
                        layers=gc["layers"], dropout=gc["dropout"])
    dimensions = dict(text_dim=text_config.get("hidden_size", text_config.get("dim")),
                      graph_dim=2 * gc["hidden_dim"], num_labels=len(labels), hidden_dim=hidden_dim)
    architecture = dict(text_config=text_config, graph_config=graph_config, dimensions=dimensions,
                        unfreeze_layers=unfreeze_layers)
    config = dict(processed_manifest=str(processed_manifest), source_manifest=str(source_manifest),
                  epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim, unfreeze_layers=unfreeze_layers,
                  text_lr=text_lr, graph_lr=graph_lr, head_lr=head_lr, weight_decay=0.01, patience=patience,
                  seed=seed, device=str(device), cpu_threads=cpu_threads, modes=list(modes),
                  graph_variant=gc["graph_variant"], max_length=tc["max_length"], label_names=labels,
                  selection="Best validation Macro-F1; earliest epoch/declaration order breaks ties",
                  head_initialization="Saved per-mode frozen heads" if head_run else "Seeded random heads")
    initial_heads = {}
    if head_run:
        for mode in modes:
            path = Path(head_run) / f"{mode}.pt"
            head = torch.load(path, map_location="cpu", weights_only=True)
            if head["mode"] != mode or head["dimensions"] != dimensions or head["label_names"] != labels:
                raise ValueError("Initial frozen head dimensions or vocabulary differ")
            for key in ("text_checkpoint_sha256", "graph_checkpoint_sha256", "graph_dataset_fingerprint"):
                if head["provenance"].get(key) != provenance[key]:
                    raise ValueError("Initial head uses different encoder/data provenance")
            initial_heads[mode] = head["state_dict"]
        config["initial_head_sha256"] = {mode: file_hash(Path(head_run) / f"{mode}.pt") for mode in modes}
    root.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(root / "tokenizer")
    write_json(root / "run_config.json", config)
    write_json(root / "provenance.json", provenance)
    datasets = {split: PairedDataset(processed_manifest, source, split, gc["graph_variant"], tokenizer, tc["max_length"])
                for split in ("train", "validation")}
    rows, run_entries = [], []
    for mode in modes:
        set_seed(seed)
        text = _load_text_encoder(text_saved, "cpu") if mode != "gnn" else None
        # Clone trainable text tensors: each mode starts from the identical checkpoint.
        if text is not None:
            for block in transformer_blocks(text)[-unfreeze_layers:]:
                for parameter in block.parameters():
                    parameter.data = parameter.detach().clone()
        graph = GraphSAGEEncoder(**graph_config) if mode != "bert" else None
        if graph is not None:
            graph.load_state_dict({k.removeprefix("encoder."): v for k, v in graph_saved["state_dict"].items()
                                   if k.startswith("encoder.")})
        model = JointClassifier(text, graph, dimensions, mode, unfreeze_layers).to(device)
        if mode in initial_heads:
            model.head.load_state_dict(initial_heads[mode])
        rates = dict(text=text_lr, graph=graph_lr, head=head_lr)
        optimizer = torch.optim.AdamW([
            {"params": [p for name, p in model.named_parameters() if p.requires_grad and parameter_group(name) == group], "lr": rate}
            for group, rate in rates.items()
            if any(p.requires_grad and parameter_group(name) == group for name, p in model.named_parameters())
        ], weight_decay=0.01)
        anchors = parameter_anchors(model)
        frozen_before = frozen_text_hash(model)
        first_step = None
        loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_pairs,
                            generator=torch.Generator().manual_seed(seed))
        best, stale, history = -1.0, 0, []
        directory = root / mode
        directory.mkdir()
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for step, batch in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["graph"].to(device), {k: v.to(device) for k, v in batch["tokens"].items()})
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"].to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Joint training produced non-finite loss")
                loss.backward()
                if first_step is None:
                    norms = {group: 0.0 for group in anchors}
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                raise ValueError(f"Missing/non-finite gradient: {name}")
                            norms[parameter_group(name)] += float(parameter.grad.detach().square().sum())
                    if any(value <= 0 for value in norms.values()):
                        raise ValueError("An active encoder/head has zero gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if first_step is None:
                    changes = changed_parameters(model, anchors)
                    if any(v["max_abs_change"] <= 0 for v in changes.values()):
                        raise ValueError("An active encoder/head did not update")
                    first_step = dict(gradient_l2={k: math.sqrt(v) for k, v in norms.items()}, changes=changes)
                total += float(loss.detach()) * len(batch["labels"])
                if step == 1 or step % 100 == 0 or step == len(loader):
                    print(f"joint {mode} epoch={epoch} batch={step}/{len(loader)} loss={float(loss.detach()):.4f}", flush=True)
            prediction = predict(model, datasets["validation"], batch_size, device)
            threshold = select_global_threshold(prediction["targets"], prediction["probabilities"])
            metrics = metric_report(prediction["targets"], prediction["probabilities"], labels, threshold)
            val_bce = float(nn.functional.binary_cross_entropy(torch.from_numpy(prediction["probabilities"]),
                                                              torch.from_numpy(prediction["targets"])))
            history.append(dict(epoch=epoch, train_loss=total / len(datasets["train"]), validation_loss=val_bce,
                                validation_macro_f1=metrics["macro_f1"], threshold=threshold))
            write_json(directory / "history.json", history)
            print(f"joint {mode} epoch={epoch} validation_macro_f1={metrics['macro_f1']:.4f}", flush=True)
            if metrics["macro_f1"] > best:
                best, stale = metrics["macro_f1"], 0
                proof = dict(first_optimizer_step=first_step, selected_checkpoint_changes=changed_parameters(model, anchors),
                             frozen_text_sha256_before=frozen_before, frozen_text_sha256_after=frozen_text_hash(model))
                if proof["frozen_text_sha256_after"] != frozen_before:
                    raise ValueError("Frozen transformer parameters changed")
                saved = dict(format="task3_joint_v1", mode=mode, state_dict=model.state_dict(), architecture=architecture,
                             config=config, provenance=provenance, label_names=labels, best_epoch=epoch,
                             threshold=threshold, training_proof=proof)
                temporary = directory / "checkpoint.tmp"
                torch.save(saved, temporary)
                temporary.replace(directory / "best_model.pt")
                np.savez_compressed(directory / "validation_predictions.npz", **prediction)
                write_json(directory / "validation_metrics.json", metrics)
                write_json(directory / "training_proof.json", proof)
            else:
                stale += 1
                if stale >= patience:
                    break
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for key in ("train_loss", "validation_loss"):
            axes[0].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        axes[0].set(xlabel="Epoch", ylabel="BCE loss"); axes[0].legend()
        axes[1].plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history])
        axes[1].set(xlabel="Epoch", ylabel="Validation Macro-F1")
        fig.suptitle(f"Joint fine-tuning: {mode}, seed {seed}")
        fig.tight_layout(); fig.savefig(directory / "training_curves.png", dpi=140); plt.close(fig)
        selected = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True, mmap=True)
        metrics = read_json(directory / "validation_metrics.json")
        row = dict(mode=mode, seed=seed, best_epoch=selected["best_epoch"], threshold=selected["threshold"],
                   epochs_run=len(history), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   training_seconds=time.perf_counter() - started)
        row.update({f"validation_{key}": metrics[key] for key in ("macro_f1", "micro_f1", "mean_average_precision", "mean_pr_auc_trapezoidal")})
        rows.append(row)
        run_entries.append(dict(mode=mode, checkpoint=f"{mode}/best_model.pt", checkpoint_sha256=file_hash(directory / "best_model.pt"),
                                validation_predictions_sha256=file_hash(directory / "validation_predictions.npz")))
        _tables(root, rows)
        del selected, saved, model, optimizer, text, graph
    if dataset_identity(processed_manifest)["sha256"] != fingerprint or file_hash(source_manifest) != provenance["source_manifest_sha256"]:
        raise ValueError("Paired dataset changed during training")
    # All choices are frozen before any test forward pass.
    chosen = max(rows, key=lambda r: r["validation_macro_f1"])["mode"]
    fusion_rows = [r for r in rows if r["mode"] in ("concat", "gated", "cross_attention")]
    selection = dict(selected_mode=chosen, selected_fusion=max(fusion_rows, key=lambda r: r["validation_macro_f1"])["mode"] if fusion_rows else None,
                     selected_on="validation_macro_f1", runs=run_entries, run_config_sha256=file_hash(root / "run_config.json"),
                     provenance_sha256=file_hash(root / "provenance.json"))
    write_json(root / "selection.json", selection)
    write_json(root / "experiment.json", dict(rows=rows, selection_sha256=file_hash(root / "selection.json"), evaluated_test=False))
    return selection


def verify_joint(output_dir, verify_dataset=True):
    root = Path(output_dir)
    config, provenance = read_json(root / "run_config.json"), read_json(root / "provenance.json")
    selection, report = read_json(root / "selection.json"), read_json(root / "experiment.json")
    for path, digest in (("selection.json", report["selection_sha256"]),
                         ("run_config.json", selection["run_config_sha256"]), ("provenance.json", selection["provenance_sha256"])):
        if file_hash(root / path) != digest:
            raise ValueError(f"Frozen {path} changed")
    processed, source = aligned_manifests(config["processed_manifest"], config["source_manifest"])
    if file_hash(config["source_manifest"]) != provenance["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if verify_dataset and dataset_identity(config["processed_manifest"])["sha256"] != provenance["graph_dataset_fingerprint"]:
        raise ValueError("Graph dataset changed")
    if [r["mode"] for r in selection["runs"]] != config["modes"] or [r["mode"] for r in report["rows"]] != config["modes"]:
        raise ValueError("Run matrix differs from declaration")
    rows_by_id = {r["sample_id"]: r for r in source["records"]}
    for run, row in zip(selection["runs"], report["rows"]):
        mode = run["mode"]
        if file_hash(root / run["checkpoint"]) != run["checkpoint_sha256"]:
            raise ValueError("Joint checkpoint changed")
        saved = torch.load(root / run["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        if saved["label_names"] != config["label_names"] or saved["config"] != config or saved["provenance"] != provenance or saved["mode"] != mode:
            raise ValueError("Joint checkpoint identity differs")
        proof = saved["training_proof"]
        expected_groups = {"head"} | ({"text"} if mode != "gnn" else set()) | ({"graph"} if mode != "bert" else set())
        if set(proof["selected_checkpoint_changes"]) != expected_groups or any(v["max_abs_change"] <= 0 for v in proof["selected_checkpoint_changes"].values()):
            raise ValueError("Encoder/head update evidence is missing")
        if proof["frozen_text_sha256_before"] != proof["frozen_text_sha256_after"]:
            raise ValueError("Frozen transformer weights changed")
        for split in (("validation", "test") if report["evaluated_test"] else ("validation",)):
            path = root / mode / f"{split}_predictions.npz"
            expected_hash = run["validation_predictions_sha256"] if split == "validation" else row["test_predictions_sha256"]
            if file_hash(path) != expected_hash:
                raise ValueError(f"{split} predictions changed")
            with np.load(path) as archive:
                ids = [r["sample_id"] for r in processed["records"] if r["split"] == split]
                if archive["sample_ids"].tolist() != ids or not np.array_equal(archive["targets"], [rows_by_id[i]["labels"] for i in ids]):
                    raise ValueError("Prediction IDs/targets differ from paired dataset")
                threshold = saved["threshold"]
                if split == "validation" and select_global_threshold(archive["targets"], archive["probabilities"]) != threshold:
                    raise ValueError("Threshold was not selected using validation")
                metrics = metric_report(archive["targets"], archive["probabilities"], config["label_names"], threshold)
            if metrics != read_json(root / mode / f"{split}_metrics.json"):
                raise ValueError("Persisted metrics differ from recomputed values")
            for key in ("macro_f1", "micro_f1", "mean_average_precision", "mean_pr_auc_trapezoidal"):
                if row[f"{split}_{key}"] != metrics[key]:
                    raise ValueError("Comparison row differs from predictions")
        if row["threshold"] != saved["threshold"] or row["best_epoch"] != saved["best_epoch"]:
            raise ValueError("Comparison checkpoint metadata differs")
    if selection["selected_mode"] != max(report["rows"], key=lambda r: r["validation_macro_f1"])["mode"]:
        raise ValueError("Model choice differs from validation selection")
    fusions = [r for r in report["rows"] if r["mode"] in ("concat", "gated", "cross_attention")]
    if selection["selected_fusion"] != (max(fusions, key=lambda r: r["validation_macro_f1"])["mode"] if fusions else None):
        raise ValueError("Fusion choice differs from validation selection")
    expected = pd.DataFrame(report["rows"])
    for column in expected:
        if column not in ("mode", "test_predictions_sha256"):
            expected[column] = pd.to_numeric(expected[column])
    pd.testing.assert_frame_equal(pd.read_csv(root / "comparison.csv"), expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    result = dict(valid=True, runs_checked=len(report["rows"]), evaluated_test=report["evaluated_test"],
                  validation_only_selection=True, encoder_updates_recorded=True, frozen_layers_unchanged=True,
                  dataset_verified=verify_dataset)
    write_json(root / "verification.json", result)
    return result


def evaluate_joint(output_dir, device="auto", cpu_threads=6):
    torch.set_num_threads(cpu_threads)
    root = Path(output_dir)
    verify_joint(root)
    config, report, selection = (read_json(root / p) for p in ("run_config.json", "experiment.json", "selection.json"))
    if report["evaluated_test"]:
        raise FileExistsError("Joint test evaluation already exists; use verify")
    for mode in config["modes"]:
        if (root / mode / "test_predictions.npz").exists():
            raise FileExistsError("Partial test predictions exist; preserve and inspect before continuing")
    source = load_manifest(config["source_manifest"])
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    dataset = PairedDataset(config["processed_manifest"], source, "test", config["graph_variant"], tokenizer, config["max_length"])
    device = resolve_device(device)
    for run, row in zip(selection["runs"], report["rows"]):
        saved = torch.load(root / run["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        model = restore_model(saved, device)
        prediction = predict(model, dataset, config["batch_size"], device)
        metrics = metric_report(prediction["targets"], prediction["probabilities"], config["label_names"], saved["threshold"])
        directory = root / run["mode"]
        np.savez_compressed(directory / "test_predictions.npz", **prediction)
        write_json(directory / "test_metrics.json", metrics)
        row.update({f"test_{key}": metrics[key] for key in ("macro_f1", "micro_f1", "mean_average_precision", "mean_pr_auc_trapezoidal")})
        row["test_predictions_sha256"] = file_hash(directory / "test_predictions.npz")
        print(f"joint test {run['mode']} macro_f1={metrics['macro_f1']:.4f}", flush=True)
        del model, saved
    report["evaluated_test"] = True
    _tables(root, report["rows"])
    write_json(root / "experiment.json", report)
    return verify_joint(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("train")
    for name in ("processed-manifest", "source-manifest", "text-checkpoint", "graph-checkpoint", "output-dir"):
        training.add_argument(f"--{name}", type=Path, required=True)
    training.add_argument("--head-run", type=Path)
    for name, default in (("epochs", 3), ("batch-size", 8), ("hidden-dim", 64), ("unfreeze-layers", 1), ("patience", 2), ("seed", 42), ("cpu-threads", 6)):
        training.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("text-lr", 1e-5), ("graph-lr", 1e-4), ("head-lr", 1e-4)):
        training.add_argument(f"--{name}", type=float, default=default)
    training.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    training.add_argument("--device", default="auto")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--cpu-threads", type=int, default=6)
    verification = sub.add_parser("verify")
    verification.add_argument("--output-dir", type=Path, required=True)
    options = vars(parser.parse_args())
    action = options.pop("action")
    result = {"train": train_joint, "evaluate": evaluate_joint, "verify": verify_joint}[action](**options)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
