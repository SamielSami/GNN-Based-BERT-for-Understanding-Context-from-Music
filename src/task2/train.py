"""Train and evaluate Task 2 GraphSAGE, pooled-MLP, or mel-CNN models."""
from __future__ import annotations

import argparse
import json
import random
import time
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ProcessedTask2Dataset, collate_task2, load_manifest, dataset_identity, file_sha256
from .graphs import GRAPH_VARIANTS
from .models import build_task2_model


@dataclass
class Task2TrainConfig:
    manifest: str
    output_dir: str = "results/task2/gnn"
    model: str = "gnn"
    graph_variant: str = "temporal_similarity"
    hidden_dim: int = 128
    layers: int = 2
    dropout: float = 0.2
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    use_pos_weight: bool = False
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    evaluate_test: bool = True
    cpu_threads: int = 6

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience", "hidden_dim", "layers", "cpu_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def forward_model(model: nn.Module, name: str, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    if name == "cnn":
        return model(batch["mel"].to(device))
    return model(batch["graph"].to(device))


def select_global_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Tune one threshold on validation data only using Macro-F1."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(
            targets, probabilities >= threshold, average="macro", zero_division=0
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if targets.ndim != 2 or targets.shape != probabilities.shape or targets.shape[1] != len(label_names) or len(targets) == 0:
        raise ValueError("Targets/probabilities must be non-empty matching [samples, labels] matrices")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Metrics require binary targets and finite probabilities in [0, 1]")
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    average_precision = []
    for column in range(targets.shape[1]):
        if targets[:, column].sum() == 0:
            average_precision.append(None)
        else:
            average_precision.append(
                float(average_precision_score(targets[:, column], probabilities[:, column]))
            )
    valid_ap = [value for value in average_precision if value is not None]
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "mean_average_precision": float(np.mean(valid_ap)) if valid_ap else None,
        "threshold": threshold,
        "samples": int(len(targets)),
        "map_label_count": len(valid_ap),
        "map_definition": "Mean per-label average precision over labels with positive support; not trapezoidal PR AUC",
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "average_precision": average_precision[index],
            }
            for index, label in enumerate(label_names)
        },
    }


def run_loader(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probabilities, all_targets, all_ids = [], [], []
    for batch in loader:
        targets = batch["labels"].to(device)
        with torch.set_grad_enabled(training):
            logits = forward_model(model, model_name, batch, device)
            loss = loss_fn(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Training/evaluation produced non-finite logits or loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(targets)
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy().astype(int))
        all_ids.extend(batch["sample_ids"])
    return {
        "loss": total_loss / len(loader.dataset),
        "probabilities": np.concatenate(all_probabilities),
        "targets": np.concatenate(all_targets),
        "sample_ids": all_ids,
    }


def compute_pos_weight(dataset: ProcessedTask2Dataset) -> torch.Tensor:
    labels = torch.stack([dataset[index]["labels"] for index in range(len(dataset))])
    positives = labels.sum(dim=0)
    negatives = len(labels) - positives
    return (negatives / positives.clamp_min(1.0)).clamp(max=20.0)


def save_history_plot(history: list[dict[str, Any]], destination: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="Task 2 loss", xlabel="Epoch", ylabel="BCE loss")
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history])
    axes[1].set(title="Validation Macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend() if axis is axes[0] else None
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _make_loader(
    dataset: ProcessedTask2Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        collate_fn=collate_task2,
    )


def train_task2(config: Task2TrainConfig) -> dict[str, Any]:
    if config.model not in {"gnn", "mlp", "cnn"}:
        raise ValueError("model must be gnn, mlp, or cnn")
    if config.graph_variant not in GRAPH_VARIANTS:
        raise ValueError(f"graph_variant must be one of {GRAPH_VARIANTS}")
    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh training output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config.manifest)
    from .audit import audit_saved_graphs
    if not audit_saved_graphs(config.manifest)["valid"]:
        raise ValueError("Graph audit failed; inspect graph_audit.json before training")
    identity = dataset_identity(config.manifest)
    split_ids = {split: [r["sample_id"] for r in manifest["records"] if r["split"] == split]
                 for split in ("train", "validation", "test")}
    provenance = {
        "dataset_fingerprint": identity["sha256"], "synthetic": manifest.get("synthetic", False),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": str(device), "cpu_threads": torch.get_num_threads(),
        "source_sha256": {path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")},
    }
    for name, value in (("run_config.json", asdict(config)), ("dataset_identity.json", identity),
                        ("split_ids.json", split_ids), ("provenance.json", provenance)):
        (output_dir / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    train_dataset = ProcessedTask2Dataset(config.manifest, "train", config.graph_variant)
    validation_dataset = ProcessedTask2Dataset(
        config.manifest, "validation", config.graph_variant
    )
    first_sample = train_dataset[0]
    input_dim = int(first_sample["graph"].x.shape[1])
    num_labels = len(manifest["label_names"])
    model = build_task2_model(
        config.model,
        input_dim=input_dim,
        num_labels=num_labels,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_loader = _make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    pos_weight = compute_pos_weight(train_dataset).to(device) if config.use_pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    best_macro_f1, stale_epochs = -1.0, 0
    checkpoint_path = output_dir / "best_model.pt"
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_result = run_loader(model, config.model, train_loader, device, loss_fn, optimizer)
        validation_result = run_loader(
            model, config.model, validation_loader, device, loss_fn
        )
        threshold = select_global_threshold(
            validation_result["targets"], validation_result["probabilities"]
        )
        validation_metrics = classification_metrics(
            validation_result["targets"],
            validation_result["probabilities"],
            manifest["label_names"],
            threshold,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_micro_f1": validation_metrics["micro_f1"],
            "validation_map": validation_metrics["mean_average_precision"],
            "threshold": threshold,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_macro_f1={row['validation_macro_f1']:.4f} threshold={threshold:.2f}"
        )
        if row["validation_macro_f1"] > best_macro_f1:
            best_macro_f1 = row["validation_macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "input_dim": input_dim,
                    "num_labels": num_labels,
                    "label_names": manifest["label_names"],
                    "threshold": threshold,
                    "best_epoch": epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "parameter_count": parameter_count,
                    "dataset_fingerprint": identity["sha256"],
                    "split_ids": split_ids,
                    "synthetic": provenance["synthetic"],
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    elapsed_seconds = time.perf_counter() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    summary: dict[str, Any] = {
        "config": asdict(config),
        "model": config.model,
        "graph_variant": config.graph_variant,
        "parameter_count": parameter_count,
        "training_seconds": elapsed_seconds,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_macro_f1": checkpoint["best_validation_macro_f1"],
        "validation_threshold": checkpoint["threshold"],
        "history": history,
        "test": None,
        "dataset_fingerprint": identity["sha256"],
        "synthetic": provenance["synthetic"],
        "device": str(device),
    }
    validation_result = run_loader(model, config.model, validation_loader, device, loss_fn)
    summary["validation"] = classification_metrics(
        validation_result["targets"], validation_result["probabilities"],
        manifest["label_names"], checkpoint["threshold"],
    )
    save_predictions(output_dir, "validation", validation_result, manifest["label_names"], checkpoint["threshold"])
    test_records = [record for record in manifest["records"] if record["split"] == "test"]
    if test_records and config.evaluate_test:
        test_dataset = ProcessedTask2Dataset(config.manifest, "test", config.graph_variant)
        test_loader = _make_loader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        test_start = time.perf_counter()
        test_result = run_loader(model, config.model, test_loader, device, loss_fn)
        summary["test_seconds"] = time.perf_counter() - test_start
        summary["test"] = classification_metrics(
            test_result["targets"],
            test_result["probabilities"],
            manifest["label_names"],
            checkpoint["threshold"],
        )
        save_predictions(output_dir, "test", test_result, manifest["label_names"], checkpoint["threshold"])
    if dataset_identity(config.manifest)["sha256"] != identity["sha256"]:
        raise ValueError("Dataset changed while training; this run cannot be compared")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_history_plot(history, output_dir / "training_curves.png")
    print(json.dumps({key: summary[key] for key in ("model", "graph_variant", "parameter_count", "training_seconds", "best_epoch", "best_validation_macro_f1", "synthetic")}, indent=2))
    return summary


def save_predictions(output_dir: Path, split: str, result: dict, label_names: list[str], threshold: float) -> None:
    np.savez_compressed(
        output_dir / f"{split}_predictions.npz", probabilities=result["probabilities"],
        targets=result["targets"], sample_ids=np.asarray(result["sample_ids"]),
        label_names=np.asarray(label_names), threshold=np.asarray(threshold),
        predictions=(result["probabilities"] >= threshold).astype(np.int8),
    )
    (output_dir / f"{split}_sample_ids.json").write_text(json.dumps(result["sample_ids"], indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="results/task2/gnn")
    parser.add_argument("--model", choices=["gnn", "mlp", "cnn"], default="gnn")
    parser.add_argument("--graph-variant", choices=GRAPH_VARIANTS, default="temporal_similarity")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate test while selecting an experimental configuration",
    )
    args = parser.parse_args()
    values = vars(args)
    values["evaluate_test"] = not values.pop("skip_test")
    train_task2(Task2TrainConfig(**values))


if __name__ == "__main__":
    main()
