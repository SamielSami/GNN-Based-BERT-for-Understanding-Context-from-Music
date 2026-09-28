"""Train and evaluate the Task 1 BERT multi-label music-tag baseline."""
from __future__ import annotations

import argparse
import json
import random
import time
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .audioset import load_prepared_manifest, sha256_file, split_by_id


@dataclass
class Task1Config:
    data_path: str
    output_dir: str = "results/task1"
    model_name: str = "distilbert-base-uncased"
    text_col: str = "text"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    lr_bert: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-5
    val_size: float = 0.15
    test_size: float = 0.15
    freeze_bert: bool = False
    patience: int = 3
    threshold: float = 0.5
    seed: int = 42
    device: str = "auto"
    split_path: str | None = None
    tune_thresholds: bool = True
    cpu_threads: int = 6
    run_baselines: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "max_length", "patience", "cpu_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not (self.val_size > 0 and self.test_size > 0 and self.val_size + self.test_size < 1):
            raise ValueError("val_size and test_size must be positive and sum to less than 1")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tag_data(path: str | Path, text_col: str = "text") -> tuple[list[str], np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    if text_col not in frame.columns:
        raise ValueError(f"Text column {text_col!r} is missing from {path}")
    tag_cols = [name for name in frame.columns if name not in {text_col, "ytid"}]
    if not tag_cols:
        raise ValueError("The CSV must contain at least one binary tag column")
    numeric = frame[tag_cols].apply(pd.to_numeric, errors="raise")
    values = set(np.unique(numeric.to_numpy()).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Tag columns must be binary 0/1; found {sorted(values)}")
    texts = frame[text_col].fillna("").astype(str).str.strip()
    if (texts.str.len() == 0).any():
        raise ValueError("Text inputs must be non-empty")
    if texts.duplicated().any():
        raise ValueError("Duplicate captions detected; remove or group them before splitting")
    if "ytid" in frame and (frame.ytid.isna().any() or frame.ytid.astype(str).str.strip().eq("").any()):
        raise ValueError("ytid values must be non-empty")
    return texts.tolist(), numeric.to_numpy(dtype=np.float32), tag_cols


def make_split_indices(
    sample_count: int,
    *,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[int]]:
    if sample_count < 7:
        raise ValueError("At least seven samples are required for train/validation/test splits")
    held_out = val_size + test_size
    if val_size <= 0 or test_size <= 0 or held_out >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")
    indices = np.arange(sample_count)
    train_idx, held_idx = train_test_split(indices, test_size=held_out, random_state=seed)
    relative_test_size = test_size / held_out
    val_idx, test_idx = train_test_split(
        held_idx, test_size=relative_test_size, random_state=seed
    )
    result = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    flat = [item for split in result.values() for item in split]
    if len(flat) != sample_count or len(set(flat)) != sample_count:
        raise RuntimeError("Generated splits are not disjoint and exhaustive")
    return result


class TagDataset(Dataset):
    def __init__(self, texts, labels, indices, tokenizer, max_length: int) -> None:
        self.texts = [texts[index] for index in indices]
        self.labels = labels[np.asarray(indices)]
        self.encoded = tokenizer(self.texts, truncation=True, padding="max_length",
                                 max_length=max_length, return_tensors="pt")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: value[index] for name, value in self.encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class BertTagClassifier(nn.Module):
    """CLS pooling followed by dropout and one logit per tag."""

    def __init__(self, model_name: str, num_labels: int, freeze_bert: bool = False) -> None:
        super().__init__()
        self.freeze_bert = freeze_bert
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        if freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bert:
            self.bert.eval()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.bert(**kwargs).last_hidden_state[:, 0]
        return self.classifier(self.dropout(hidden))


def evaluate_epoch(model, loader, device, threshold: float, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probabilities, all_targets = [], []
    for batch in loader:
        targets = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}
        with torch.set_grad_enabled(training):
            logits = model(**inputs)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        total_loss += loss.item() * len(targets)
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets).astype(int)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "probabilities": probabilities,
        "targets": targets,
    }


def select_thresholds(validation_targets, validation_probabilities) -> dict:
    """Maximize per-label validation F1 on a fixed grid; ties favor sparsity.

    A label without validation positives uses the validation-selected global
    threshold. Test targets/probabilities are deliberately not accepted here.
    """
    targets = np.asarray(validation_targets)
    probabilities = np.asarray(validation_probabilities)
    if targets.shape != probabilities.shape or targets.ndim != 2 or len(targets) == 0:
        raise ValueError("Validation arrays must be nonempty matching 2D matrices")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)) or not np.isin(targets, [0, 1]).all():
        raise ValueError("Invalid validation probabilities or targets")
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()
    global_scores = [f1_score(targets, probabilities >= t, average="macro", zero_division=0) for t in grid]
    global_threshold = grid[len(grid) - 1 - int(np.argmax(global_scores[::-1]))]
    thresholds, unsupported = [], []
    for column in range(targets.shape[1]):
        if targets[:, column].sum() == 0:
            thresholds.append(global_threshold)
            unsupported.append(column)
        else:
            scores = [f1_score(targets[:, column], probabilities[:, column] >= t, zero_division=0) for t in grid]
            thresholds.append(grid[len(grid) - 1 - int(np.argmax(scores[::-1]))])
    return {"fit_split": "validation", "objective": "per_label_f1", "grid": grid,
            "tie_break": "highest_threshold", "global_threshold": global_threshold,
            "thresholds": thresholds, "no_validation_positives": unsupported}


def classification_report(targets, probabilities, tags, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    auc_pr = []
    for column in range(len(tags)):
        if targets[:, column].sum() == 0:
            auc_pr.append(None)
        else:
            auc_pr.append(float(average_precision_score(targets[:, column], probabilities[:, column])))
    valid_auc_pr = [value for value in auc_pr if value is not None]
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "mean_auc_pr": float(np.mean(valid_auc_pr)) if valid_auc_pr else None,
        "threshold": np.asarray(threshold).tolist(),
        "mean_average_precision": float(np.mean(valid_auc_pr)) if valid_auc_pr else None,
        "ap_label_count": len(valid_auc_pr),
        "samples": int(len(targets)),
        "per_tag": {
            tag: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auc_pr": auc_pr[i],
            }
            for i, tag in enumerate(tags)
        },
    }


def save_curve(history: list[dict], destination: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_macro_f1"] for row in history], label="Train Macro-F1")
    plt.plot(epochs, [row["val_macro_f1"] for row in history], label="Validation Macro-F1")
    plt.plot(
        epochs,
        [row["val_micro_f1"] for row in history],
        "--",
        label="Validation Micro-F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Task 1: BERT Tag Classifier")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=150)
    plt.close()


def train(config: Task1Config) -> dict:
    started = time.perf_counter()
    set_seed(config.seed)
    device = resolve_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve existing work: {output_dir}")
    texts, labels, tags = load_tag_data(config.data_path, config.text_col)
    frame = pd.read_csv(config.data_path, dtype={"ytid": str})
    manifest = load_prepared_manifest(config.data_path)
    if "ytid" in frame and any(tag.startswith(("/m/", "/t/")) for tag in tags) and manifest is None:
        raise ValueError("AudioSet training requires its preparation manifest")
    if manifest:
        splits = manifest["split_indices"]
        if config.split_path and json.loads(Path(config.split_path).read_text(encoding="utf-8")) != splits:
            raise ValueError("Explicit splits differ from the training-fitted vocabulary partition")
    elif config.split_path:
        splits = json.loads(Path(config.split_path).read_text(encoding="utf-8"))
        if set(splits) != {"train", "validation", "test"} or not all(splits.values()):
            raise ValueError("Explicit splits must define non-empty train, validation, and test")
        flat = [i for indices in splits.values() for i in indices]
        if any(type(i) is not int for i in flat) or sorted(flat) != list(range(len(texts))):
            raise ValueError("Explicit splits must be disjoint and cover each CSV row exactly once")
    elif "ytid" in frame:
        id_splits = split_by_id(frame.ytid, seed=config.seed, val_size=config.val_size, test_size=config.test_size)
        splits = {name: frame.index[frame.ytid.isin(ids)].tolist() for name, ids in id_splits.items()}
    else:
        splits = make_split_indices(
            len(texts), val_size=config.val_size, test_size=config.test_size, seed=config.seed
        )
    if "ytid" in frame:
        id_sets = [set(frame.iloc[indices].ytid) for indices in splits.values()]
        if any(a & b for i, a in enumerate(id_sets) for b in id_sets[i + 1:]):
            raise ValueError("ytid overlap across splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_indices.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    if manifest:
        (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    datasets = {
        name: TagDataset(texts, labels, indices, tokenizer, config.max_length)
        for name, indices in splits.items()
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.batch_size, shuffle=True, generator=generator
        ),
        "validation": DataLoader(datasets["validation"], batch_size=config.batch_size),
        "test": DataLoader(datasets["test"], batch_size=config.batch_size),
    }

    model = BertTagClassifier(config.model_name, len(tags), config.freeze_bert).to(device)
    parameter_groups = [
        {"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": config.lr_bert},
        {"params": list(model.classifier.parameters()), "lr": config.lr_head},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    total_steps = len(loaders["train"]) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    history: list[dict] = []
    best_f1 = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = evaluate_epoch(
            model, loaders["train"], device, config.threshold, optimizer, scheduler
        )
        val_metrics = evaluate_epoch(model, loaders["validation"], device, config.threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_micro_f1": train_metrics["micro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
            f"val_micro_f1={row['val_micro_f1']:.4f}"
        )
        if row["val_loss"] < best_loss:
            best_loss = row["val_loss"]
            best_f1 = row["val_macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "tags": tags,
                    "best_epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "selection_metric": "validation_bce_loss",
                    "best_val_loss": best_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after {epoch} epochs")
                break
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation = evaluate_epoch(model, loaders["validation"], device, config.threshold)
    calibration = select_thresholds(validation["targets"], validation["probabilities"]) if config.tune_thresholds else {
        "fit_split": None, "thresholds": [config.threshold] * len(tags), "objective": "fixed"}
    thresholds = np.asarray(calibration["thresholds"])
    checkpoint["thresholds"] = thresholds.tolist()
    checkpoint["calibration"] = calibration
    checkpoint["dataset_sha256"] = sha256_file(config.data_path)
    checkpoint["label_source"] = manifest["label_source"] if manifest else "legacy_unverified"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "thresholds.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    baseline_models = None
    if config.run_baselines and manifest:
        from .baselines import fit_baselines
        baseline_models = fit_baselines(texts, labels, tags, splits, config.seed)
    # All fitting, checkpoint selection and calibration finish before test inference.
    (output_dir / "selection.json").write_text(json.dumps({
        "best_epoch": checkpoint["best_epoch"], "selection_metric": "validation_bce_loss",
        "calibration": calibration, "test_used_for_selection": False,
        "baseline_calibration": {name: value[1] for name, value in (baseline_models or {}).items()},
    }, indent=2), encoding="utf-8")
    test_epoch = evaluate_epoch(model, loaders["test"], device, thresholds)
    test_report = classification_report(
        test_epoch["targets"], test_epoch["probabilities"], tags, thresholds
    )
    test_report["loss"] = test_epoch["loss"]
    metrics = {
        "dataset": {
            "path": config.data_path,
            "samples": len(texts),
            "num_tags": len(tags),
            "label_source": manifest["label_source"] if manifest else "legacy_unverified",
            "sha256": sha256_file(config.data_path),
            "proxy_label_warning": None if manifest else (
                "Labels are deterministic keyword matches from the same input captions; "
                "scores measure recovery of lexical proxy rules, not independent semantic annotation."
            ),
        },
        "config": asdict(config),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "history": history,
        "best_epoch": checkpoint["best_epoch"],
        "selection_metric": "validation_bce_loss",
        "calibration": calibration,
        "validation": classification_report(validation["targets"], validation["probabilities"], tags, thresholds),
        "test_fixed_0_5": classification_report(test_epoch["targets"], test_epoch["probabilities"], tags, 0.5),
        "test": test_report,
    }
    for name, values in (("validation", validation), ("test", test_epoch)):
        ids = frame.iloc[splits[name]].ytid.to_numpy(dtype=str) if "ytid" in frame else np.asarray(splits[name]).astype(str)
        np.savez_compressed(output_dir / f"{name}_predictions.npz", ytid=ids,
                            row_indices=np.asarray(splits[name]), tags=np.asarray(tags),
                            targets=values["targets"], probabilities=values["probabilities"], thresholds=thresholds)
    if baseline_models:
        from .baselines import evaluate_baselines
        metrics["baselines"] = evaluate_baselines(baseline_models, texts, labels, tags, splits, frame, output_dir)
    metrics["runtime"] = {"seconds": time.perf_counter() - started, "device": str(device),
                          "cpu_threads": torch.get_num_threads(), "platform": platform.platform(),
                          "torch": str(torch.__version__)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_curve(history, output_dir / "f1_curve.png")

    test_texts = datasets["test"].texts
    examples = []
    for index in range(min(5, len(test_texts))):
        probabilities = test_epoch["probabilities"][index]
        top_indices = np.argsort(-probabilities)[:5]
        examples.append(
            {
                "ytid": str(frame.iloc[splits["test"][index]]["ytid"]) if "ytid" in frame else None,
                "text": test_texts[index],
                "top_predicted": [
                    {"tag": tags[i], "probability": round(float(probabilities[i]), 4)}
                    for i in top_indices
                ],
                "true_tags": [tags[i] for i, value in enumerate(test_epoch["targets"][index]) if value],
            }
        )
    (output_dir / "example_predictions.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in test_report.items() if key != "per_tag"}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="results/task1")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-bert", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-path", help="Use explicit aligned train/validation/test row indices")
    parser.add_argument("--no-tune-thresholds", dest="tune_thresholds", action="store_false")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--no-baselines", dest="run_baselines", action="store_false")
    args = parser.parse_args()
    train(Task1Config(**vars(args)))


if __name__ == "__main__":
    main()
