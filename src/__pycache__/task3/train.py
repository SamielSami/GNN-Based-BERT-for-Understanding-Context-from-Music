"""Run five frozen-feature ablations; select on validation and evaluate one winner."""
import argparse
import csv
import json
import time
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.task2.train import classification_metrics, resolve_device, select_global_threshold, set_seed
from .data import load_features, make_demo, feature_fingerprint
from .models import FusionClassifier, MODES


def run_experiments(features, output_dir, epochs=20, batch_size=16, hidden_dim=64,
                    seed=42, device="auto", evaluate_test=False, patience=5, learning_rate=1e-3,
                    cpu_threads=6):
    if min(epochs, batch_size, hidden_dim) < 1:
        raise ValueError("epochs, batch_size, and hidden_dim must be positive")
    if patience < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("patience and learning_rate must be positive and finite")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    data = load_features(features, mmap=True)
    output = Path(output_dir)
    # Keep checkpoints and reports from different invocations separate.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty output directory for each experiment")
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    fingerprint = feature_fingerprint(features)
    config = dict(epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim, seed=seed,
                  device=str(device), patience=patience, learning_rate=learning_rate, cpu_threads=cpu_threads,
                  features=str(Path(features).resolve()), feature_sha256=fingerprint)
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    inputs = [data["tokens"].float(), data["mask"].bool(), data["graph"].float(), data["labels"].float()]
    split_indices = {s: torch.tensor([i for i, value in enumerate(data["splits"]) if value == s])
                     for s in ("train", "validation", "test")}
    dims = dict(text_dim=inputs[0].shape[-1], graph_dim=inputs[2].shape[-1],
                num_labels=len(data["label_names"]), hidden_dim=hidden_dim)

    def mode_inputs(mode):
        if mode == "cross_attention":
            return inputs
        return [inputs[0][:, :1], inputs[1][:, :1], *inputs[2:]]

    def infer(model, split):
        model.eval()
        indices = split_indices[split]
        probs, embeddings = [], []
        with torch.no_grad():
            for chunk in indices.split(batch_size):
                logits, embedding = model(*(x[chunk].to(device) for x in mode_inputs(model.mode)[:3]), return_embedding=True)
                probs.append(logits.sigmoid().cpu())
                embeddings.append(embedding.cpu())
        return torch.cat(probs).numpy(), torch.cat(embeddings).numpy()

    rows = []
    for mode in MODES:
        set_seed(seed)
        model = FusionClassifier(**dims, mode=mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loader = DataLoader(Subset(TensorDataset(*mode_inputs(mode)), split_indices["train"].tolist()),
                            batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        best, history, stale_epochs = -1.0, [], 0
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for batch in loader:
                t, m, g, y = (x.to(device) for x in batch)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(t, m, g), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(y)
            probabilities, _ = infer(model, "validation")
            targets = inputs[3][split_indices["validation"]].numpy()
            threshold = select_global_threshold(targets, probabilities)
            metrics = classification_metrics(targets, probabilities, data["label_names"], threshold)
            history.append(dict(epoch=epoch, train_loss=total / len(loader.dataset),
                                validation_macro_f1=metrics["macro_f1"], threshold=threshold))
            print(f"{mode} epoch={epoch} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                stale_epochs = 0
                torch.save(dict(state_dict=model.state_dict(), dimensions=dims, mode=mode,
                                label_names=data["label_names"], threshold=threshold, best_epoch=epoch,
                                config=config, feature_sha256=fingerprint,
                                provenance=data.get("provenance", {})), output / f"{mode}.pt")
                best_metrics = metrics
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        training_seconds = time.perf_counter() - started
        saved = torch.load(output / f"{mode}.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        probabilities, embeddings = infer(model, "validation")
        validation_indices = split_indices["validation"].tolist()
        np.savez_compressed(output / f"{mode}_validation_predictions.npz", probabilities=probabilities,
                            targets=targets, embeddings=embeddings,
                            sample_ids=np.array([data["sample_ids"][i] for i in validation_indices]))
        (output / f"{mode}_validation_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        (output / f"{mode}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        plt.plot([r["epoch"] for r in history], [r["validation_macro_f1"] for r in history], label=mode)
        rows.append(dict(mode=mode, validation_macro_f1=best,
                         epochs_run=len(history),
                         validation_micro_f1=best_metrics["micro_f1"],
                         validation_map=best_metrics["mean_average_precision"],
                         best_epoch=saved["best_epoch"], threshold=saved["threshold"],
                         seconds=training_seconds,
                         parameters=sum(p.numel() for p in model.parameters())))
    plt.xlabel("Epoch"); plt.ylabel("Validation Macro-F1"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "validation_curves.png"); plt.close()
    for mode in MODES:
        history = json.loads((output / f"{mode}_history.json").read_text(encoding="utf-8"))
        plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label=mode)
    plt.xlabel("Epoch"); plt.ylabel("Training BCE loss"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "training_curves.png"); plt.close()
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    selected = max(rows, key=lambda row: row["validation_macro_f1"])["mode"]
    (output / "selection.json").write_text(json.dumps(
        dict(mode=selected, checkpoint=f"{selected}.pt", selected_on="validation_macro_f1",
             feature_sha256=fingerprint), indent=2), encoding="utf-8")
    checkpoint = torch.load(output / f"{selected}.pt", map_location=device, weights_only=True)
    model = FusionClassifier(**dims, mode=selected).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    split = "test" if evaluate_test else "validation"
    probabilities, embeddings = infer(model, split)
    indices = split_indices[split].tolist()
    targets = inputs[3][indices].numpy()
    report = dict(selected_mode=selected, evaluated_split=split, seed=seed,
                  provenance=data.get("provenance", {}),
                  metrics=classification_metrics(targets, probabilities, data["label_names"], checkpoint["threshold"]))
    (output / "selected_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "selected_predictions.npz", probabilities=probabilities, targets=targets,
                        embeddings=embeddings, sample_ids=np.array([data["sample_ids"][i] for i in indices]))
    cases = [dict(sample_id=data["sample_ids"][i], text=data["texts"][i],
                  true_tags=[name for j, name in enumerate(data["label_names"]) if targets[k, j]],
                  predicted_tags=[name for j, name in enumerate(data["label_names"]) if probabilities[k, j] >= checkpoint["threshold"]])
             for k, i in enumerate(indices[:3])]
    (output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    if len(indices) >= 3:
        points = TSNE(n_components=2, perplexity=min(30, len(indices) - 1), random_state=seed,
                      init="random", learning_rate="auto").fit_transform(embeddings)
        plt.scatter(points[:, 0], points[:, 1], c=targets.sum(1), cmap="viridis")
        plt.colorbar(label="Number of true labels")
        plt.title(f"{selected}: {split} embeddings (descriptive t-SNE)")
        plt.tight_layout(); plt.savefig(output / "embeddings.png"); plt.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.demo == bool(args.features):
        parser.error("Choose exactly one of --demo or --features")
    features = make_demo(args.output_dir.with_suffix(".demo.pt"), args.seed) if args.demo else args.features
    print(json.dumps(run_experiments(features, args.output_dir, epochs=args.epochs,
                                    batch_size=args.batch_size, seed=args.seed, device=args.device,
                                    hidden_dim=args.hidden_dim, patience=args.patience,
                                    learning_rate=args.learning_rate,
                                    cpu_threads=args.cpu_threads,
                                    evaluate_test=args.evaluate_test), indent=2))


if __name__ == "__main__":
    main()
