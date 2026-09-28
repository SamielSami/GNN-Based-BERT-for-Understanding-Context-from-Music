"""Train projection heads and retrieve paired captions/graphs from frozen features."""
import argparse
import json
import math
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.task2.train import resolve_device, set_seed
from src.task3.data import load_features, feature_fingerprint, make_demo
from .model import RetrievalModel, contrastive_loss
from .metrics import retrieval_metrics, normalized_pairs, paired_ranks


def paired_inputs(features):
    data = load_features(features, mmap=True)
    for split in ("train", "validation", "test"):
        if data["splits"].count(split) < 2:
            raise ValueError(f"Retrieval needs at least two pairs in {split}")
    # CLS pooling matches the Task 1 encoder convention; labels never enter the loss.
    return data, data["tokens"][:, 0].detach().float().contiguous(), data["graph"].detach().float().contiguous()


def encode(model, text, graph, indices, device, batch_size=256):
    model.eval()
    texts, graphs = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            t, g = model(text[chunk].to(device), graph[chunk].to(device))
            texts.append(t.cpu().numpy()); graphs.append(g.cpu().numpy())
    return np.concatenate(texts), np.concatenate(graphs)


def empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Use a new, empty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate(checkpoint, features, output_dir, split="test", device="auto"):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Invalid split")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["feature_sha256"] != feature_fingerprint(features):
        raise ValueError("Feature file differs from checkpoint provenance")
    data, text, graph = paired_inputs(features)
    device = resolve_device(device)
    model = RetrievalModel(**saved["dimensions"]).to(device)
    model.load_state_dict(saved["state_dict"])
    indices = [i for i, s in enumerate(data["splits"]) if s == split]
    t, g = encode(model, text, graph, indices, device)
    report = dict(split=split, best_epoch=saved["best_epoch"], provenance=data.get("provenance", {}),
                  feature_sha256=saved["feature_sha256"], metrics=retrieval_metrics(t, g))
    baseline_embeddings = None
    if "initial_state_dict" in saved:
        model.load_state_dict(saved["initial_state_dict"])
        baseline_embeddings = encode(model, text, graph, indices, device)
        report["untrained_projection_metrics"] = retrieval_metrics(*baseline_embeddings)
        report["mean_recall_at_1_gain"] = (
            report["metrics"]["mean_recall_at_1"] - report["untrained_projection_metrics"]["mean_recall_at_1"]
        )
    output = empty_output(output_dir)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ids = [data["sample_ids"][i] for i in indices]
    np.savez_compressed(output / "embeddings.npz", text=t, graph=g, sample_ids=np.asarray(ids))
    if baseline_embeddings is not None:
        np.savez_compressed(output / "untrained_embeddings.npz", text=baseline_embeddings[0],
                            graph=baseline_embeddings[1], sample_ids=np.asarray(ids))
    examples, rank_records = [], {}
    nt, ng = normalized_pairs(t, g)
    for direction, queries, gallery in (("text_to_graph", nt, ng), ("graph_to_text", ng, nt)):
        ranks = paired_ranks(queries, gallery)
        rank_records[direction] = ranks.tolist()
        # Fixed queries plus explicitly outcome-selected best/worst diagnostics.
        selected = {i: "fixed_first_three" for i in range(min(3, len(ids)))}
        selected.setdefault(int(np.argmin(ranks)), "best_rank_diagnostic")
        selected.setdefault(int(np.argmax(ranks)), "worst_rank_diagnostic")
        for query, selection in selected.items():
            scores = queries[query] @ gallery.T
            # Put tied negatives before this positive, matching aggregate metrics.
            order = np.lexsort((np.arange(len(ids)) == query, -scores))[:min(10, len(ids))]
            examples.append(dict(direction=direction, query_id=ids[query], caption=data["texts"][indices[query]],
                                 selection=selection, paired_rank=int(ranks[query]),
                                 ranked_matches=[dict(sample_id=ids[i], score=float(scores[i]),
                                                      caption=data["texts"][indices[i]], rank=rank,
                                                      is_paired_positive=bool(i == query)) for rank, i in enumerate(order, 1)]))
    (output / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    (output / "ranks.json").write_text(json.dumps(dict(sample_ids=ids, **rank_records), indent=2), encoding="utf-8")
    return report


def train(features, output_dir, epochs=20, batch_size=16, projection_dim=64,
          temperature=0.07, learning_rate=1e-3, patience=5, seed=42, device="auto", cpu_threads=6):
    if min(epochs, projection_dim, patience) < 1 or batch_size < 2:
        raise ValueError("Positive epochs/dimensions/patience and batch_size >= 2 required")
    if any(not math.isfinite(v) or v <= 0 for v in (temperature, learning_rate)):
        raise ValueError("Temperature and learning rate must be positive and finite")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    data, text, graph = paired_inputs(features)
    output = empty_output(output_dir)
    set_seed(seed)
    device = resolve_device(device)
    dims = dict(text_dim=text.shape[1], graph_dim=graph.shape[1], projection_dim=projection_dim)
    model = RetrievalModel(**dims).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_ids = torch.tensor([i for i, s in enumerate(data["splits"]) if s == "train"])
    val_ids = [i for i, s in enumerate(data["splits"]) if s == "validation"]
    config = dict(epochs=epochs, batch_size=batch_size, projection_dim=projection_dim, temperature=temperature, learning_rate=learning_rate,
                  patience=patience, seed=seed, device=str(device), dimensions=dims,
                  cpu_threads=cpu_threads, weight_decay=0.01,
                  parameter_count=sum(p.numel() for p in model.parameters()), frozen_encoders=True)
    fingerprint = feature_fingerprint(features)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    initial_t, initial_g = encode(model, text, graph, val_ids, device)
    initial_metrics = retrieval_metrics(initial_t, initial_g)
    (output / "initial_validation.json").write_text(json.dumps(initial_metrics, indent=2), encoding="utf-8")
    best, stale, history = -1.0, 0, []
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = train_ids[torch.randperm(len(train_ids), generator=generator)]
        batches = list(shuffled.split(batch_size))
        # A singleton has no negatives: merge it into the preceding batch.
        if len(batches) > 1 and len(batches[-1]) == 1:
            tail = batches.pop()
            batches[-1] = torch.cat([batches[-1], tail])
        total = 0.0
        for batch in batches:
            t, g = model(text[batch].to(device), graph[batch].to(device))
            loss = contrastive_loss(t, g, temperature)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        t, g = encode(model, text, graph, val_ids, device)
        metrics = retrieval_metrics(t, g)
        score = metrics["mean_recall_at_1"]
        history.append(dict(epoch=epoch, train_loss=total / len(train_ids), validation=metrics))
        print(f"epoch={epoch} loss={total / len(train_ids):.4f} validation_mean_R@1={score:.4f}", flush=True)
        if score > best:
            best, stale = score, 0
            torch.save(dict(state_dict=model.state_dict(), dimensions=dims, config=config,
                            initial_state_dict=initial_state,
                            feature_sha256=fingerprint, best_epoch=epoch, provenance=data.get("provenance", {})),
                       output / "best_model.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    (output / "timing.json").write_text(json.dumps(dict(training_seconds=time.perf_counter() - started)), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="InfoNCE loss")
    for direction in ("text_to_graph", "graph_to_text"):
        axes[1].plot([r["epoch"] for r in history], [r["validation"][direction]["recall_at_1"] for r in history], label=direction)
    axes[1].axhline(initial_metrics["mean_recall_at_1"], linestyle="--", color="gray", label="Untrained projections")
    axes[1].axhline(1 / len(val_ids), linestyle=":", color="orange", label="Random ranking expectation")
    axes[1].legend()
    axes[1].set(xlabel="Epoch", ylabel="Validation Recall@1", ylim=(0, None))
    fig.tight_layout(); fig.savefig(output / "learning_curves.png"); plt.close(fig)
    return evaluate(output / "best_model.pt", features, output / "validation", split="validation", device=str(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    group = training.add_mutually_exclusive_group(required=True)
    group.add_argument("--features", type=Path)
    group.add_argument("--demo", action="store_true")
    for name, default in (("epochs", 20), ("batch-size", 16), ("projection-dim", 64), ("patience", 5), ("seed", 42), ("cpu-threads", 6)):
        training.add_argument(f"--{name}", type=int, default=default)
    training.add_argument("--temperature", type=float, default=0.07)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--features", type=Path, required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    for command in (training, evaluation):
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--device", default="auto")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "train":
        if args.pop("demo"):
            demo_path = args["output_dir"].with_suffix(".demo.pt")
            if demo_path.exists():
                parser.error("Demo feature file already exists; use a new output path")
            args["features"] = make_demo(demo_path, args["seed"])
        result = train(**args)
    else:
        result = evaluate(**args)
    print(json.dumps(result, indent=2))
