"""Train BERT, GNN, or fusion classifiers from processed sample dictionaries."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from .bert_encoder import BertTagClassifier, BertTextEncoder
from .fusion_model import GNNBertFusion
from .task2.models import GNNClassifier, GraphSAGEEncoder

class MusicDataset(Dataset):
    def __init__(self, split_file: str):
        self.paths = json.loads(Path(split_file).read_text(encoding="utf-8"))
    def __len__(self): return len(self.paths)
    def __getitem__(self, index): return torch.load(self.paths[index], weights_only=False)

def collate(samples):
    return {"graph": Batch.from_data_list([s["graph"] for s in samples]), "text": [s.get("text", "") for s in samples], "labels": torch.stack([torch.as_tensor(s["labels"], dtype=torch.float32) for s in samples])}

def build_model(cfg, choice):
    text = BertTextEncoder(cfg["model"]["text_model"], cfg["model"]["freeze_text_encoder"])
    graph = GraphSAGEEncoder(cfg["model"]["audio_feature_dim"], cfg["model"]["hidden_dim"], cfg["model"]["gnn_layers"], cfg["model"]["dropout"])
    labels = cfg["data"]["num_labels"]
    if choice == "bert": return BertTagClassifier(text, labels, cfg["model"]["dropout"])
    if choice == "gnn": return GNNClassifier(graph, labels)
    return GNNBertFusion(graph, text, labels, cfg["model"]["fusion"], cfg["model"]["dropout"])

def forward(model, choice, batch, max_length, device):
    graph = batch["graph"].to(device)
    if choice == "gnn": return model(graph)
    encoder = model.encoder if choice == "bert" else model.text_encoder
    tokens = encoder.tokenize(batch["text"], max_length, device)
    return model(tokens) if choice == "bert" else model(graph, tokens)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config.yaml"); parser.add_argument("--train-split", required=True); parser.add_argument("--val-split", required=True); parser.add_argument("--model", choices=["bert", "gnn", "fusion"])
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text()); choice = args.model or cfg["training"]["model"]
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else "cpu")
    model = build_model(cfg, choice).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]); loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(MusicDataset(args.train_split), batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(MusicDataset(args.val_split), batch_size=cfg["training"]["batch_size"], collate_fn=collate)
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"]); checkpoint_dir.mkdir(parents=True, exist_ok=True); best = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train(); train_losses = []
        for batch in train_loader:
            optimiser.zero_grad(); logits = forward(model, choice, batch, cfg["data"]["max_text_length"], device); loss = loss_fn(logits, batch["labels"].to(device)); loss.backward(); optimiser.step(); train_losses.append(loss.item())
        model.eval(); val_losses = []
        with torch.no_grad():
            for batch in val_loader: val_losses.append(loss_fn(forward(model, choice, batch, cfg["data"]["max_text_length"], device), batch["labels"].to(device)).item())
        val_loss = float(np.mean(val_losses)); print(f"epoch={epoch} train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f}")
        if val_loss < best: best = val_loss; torch.save({"state_dict": model.state_dict(), "config": cfg, "model": choice}, checkpoint_dir / "best.pt")

if __name__ == "__main__": main()
