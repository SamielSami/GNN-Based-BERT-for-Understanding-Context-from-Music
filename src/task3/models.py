"""Fusion heads over frozen graph embeddings and BERT token embeddings."""
import torch
from torch import nn

MODES = ("bert", "gnn", "concat", "gated", "cross_attention")


class FusionClassifier(nn.Module):
    def __init__(self, text_dim, graph_dim, num_labels, mode="concat", hidden_dim=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.text_projection = nn.Linear(text_dim, hidden_dim) if mode != "gnn" else None
        self.graph_projection = nn.Linear(graph_dim, hidden_dim) if mode != "bert" else None
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim) if mode == "gated" else None
        self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True) if mode == "cross_attention" else None
        width = 2 * hidden_dim if mode == "concat" else hidden_dim
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, num_labels))

    def forward(self, tokens, mask, graph, return_embedding=False):
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or graph.ndim != 2 or len(graph) != len(tokens):
            raise ValueError("Expected tokens [B,T,D], mask [B,T], graph [B,G]")
        mask = mask.bool()
        if not mask.any(dim=1).all() or not mask[:, 0].all():
            raise ValueError("Each caption must include a valid first/CLS token")
        # Remove padded values before projection, including arbitrary cache padding.
        # Only cross-attention consumes non-CLS tokens. Avoid projecting the full
        # sequence for the four pooled controls; this preserves their outputs.
        text_input = tokens if self.mode == "cross_attention" else tokens[:, :1]
        text_mask = mask if self.mode == "cross_attention" else mask[:, :1]
        text = self.text_projection(text_input.masked_fill(~text_mask.unsqueeze(-1), 0)) if self.text_projection else None
        t = text[:, 0] if text is not None else None
        g = self.graph_projection(graph) if self.graph_projection else None
        if self.mode == "bert":
            embedding = t
        elif self.mode == "gnn":
            embedding = g
        elif self.mode == "concat":
            embedding = torch.cat([t, g], dim=-1)
        elif self.mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([t, g], dim=-1)))
            embedding = gate * t + (1 - gate) * g
        else:
            attended, _ = self.attention(g[:, None], text, text, key_padding_mask=~mask, need_weights=False)
            embedding = g + attended[:, 0]
        logits = self.head(embedding)
        return (logits, embedding) if return_embedding else logits
