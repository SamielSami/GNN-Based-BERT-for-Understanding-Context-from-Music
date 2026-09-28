import math
import torch
from torch import nn
from torch.nn import functional as F


class RetrievalModel(nn.Module):
    def __init__(self, text_dim, graph_dim, projection_dim=64):
        super().__init__()
        if min(text_dim, graph_dim, projection_dim) < 1:
            raise ValueError("Embedding dimensions must be positive")
        self.text = nn.Linear(text_dim, projection_dim)
        self.graph = nn.Linear(graph_dim, projection_dim)

    def forward(self, text, graph):
        return F.normalize(self.text(text), dim=-1), F.normalize(self.graph(graph), dim=-1)


def contrastive_loss(text, graph, temperature=0.07):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if text.ndim != 2 or text.shape != graph.shape or len(text) < 2:
        raise ValueError("InfoNCE needs at least two aligned pairs with matching dimensions")
    logits = F.normalize(text, dim=-1) @ F.normalize(graph, dim=-1).T / temperature
    targets = torch.arange(len(text), device=text.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
