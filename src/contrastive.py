"""Optional Task 4 graph-text contrastive objective."""
from __future__ import annotations
import torch
from torch import nn
from src.task4.model import contrastive_loss

class GraphTextContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, graph_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        return contrastive_loss(text_embeddings, graph_embeddings, self.temperature)
