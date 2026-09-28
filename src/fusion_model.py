"""Graph-text fusion models for music context prediction."""
from __future__ import annotations
import torch
from torch import nn
from .bert_encoder import BertTextEncoder
from .task2.models import GraphSAGEEncoder

class GNNBertFusion(nn.Module):
    def __init__(self, graph_encoder: GraphSAGEEncoder, text_encoder: BertTextEncoder, num_labels: int, fusion: str = "cross_attention", dropout: float = 0.2) -> None:
        super().__init__()
        self.graph_encoder, self.text_encoder, self.fusion = graph_encoder, text_encoder, fusion
        self.graph_projection = nn.Linear(graph_encoder.output_dim, text_encoder.hidden_size)
        self.attention = nn.MultiheadAttention(text_encoder.hidden_size, num_heads=8, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(text_encoder.hidden_size * 2, num_labels))

    def forward(self, graph, tokens):
        graph_embedding = self.graph_projection(self.graph_encoder(graph))
        text_cls, text_tokens = self.text_encoder(tokens)
        if self.fusion == "concat": fused = torch.cat((graph_embedding, text_cls), dim=-1)
        elif self.fusion == "cross_attention":
            attended, _ = self.attention(graph_embedding.unsqueeze(1), text_tokens, text_tokens, need_weights=False)
            fused = torch.cat((graph_embedding, attended.squeeze(1)), dim=-1)
        else: raise ValueError("fusion must be 'concat' or 'cross_attention'")
        return self.head(fused)
