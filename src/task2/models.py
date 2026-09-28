"""Task 2 GraphSAGE, pooled-feature MLP, and mel-CNN classifiers."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool


def _batch_vector(graph) -> torch.Tensor:
    batch = getattr(graph, "batch", None)
    if batch is None:
        batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
    return batch


class GraphSAGEEncoder(nn.Module):
    """Inductive segment-graph encoder with mean/max graph readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or layers < 1:
            raise ValueError("input_dim, hidden_dim, and layers must be positive")
        dimensions = [input_dim] + [hidden_dim] * layers
        self.convs = nn.ModuleList(
            SAGEConv(dimensions[index], dimensions[index + 1]) for index in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim

    def forward(self, graph) -> torch.Tensor:
        x = graph.x.float()
        for convolution, normalization in zip(self.convs, self.norms):
            x = convolution(x, graph.edge_index)
            x = self.dropout(torch.relu(normalization(x)))
        batch = _batch_vector(graph)
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)


class GNNClassifier(nn.Module):
    def __init__(self, encoder: GraphSAGEEncoder, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(encoder.output_dim, num_labels)
        )

    def forward(self, graph) -> torch.Tensor:
        return self.head(self.encoder(graph))


class PooledMLPClassifier(nn.Module):
    """No-message-passing control using mean/max pooled node features."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, graph) -> torch.Tensor:
        batch = _batch_vector(graph)
        pooled = torch.cat(
            [global_mean_pool(graph.x.float(), batch), global_max_pool(graph.x.float(), batch)],
            dim=-1,
        )
        return self.network(pooled)


class MelCNNClassifier(nn.Module):
    """Compact 2-D CNN baseline over clip-level log-mel images."""

    def __init__(self, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        channels = (1, 16, 32, 64)
        blocks = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_labels),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 4 or mel.shape[1] != 1:
            raise ValueError("mel input must have shape [batch, 1, mel_bins, frames]")
        return self.head(self.features(mel.float()))


def build_task2_model(
    name: str,
    *,
    input_dim: int,
    num_labels: int,
    hidden_dim: int = 128,
    layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    if name == "gnn":
        return GNNClassifier(
            GraphSAGEEncoder(input_dim, hidden_dim, layers, dropout), num_labels, dropout
        )
    if name == "mlp":
        return PooledMLPClassifier(input_dim, hidden_dim, num_labels, dropout)
    if name == "cnn":
        return MelCNNClassifier(num_labels, dropout)
    raise ValueError("model name must be one of: gnn, mlp, cnn")
