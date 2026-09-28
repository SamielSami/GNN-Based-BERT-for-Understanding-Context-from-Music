"""HuggingFace text encoder and multi-label text baseline."""
from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class BertTextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        if freeze:
            for parameter in self.model.parameters(): parameter.requires_grad = False

    def tokenize(self, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {key: value.to(device) for key, value in tokens.items()}

    def forward(self, tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model(**tokens).last_hidden_state
        return hidden[:, 0], hidden

class BertTagClassifier(nn.Module):
    def __init__(self, encoder: BertTextEncoder, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.hidden_size, num_labels))

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        cls, _ = self.encoder(tokens)
        return self.head(cls)
