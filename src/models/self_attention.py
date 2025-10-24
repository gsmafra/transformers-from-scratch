from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess
from .metrics import summarize_stats


class SimpleSelfAttentionClassifier(Module):
    def __init__(self, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = Linear(n_features, d_model)
        self.post_attn = Linear(d_model, d_model)
        self.classifier = Linear(d_model, 1)

    def forward(self, x_in: Tensor) -> Tensor:
        hidden = torch.tanh(self.proj(x_in))  # (N, T, d)
        similarity = torch.matmul(hidden, hidden.transpose(1, 2)) / (self.d_model ** 0.5)  # (N, T, T)
        attn = torch.softmax(similarity, dim=2)  # (N, T, T)
        hidden_c = attn @ hidden  # (N, T, d)
        pooled = hidden_c.mean(dim=1)  # (N, d)
        pooled = torch.tanh(self.post_attn(pooled))
        return self.classifier(pooled)


class SelfAttentionAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.4
    LR_END = 0.2

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="self_attention",
            backbone=SimpleSelfAttentionClassifier(n_features=n_features, d_model=self.D_MODEL),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        hidden = torch.tanh(self.backbone.proj(x))
        logits = torch.matmul(hidden, hidden.transpose(1, 2)) / (self.backbone.d_model ** 0.5)
        return summarize_stats("attn_logits", logits)
