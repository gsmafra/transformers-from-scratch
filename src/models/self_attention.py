from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess, make_tanh_classifier_head
from .metrics import summarize_stats


class SimpleSelfAttentionClassifier(Module):
    def __init__(self, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = Linear(n_features, d_model)
        # Tanh MLP head ending in Linear; output logits (no Sigmoid)
        self.classifier = make_tanh_classifier_head(d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        hidden = torch.tanh(self.proj(x_in))  # (N, T, d)
        similarity = torch.matmul(hidden, hidden.transpose(1, 2)) / (self.d_model ** 0.5)  # (N, T, T)
        attn = torch.softmax(similarity, dim=2)  # (N, T, T)
        hidden_c = attn @ hidden  # (N, T, d)
        pooled = hidden_c.mean(dim=1)  # (N, d)
        return self.classifier(pooled)


class SelfAttentionAccess(ModelAccess):
    def __init__(
        self,
        n_features: int = 2,
        *,
        epochs: int = 1000,
        d_model: int = 16,
        lr_start: Optional[float] = 0.4,
        lr_end: Optional[float] = 0.2,
    ) -> None:
        super().__init__(
            name="self_attention",
            backbone=SimpleSelfAttentionClassifier(n_features=n_features, d_model=d_model),
            epochs=epochs,
            lr_start=lr_start,
            lr_end=lr_end,
        )

    def final_linear(self) -> Linear:
        # final linear is the last layer of the head Sequential
        return self.backbone.classifier[2]

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        hidden = torch.tanh(self.backbone.proj(x))
        logits = torch.matmul(hidden, hidden.transpose(1, 2)) / (self.backbone.d_model ** 0.5)
        return summarize_stats("attn_logits", logits)
