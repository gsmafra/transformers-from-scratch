from typing import Dict, Optional

import torch
from torch import Tensor, softmax, tanh
from torch.nn import Linear, Module

from .base import ModelAccess, make_tanh_classifier_head
from .metrics import summarize_stats


class AttentionPoolingClassifier(Module):
    def __init__(self, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.proj = Linear(n_features, d_model)
        self.scorer = Linear(d_model, 1)
        # Tanh MLP head ending in Linear; output logits (no Sigmoid)
        self.classifier = make_tanh_classifier_head(d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        hidden = tanh(self.proj(x_in))  # (N, T, d)
        # Ablation: use mean over features as score (no learnable scorer)
        scores = torch.mean(hidden, dim=2)  # (N, T)
        attn = softmax(scores, dim=1)  # (N, T)
        pooled = (attn.unsqueeze(-1) * hidden).sum(dim=1)  # (N, d)
        return self.classifier(pooled)  # (N, 1)


class AttentionAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.5
    LR_END = 0.1

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="attention",
            backbone=AttentionPoolingClassifier(n_features=n_features, d_model=self.D_MODEL),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        # final linear is the last layer of the head Sequential
        return self.backbone.classifier[2]

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        hidden = tanh(self.backbone.proj(x))
        scores = torch.mean(hidden, dim=2)
        return summarize_stats("attn_logits", scores)
