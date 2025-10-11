from typing import Optional

import torch
from torch import Tensor, softmax, tanh
from torch.nn import Linear, Module

from .base import ModelAccess, make_tanh_classifier_head


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
    def __init__(
        self,
        n_features: int = 2,
        *,
        epochs: int = 1000,
        d_model: int = 16,
        lr_start: Optional[float] = 0.5,
        lr_end: Optional[float] = 0.1,
    ) -> None:
        super().__init__(
            name="attention",
            backbone=AttentionPoolingClassifier(n_features=n_features, d_model=d_model),
            epochs=epochs,
            lr_start=lr_start,
            lr_end=lr_end,
        )

    def final_linear(self) -> Linear:  # type: ignore[override]
        # final linear is the last layer of the head Sequential
        return self.backbone.classifier[2]  # type: ignore[attr-defined,index]
