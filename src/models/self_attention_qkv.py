from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess, make_tanh_classifier_head
from .metrics import summarize_stats


class SelfAttentionQKVClassifier(Module):
    def __init__(self, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_proj = Linear(n_features, d_model)
        self.k_proj = Linear(n_features, d_model)
        self.v_proj = Linear(n_features, d_model)
        self.classifier = make_tanh_classifier_head(d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        q = torch.tanh(self.q_proj(x_in))  # (N, T, d)
        k = torch.tanh(self.k_proj(x_in))  # (N, T, d)
        v = torch.tanh(self.v_proj(x_in))  # (N, T, d)
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.d_model ** 0.5)  # (N, T, T)
        attn = torch.softmax(scores, dim=2)  # (N, T, T)
        context = attn @ v  # (N, T, d)
        pooled = context.mean(dim=1)  # (N, d)
        return self.classifier(pooled)


class SelfAttentionQKVAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.4
    LR_END = 0.2

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="self_attention_qkv",
            backbone=SelfAttentionQKVClassifier(n_features=n_features, d_model=self.D_MODEL),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[2]

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        q = torch.tanh(self.backbone.q_proj(x))
        k = torch.tanh(self.backbone.k_proj(x))
        logits = torch.matmul(q, k.transpose(1, 2)) / (self.backbone.d_model ** 0.5)
        return summarize_stats("attn_logits", logits)
