from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess
from .metrics import summarize_stats
from .mha import MultiHeadSelfAttention


def sinusoidal_positional_encoding(T: int, d_model: int) -> Tensor:
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)  # (1, d)
    denom = torch.pow(10000.0, (2 * (i // 2)) / float(d_model))  # (1, d)
    angles = pos / denom  # (T, d)
    pe = torch.empty(T, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class SingleLayerTransformerClassifier(Module):
    def __init__(
        self,
        n_features: int = 2,
        d_model: int = 16,
        pe_scale: float = 0.1,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.pe_scale = float(pe_scale)
        self.n_heads = int(n_heads)
        # Project inputs from feature space to model space
        self.in_proj = Linear(n_features, d_model)
        # Learnable [CLS] token in input feature space (same dimensionality as x_in features)
        self.cls_token = torch.nn.Parameter(torch.zeros(n_features))
        # Multi-head self-attention in model space
        self.mha = MultiHeadSelfAttention(d_model=d_model, n_heads=self.n_heads)
        self.post_attn = Linear(d_model, d_model)
        self.classifier = Linear(d_model, 1)

    def forward(self, x_in: Tensor) -> Tensor:
        N, T, _ = x_in.shape
        # Always prepend a learned [CLS] token in input feature space
        cls = self.cls_token.view(1, 1, -1).expand(N, 1, -1)  # (N, 1, F)
        x_work = torch.cat([cls, x_in], dim=1)  # (N, T+1, F)
        T_eff = T + 1

        # Add positional encoding in input space, then project to model space and apply MHA
        pe_in = sinusoidal_positional_encoding(T_eff, self.n_features).unsqueeze(0)  # (1, T_eff, F)
        x_plus = x_work + self.pe_scale * pe_in
        z = self.in_proj(x_plus)  # (N, T_eff, d)
        context = self.mha(z)  # (N, T_eff, d)
        context = torch.tanh(self.post_attn(context))  # (N, T_eff, d)

        pooled = context[:, 0, :]  # (N, d)
        return self.classifier(pooled)


class SingleLayerTransformerAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.04
    LR_END = 0.02

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="singlelayer_transformer",
            backbone=SingleLayerTransformerClassifier(n_features=n_features, d_model=self.D_MODEL),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        # Summarize most recent attention logits captured during forward
        logits = self.backbone.mha.last_logits if self.backbone.mha.last_logits is not None else torch.empty(0)
        return summarize_stats("attn_logits", logits)
