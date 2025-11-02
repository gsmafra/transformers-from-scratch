from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, Module, LayerNorm

from .base import ModelAccess
from .metrics import summarize_stats


def sinusoidal_positional_encoding(T: int, d_model: int) -> Tensor:
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    denom = torch.pow(10000.0, (2 * (i // 2)) / float(d_model))
    angles = pos / denom
    pe = torch.empty(T, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class TwoLayerSelfAttentionQKVPosClassifier(Module):
    def __init__(self, n_features: int = 12, d_model: int = 16, pe_scale: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.pe_scale = float(pe_scale)
        # Input projection to model dimension
        self.in_proj = Linear(n_features, d_model)
        # Pre-norm layers
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        # Layer 1 projections and position-wise MLP
        self.q1 = Linear(d_model, d_model)
        self.k1 = Linear(d_model, d_model)
        self.v1 = Linear(d_model, d_model)
        self.post1 = Linear(d_model, d_model)
        # Layer 2 projections and position-wise MLP
        self.q2 = Linear(d_model, d_model)
        self.k2 = Linear(d_model, d_model)
        self.v2 = Linear(d_model, d_model)
        self.post2 = Linear(d_model, d_model)
        # Classifier
        self.classifier = Linear(d_model, 1)

    def forward(self, x_in: Tensor) -> Tensor:
        N, T, _ = x_in.shape
        # Project input to model space, then add PE and pre-norm
        z0 = self.in_proj(x_in)
        pe1 = sinusoidal_positional_encoding(T, self.d_model).unsqueeze(0)
        x1 = z0 + self.pe_scale * pe1
        x1n = self.ln1(x1)
        q1 = self.q1(x1n)
        k1 = self.k1(x1n)
        v1 = self.v1(x1n)
        s1 = torch.matmul(q1, k1.transpose(1, 2)) / (self.d_model ** 0.5)
        a1 = torch.softmax(s1, dim=2)
        c1 = a1 @ v1
        h1 = torch.tanh(self.post1(c1))
        x1_out = x1 + h1

        x2 = x1_out
        x2n = self.ln2(x2)
        q2 = self.q2(x2n)
        k2 = self.k2(x2n)
        v2 = self.v2(x2n)
        s2 = torch.matmul(q2, k2.transpose(1, 2)) / (self.d_model ** 0.5)
        a2 = torch.softmax(s2, dim=2)
        c2 = a2 @ v2
        h2 = torch.tanh(self.post2(c2))
        x2_out = x2 + h2

        pooled = x2_out.mean(dim=1)
        return self.classifier(pooled)


class MultilayerAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.004
    LR_END = 0.002

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="multilayer",
            backbone=TwoLayerSelfAttentionQKVPosClassifier(n_features=n_features, d_model=self.D_MODEL),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        # Report first-layer attention logits on current input with PE1
        N, T, _ = x.shape
        z0 = self.backbone.in_proj(x)
        pe1 = sinusoidal_positional_encoding(T, self.backbone.d_model).unsqueeze(0)
        x1 = z0 + self.backbone.pe_scale * pe1
        x1n = self.backbone.ln1(x1)
        q1 = self.backbone.q1(x1n)
        k1 = self.backbone.k1(x1n)
        logits = torch.matmul(q1, k1.transpose(1, 2)) / (self.backbone.d_model ** 0.5)
        return summarize_stats("attn_logits", logits)
