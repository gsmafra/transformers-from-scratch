from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess
from .metrics import summarize_stats


def sinusoidal_positional_encoding(T: int, d_model: int, device=None) -> Tensor:
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(d_model, dtype=torch.float32, device=device).unsqueeze(0)
    denom = torch.pow(10000.0, (2 * (i // 2)) / float(d_model))
    angles = pos / denom
    pe = torch.empty(T, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class TwoLayerSelfAttentionQKVPosClassifier(Module):
    def __init__(self, n_features: int = 12, d_model: int = 16, pe_scale: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.pe_scale = float(pe_scale)
        # Layer 1 projections and position-wise MLP
        self.q1 = Linear(n_features, d_model)
        self.k1 = Linear(n_features, d_model)
        self.v1 = Linear(n_features, d_model)
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
        # Layer 1: add PE in input feature space
        pe1 = sinusoidal_positional_encoding(T, self.n_features, device=x_in.device).unsqueeze(0)
        x1 = x_in + self.pe_scale * pe1
        q1 = self.q1(x1)
        k1 = self.k1(x1)
        v1 = self.v1(x1)
        s1 = torch.matmul(q1, k1.transpose(1, 2)) / (self.d_model ** 0.5)
        a1 = torch.softmax(s1, dim=2)
        c1 = a1 @ v1
        h1 = torch.tanh(self.post1(c1))

        # Layer 2: add PE in model space
        pe2 = sinusoidal_positional_encoding(T, self.d_model, device=x_in.device).unsqueeze(0)
        x2 = h1 + self.pe_scale * pe2
        q2 = self.q2(x2)
        k2 = self.k2(x2)
        v2 = self.v2(x2)
        s2 = torch.matmul(q2, k2.transpose(1, 2)) / (self.d_model ** 0.5)
        a2 = torch.softmax(s2, dim=2)
        c2 = a2 @ v2
        h2 = torch.tanh(self.post2(c2))

        pooled = h2.mean(dim=1)
        return self.classifier(pooled)


class MultilayerAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.04
    LR_END = 0.02

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
        pe1 = sinusoidal_positional_encoding(T, self.backbone.n_features, device=x.device).unsqueeze(0)
        x1 = x + self.backbone.pe_scale * pe1
        q1 = self.backbone.q1(x1)
        k1 = self.backbone.k1(x1)
        logits = torch.matmul(q1, k1.transpose(1, 2)) / (self.backbone.d_model ** 0.5)
        return summarize_stats("attn_logits", logits)
