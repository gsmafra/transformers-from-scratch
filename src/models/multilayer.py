from typing import Dict

import torch
import torch.nn.functional as F
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


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    """Reshape (N,T,C) -> (N,H,T,dk) with dk=C//H, then swap to heads-first."""
    N, T, C = x.shape
    dk = C // n_heads
    return x.view(N, T, n_heads, dk).transpose(1, 2)


def _combine_heads(x: Tensor) -> Tensor:
    """Reshape (N,H,T,dk) -> (N,T,C) combining heads back to channels."""
    N, H, T, dk = x.shape
    return x.transpose(1, 2).contiguous().view(N, T, H * dk)


class TransformerClassifier(Module):
    def __init__(self, n_features: int = 12, d_model: int = 16, pe_scale: float = 0.1, n_heads: int = 1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.pe_scale = float(pe_scale)
        self.n_heads = int(n_heads)
        assert self.n_heads >= 1 and (self.d_model % self.n_heads == 0), "d_model must be divisible by n_heads"
        # Input projection to model dimension
        self.in_proj = Linear(n_features, d_model)
        # Learnable CLS token in model space
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
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
        # Project input to model space, prepend CLS, then add PE and pre-norm
        z0 = self.in_proj(x_in)
        cls = self.cls_token.expand(N, -1, -1)
        z0_with_cls = torch.cat([cls, z0], dim=1)
        pe1 = sinusoidal_positional_encoding(T + 1, self.d_model).unsqueeze(0)
        x1 = z0_with_cls + self.pe_scale * pe1
        x1n = self.ln1(x1)
        H = self.n_heads
        dk = self.d_model // H
        q1 = _split_heads(self.q1(x1n), H)
        k1 = _split_heads(self.k1(x1n), H)
        v1 = _split_heads(self.v1(x1n), H)
        s1 = torch.matmul(q1, k1.transpose(-2, -1)) / (dk ** 0.5)
        a1 = torch.softmax(s1, dim=-1)
        c1 = torch.matmul(a1, v1)
        c1 = _combine_heads(c1)
        h1 = F.gelu(self.post1(c1))
        x1_out = x1 + h1

        x2 = x1_out
        x2n = self.ln2(x2)
        q2 = _split_heads(self.q2(x2n), H)
        k2 = _split_heads(self.k2(x2n), H)
        v2 = _split_heads(self.v2(x2n), H)
        s2 = torch.matmul(q2, k2.transpose(-2, -1)) / (dk ** 0.5)
        a2 = torch.softmax(s2, dim=-1)
        c2 = torch.matmul(a2, v2)
        c2 = _combine_heads(c2)
        h2 = F.gelu(self.post2(c2))
        x2_out = x2 + h2

        cls_repr = x2_out[:, 0, :]
        return self.classifier(cls_repr)


class MultilayerAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.004
    LR_END = 0.002
    N_HEADS = 2

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="multilayer",
            backbone=TransformerClassifier(n_features=n_features, d_model=self.D_MODEL, n_heads=self.N_HEADS),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        # Report first-layer attention logits on current input with PE1
        N, T, _ = x.shape
        z0 = self.backbone.in_proj(x)
        cls = self.backbone.cls_token.expand(N, -1, -1)
        z0_with_cls = torch.cat([cls, z0], dim=1)
        pe1 = sinusoidal_positional_encoding(T + 1, self.backbone.d_model).unsqueeze(0)
        x1 = z0_with_cls + self.backbone.pe_scale * pe1
        x1n = self.backbone.ln1(x1)
        H = self.backbone.n_heads
        dk = self.backbone.d_model // H
        q1 = _split_heads(self.backbone.q1(x1n), H)
        k1 = _split_heads(self.backbone.k1(x1n), H)
        logits = torch.matmul(q1, k1.transpose(-2, -1)) / (dk ** 0.5)
        return summarize_stats("attn_logits", logits)
