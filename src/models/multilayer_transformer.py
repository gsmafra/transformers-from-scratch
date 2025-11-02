from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, LayerNorm

from .base import ModelAccess
from .metrics import summarize_stats
from .mha import MultiHeadSelfAttention


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


class MultilayerTransformerClassifier(Module):
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
        # Multi-head attention layers + position-wise MLPs
        self.mha1 = MultiHeadSelfAttention(d_model=d_model, n_heads=self.n_heads)
        self.post1 = Linear(d_model, d_model)
        self.mha2 = MultiHeadSelfAttention(d_model=d_model, n_heads=self.n_heads)
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
        c1 = self.mha1(x1n)
        h1 = F.gelu(self.post1(c1))
        x1_out = x1 + h1

        x2 = x1_out
        x2n = self.ln2(x2)
        c2 = self.mha2(x2n)
        h2 = F.gelu(self.post2(c2))
        x2_out = x2 + h2

        cls_repr = x2_out[:, 0, :]
        return self.classifier(cls_repr)


class MultilayerTransformerAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.004
    LR_END = 0.002
    N_HEADS = 2

    def __init__(self, n_features: int) -> None:
        super().__init__(
            name="multilayer_transformer",
            backbone=MultilayerTransformerClassifier(n_features=n_features, d_model=self.D_MODEL, n_heads=self.N_HEADS),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        # Summarize the most recent first-layer attention logits captured during forward
        logits = self.backbone.mha1.last_logits if self.backbone.mha1.last_logits is not None else torch.empty(0)
        return summarize_stats("attn_logits", logits)
