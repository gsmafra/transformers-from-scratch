from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess
from .metrics import summarize_stats
from .transformer_block import TransformerBlock


def sinusoidal_positional_encoding(T: int, d_model: int) -> Tensor:
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    denom = torch.pow(10000.0, (2 * (i // 2)) / float(d_model))
    angles = pos / denom
    pe = torch.empty(T, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


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
        self.block1 = TransformerBlock(d_model=d_model, n_heads=self.n_heads, expansion=4)
        self.block2 = TransformerBlock(d_model=d_model, n_heads=self.n_heads, expansion=4)
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
        x1_out = self.block1(x1)
        x2_out = self.block2(x1_out)

        cls_repr = x2_out[:, 0, :]
        return self.classifier(cls_repr)


class MultilayerTransformerAccess(ModelAccess):
    D_MODEL = 32
    LR_START = 0.004
    LR_END = 0.002
    N_HEADS = 4

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
        # Summarize the most recent first block attention logits captured during forward
        logits = (
            self.backbone.block1.mha.last_logits
            if getattr(self.backbone.block1.mha, "last_logits", None) is not None
            else torch.empty(0)
        )
        return summarize_stats("attn_logits", logits)
