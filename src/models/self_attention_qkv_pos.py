from typing import Dict

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import ModelAccess, make_tanh_classifier_head
from .metrics import summarize_stats


def sinusoidal_positional_encoding(T: int, d_model: int, device=None) -> Tensor:
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)  # (T, 1)
    i = torch.arange(d_model, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d)
    denom = torch.pow(10000.0, (2 * (i // 2)) / float(d_model))  # (1, d)
    angles = pos / denom  # (T, d)
    pe = torch.empty(T, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class SelfAttentionQKVPosClassifier(Module):
    def __init__(
        self,
        n_features: int = 2,
        d_model: int = 16,
        pe_scale: float = 0.1,
        *,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.pe_scale = float(pe_scale)
        self.use_cls_token = bool(use_cls_token)
        self.q_proj = Linear(n_features, d_model)
        self.k_proj = Linear(n_features, d_model)
        self.v_proj = Linear(n_features, d_model)
        # Learnable [CLS] token in input feature space (same dimensionality as x_in features)
        if self.use_cls_token:
            self.cls_token = torch.nn.Parameter(torch.zeros(n_features))
        self.classifier = make_tanh_classifier_head(d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        N, T, _ = x_in.shape
        # Optionally prepend a learned [CLS] token in input feature space
        if self.use_cls_token:
            cls = self.cls_token.view(1, 1, -1).expand(N, 1, -1)  # (N, 1, F)
            x_work = torch.cat([cls, x_in], dim=1)  # (N, T+1, F)
            T_eff = T + 1
        else:
            x_work = x_in
            T_eff = T

        # Add positional encoding in input space before projections
        pe_in = sinusoidal_positional_encoding(T_eff, self.n_features, device=x_in.device).unsqueeze(0)  # (1, T_eff, F)
        x_plus = x_work + self.pe_scale * pe_in

        q = self.q_proj(x_plus)  # (N, T_eff, d)
        k = self.k_proj(x_plus)  # (N, T_eff, d)
        v = self.v_proj(x_plus)  # (N, T_eff, d)
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.d_model ** 0.5)  # (N, T_eff, T_eff)
        attn = torch.softmax(scores, dim=2)  # (N, T_eff, T_eff)
        context = attn @ v  # (N, T_eff, d)

        if self.use_cls_token:
            pooled = context[:, 0, :]  # (N, d) — [CLS] representation
        else:
            pooled = context.mean(dim=1)  # (N, d)

        return self.classifier(pooled)


class SelfAttentionQKVPosAccess(ModelAccess):
    D_MODEL = 16
    LR_START = 0.04
    LR_END = 0.02

    def __init__(self, n_features: int, *, use_cls_token: bool = False) -> None:
        super().__init__(
            name="self_attention_qkv_pos",
            backbone=SelfAttentionQKVPosClassifier(
                n_features=n_features, d_model=self.D_MODEL, use_cls_token=use_cls_token
            ),
            lr_start=self.LR_START,
            lr_end=self.LR_END,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[2]

    def extra_metrics(self, x: Tensor) -> Dict[str, float]:
        N, T, _ = x.shape
        if self.backbone.use_cls_token:
            cls = self.backbone.cls_token.view(1, 1, -1).expand(N, 1, -1)
            x_work = torch.cat([cls, x], dim=1)
            T_eff = T + 1
        else:
            x_work = x
            T_eff = T
        pe_in = sinusoidal_positional_encoding(T_eff, self.backbone.n_features, device=x.device).unsqueeze(0)
        x_plus = x_work + self.backbone.pe_scale * pe_in
        q = self.backbone.q_proj(x_plus)
        k = self.backbone.k_proj(x_plus)
        logits = torch.matmul(q, k.transpose(1, 2)) / (self.backbone.d_model ** 0.5)
        return summarize_stats("attn_logits", logits)
