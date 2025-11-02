from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module


class MultiHeadSelfAttention(Module):
    """Multi-head self-attention module with internal Q/K/V projections.

    Exposes `last_logits` and `last_attn` from the most recent forward for instrumentation.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert n_heads >= 1 and (d_model % n_heads == 0), "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dk = self.d_model // self.n_heads

        self.q = Linear(d_model, d_model)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)

        # Exposed intermediates for hooks/metrics (populated on forward)
        self.last_logits: Optional[Tensor] = None  # (N,H,T,T)
        self.last_attn: Optional[Tensor] = None    # (N,H,T,T)

    def _split_heads(self, x: Tensor) -> Tensor:
        N, T, C = x.shape
        return x.view(N, T, self.n_heads, self.dk).transpose(1, 2)  # (N,H,T,dk)

    def _combine_heads(self, x: Tensor) -> Tensor:
        # x: (N,H,T,dk)
        N, H, T, dk = x.shape
        return x.transpose(1, 2).contiguous().view(N, T, H * dk)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N,T,d_model)
        q = self._split_heads(self.q(x))  # (N,H,T,dk)
        k = self._split_heads(self.k(x))  # (N,H,T,dk)
        v = self._split_heads(self.v(x))  # (N,H,T,dk)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5)  # (N,H,T,T)
        attn = torch.softmax(logits, dim=-1)
        context = torch.matmul(attn, v)  # (N,H,T,dk)
        out = self._combine_heads(context)  # (N,T,d_model)

        # Save for instrumentation
        self.last_logits = logits.detach()
        self.last_attn = attn.detach()
        return out

