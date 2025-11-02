import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, LayerNorm

from .mha import MultiHeadSelfAttention


class TransformerBlock(Module):
    """Pre-norm Transformer block: MHA + FFN with residuals.

    - LayerNorm before MHA and FFN
    - FFN expansion ratio configurable (default 4)
    """

    def __init__(self, d_model: int, n_heads: int, *, expansion: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.expansion = int(expansion)

        # Attention sublayer
        self.ln_attn = LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)

        # Feed-forward sublayer
        self.ln_ff = LayerNorm(d_model)
        hidden = d_model * self.expansion
        self.fc1 = Linear(d_model, hidden)
        self.fc2 = Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # MHA with pre-norm and residual
        a = self.ln_attn(x)
        x = x + self.mha(a)
        # FFN with pre-norm and residual
        f = self.ln_ff(x)
        x = x + self.fc2(F.gelu(self.fc1(f)))
        return x
