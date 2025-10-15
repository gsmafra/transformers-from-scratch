from typing import Optional

from torch import Tensor, tanh
from torch.nn import Linear, Module

from .base import ModelAccess


class SimpleTemporalPoolingClassifier(Module):
    def __init__(self, sequence_length: int, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.proj = Linear(n_features, d_model)
        # Output raw logits (no Sigmoid)
        self.classifier = Linear(sequence_length, 1)

    def forward(self, x_in: Tensor) -> Tensor:
        h = tanh(self.proj(x_in))  # (N, T, d)
        pooled = h.mean(dim=-1)  # (N, T)
        return self.classifier(pooled)  # (N, 1)


def build_model(sequence_length: int, n_features: int) -> Module:
    return SimpleTemporalPoolingClassifier(sequence_length=sequence_length, n_features=n_features, d_model=16)


class TemporalAccess(ModelAccess):
    def __init__(
        self,
        sequence_length: int,
        n_features: int = 2,
        *,
        epochs: int = 1000,
        lr_start: Optional[float] = 1.0,
        lr_end: Optional[float] = None,
        mini_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="temporal",
            backbone=build_model(sequence_length, n_features),
            epochs=epochs,
            lr_start=lr_start,
            lr_end=lr_end,
            mini_batch_size=mini_batch_size,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier
