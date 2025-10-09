from typing import Optional
from torch.nn import Linear, Module, Sequential, Sigmoid, Flatten

from .base import ModelAccess


def build_logreg(sequence_length: int, n_features: int) -> Module:
    return Sequential(Flatten(start_dim=1), Linear(sequence_length * n_features, 1), Sigmoid())


class LogRegAccess(ModelAccess):
    def __init__(
        self,
        sequence_length: int,
        n_features: int = 2,
        *,
        epochs: int = 1000,
        lr_start: Optional[float] = 1.0,
        lr_end: Optional[float] = None,
    ) -> None:
        super().__init__(
            name="logreg",
            backbone=build_logreg(sequence_length, n_features),
            epochs=epochs,
            lr_start=lr_start,
            lr_end=lr_end,
        )

    def final_linear(self) -> Linear:  # type: ignore[override]
        return self.backbone[1]  # type: ignore[index]
