from typing import Optional

from torch.nn import Flatten, Linear, Module, Sequential

from .base import ModelAccess


def build_logreg(sequence_length: int, n_features: int) -> Module:
    # Output raw logits (no Sigmoid) for numerical stability with BCEWithLogitsLoss
    return Sequential(Flatten(start_dim=1), Linear(sequence_length * n_features, 1))


class LogRegAccess(ModelAccess):
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
            name="logreg",
            backbone=build_logreg(sequence_length, n_features),
            epochs=epochs,
            lr_start=lr_start,
            lr_end=lr_end,
            mini_batch_size=mini_batch_size,
        )

    def final_linear(self) -> Linear:
        return self.backbone[1]
