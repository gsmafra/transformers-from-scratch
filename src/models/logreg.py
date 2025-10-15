from typing import Optional

from torch.nn import Flatten, Linear, Module, Sequential

from .base import ModelAccess


def build_logreg(sequence_length: int, n_features: int) -> Module:
    # Output raw logits (no Sigmoid) for numerical stability with BCEWithLogitsLoss
    return Sequential(Flatten(start_dim=1), Linear(sequence_length * n_features, 1))


class LogRegAccess(ModelAccess):
    LR_START = 1.0

    def __init__(self, sequence_length: int, n_features: int) -> None:
        super().__init__(
            name="logreg",
            backbone=build_logreg(sequence_length, n_features),
            lr_start=self.LR_START,
            lr_end=None,  # use parent's default to keep constant LR
        )

    def final_linear(self) -> Linear:
        return self.backbone[1]
