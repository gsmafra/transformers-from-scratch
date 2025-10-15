from torch.nn import Flatten, Linear, Module, Sequential, Tanh

from .base import ModelAccess


def build_mlp(sequence_length: int, n_features: int, hidden: int) -> Module:
    in_features = sequence_length * n_features
    # One hidden layer MLP over the flattened sequence
    return Sequential(
        Flatten(start_dim=1),
        Linear(in_features, hidden),
        Tanh(),
        Linear(hidden, 1),
    )


class MLPAccess(ModelAccess):
    HIDDEN = 16

    def __init__(self, sequence_length: int, n_features: int) -> None:
        super().__init__(
            name="mlp",
            backbone=build_mlp(sequence_length, n_features, hidden=self.HIDDEN),
        )

    def final_linear(self) -> Linear:
        # Sequential indices: 0=Flatten, 1=Linear(hidden), 2=Tanh, 3=Linear(1)
        return self.backbone[3]
