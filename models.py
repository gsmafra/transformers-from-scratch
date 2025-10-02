from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, softmax, tanh
from torch.nn import Linear, Module, Sequential, Sigmoid


class SimpleTemporalPoolingClassifier(Module):
    def __init__(self, sequence_length: int, d_model: int = 16) -> None:
        super().__init__()
        self.proj = Linear(1, d_model)
        self.classifier = Sequential(Linear(sequence_length, 1), Sigmoid())

    def forward(self, x_flat: Tensor) -> Tensor:
        # x_flat: (N, T)
        h = tanh(self.proj(x_flat.unsqueeze(-1)))  # (N, T, d)
        pooled = h.mean(dim=-1)  # (N, T)
        return self.classifier(pooled)  # (N, 1)


class AttentionPoolingClassifier(Module):
    def __init__(self, d_model: int = 16) -> None:
        super().__init__()
        self.proj = Linear(1, d_model)
        self.scorer = Linear(d_model, 1)
        self.classifier = Sequential(Linear(d_model, 1), Sigmoid())

    def forward(self, x_flat: Tensor) -> Tensor:
        # x_flat: (N, T)
        hidden = tanh(self.proj(x_flat.unsqueeze(-1)))  # (N, T, d)
        scores = self.scorer(hidden).squeeze(-1)  # (N, T)
        attn = softmax(scores, dim=1)  # (N, T)
        pooled = (attn.unsqueeze(-1) * hidden).sum(dim=1)  # (N, d)
        return self.classifier(pooled)  # (N, 1)


def build_model(sequence_length: int) -> Module:
    return SimpleTemporalPoolingClassifier(sequence_length=sequence_length, d_model=16)


def build_logreg(sequence_length: int) -> Module:
    return Sequential(Linear(sequence_length, 1), Sigmoid())


# --- Model accessors -------------------------------------------------------


class ModelAccess(ABC):
    name: str
    backbone: Module
    epochs: int
    lr: float

    def __init__(self, name: str, backbone: Module, *, epochs: int, lr: float) -> None:
        self.name = name
        self.backbone = backbone
        self.epochs = epochs
        self.lr = lr

    def forward(self, x_flat: Tensor) -> Tensor:
        return self.backbone(x_flat)

    @abstractmethod
    def final_linear(self) -> Linear:
        raise NotImplementedError


class LogRegAccess(ModelAccess):
    def __init__(self, sequence_length: int, *, epochs: int = 1000, lr: float = 1.0) -> None:
        super().__init__(
            name="logreg",
            backbone=build_logreg(sequence_length),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone[0]  # type: ignore[index]


class TemporalAccess(ModelAccess):
    def __init__(self, sequence_length: int, *, epochs: int = 1000, lr: float = 1.0) -> None:
        super().__init__(
            name="temporal",
            backbone=build_model(sequence_length),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[0]  # type: ignore[attr-defined,index]


class AttentionAccess(ModelAccess):
    def __init__(self, *, epochs: int = 1000, lr: float = 1.0, d_model: int = 16) -> None:
        super().__init__(
            name="attention",
            backbone=AttentionPoolingClassifier(d_model=d_model),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[0]  # type: ignore[attr-defined,index]

