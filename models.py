from abc import ABC, abstractmethod

import torch
from torch import Tensor, softmax, tanh
from torch.nn import Linear, Module, Sequential, Sigmoid, Flatten, Tanh


def make_tanh_classifier_head(d_model: int) -> Sequential:
    """Binary classifier head: Linear -> Tanh -> Linear -> Sigmoid.

    Returns a `Sequential` of length 4 where index 2 is the final Linear layer.
    """
    return Sequential(
        Linear(d_model, d_model),
        Tanh(),
        Linear(d_model, 1),
        Sigmoid(),
    )


class SimpleTemporalPoolingClassifier(Module):
    def __init__(self, sequence_length: int, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        # Project per‑timestep features (n_features) to d_model
        self.proj = Linear(n_features, d_model)
        self.classifier = Sequential(Linear(sequence_length, 1), Sigmoid())

    def forward(self, x_in: Tensor) -> Tensor:
        # x_in: (N, T, F)
        h = tanh(self.proj(x_in))  # (N, T, d)
        pooled = h.mean(dim=-1)  # (N, T)
        return self.classifier(pooled)  # (N, 1)


class AttentionPoolingClassifier(Module):
    def __init__(self, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.proj = Linear(n_features, d_model)
        self.scorer = Linear(d_model, 1)
        self.classifier = make_tanh_classifier_head(d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        # x_in: (N, T, F)
        hidden = tanh(self.proj(x_in))  # (N, T, d)
        scores = self.scorer(hidden).squeeze(-1)  # (N, T)
        attn = softmax(scores, dim=1)  # (N, T)
        pooled = (attn.unsqueeze(-1) * hidden).sum(dim=1)  # (N, d)
        return self.classifier(pooled)  # (N, 1)


class SimpleSelfAttentionClassifier(Module):
    def __init__(self, n_features: int = 2, d_model: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = Linear(n_features, d_model)
        self.classifier = make_tanh_classifier_head(d_model)

    def forward(self, x_in: Tensor) -> Tensor:
        # x_in: (N, T, F)
        hidden = torch.tanh(self.proj(x_in))  # (N, T, d)
        similarity = torch.matmul(hidden, hidden.transpose(1, 2)) / (self.d_model ** 0.5)  # (N, T, T)
        attn = torch.softmax(similarity, dim=2)  # (N, T, T)
        hidden_c = attn @ hidden  # (N, T, d)
        pooled = hidden_c.mean(dim=1)  # (N, d)
        return self.classifier(pooled)


def build_model(sequence_length: int, n_features: int) -> Module:
    return SimpleTemporalPoolingClassifier(sequence_length=sequence_length, n_features=n_features, d_model=16)


def build_logreg(sequence_length: int, n_features: int) -> Module:
    # Flatten per-timestep features and run a logistic regression
    return Sequential(Flatten(start_dim=1), Linear(sequence_length * n_features, 1), Sigmoid())


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
    def __init__(self, sequence_length: int, n_features: int = 2, *, epochs: int = 1000, lr: float = 1.0) -> None:
        super().__init__(
            name="logreg",
            backbone=build_logreg(sequence_length, n_features),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone[1]  # type: ignore[index]


class TemporalAccess(ModelAccess):
    def __init__(self, sequence_length: int, n_features: int = 2, *, epochs: int = 1000, lr: float = 1.0) -> None:
        super().__init__(
            name="temporal",
            backbone=build_model(sequence_length, n_features),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[0]  # type: ignore[attr-defined,index]


class AttentionAccess(ModelAccess):
    def __init__(self, n_features: int = 2, *, epochs: int = 1000, lr: float = 0.5, d_model: int = 16) -> None:
        super().__init__(
            name="attention",
            backbone=AttentionPoolingClassifier(n_features=n_features, d_model=d_model),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[2]  # type: ignore[attr-defined,index]


class SelfAttentionAccess(ModelAccess):
    def __init__(self, n_features: int = 2, *, epochs: int = 1000, lr: float = 0.2, d_model: int = 16) -> None:
        super().__init__(
            name="self_attention",
            backbone=SimpleSelfAttentionClassifier(n_features=n_features, d_model=d_model),
            epochs=epochs,
            lr=lr,
        )

    def final_linear(self) -> Linear:
        return self.backbone.classifier[2]  # type: ignore[attr-defined,index]
