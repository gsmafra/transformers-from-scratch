from abc import ABC, abstractmethod

from torch.nn import Linear, Module, Sequential, Tanh


def make_tanh_classifier_head(d_model: int) -> Sequential:
    """Binary classifier head: Linear -> Tanh -> Linear.

    The caller typically adds a Sigmoid at the end inside a larger Sequential.
    Here we return the full head with a Sigmoid-compatible last Linear.

    Layout (indices):
      0: Linear(d_model, d_model)
      1: Tanh()
      2: Linear(d_model, 1)  # final linear, useful for diagnostics
    """
    return Sequential(Linear(d_model, d_model), Tanh(), Linear(d_model, 1))


class ModelAccess(ABC):
    name: str
    backbone: Module
    epochs: int
    lr: float
    momentum: float

    def __init__(self, name: str, backbone: Module, *, epochs: int, lr: float, momentum: float = 0.9) -> None:
        self.name = name
        self.backbone = backbone
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum

    def forward(self, x):  # passthrough to module
        return self.backbone(x)

    @abstractmethod
    def final_linear(self) -> Linear:
        raise NotImplementedError
