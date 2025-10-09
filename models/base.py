from abc import ABC, abstractmethod
from typing import Optional

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
    momentum: float
    lr_start: Optional[float]
    lr_end: Optional[float]

    def __init__(
        self,
        name: str,
        backbone: Module,
        *,
        epochs: int,
        momentum: float = 0.9,
        lr_start: float = 0.1,
        lr_end: float = 0.1,
    ) -> None:
        self.name = name
        self.backbone = backbone
        self.epochs = epochs
        self.momentum = momentum
        self.lr_start = lr_start
        self.lr_end = lr_end

    def forward(self, x):  # passthrough to module
        return self.backbone(x)

    @abstractmethod
    def final_linear(self) -> Linear:
        raise NotImplementedError
