from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
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
    lr_start: float
    lr_end: float

    def __init__(
        self,
        name: str,
        backbone: Module,
        *,
        epochs: int,
        lr_start: Optional[float] = None,
        lr_end: Optional[float] = None,
    ) -> None:
        self.name = name
        self.backbone = backbone
        self.epochs = epochs
        # Coalesce schedule to valid floats
        base = 0.1 if lr_start is None else float(lr_start)
        self.lr_start = base
        self.lr_end = base if lr_end is None else float(lr_end)

        # Initialize module weights
        self._init_weights()

    def forward(self, x):  # passthrough to module
        return self.backbone(x)

    @abstractmethod
    def final_linear(self) -> Linear:
        raise NotImplementedError

    def _init_weights(self) -> None:
        """Apply Xavier initialization to Linear layers and zero biases.

        This standardizes initialization across all models built on ModelAccess.
        """
        for m in self.backbone.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    # Extra scalar metrics derived from model internals; override in subclasses
    def extra_metrics(self, x) -> Dict[str, float]:
        return {}
