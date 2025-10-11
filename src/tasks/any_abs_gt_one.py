from torch import Tensor, randn

from .base import Task


class AnyAbsGreaterThanOneTask(Task):
    def label(self, x: Tensor) -> Tensor:
        # Positive if any raw element exceeds unit magnitude
        return (x.abs() > 1.0).any(dim=(1, 2)).float()

    def generate_candidates(self, n: int, T: int, F: int):
        x = randn(n, T, F)
        y = self.label(x)
        return x, y

