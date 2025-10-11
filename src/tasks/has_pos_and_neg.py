from torch import Tensor, randn

from .base import Task


class HasPosAndNegTask(Task):
    def label(self, x: Tensor) -> Tensor:
        x2d = x.sum(dim=-1)
        has_pos = (x2d > 0).any(dim=1)
        has_neg = (x2d < 0).any(dim=1)
        return (has_pos & has_neg).float()

    def generate_candidates(self, n: int, T: int, F: int):
        x = randn(n, T, F)
        y = self.label(x)
        return x, y

