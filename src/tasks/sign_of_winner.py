from typing import Tuple

import torch
from torch import Tensor, randn

from .base import Task


class SignOfWinnerTask(Task):
    def label(self, x: Tensor) -> Tensor:
        x2d = x.sum(dim=-1)
        winner_idx = torch.argmax(torch.abs(x2d), dim=1)
        winner_vals = x2d.gather(1, winner_idx.unsqueeze(1)).squeeze(1)
        return (winner_vals > 0).float()

    def generate_candidates(self, n: int, T: int, F: int) -> Tuple[Tensor, Tensor]:
        x = randn(n, T, F)
        y = self.label(x)
        return x, y
