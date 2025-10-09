import torch
from torch import Tensor

from .base import Task


class HasAllTokensTask(Task):
    def label(self, x: Tensor) -> Tensor:
        present = (x.sum(dim=1) > 0).all(dim=1)
        return present.float()

    def generate_candidates(self, n: int, T: int, V: int):
        if V < 2:
            raise ValueError("has_all_tokens requires n_features (vocab_size) >= 2")
        if T < V:
            raise ValueError("sequence_length must be >= n_features for 'has_all_tokens'")

        tokens = torch.randint(0, V, (n, T))
        x = torch.zeros(tokens.size(0), T, V)
        x.scatter_(2, tokens.unsqueeze(-1).long(), 1.0)
        y = self.label(x)
        return x, y

