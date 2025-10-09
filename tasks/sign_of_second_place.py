from torch import Tensor, randn

from .base import Task


class SignOfSecondPlaceTask(Task):
    def label(self, x: Tensor) -> Tensor:
        # Sum features per timestep, then take the sign of the 2nd largest |value|
        x2d = x.sum(dim=-1)  # (n, T)
        # Indices of the top-2 by absolute value per row
        _, topk_idx = x2d.abs().topk(k=2, dim=1, largest=True, sorted=True)
        second_idx = topk_idx[:, 1]  # (n,)
        second_vals = x2d.gather(1, second_idx.unsqueeze(1)).squeeze(1)
        return (second_vals > 0).float()

    def generate_candidates(self, n: int, T: int, F: int):
        if T < 2:
            raise ValueError("sign_of_second_place requires sequence_length >= 2")
        x = randn(n, T, F)
        y = self.label(x)
        return x, y

