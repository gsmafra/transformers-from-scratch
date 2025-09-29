from typing import Tuple

import torch
from torch import manual_seed, randn, Tensor


def prepare_data(sequence_length: int, n_samples: int, seed: int = 0) -> Tuple[Tensor, Tensor]:
    """Generate sequences and labels for the "sign of the winner" task.

    Definition (0-based indexing):
    - winner = argmax_t |x_t|
    - y = 1 if x[winner] > 0 else 0

    Returns:
      x: (n_samples, sequence_length, 1) drawn from N(0, 1)
      y: (n_samples,)
    """
    manual_seed(seed)

    # Zero-mean data keeps the task roughly balanced
    x = randn(n_samples, sequence_length, 1)

    x2d = x.squeeze(-1)  # (n, T)
    # Index of the maximum absolute value per sample
    winner_idx = torch.argmax(torch.abs(x2d), dim=1)  # (n,)
    # Gather the winning values and set label by sign
    winner_vals = x2d.gather(1, winner_idx.unsqueeze(1)).squeeze(1)  # (n,)
    y = (winner_vals > 0).float()

    return x, y
