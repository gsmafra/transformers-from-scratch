from typing import Tuple

import torch
from torch import manual_seed, randn, Tensor


def prepare_data(sequence_length: int, n_samples: int, seed: int = 0) -> Tuple[Tensor, Tensor]:
    """Generate sequences and labels for a mixed-rule scoring task.

    Rules (0-based indexing):
    - Odd indices (1, 3, ...): condition is x_t > 0
    - Even indices (0, 2, ...): condition is |x_t| > 0.67448975
    Scoring: +1 if the index's condition is satisfied, -1 otherwise (0 contributes 0).
    Label: y = 1 if the total score across timesteps is > 0, else 0.

    Returns:
      x: (n_samples, sequence_length, 1) drawn from N(0, 1)
      y: (n_samples,)
    """
    manual_seed(seed)

    # Zero-mean data keeps the task roughly balanced
    x = randn(n_samples, sequence_length, 1)

    x2d = x.squeeze(-1)  # (n, T)
    idx = torch.arange(sequence_length, device=x.device)
    odd_mask = (idx % 2 == 1)
    even_mask = ~odd_mask

    # Scores: +1 if condition met, -1 otherwise
    odd_scores = torch.where(x2d[:, odd_mask] > 0, torch.ones_like(x2d[:, odd_mask]), -torch.ones_like(x2d[:, odd_mask]))
    even_scores = torch.where(
        torch.abs(x2d[:, even_mask]) > 0.67448975,
        torch.ones_like(x2d[:, even_mask]),
        -torch.ones_like(x2d[:, even_mask]),
    )

    points = odd_scores.sum(dim=1) + even_scores.sum(dim=1)
    y = (points > 0).float()

    return x, y
