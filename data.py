from typing import Tuple

import torch
from torch import manual_seed, rand, randn, tensor, Tensor, where
from torch.distributions import Normal


def prepare_data(sequence_length: int, n_samples: int, seed: int = 0) -> Tuple[Tensor, Tensor]:
    """Generate sequences and noisy labels.

    - X ~ Normal(mu, 1) with mu chosen so P(any x > 0) ≈ 0.5
    - y = 1 if any timestep > 0, else 0
    - 10% label noise (flip)
    Returns:
      x: (n_samples, sequence_length, 1)
      y: (n_samples,)
    """
    manual_seed(seed)

    # Choose per-timestep positive rate p so that 1 - (1 - p)^T ≈ 0.5
    p_target = 1.0 - 0.5 ** (1.0 / float(sequence_length))
    mu = Normal(0.0, 1.0).icdf(tensor(p_target)).item()

    x = randn(n_samples, sequence_length, 1) + mu
    y = (x.squeeze(-1).max(dim=1).values > 0).float()

    # Inject label noise: flip label with 10% probability per sample
    flip = (rand(n_samples) < 0.10).float()
    y = where(flip > 0, 1.0 - y, y)

    return x, y

