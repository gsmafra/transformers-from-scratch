from typing import Callable, Dict, Tuple, Optional

import torch
from torch import Tensor, manual_seed, randn


# --- Task collection --------------------------------------------------------

def label_sign_of_winner(x2d: Tensor) -> Tensor:
    """y=1 if the argmax of |x_t| is positive, else 0."""
    winner_idx = torch.argmax(torch.abs(x2d), dim=1)  # (n,)
    winner_vals = x2d.gather(1, winner_idx.unsqueeze(1)).squeeze(1)  # (n,)
    return (winner_vals > 0).float()


def label_has_pos_and_neg(x2d: Tensor) -> Tensor:
    """y=1 if there is at least one positive AND one negative in the sequence, else 0."""
    has_pos = (x2d > 0).any(dim=1)
    has_neg = (x2d < 0).any(dim=1)
    return (has_pos & has_neg).float()


TASKS: Dict[str, Callable[[Tensor], Tensor]] = {
    "sign_of_winner": label_sign_of_winner,
    "has_pos_and_neg": label_has_pos_and_neg,
}

# Set which task the code points to by default
DEFAULT_TASK = "has_pos_and_neg"


def prepare_data(
    sequence_length: int,
    n_samples: int,
    seed: int = 0,
    *,
    task: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Generate sequences and labels for a selected dummy task.

    Tasks:
      - sign_of_winner: y=1 if argmax |x_t| is positive else 0
      - has_pos_and_neg: y=1 if at least one positive and one negative exist

    Returns:
      x: (n_samples, sequence_length, 1) from N(0, 1)
      y: (n_samples,)
    """
    manual_seed(seed)

    task_name = task or DEFAULT_TASK
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")

    # Special balanced generation for has_pos_and_neg
    if task_name == "has_pos_and_neg":
        x2d, y = _generate_balanced_has_pos_and_neg(n_samples, sequence_length)
        return x2d.unsqueeze(-1), y

    # Default: iid Gaussian sequences + task labeler
    x = randn(n_samples, sequence_length, 1)
    x2d = x.squeeze(-1)  # (n, T)
    y = TASKS[task_name](x2d)
    return x, y


def _generate_balanced_has_pos_and_neg(n_samples: int, sequence_length: int) -> Tuple[Tensor, Tensor]:
    """Construct a 50/50 balanced dataset for the has_pos_and_neg task.

    Half the samples contain at least one positive and one negative; half are
    single-sign (all positive or all negative). Returns `(x2d, y)` where
    `x2d` has shape `(n_samples, T)` and `y` is `(n_samples,)` of floats.
    """
    T = sequence_length
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # Positive class: sequences with both signs
    mags_pos = torch.abs(randn(n_pos, T))
    signs_pos = torch.sign(randn(n_pos, T))  # +/-1 almost surely
    # Fix any rows that are all one sign by flipping one random position
    all_same = signs_pos.abs().sum(dim=1) == T
    if all_same.any():
        idx = torch.nonzero(all_same, as_tuple=False).squeeze(1)
        j = torch.randint(0, T, (idx.numel(),))
        signs_pos[idx, j] *= -1
    x_pos = mags_pos * signs_pos

    # Negative class: sequences with a single sign
    mags_neg = torch.abs(randn(n_neg, T))
    row_signs = (torch.randint(0, 2, (n_neg,)) * 2 - 1).float().unsqueeze(1)  # +/-1 per row
    x_neg = mags_neg * row_signs

    # Stack and shuffle
    x2d = torch.cat([x_pos, x_neg], dim=0)
    y = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    perm = torch.randperm(n_samples)
    return x2d[perm], y[perm]
