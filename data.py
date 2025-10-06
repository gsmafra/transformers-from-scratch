from typing import Callable, Dict, Tuple, Optional
from abc import ABC, abstractmethod

import torch
from torch import Tensor, manual_seed, randn


# Set which task the code points to by default
DEFAULT_TASK = "sign_of_second_place"


def prepare_data(
    sequence_length: int,
    n_samples: int,
    seed: int = 0,
    *,
    task: Optional[str] = None,
    n_features: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Generate sequences and labels for a selected dummy task.

    Inputs have `n_features` per timestep. Each Task implements `label(x)` so
    labeling logic lives within the task, not as free functions.

    Returns `(x, y)` with shapes `(n_samples, sequence_length, n_features)` and
    `(n_samples,)` respectively.
    """
    manual_seed(seed)

    task_name = task or DEFAULT_TASK

    TASK_REGISTRY: Dict[str, "Task"] = {
        "sign_of_winner": SignOfWinnerTask(),
        "sign_of_second_place": SignOfSecondPlaceTask(),
        "has_pos_and_neg": HasPosAndNegTask(),
        "has_all_tokens": HasAllTokensTask(),
    }

    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}")

    # Generate candidates then stratified sample 50/50 with replacement
    n_cand = max(10 * n_samples, 512)
    x_cand, y_cand = TASK_REGISTRY[task_name].generate_candidates(n_cand, sequence_length, n_features)
    # If a class is missing, fail fast and surface a clear error
    if (y_cand > 0.5).sum() == 0 or (y_cand <= 0.5).sum() == 0:
        raise ValueError("Candidate pool missing a class; increase candidate size or adjust task parameters.")
    return stratified_sample_balanced(x_cand, y_cand, n_samples)

# --- Task classes -----------------------------------------------------------

class Task(ABC):
    @abstractmethod
    def label(self, x: Tensor) -> Tensor:
        """Compute labels for inputs `x`.

        x: (n, T, F) features (or one-hot token features for token tasks)
        return: (n,) floats in {0.0, 1.0}
        """
        raise NotImplementedError

    @abstractmethod
    def generate_candidates(self, n: int, T: int, F: int) -> Tuple[Tensor, Tensor]:
        """Return candidate inputs x and labels y.

        x: (n, T, F) float features or one-hot token features
        y: (n,) floats in {0.0, 1.0}
        """
        raise NotImplementedError


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


class SignOfSecondPlaceTask(Task):
    def label(self, x: Tensor) -> Tensor:
        # Sum features per timestep, then take the sign of the 2nd largest |value|
        x2d = x.sum(dim=-1)  # (n, T)
        # Indices of the top-2 by absolute value per row
        _, topk_idx = x2d.abs().topk(k=2, dim=1, largest=True, sorted=True)
        second_idx = topk_idx[:, 1]  # (n,)
        second_vals = x2d.gather(1, second_idx.unsqueeze(1)).squeeze(1)
        return (second_vals > 0).float()

    def generate_candidates(self, n: int, T: int, F: int) -> Tuple[Tensor, Tensor]:
        if T < 2:
            raise ValueError("sign_of_second_place requires sequence_length >= 2")
        x = randn(n, T, F)
        y = self.label(x)
        return x, y


class HasPosAndNegTask(Task):
    def label(self, x: Tensor) -> Tensor:
        x2d = x.sum(dim=-1)
        has_pos = (x2d > 0).any(dim=1)
        has_neg = (x2d < 0).any(dim=1)
        return (has_pos & has_neg).float()

    def generate_candidates(self, n: int, T: int, F: int) -> Tuple[Tensor, Tensor]:
        x = randn(n, T, F)
        y = self.label(x)
        return x, y


class HasAllTokensTask(Task):
    def label(self, x: Tensor) -> Tensor:
        present = (x.sum(dim=1) > 0).all(dim=1)
        return present.float()

    def generate_candidates(self, n: int, T: int, V: int) -> Tuple[Tensor, Tensor]:
        if V < 2:
            raise ValueError("has_all_tokens requires n_features (vocab_size) >= 2")
        if T < V:
            raise ValueError("sequence_length must be >= n_features for 'has_all_tokens'")

        tokens = torch.randint(0, V, (n, T))
        x = torch.zeros(tokens.size(0), T, V)
        x.scatter_(2, tokens.unsqueeze(-1).long(), 1.0)
        y = self.label(x)
        return x, y


def stratified_sample_balanced(x_cand: Tensor, y_cand: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    target_pos = n // 2
    target_neg = n - target_pos

    pos_idx = torch.nonzero(y_cand > 0.5, as_tuple=False).squeeze(1)
    neg_idx = torch.nonzero(y_cand <= 0.5, as_tuple=False).squeeze(1)

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        raise ValueError("Candidate pool missing a class; increase n_cand or adjust seed.")

    pos_sel = pos_idx[torch.randint(0, pos_idx.numel(), (target_pos,))]
    neg_sel = neg_idx[torch.randint(0, neg_idx.numel(), (target_neg,))]

    x = torch.cat([x_cand[pos_sel], x_cand[neg_sel]], dim=0)
    y = torch.cat([torch.ones(target_pos), torch.zeros(target_neg)], dim=0)
    perm = torch.randperm(n)
    return x[perm], y[perm]
