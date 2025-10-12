from typing import Dict

import torch
from torch import Tensor


def summarize_stats(name: str, t: Tensor) -> Dict[str, float]:
    """Return mean/std/maxabs for a tensor, keyed by `<name>_...`.

    Uses population std (unbiased=False). Returns empty dict if no elements.
    """
    if not isinstance(t, Tensor):
        return {}
    flat = t.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {}
    return {
        f"{name}_mean": float(flat.mean().item()),
        f"{name}_std": float(flat.std(unbiased=False).item()),
        f"{name}_maxabs": float(flat.abs().max().item()),
    }

