from typing import Optional, Tuple

from torch import Tensor, manual_seed

from .registry import TASK_REGISTRY
from .sampling import stratified_sample_balanced


# Set which task the code points to by default
DEFAULT_TASK = "single_digit_string_sum"


def prepare_data(
    sequence_length: int,
    n_samples: int,
    seed: int = 0,
    *,
    task: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Generate sequences and labels for a selected task.

    Returns `(x, y)` with shapes `(n_samples, sequence_length, F)` and `(n_samples,)`.
    Task determines its feature dimension F via the `feature_dim` attribute.
    """
    manual_seed(seed)

    task_name = task or DEFAULT_TASK

    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}")

    # Generate candidates then stratified sample 50/50 with replacement
    n_cand = max(10 * n_samples, 512)
    x_cand, y_cand = TASK_REGISTRY[task_name].generate_candidates(n_cand, sequence_length)
    # If a class is missing, fail fast and surface a clear error
    if (y_cand > 0.5).sum() == 0 or (y_cand <= 0.5).sum() == 0:
        raise ValueError("Candidate pool missing a class; increase candidate size or adjust task parameters.")
    return stratified_sample_balanced(x_cand, y_cand, n_samples)
