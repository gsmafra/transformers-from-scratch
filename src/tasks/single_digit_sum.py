from typing import Tuple

from torch import Tensor

from .arithmetic_common import (
    TOKENS as _TOKENS,
    TOK2IDX as _TOK2IDX,
    FORM_A_PLUS_B_EQ_C,
    label_equations_from_indices,
    generate_candidates_equations,
)
from .base import Task


class SingleDigitSumTask(Task):
    feature_dim: int = len(_TOKENS)

    def label(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(1) != 5 or x.size(2) < self.feature_dim:
            raise ValueError("Expected shape (N, 5, V>=12) with one-hot tokens")

        idx = x.argmax(dim=2)  # (N, 5)
        return label_equations_from_indices(idx, [FORM_A_PLUS_B_EQ_C])

    def generate_candidates(self, n: int, T: int) -> Tuple[Tensor, Tensor]:
        """Generate pairs: for each random (a,b) with a+b <= 9, emit one correct and one incorrect sample.

        If a+b > 9, discard that pair (emit nothing). Truncate to n after shuffling.
        Output one-hot tokens with shape (N, 5, self.feature_dim).
        """
        return generate_candidates_equations(n, T, [FORM_A_PLUS_B_EQ_C])
