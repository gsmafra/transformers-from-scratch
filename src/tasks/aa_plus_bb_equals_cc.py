from typing import Tuple

from torch import Tensor

from .base import Task
from .multi_digit_sum import MultiDigitSumTask
from .arithmetic_common import TOKENS as _TOKENS


class AAPlusBBEqualsCCTask(Task):
    """Wrapper task for the equation aa + bb = cc using token format.

    Delegates to MultiDigitSumTask with D=2, N_left=2, N_right=1.
    """

    feature_dim: int = len(_TOKENS)

    def __init__(self) -> None:
        super().__init__()
        self._inner = MultiDigitSumTask(D=2, N_left=2, N_right=1)
        # Align sequence length with the inner task
        self.sequence_length = self._inner.sequence_length

    def label(self, x: Tensor) -> Tensor:
        return self._inner.label(x)

    def generate_candidates(self, n: int, T: int) -> Tuple[Tensor, Tensor]:
        return self._inner.generate_candidates(n, T)
