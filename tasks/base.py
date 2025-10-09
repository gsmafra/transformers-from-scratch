from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor


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
