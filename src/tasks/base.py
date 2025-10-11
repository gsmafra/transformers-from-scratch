from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor


class Task(ABC):
    # Default per-timestep feature dimension. Token tasks override this.
    feature_dim: int = 2
    @abstractmethod
    def label(self, x: Tensor) -> Tensor:
        """Compute labels for inputs `x`.

        x: (n, T, F) features (or one-hot token features for token tasks)
        return: (n,) floats in {0.0, 1.0}
        """
        raise NotImplementedError

    @abstractmethod
    def generate_candidates(self, n: int, T: int) -> Tuple[Tensor, Tensor]:
        """Return candidate inputs x and labels y using this task's feature_dim.

        x: (n, T, F) float or one-hot features where F == self.feature_dim
        y: (n,) floats in {0.0, 1.0}
        """
        raise NotImplementedError
