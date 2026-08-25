from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegressionDataset:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional arrays.")
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("x and y must have the same positive length.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("x and y must contain only finite values.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @staticmethod
    def demo(n: int = 121) -> "RegressionDataset":
        x = np.linspace(-2.0, 2.0, n)
        y = 0.75 * x**3 - 1.25 * x**2 + 0.5 * x + 2.0
        return RegressionDataset(x=x, y=y)
