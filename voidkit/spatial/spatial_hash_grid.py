"""N-dimensional spatial hashing for local-radius queries."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, DefaultDict, List, Optional, Tuple

import numpy as np


class SpatialHashGrid:
    """Sparse spatial hash supporting exact Euclidean radius queries."""

    def __init__(self, cell_size: float):
        if not np.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("cell_size must be a positive finite value.")
        self.cell_size = float(cell_size)
        self.grid: DefaultDict[Tuple[int, ...], List[Tuple[np.ndarray, Any]]] = defaultdict(list)
        self._dimension: Optional[int] = None

    def _coerce_point(self, point: np.ndarray) -> np.ndarray:
        value = np.asarray(point, dtype=float)
        if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("point must be a non-empty finite one-dimensional array.")
        if self._dimension is None:
            self._dimension = value.size
        elif value.size != self._dimension:
            raise ValueError(f"point dimension must remain {self._dimension}.")
        return value

    def _hash(self, point: np.ndarray) -> Tuple[int, ...]:
        value = self._coerce_point(point)
        return tuple(np.floor(value / self.cell_size).astype(int))

    def insert(self, point: np.ndarray, obj: Any) -> None:
        value = self._coerce_point(point).copy()
        self.grid[self._hash(value)].append((value, obj))

    def query(self, point: np.ndarray, radius: float) -> List[Any]:
        center = self._coerce_point(point)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be a non-negative finite value.")

        min_cell = np.floor((center - radius) / self.cell_size).astype(int)
        max_cell = np.floor((center + radius) / self.cell_size).astype(int)
        ranges = [range(lo, hi + 1) for lo, hi in zip(min_cell, max_cell)]

        results: List[Any] = []
        radius_squared = float(radius) ** 2
        for cell in product(*ranges):
            for stored_point, obj in self.grid.get(tuple(cell), []):
                if float(np.sum((stored_point - center) ** 2)) <= radius_squared:
                    results.append(obj)
        return results

    def get_collisions(self, point: np.ndarray) -> List[Any]:
        """Return objects occupying the same hash cell as ``point``."""
        return [obj for _, obj in self.grid.get(self._hash(point), [])]
