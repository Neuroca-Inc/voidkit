"""Persistent-homology adapter with an optional Ripser dependency."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_persistent_homology(
    data: np.ndarray,
    max_dim: int = 1,
    is_distance_matrix: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute persistent homology for a point cloud or distance matrix via Ripser."""
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError(
            "compute_persistent_homology requires the optional 'ripser' dependency; "
            "install VoidKit with the 'tda' extra."
        ) from exc

    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values.")
    if max_dim < 0:
        raise ValueError("max_dim must be non-negative.")
    if is_distance_matrix:
        if values.shape[0] != values.shape[1]:
            raise ValueError("A distance matrix must be square.")
        if not np.allclose(values, values.T, atol=1e-10, rtol=1e-8):
            raise ValueError("A distance matrix must be symmetric.")
        if np.any(values < 0.0) or not np.allclose(np.diag(values), 0.0, atol=1e-10):
            raise ValueError("A distance matrix must be non-negative with a zero diagonal.")

    return ripser(values, maxdim=max_dim, distance_matrix=is_distance_matrix)
