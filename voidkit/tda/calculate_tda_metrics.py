"""Persistence-diagram summary metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def calculate_tda_metrics(persistence_diagrams: List[np.ndarray]) -> Dict[str, float]:
    """Summarize H0/H1 persistence diagrams without letting essential bars pollute totals."""
    if not isinstance(persistence_diagrams, list) or not persistence_diagrams:
        raise TypeError("persistence_diagrams must be a non-empty list of arrays.")

    diagrams = []
    for diagram in persistence_diagrams:
        values = np.asarray(diagram, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Each persistence diagram must have shape (n_features, 2).")
        if np.any(np.isnan(values)):
            raise ValueError("Persistence diagrams may not contain NaN values.")
        diagrams.append(values)

    h0 = diagrams[0]
    essential_h0 = int(np.sum(np.isinf(h0[:, 1]))) if h0.size else 0

    total_b1 = 0.0
    essential_b1 = 0
    if len(diagrams) > 1 and diagrams[1].size:
        h1 = diagrams[1]
        finite = np.isfinite(h1[:, 1])
        total_b1 = float(np.sum(h1[finite, 1] - h1[finite, 0]))
        essential_b1 = int(np.sum(~finite))

    return {
        "component_count": essential_h0,
        "essential_h0_count": essential_h0,
        "total_b1_persistence": total_b1,
        "essential_b1_count": essential_b1,
    }
