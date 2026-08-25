"""Mutation operators."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_mutation(
    weights: np.ndarray,
    mutation_rate: float = 0.01,
    mutation_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply independent Gaussian mutations to an array."""
    values = np.asarray(weights)
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must satisfy 0 <= mutation_rate <= 1.")
    if not np.isfinite(mutation_scale) or mutation_scale < 0.0:
        raise ValueError("mutation_scale must be non-negative and finite.")
    generator = rng if rng is not None else np.random.default_rng()
    mask = generator.random(values.shape) < mutation_rate
    mutations = generator.normal(0.0, mutation_scale, values.shape)
    result = values.copy()
    result[mask] = result[mask] + mutations[mask]
    return result
