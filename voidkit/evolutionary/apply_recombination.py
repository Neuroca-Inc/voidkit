"""Recombination operators."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_recombination(
    weights1: np.ndarray,
    weights2: np.ndarray,
    recombination_prob: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Perform elementwise crossover between equal-shaped parent arrays."""
    first = np.asarray(weights1)
    second = np.asarray(weights2)
    if first.shape != second.shape:
        raise ValueError("Weight arrays must have the same shape.")
    if not 0.0 <= recombination_prob <= 1.0:
        raise ValueError("recombination_prob must satisfy 0 <= p <= 1.")
    generator = rng if rng is not None else np.random.default_rng()
    choose_first = generator.random(first.shape) < recombination_prob
    return np.where(choose_first, first, second)
