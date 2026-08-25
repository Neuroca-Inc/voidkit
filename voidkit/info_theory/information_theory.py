"""Discrete information-theory utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import entropy as scipy_entropy


def _validate_probability_array(values: np.ndarray, *, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Probability array must be {ndim}-dimensional.")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Probabilities must be non-empty and finite.")
    if np.any(array < 0.0):
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(array.sum(), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    return array


def _validate_base(base: float) -> None:
    if not np.isfinite(base) or base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError("base must be positive and different from 1.")


def calculate_entropy(pk: np.ndarray, base: float = 2) -> float:
    """Calculate Shannon entropy of a discrete probability distribution."""
    _validate_base(base)
    distribution = _validate_probability_array(pk, ndim=1)
    return float(scipy_entropy(distribution, base=base))


def calculate_mutual_information(p_xy: np.ndarray, base: float = 2) -> float:
    """Calculate mutual information ``I(X;Y)`` from a joint distribution."""
    _validate_base(base)
    joint = _validate_probability_array(p_xy, ndim=2)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    independent = np.outer(p_x, p_y)
    non_zero = joint > 0.0
    return float(
        np.sum(joint[non_zero] * np.log(joint[non_zero] / independent[non_zero]))
        / np.log(base)
    )


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2) -> float:
    """Calculate ``D_KL(P || Q)`` for equal-sized discrete distributions."""
    _validate_base(base)
    first = _validate_probability_array(p, ndim=1)
    second = _validate_probability_array(q, ndim=1)
    if first.shape != second.shape:
        raise ValueError("Distributions must have the same shape.")
    return float(scipy_entropy(first, second, base=base))
