"""Fractional-calculus operators."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


def caputo_derivative(
    f: np.ndarray,
    alpha: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Approximate the Caputo derivative of order ``alpha`` on a uniform grid.

    The implementation uses the classical L1 discretization for ``0 < alpha < 1``.
    For samples ``f[n] = f(t_n)`` with ``t_n = n * dt``, the approximation is

    ``D_C^alpha f(t_n) ~= dt^-alpha / Gamma(2-alpha) *
    sum_k b_k (f[n-k] - f[n-k-1])``

    with ``b_k = (k+1)^(1-alpha) - k^(1-alpha)``.

    This formulation preserves the defining Caputo property that constants have
    zero fractional derivative.
    """
    values = np.asarray(f, dtype=float)
    if values.ndim != 1:
        raise ValueError("f must be a one-dimensional array.")
    if values.size == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("f must contain only finite values.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value.")

    result = np.zeros_like(values, dtype=float)
    increments = np.diff(values)
    prefactor = dt ** (-alpha) / gamma(2.0 - alpha)

    for n in range(1, values.size):
        k = np.arange(n, dtype=float)
        weights = (k + 1.0) ** (1.0 - alpha) - k ** (1.0 - alpha)
        result[n] = prefactor * np.dot(weights, increments[n - 1 :: -1])

    return result
