"""Fixed-point finding for continuous-time dynamical systems."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from scipy.optimize import root


def find_fixed_points(
    func: Callable[[np.ndarray], np.ndarray],
    initial_guesses: Iterable[np.ndarray],
    tol: float = 1e-9,
    dedup_tol: float = 1e-7,
) -> np.ndarray:
    """Find converged equilibria ``func(x) = 0`` from multiple initial guesses.

    Failed nonlinear solves are rejected rather than returned as fixed points.
    Converged roots that differ by at most ``dedup_tol`` are returned once.
    """
    if tol <= 0.0 or dedup_tol <= 0.0:
        raise ValueError("tol and dedup_tol must be positive.")

    guesses = [np.atleast_1d(np.asarray(g, dtype=float)) for g in initial_guesses]
    if not guesses:
        return np.empty((0, 0), dtype=float)

    dimension = guesses[0].size
    if dimension == 0:
        raise ValueError("Initial guesses must be non-empty.")
    if any(g.ndim != 1 or g.size != dimension for g in guesses):
        raise ValueError("All initial guesses must be one-dimensional and equal-sized.")
    if any(not np.all(np.isfinite(g)) for g in guesses):
        raise ValueError("Initial guesses must contain only finite values.")

    roots: list[np.ndarray] = []
    for guess in guesses:
        result = root(func, guess, tol=tol)
        candidate = np.asarray(result.x, dtype=float)
        residual = np.atleast_1d(np.asarray(func(candidate), dtype=float))
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))

        if not result.success or not np.isfinite(residual_norm) or residual_norm > tol:
            continue
        if not any(np.linalg.norm(candidate - existing, ord=np.inf) <= dedup_tol for existing in roots):
            roots.append(candidate)

    if not roots:
        return np.empty((0, dimension), dtype=float)
    return np.vstack(roots)
