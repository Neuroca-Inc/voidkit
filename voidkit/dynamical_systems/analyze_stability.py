"""Local linear stability classification for continuous-time systems."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def analyze_stability(jacobian: np.ndarray, tol: float = 1e-10) -> Dict[str, Any]:
    """Classify a fixed point from Jacobian eigenvalues.

    Hyperbolic equilibria are classified from eigenvalue real parts. If one or
    more eigenvalues have real part within ``tol`` of zero, linearization alone
    is not sufficient to establish nonlinear stability and the result is marked
    nonhyperbolic/inconclusive unless a positive-real-part mode already proves
    instability.
    """
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("jacobian must be a square matrix.")
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be non-empty and finite.")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    eigenvalues = np.linalg.eigvals(matrix)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    has_positive = np.any(real_parts > tol)
    has_negative = np.any(real_parts < -tol)
    has_neutral = np.any(np.abs(real_parts) <= tol)
    has_complex = np.any(np.abs(imag_parts) > tol)

    if has_positive and has_negative:
        stability_type = "Saddle Point"
    elif has_positive:
        if has_neutral:
            stability_type = "Unstable (Nonhyperbolic)"
        else:
            stability_type = "Unstable Spiral" if has_complex else "Unstable Node"
    elif has_neutral:
        stability_type = "Nonhyperbolic (Linearization Inconclusive)"
    else:
        stability_type = "Stable Spiral" if has_complex else "Stable Node"

    return {
        "eigenvalues": eigenvalues,
        "stability_type": stability_type,
        "hyperbolic": not has_neutral,
    }
