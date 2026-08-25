"""Power-law diagnostics."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_power_law(data: np.ndarray) -> Tuple[float, float]:
    """Estimate a continuous power-law exponent and a CCDF fit diagnostic.

    Returns ``(alpha, r_squared)`` for ``p(x) proportional to x**(-alpha)`` using
    the continuous maximum-likelihood estimator with ``xmin = min(data)``. The
    reported R² compares the empirical log-CCDF to the corresponding fitted
    asymptotic CCDF and is a diagnostic, not a goodness-of-fit hypothesis test.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("data must be a one-dimensional array with at least 3 samples.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Power-law samples must be finite and strictly positive.")

    xmin = float(np.min(values))
    log_ratios = np.log(values / xmin)
    denominator = float(np.sum(log_ratios))
    if denominator <= 0.0:
        raise ValueError("Power-law fitting requires at least two distinct positive values.")

    alpha = 1.0 + values.size / denominator

    unique = np.unique(values)
    if unique.size < 2:
        return float(alpha), 1.0
    empirical_ccdf = np.array([np.mean(values >= x) for x in unique], dtype=float)
    predicted_ccdf = (unique / xmin) ** (-(alpha - 1.0))

    observed = np.log(empirical_ccdf)
    predicted = np.log(predicted_ccdf)
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(alpha), float(r_squared)
