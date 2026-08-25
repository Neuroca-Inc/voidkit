"""Causal and directed-information adapters."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def granger_causality(
    data: np.ndarray,
    max_lag: int,
    test: str = "ssr_chi2test",
) -> Dict[str, Any]:
    """Run Statsmodels' bivariate Granger-causality lag tests."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("data must have shape (n_observations, 2).")
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    results = grangercausalitytests(values, maxlag=max_lag, verbose=False)
    if any(test not in lag_result[0] for lag_result in results.values()):
        raise ValueError(f"Unknown Granger test key: {test!r}.")
    return results


def _discretize_equal_width(values: np.ndarray, n_bins: int) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return np.zeros(values.size, dtype=np.int64)
    internal_edges = np.linspace(minimum, maximum, n_bins + 1)[1:-1]
    return np.searchsorted(internal_edges, values, side="right").astype(np.int64)


def calculate_transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    n_bins: int = 10,
) -> float:
    """Estimate discrete transfer entropy from source ``x`` to target ``y``.

    The estimator discretizes both series into equal-width bins and evaluates
    ``I(Y_t ; X_(t-lag) | Y_(t-lag))`` with the plug-in empirical distribution.
    """
    source = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Time series must be one-dimensional.")
    if source.size != target.size:
        raise ValueError("Time series must have the same length.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Time series must contain only finite values.")
    if lag < 1 or lag >= source.size:
        raise ValueError("lag must satisfy 1 <= lag < len(x).")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    x_binned = _discretize_equal_width(source, n_bins)
    y_binned = _discretize_equal_width(target, n_bins)

    y_t = y_binned[lag:]
    y_past = y_binned[:-lag]
    x_past = x_binned[:-lag]
    n_obs = y_t.size

    triple = np.zeros((n_bins, n_bins, n_bins), dtype=float)
    ypast_xpast = np.zeros((n_bins, n_bins), dtype=float)
    yt_ypast = np.zeros((n_bins, n_bins), dtype=float)
    ypast = np.zeros(n_bins, dtype=float)

    np.add.at(triple, (y_t, y_past, x_past), 1.0)
    np.add.at(ypast_xpast, (y_past, x_past), 1.0)
    np.add.at(yt_ypast, (y_t, y_past), 1.0)
    np.add.at(ypast, y_past, 1.0)

    triple /= n_obs
    ypast_xpast /= n_obs
    yt_ypast /= n_obs
    ypast /= n_obs

    positive = np.argwhere(triple > 0.0)
    te = 0.0
    for i, j, k in positive:
        p_cond_source = triple[i, j, k] / ypast_xpast[j, k]
        p_cond_target = yt_ypast[i, j] / ypast[j]
        te += triple[i, j, k] * np.log2(p_cond_source / p_cond_target)
    return float(max(0.0, te))
