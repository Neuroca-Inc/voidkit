"""Generic time-series transforms and discrete transfer entropy.
`transfer_entropy_discrete` returns bits, matching the source implementation.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

def safe_zscore(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def implied_timescale(lam: float) -> float:
    if lam <= 0 or lam >= 1:
        return float("inf")
    return -1.0 / math.log(lam)


def discretize_quantiles(x, n_bins=8):
    x = np.asarray(x)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(x, dtype=int), edges
    bins = np.digitize(x, edges[1:-1], right=False)
    return bins.astype(int), edges


def transfer_entropy_discrete(x, y, lag=1, n_bins=8) -> float:
    """
    TE(X->Y) = I(X_{t-lag}; Y_t | Y_{t-lag}) in bits, using quantile discretization.
    """
    x_b, _ = discretize_quantiles(x, n_bins)
    y_b, _ = discretize_quantiles(y, n_bins)

    x_prev = x_b[:-lag]
    y_prev = y_b[:-lag]
    y_t = y_b[lag:]

    bx = max(x_b.max() + 1, n_bins)
    by = max(y_b.max() + 1, n_bins)

    triple = y_t + by * (y_prev + by * x_prev)
    pair_yx = y_prev + by * x_prev
    pair_yy = y_t + by * y_prev

    ct_triple = Counter(triple)
    ct_pair_yx = Counter(pair_yx)
    ct_pair_yy = Counter(pair_yy)
    ct_y_prev = Counter(y_prev)

    n = len(y_t)
    te = 0.0
    for key, c in ct_triple.items():
        p_xyz = c / n
        y_t_val = key % by
        tmp = key // by
        y_prev_val = tmp % by
        x_prev_val = tmp // by

        denom_yx = ct_pair_yx[y_prev_val + by * x_prev_val]
        p_y_given_yx = c / denom_yx if denom_yx else 0.0

        denom_y = ct_y_prev[y_prev_val]
        p_y_given_y = (ct_pair_yy[y_t_val + by * y_prev_val] / denom_y) if denom_y else 0.0

        te += p_xyz * math.log((p_y_given_yx + 1e-12) / (p_y_given_y + 1e-12))
    return te / math.log(2)


def event_triggered_average(series, event_idxs, window):
    n = len(series)
    w = window
    valid = [idx for idx in event_idxs if idx - w >= 0 and idx + w < n]
    if not valid:
        return None, 0
    mat = np.vstack([series[idx - w: idx + w + 1] for idx in valid])
    return mat.mean(axis=0), len(valid)

transfer_entropy_bits = transfer_entropy_discrete
__all__=["safe_zscore","implied_timescale","discretize_quantiles","transfer_entropy_discrete","transfer_entropy_bits","event_triggered_average"]
