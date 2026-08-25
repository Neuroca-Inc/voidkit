"""Fractal-dimension estimators."""

from __future__ import annotations

import numpy as np


def calculate_fractal_dimension(points: np.ndarray, threshold: float = 0.9) -> float:
    """Estimate box-counting dimension of a finite point cloud.

    Coordinates are normalized to the unit box before counting occupied boxes at
    dyadic scales. ``threshold`` is the maximum occupied-box fraction used in the
    regression; it suppresses scales where the count has saturated at roughly one
    occupied box per sample.
    """
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] < 1:
        raise ValueError("points must have shape (n_points, n_features) with at least 4 points.")
    if not np.all(np.isfinite(data)):
        raise ValueError("points must contain only finite values.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must satisfy 0 < threshold <= 1.")

    mins = data.min(axis=0)
    spans = data.max(axis=0) - mins
    active = spans > 0.0
    if not np.any(active):
        return 0.0

    normalized = (data[:, active] - mins[active]) / spans[active]
    n_points = normalized.shape[0]
    max_k = max(3, min(16, int(np.ceil(np.log2(max(n_points, 2)))) + 2))

    inverse_scales: list[float] = []
    counts: list[int] = []
    for k in range(1, max_k + 1):
        inv_eps = float(2**k)
        indices = np.floor(normalized * inv_eps).astype(np.int64)
        indices = np.minimum(indices, int(inv_eps) - 1)
        count = int(np.unique(indices, axis=0).shape[0])
        inverse_scales.append(inv_eps)
        counts.append(count)

    inverse_scales_arr = np.asarray(inverse_scales, dtype=float)
    counts_arr = np.asarray(counts, dtype=float)
    usable = (counts_arr > 1.0) & (counts_arr < threshold * n_points)

    if np.count_nonzero(usable) < 3:
        usable = (counts_arr > 1.0) & (counts_arr < n_points)
    if np.count_nonzero(usable) < 2:
        return 0.0

    slope, _ = np.polyfit(
        np.log(inverse_scales_arr[usable]),
        np.log(counts_arr[usable]),
        1,
    )
    return float(max(0.0, slope))
